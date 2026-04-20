import json
import re
import os


# ── Keyword Hit Rate ───────────────────────────────────────────────────────────
def keyword_hit_rate(keywords_en: str, answer_hi: str) -> float:
    """
    Check what fraction of gold keywords appear in the Hindi answer.

    WHY on Hindi directly (not back-translated English):
    Government keywords — numbers (34, 5.99), scheme names (AYURGYAN, ABIHR),
    amounts (₹47 crore) — appear VERBATIM in Hindi government text.
    Checking on Hindi avoids back-translation noise entirely.
    """
    if not keywords_en or keywords_en.strip().upper() in ("MISSING", "NAN", ""):
        return None
    keys = [k.strip() for k in re.split(r",\s*|\|", keywords_en) if k.strip()]
    if not keys:
        return None
    low  = (answer_hi or "").lower()
    hits = sum(1 for k in keys if k.lower() in low)
    return round(hits / len(keys), 4)


# ── ROUGE-L ────────────────────────────────────────────────────────────────────
def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1. Both strings should be English (gold vs back-translated answer)."""
    from rouge_score import rouge_scorer
    if not reference or not hypothesis:
        return 0.0
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return round(sc.score(reference, hypothesis)["rougeL"].fmeasure, 4)


# ── BERTScore (multilingual, cross-lingual) ────────────────────────────────────
def bertscore_multilingual_batch(refs: list, hyps: list) -> list:
    """
    Compute BERTScore F1 for a batch.

    refs = English gold answers
    hyps = Hindi model answers

    WHY xlm-roberta-large with lang=None:
    - lang="hi" uses a Hindi-only model, but our refs are English
    - xlm-roberta-large is trained cross-lingually: it places Hindi and
      English semantically equivalent text in nearby vector positions
    - lang=None is required when specifying model_type explicitly
    """
    import evaluate
    if not refs or not hyps:
        return []
    # Clean inputs
    refs = [str(r) if r and str(r).strip() else "empty" for r in refs]
    hyps = [str(h) if h and str(h).strip() else "empty" for h in hyps]

    metric = evaluate.load("bertscore")
    results = metric.compute(
        predictions=hyps,
        references=refs,
        model_type="xlm-roberta-large",
        lang=None,          # required when model_type is specified
        batch_size=32,
        verbose=False,
    )
    return [round(float(f), 4) for f in results["f1"]]


# ── Document Fidelity (chrF) ───────────────────────────────────────────────────
def doc_fidelity_chrf(ref: str, hyp: str) -> float:
    """
    chrF between a machine-translated doc and the official Hindi doc.
    Used to measure translation quality for C2 (Sarvam) and C5 (IndicTrans2).
    Higher = closer to official government Hindi.
    """
    from sacrebleu.metrics import CHRF
    if not ref or not hyp:
        return float("nan")
    return round(CHRF().corpus_score([hyp], [[ref]]).score, 4)


# ── Local LLM Judge (no API) ───────────────────────────────────────────────────
# Uses a small local model via vLLM or HuggingFace transformers.
# No Gemini, no API key required.

_judge_llm  = None
_judge_tok  = None


def _load_judge():
    global _judge_llm, _judge_tok
    if _judge_llm is not None:
        return
    # Use Qwen-2.5-7B-Instruct — small enough to run alongside evaluation
    # Override with JUDGE_MODEL env var if needed
    model_id = os.getenv("JUDGE_MODEL_HF", "Qwen/Qwen2.5-7B-Instruct")
    print(f"  [judge] Loading {model_id} ...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    _judge_tok = AutoTokenizer.from_pretrained(model_id)
    _judge_llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
        trust_remote_code=True,
    )
    print("  [judge] Ready")


JUDGE_PROMPT = """You are evaluating an AI answer for an Indian government parliamentary question.

Gold answer (English): {gold}
Generated answer (English, back-translated from Hindi): {generated}

Rate the generated answer:
- "grounded": all key facts correct (numbers, names, amounts match gold)
- "minor": mostly correct but one detail missing or slightly off
- "major": key facts wrong, hallucinated, or answer is empty/irrelevant

Reply ONLY with valid JSON, no markdown:
{{"label": "grounded" | "minor" | "major", "rationale": "one sentence"}}"""


def llm_judge_hallucination(gold_en: str, back_en: str, api_key: str = "") -> dict:
    """
    Judge factual accuracy using a local LLM (Qwen-2.5-7B).
    api_key param kept for API compatibility but is ignored.
    """
    if not gold_en or not back_en or "BT_ERROR" in str(back_en):
        return {"label": "unknown", "rationale": "missing input"}

    try:
        _load_judge()
        from vllm import SamplingParams

        prompt = JUDGE_PROMPT.format(
            gold=str(gold_en)[:500],
            generated=str(back_en)[:500],
        )
        msgs = [{"role": "user", "content": prompt}]
        formatted = _judge_tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        outputs = _judge_llm.generate(
            [formatted],
            SamplingParams(temperature=0.0, max_tokens=80),
        )
        raw = outputs[0].outputs[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)

    except json.JSONDecodeError:
        # Salvage label from malformed response
        for label in ("grounded", "minor", "major"):
            if label in raw.lower():
                return {"label": label, "rationale": "parsed from malformed JSON"}
        return {"label": "unknown", "rationale": "JSON parse failed"}
    except Exception as e:
        return {"label": "unknown", "rationale": str(e)}


def hallucination_numeric(label: str) -> float:
    """Convert label → numeric score: grounded=1.0, minor=0.5, major=0.0."""
    return {"grounded": 1.0, "minor": 0.5, "major": 0.0}.get(
        (label or "").lower(), float("nan")
    )
