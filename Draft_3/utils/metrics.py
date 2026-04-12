# utils/metrics.py
# All evaluation metrics in one place.
# chrF, BERTScore, Named Entity preservation, Semantic score, Numeric fidelity

import re
from typing import Optional


# ── chrF ─────────────────────────────────────────────────────────────────────

def chrf_score(hypothesis: str, reference: str) -> float:
    """Character n-gram F-score. Range 0-100. Higher = better."""
    try:
        import sacrebleu
        return sacrebleu.sentence_chrf(hypothesis, [reference]).score
    except Exception:
        return 0.0


# ── BERTScore ─────────────────────────────────────────────────────────────────

def bertscore_f1(hypothesis: str, reference: str) -> float:
    """
    Semantic similarity using multilingual BERT.
    Range 0-1. Higher = better.
    Note: runs on CPU if no GPU, still gives valid scores.
    """
    try:
        from bert_score import score
        _, _, F = score([hypothesis], [reference], lang="en", verbose=False)
        return float(F[0])
    except Exception:
        return 0.0


def bertscore_batch(hypotheses: list, references: list) -> list[float]:
    """Batch BERTScore — much faster than one-by-one."""
    try:
        from bert_score import score
        _, _, F = score(hypotheses, references, lang="en", verbose=False)
        return [float(f) for f in F]
    except Exception:
        return [0.0] * len(hypotheses)


# ── Named Entity Preservation ─────────────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def ne_preservation(original: str, back_translated: str) -> Optional[float]:
    """
    What fraction of named entities in the original survive back-translation?
    Returns None if original has no named entities.
    Range 0-1. 1.0 = all entities preserved.
    """
    nlp = _get_nlp()
    orig_ents  = set(e.text.lower() for e in nlp(original).ents)
    back_ents  = set(e.text.lower() for e in nlp(back_translated).ents)
    if not orig_ents:
        return None
    return len(orig_ents & back_ents) / len(orig_ents)


# ── Numeric Fidelity ──────────────────────────────────────────────────────────

def extract_numbers(text: str) -> set:
    """Extract all numbers from text (handles commas like 12,65,764)."""
    # Remove commas from numbers first
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def numeric_fidelity(original: str, back_translated: str) -> Optional[float]:
    """
    What fraction of numbers in the original appear in the back-translation?
    Returns None if original has no numbers.
    """
    orig_nums = extract_numbers(original)
    if not orig_nums:
        return None
    back_nums = extract_numbers(back_translated)
    preserved = orig_nums & back_nums
    return len(preserved) / len(orig_nums)


# ── Acronym Fidelity ──────────────────────────────────────────────────────────

def extract_acronyms(text: str) -> set:
    """Extract uppercase acronyms of 2+ letters."""
    return set(re.findall(r"\b[A-Z]{2,}\b", text))


def acronym_fidelity(original: str, back_translated: str) -> Optional[float]:
    """
    What fraction of acronyms in the original appear in the back-translation?
    Returns None if original has no acronyms.
    """
    orig_acr = extract_acronyms(original)
    if not orig_acr:
        return None
    back_acr = extract_acronyms(back_translated)
    preserved = orig_acr & back_acr
    return len(preserved) / len(orig_acr)


# ── Semantic Equivalence (GPT-as-judge) ───────────────────────────────────────

def semantic_score_gpt(gold_answer_en: str, generated_answer_en: str,
                       openai_key: str) -> dict:
    """
    Use GPT-4o to score semantic equivalence between gold and generated answers.
    Returns {'score': 0-5, 'reasoning': str}

    Score rubric:
      5 — Identical meaning, all facts preserved
      4 — Same meaning, minor wording difference
      3 — Mostly correct, one detail missing or slightly off
      2 — Partially correct, key detail wrong or missing
      1 — Wrong answer, some overlap with gold
      0 — Completely wrong or no answer
    """
    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        prompt = f"""You are evaluating factual accuracy of a generated answer against a gold answer for Indian government document QA.

Gold answer: {gold_answer_en}

Generated answer: {generated_answer_en}

Rate the semantic equivalence on a scale of 0-5:
5 = Identical meaning, all facts preserved
4 = Same meaning, minor wording difference
3 = Mostly correct, one minor detail wrong or missing
2 = Partially correct, key fact wrong or missing
1 = Wrong answer with some overlap
0 = Completely wrong or empty

Respond with JSON only:
{{"score": <integer 0-5>, "reasoning": "<one sentence>"}}"""

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        import json
        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception as e:
        return {"score": -1, "reasoning": f"ERROR: {e}"}


# ── Compute all metrics for a pair ────────────────────────────────────────────

def compute_all_metrics(
    original_en: str,
    back_translated_en: str,
    gold_answer_en: str = None,
    generated_answer_en: str = None,
    openai_key: str = None
) -> dict:
    """
    Compute all metrics for one sentence pair.
    For translation experiment: pass original_en and back_translated_en.
    For QA experiment: also pass gold_answer_en and generated_answer_en.
    """
    metrics = {
        "chrf":             round(chrf_score(back_translated_en, original_en), 2),
        "bertscore_f1":     round(bertscore_f1(back_translated_en, original_en), 4),
        "ne_preservation":  ne_preservation(original_en, back_translated_en),
        "numeric_fidelity": numeric_fidelity(original_en, back_translated_en),
        "acronym_fidelity": acronym_fidelity(original_en, back_translated_en),
    }

    # Round optional float metrics
    for k in ["ne_preservation", "numeric_fidelity", "acronym_fidelity"]:
        if metrics[k] is not None:
            metrics[k] = round(metrics[k], 3)

    # QA semantic score (only if we have gold and generated answers)
    if gold_answer_en and generated_answer_en and openai_key:
        sem = semantic_score_gpt(gold_answer_en, generated_answer_en, openai_key)
        metrics["semantic_score"]     = sem.get("score", -1)
        metrics["semantic_reasoning"] = sem.get("reasoning", "")

    return metrics
