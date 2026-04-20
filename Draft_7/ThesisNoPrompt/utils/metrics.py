import re
import os

# ── Keyword Hit Rate ───────────────────────────────────────────────────────────
def keyword_hit_rate(keywords_en: str, answer_hi: str) -> float:
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
    from rouge_score import rouge_scorer
    if not reference or not hypothesis:
        return 0.0
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return round(sc.score(reference, hypothesis)["rougeL"].fmeasure, 4)

# ── BERTScore (Multilingual) ───────────────────────────────────────────────────
def bertscore_multilingual_batch(refs: list, hyps: list) -> list:
    if not refs or not hyps: return []
    import evaluate
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(
            predictions=hyps,
            references=refs,
            model_type="xlm-roberta-large",
            lang="hi",
            batch_size=32
        )
    return [round(f, 4) for f in results["f1"]]

# ── chrF (Document Fidelity) ───────────────────────────────────────────────────
def doc_fidelity_chrf(ref_hi: str, hyp_hi: str) -> float:
    from sacrebleu.metrics import CHRF
    if not ref_hi or not hyp_hi:
        return float("nan")
    chrf = CHRF(word_order=2)
    return round(chrf.corpus_score([hyp_hi], [[ref_hi]]).score, 2)