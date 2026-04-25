"""
utils/metrics.py — Evaluation metrics for HindiParl-QA.

Three primary metrics:
  1. keyword_hit_rate  — factual precision (numbers, scheme names, amounts)
  2. rouge_l_f1        — lexical overlap (LCS-based)
  3. bertscore_batch   — semantic similarity (cross-lingual, xlm-roberta-large)

CRITICAL BERTScore note
───────────────────────
Always use model_type="xlm-roberta-large" with lang=None.
Using lang="hi" triggers a Hindi-only encoder path inside the `evaluate`
library and produces scores on a different scale — making no-prompt and
prompted results incomparable.  This was a bug in the original no-prompt
pipeline and is corrected here.

Response classification
───────────────────────
classify_response() labels each row as "valid", "refusal", or "null".
Metrics are computed ONLY on "valid" rows.  Coverage (valid rate) is
reported as a separate dimension.  Including null / refusal rows in
metric averages produces misleading numbers.
"""

import re
import os
import json

# Regex for detecting model refusals in back-translated English
# ── TUNABLE: Add more patterns if you observe new refusal phrasings ───────────
_REFUSAL_RE = re.compile(
    r"not available|not provided|"
    r"no (?:information|context|data|content)|"
    r"insufficient (?:information|context|data)|"
    r"cannot (?:answer|provide|find)|"
    r"document (?:is not|not) available|"
    r"the (?:provided|given) (?:text|document|context) does not|"
    r"unable to (?:answer|provide|find)|"
    r"information (?:is|was) not|"
    r"no relevant|"
    r"I don'?t have (?:enough|sufficient|the)|"
    r"based on the (?:provided|given).{0,40}(?:not|no|cannot|insufficient)",
    re.IGNORECASE,
)

# Hindi refusal phrases (checked on raw answer_hi before back-translation)
_HINDI_REFUSAL_RE = re.compile(
    r"उपलब्ध नहीं|पर्याप्त जानकारी नहीं|जानकारी नहीं|संदर्भ में नहीं",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_response(answer_hi: str, back_en: str) -> str:
    """
    Label a model response as "valid", "refusal", or "null".

    null    — model returned nothing (empty string or NaN)
    refusal — model produced text but said it cannot answer from context
    valid   — everything else (model attempted a genuine answer)

    Parameters
    ----------
    answer_hi : Raw Hindi model output from all_runs.csv
    back_en   : Back-translated English version of that output
    """
    if not answer_hi or not str(answer_hi).strip():
        return "null"

    back = str(back_en).strip() if back_en else ""
    if _REFUSAL_RE.search(back):
        return "refusal"
    if _HINDI_REFUSAL_RE.search(str(answer_hi)):
        return "refusal"

    return "valid"


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD HIT RATE
# ─────────────────────────────────────────────────────────────────────────────

def keyword_hit_rate(keywords_en: str, back_en: str) -> float | None:
    """
    Fraction of gold keywords found verbatim in the back-translated answer.

    Why back_en (not raw Hindi)?
    Government keywords — numbers (34), scheme names (AYURGYAN), amounts
    (47.07 crore) — appear as-is in the back-translation.  Checking on
    the back-translation avoids false negatives from Devanagari numerals.

    Returns None if keywords field is missing (excluded from averages).
    """
    if not keywords_en or str(keywords_en).strip().upper() in ("MISSING", "NAN", ""):
        return None

    keys = [k.strip() for k in re.split(r",\s*|\|", str(keywords_en)) if k.strip()]
    if not keys:
        return None

    text = (back_en or "").lower()
    hits = sum(1 for k in keys if k.lower() in text)
    return round(hits / len(keys), 4)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────

def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """
    ROUGE-L F1 between English gold answer and back-translated answer.

    Use stemmer=True so "projects" and "project" are treated as the same token.
    Returns 0.0 for empty inputs rather than crashing.
    """
    from rouge_score import rouge_scorer
    if not reference or not hypothesis:
        return 0.0
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return round(sc.score(str(reference), str(hypothesis))["rougeL"].fmeasure, 4)


# ─────────────────────────────────────────────────────────────────────────────
# BERTSCORE (cross-lingual, fixed configuration)
# ─────────────────────────────────────────────────────────────────────────────

def bertscore_batch(refs: list, hyps: list, batch_size: int = 32) -> list:
    """
    BERTScore F1 for a batch of (reference, hypothesis) pairs.

    Both refs and hyps should be English strings:
      refs = English gold answers
      hyps = back-translated English answers (or direct English answers)

    CRITICAL configuration:
      model_type = "xlm-roberta-large"
        Cross-lingual model handling both English and Hindi in the same
        semantic space.

      lang = None    ← DO NOT CHANGE TO "hi"
        When model_type is specified explicitly, lang must be None.
        Setting lang="hi" forces a Hindi-only encoder path that cannot
        handle English references, producing incomparable scores.

    ── TUNABLE: Reduce batch_size to 16 if you get OOM during evaluation ─────
    """
    import evaluate
    import warnings

    if not refs or not hyps:
        return []

    # Guard: empty strings cause silent NaN bugs in BERTScore
    refs = [str(r) if str(r).strip() else "empty" for r in refs]
    hyps = [str(h) if str(h).strip() else "empty" for h in hyps]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metric = evaluate.load("bertscore")
        results = metric.compute(
            predictions=hyps,
            references=refs,
            model_type="xlm-roberta-large",
            lang=None,            # REQUIRED — see docstring above
            batch_size=batch_size,
            verbose=False,
        )
    return [round(float(f), 4) for f in results["f1"]]


# Legacy alias used in old 04_evaluate.py imports
bertscore_multilingual_batch = bertscore_batch


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT FIDELITY (chrF) — translation quality metric
# ─────────────────────────────────────────────────────────────────────────────

def doc_fidelity_chrf(ref_hi: str, hyp_hi: str) -> float:
    """
    chrF+2 between official Hindi document and a machine-translated version.

    Used to measure translation quality for C2 (Sarvam-MT) vs C5 (IndicTrans2).
    Higher score = machine translation is closer to official government Hindi.
    This metric is optional and run separately (not part of main evaluation).
    """
    from sacrebleu.metrics import CHRF
    if not ref_hi or not hyp_hi:
        return float("nan")
    return round(CHRF(word_order=2).corpus_score([str(hyp_hi)], [[str(ref_hi)]]).score, 4)
