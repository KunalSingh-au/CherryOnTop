# Evaluation: keyword hit rate, ROUGE-L, multilingual BERTScore, LLM judge.

from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np


def parse_keyword_phrases(keywords_en: str) -> list[str]:
    if not keywords_en or keywords_en.strip().upper() in ("MISSING", ""):
        return []
    s = keywords_en.strip()
    if "|" in s and "," not in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    parts = re.split(r",\s*", s)
    return [p.strip() for p in parts if p.strip()]


def keyword_hit_rate(keywords_en: str, text_en: str) -> Optional[float]:
    """Fraction of curated keyword phrases found in English text (case-insensitive)."""
    keys = parse_keyword_phrases(keywords_en)
    if not keys:
        return None
    low = (text_en or "").lower()
    hits = sum(1 for k in keys if k.lower() in low)
    return round(hits / len(keys), 4)


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    try:
        from rouge_score import rouge_scorer

        if not reference or not hypothesis:
            return 0.0
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return round(float(sc.score(reference, hypothesis)["rougeL"].fmeasure), 4)
    except Exception:
        return 0.0


def bertscore_multilingual_batch(
    hypotheses: list[str],
    references: list[str],
    model: str,
) -> list[float]:
    try:
        from bert_score import score

        h = [x if x else " " for x in hypotheses]
        r = [x if x else " " for x in references]
        _, _, f1 = score(h, r, model_type=model, verbose=False)
        return [round(float(x), 4) for x in f1]
    except Exception:
        return [0.0] * len(hypotheses)


def chrf_sentence(hypothesis: str, reference: str) -> float:
    try:
        import sacrebleu

        if not hypothesis or not reference:
            return 0.0
        return round(float(sacrebleu.sentence_chrf(hypothesis, [reference]).score), 2)
    except Exception:
        return 0.0


def doc_fidelity_chrf(machine_hi: str, human_hi: str) -> Optional[float]:
    """chrF between Sarvam-translated doc snippet and official Hindi doc (translation quality proxy)."""
    if not machine_hi or not human_hi:
        return None
    return chrf_sentence(machine_hi, human_hi)


_HALLUCINATION_PROMPT = """You evaluate Hindi government QA answers against a gold English answer.

Gold (English, authoritative): {gold_en}

Model answer (back-translated to English for judging): {back_en}

Document context was provided to the model as: {ctx_note}

Classify hallucination severity for the model answer relative to the gold facts:
- "grounded": Factually aligned with gold; no invented policy numbers or entities.
- "minor": Small wording slip or omits a secondary detail; core facts intact.
- "major": Wrong numbers, wrong scheme names, invents facts, or contradicts gold.

Return JSON only: {{"label": "grounded|minor|major", "rationale": "<one sentence>"}}"""


def llm_judge_hallucination(
    gold_en: str,
    back_en: str,
    condition: str,
    google_api_key: str,
    model_name: str,
) -> dict:
    if not google_api_key or not gold_en or not back_en:
        return {"label": "unknown", "rationale": "missing key or text"}

    ctx_note = {
        "C1": "English official document",
        "C2": "Machine-translated Hindi document (Sarvam)",
        "C3": "No document (model knowledge only)",
        "C4": "Official Hindi government document",
    }.get(condition, condition)

    prompt = _HALLUCINATION_PROMPT.format(
        gold_en=gold_en[:2500],
        back_en=back_en[:2500],
        ctx_note=ctx_note,
    )
    try:
        import google.generativeai as genai

        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 128},
        )
        raw = (resp.text or "").strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            return {"label": "unknown", "rationale": "json_parse_failed"}
    except Exception as e:
        return {"label": "unknown", "rationale": str(e)}


def hallucination_numeric(label: str) -> float:
    return {"grounded": 1.0, "minor": 0.5, "major": 0.0, "unknown": float("nan")}.get(
        (label or "").lower(), float("nan")
    )


def pearson_r(xs: list[float], ys: list[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]  # not nan
    if len(pairs) < 5:
        return None
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)
