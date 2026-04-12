#!/usr/bin/env python3
"""
Evaluate all rows in outputs/runs/all_runs.csv:

  - Back-translate Hindi answers → English (Sarvam)
  - Keyword hit rate (curated phrases vs back-English)
  - ROUGE-L (back-English vs gold_answer_en)
  - BERTScore F1 multilingual (same pair)
  - LLM-as-judge: grounded / minor / major (Gemini)

Also attaches doc_fidelity_chrf (Sarvam-translated doc vs official Hindi doc) per doc_id when available.

Output: outputs/evaluation/all_results.csv
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    RUNS_CSV,
    EVAL_CSV,
    TRANSLATED_SARVAM,
    EXTRACTED_HI,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    BERTSCORE_MODEL,
)
from utils.llm import sarvam_hi_to_en
from utils.metrics import (
    keyword_hit_rate,
    rouge_l_f1,
    bertscore_multilingual_batch,
    doc_fidelity_chrf,
    llm_judge_hallucination,
    hallucination_numeric,
)


def load_doc_map(path: str, field: str) -> dict:
    if not os.path.exists(path):
        return {}
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m[r["doc_id"]] = r.get(field, "") or ""
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Gemini hallucination judge (faster; use for dry runs).",
    )
    args = ap.parse_args()

    if not os.path.exists(RUNS_CSV):
        print(f"Missing {RUNS_CSV}. Run scripts/03_run_qa.py first.")
        sys.exit(1)

    with open(RUNS_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    sarv_docs = load_doc_map(TRANSLATED_SARVAM, "answer_hi")
    hi_docs = load_doc_map(EXTRACTED_HI, "answer_hi")

    print(f"Back-translating {len(rows)} answers...")
    backs = []
    for i, row in enumerate(rows, 1):
        ah = row.get("answer_hi", "")
        if ah and "ERROR" not in ah:
            backs.append(sarvam_hi_to_en(ah))
        else:
            backs.append("")
        print(f"  {i}/{len(rows)}", end="\r")
    print()

    golds = [r.get("gold_answer_en", "") for r in rows]
    print(f"BERTScore batch ({BERTSCORE_MODEL})...")
    berts = bertscore_multilingual_batch(backs, golds, BERTSCORE_MODEL)

    out_rows = []
    for i, row in enumerate(rows):
        gold = row.get("gold_answer_en", "")
        back = backs[i]
        kw = row.get("keywords_en", "")
        did = row.get("doc_id", "")
        df = doc_fidelity_chrf(sarv_docs.get(did, ""), hi_docs.get(did, ""))

        metrics = {
            "back_en": back,
            "keyword_hit_rate": keyword_hit_rate(kw, back),
            "rougeL_f1": rouge_l_f1(gold, back),
            "bertscore_f1": berts[i],
            "doc_fidelity_chrf": df,
        }

        if args.skip_judge:
            metrics["hallucination_label"] = "unknown"
            metrics["hallucination_rationale"] = ""
            metrics["hallucination_score"] = float("nan")
        else:
            judge = llm_judge_hallucination(
                gold,
                back,
                row.get("condition", ""),
                GOOGLE_API_KEY,
                GEMINI_MODEL,
            )
            metrics["hallucination_label"] = judge.get("label", "unknown")
            metrics["hallucination_rationale"] = judge.get("rationale", "")
            metrics["hallucination_score"] = hallucination_numeric(
                metrics["hallucination_label"]
            )

        out_rows.append({**row, **metrics})
        print(f"  judge {i+1}/{len(rows)}", end="\r")
    print()

    os.makedirs(os.path.dirname(EVAL_CSV), exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with open(EVAL_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows → {EVAL_CSV}")
    print("Next: python scripts/05_analyze.py")


if __name__ == "__main__":
    main()
