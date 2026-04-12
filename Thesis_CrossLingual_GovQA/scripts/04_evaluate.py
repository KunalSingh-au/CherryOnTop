#!/usr/bin/env python3
"""
Score rows in a runs CSV (default outputs/runs/all_runs.csv).

Back-translate (Sarvam), keyword hit rate, ROUGE-L, multilingual BERTScore,
optional Gemini hallucination judge, doc_fidelity_chrf.

See RUNBOOK.txt for usage.
"""

import argparse
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
from utils.jsonl import load_docs_jsonl
from utils.llm import sarvam_hi_to_en
from utils.metrics import (
    keyword_hit_rate,
    rouge_l_f1,
    bertscore_multilingual_batch,
    doc_fidelity_chrf,
    llm_judge_hallucination,
    hallucination_numeric,
)

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-csv",
        default=RUNS_CSV,
        help="Input generations CSV (from 03_run_qa.py).",
    )
    ap.add_argument(
        "--out",
        default=EVAL_CSV,
        help="Output scored CSV.",
    )
    ap.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Gemini hallucination judge (faster for smoke tests).",
    )
    args = ap.parse_args()

    if not os.path.exists(args.runs_csv):
        print(f"Missing {args.runs_csv}. Run scripts/03_run_qa.py first.")
        sys.exit(1)

    df = pd.read_csv(args.runs_csv, encoding="utf-8-sig")
    if df.empty:
        print("Runs CSV is empty.")
        sys.exit(1)

    rows = df.to_dict("records")
    sarv_docs = load_docs_jsonl(TRANSLATED_SARVAM, "answer_hi")
    hi_docs = load_docs_jsonl(EXTRACTED_HI, "answer_hi")

    print(f"Back-translating {len(rows)} answers...")
    backs = []
    for i, row in enumerate(rows, 1):
        ah = row.get("answer_hi", "")
        if isinstance(ah, float):
            ah = ""
        ah = str(ah) if ah is not None else ""
        if ah and "ERROR" not in ah:
            backs.append(sarvam_hi_to_en(ah))
        else:
            backs.append("")
        print(f"  {i}/{len(rows)}", end="\r")
    print()

    golds = [str(r.get("gold_answer_en", "") or "") for r in rows]
    print(f"BERTScore batch ({BERTSCORE_MODEL})...")
    berts = bertscore_multilingual_batch(backs, golds, BERTSCORE_MODEL)

    out_rows = []
    for i, row in enumerate(rows):
        gold = str(row.get("gold_answer_en", "") or "")
        back = backs[i]
        kw = str(row.get("keywords_en", "") or "")
        did = str(row.get("doc_id", "") or "")
        df_val = doc_fidelity_chrf(sarv_docs.get(did, ""), hi_docs.get(did, ""))

        metrics = {
            "back_en": back,
            "keyword_hit_rate": keyword_hit_rate(kw, back),
            "rougeL_f1": rouge_l_f1(gold, back),
            "bertscore_f1": berts[i],
            "doc_fidelity_chrf": df_val,
        }

        if args.skip_judge:
            metrics["hallucination_label"] = "unknown"
            metrics["hallucination_rationale"] = ""
            metrics["hallucination_score"] = float("nan")
        else:
            judge = llm_judge_hallucination(
                gold,
                back,
                str(row.get("condition", "") or ""),
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

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(out_rows)} rows → {args.out}")
    print("Next: python scripts/05_analyze.py")


if __name__ == "__main__":
    main()
