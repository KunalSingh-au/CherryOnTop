#!/usr/bin/env python3
"""
Aggregate outputs/evaluation/all_results.csv (or --eval-csv) into tables under outputs/analysis/.

See RUNBOOK.txt.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from config import EVAL_CSV, ANALYSIS_DIR, SUMMARY_CSV
from utils.metrics import pearson_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", default=EVAL_CSV, help="Scored results from 04_evaluate.py")
    args = ap.parse_args()

    if not os.path.exists(args.eval_csv):
        print(f"Missing {args.eval_csv}. Run scripts/04_evaluate.py first.")
        sys.exit(1)

    df = pd.read_csv(args.eval_csv, encoding="utf-8-sig")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    metrics = ["keyword_hit_rate", "rougeL_f1", "bertscore_f1", "hallucination_score"]
    g = (
        df.groupby(["model", "condition"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    g.to_csv(SUMMARY_CSV, index=False)
    print("=== Mean metrics by model × condition ===\n")
    print(g.to_string(index=False))

    lang = df.groupby("condition", dropna=False)[metrics].mean().reset_index()
    print("\n=== Mean by condition (all models) ===\n")
    print(lang.to_string(index=False))
    lang.to_csv(os.path.join(ANALYSIS_DIR, "by_condition_only.csv"), index=False)

    c2c4 = df[df["condition"].isin(["C2", "C4"])].groupby("condition")[metrics].mean()
    print("\n=== Translation effect (C2 vs C4) ===\n")
    print(c2c4)

    ministry = (
        df.groupby(["ministry", "condition"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    ministry.to_csv(os.path.join(ANALYSIS_DIR, "by_ministry_condition.csv"), index=False)
    print("\n=== Hardest ministries (mean hallucination_score) ===\n")
    hm = df.groupby("ministry")["hallucination_score"].mean().sort_values()
    print(hm)

    sub = df[df["doc_fidelity_chrf"].notna()].copy()
    sub["doc_fidelity_chrf"] = pd.to_numeric(sub["doc_fidelity_chrf"], errors="coerce")
    if len(sub) > 10:
        print("\n=== Pearson r: doc_fidelity_chrf vs metrics ===\n")
        for m in ["keyword_hit_rate", "bertscore_f1", "hallucination_score"]:
            xs = sub["doc_fidelity_chrf"].tolist()
            ys = pd.to_numeric(sub[m], errors="coerce").tolist()
            r = pearson_r(xs, ys)
            print(f"  doc_fidelity_chrf vs {m}: r = {r}")
        sub.to_csv(os.path.join(ANALYSIS_DIR, "subset_with_doc_fidelity.csv"), index=False)

    lb = df.groupby("model")[metrics].mean().sort_values("hallucination_score", ascending=False)
    print("\n=== Model leaderboard (hallucination_score, higher=better) ===\n")
    print(lb)
    lb.to_csv(os.path.join(ANALYSIS_DIR, "model_leaderboard.csv"))

    print(f"\nSummary → {SUMMARY_CSV}")
    print(f"Other tables → {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
