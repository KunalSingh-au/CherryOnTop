#!/usr/bin/env python3
"""
Thesis analysis tables and correlations.

  - Model × condition means (keyword_hit_rate, rougeL_f1, bertscore_f1, hallucination_score)
  - Language effect: C1 vs C2 vs C4 (paired summaries)
  - Translation effect: C2 vs C4
  - Ministry × condition difficulty
  - Pearson: doc_fidelity_chrf vs keyword_hit_rate / hallucination_score (per row, grouped notes)

Writes outputs/analysis/*.csv and prints tables to stdout.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from config import EVAL_CSV, ANALYSIS_DIR, SUMMARY_CSV
from utils.metrics import pearson_r


def main():
    if not os.path.exists(EVAL_CSV):
        print(f"Missing {EVAL_CSV}. Run scripts/04_evaluate.py first.")
        sys.exit(1)

    df = pd.read_csv(EVAL_CSV, encoding="utf-8-sig")
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
    g.to_csv(os.path.join(ANALYSIS_DIR, "by_model_condition.csv"), index=False)

    # Language / translation effects (aggregate over models)
    lang = (
        df.groupby("condition", dropna=False)[metrics].mean().reset_index()
    )
    print("\n=== Mean by condition (all models) ===\n")
    print(lang.to_string(index=False))
    lang.to_csv(os.path.join(ANALYSIS_DIR, "by_condition_only.csv"), index=False)

    c2c4 = df[df["condition"].isin(["C2", "C4"])].groupby("condition")[metrics].mean()
    print("\n=== Translation effect (C2 machine Hindi doc vs C4 official Hindi doc) ===\n")
    print(c2c4)

    ministry = (
        df.groupby(["ministry", "condition"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    ministry.to_csv(os.path.join(ANALYSIS_DIR, "by_ministry_condition.csv"), index=False)
    print("\n=== Hardest ministries (mean hallucination_score, all conditions) ===\n")
    hm = df.groupby("ministry")["hallucination_score"].mean().sort_values()
    print(hm)

    # Correlation: doc fidelity vs outcomes (rows where doc_fidelity defined)
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

    # Model leaderboard (overall mean hallucination_score + keyword)
    lb = df.groupby("model")[metrics].mean().sort_values("hallucination_score", ascending=False)
    print("\n=== Model leaderboard (mean hallucination_score, higher=better) ===\n")
    print(lb)
    lb.to_csv(os.path.join(ANALYSIS_DIR, "model_leaderboard.csv"))

    print(f"\nAnalysis CSVs → {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
