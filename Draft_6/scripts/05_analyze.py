#!/usr/bin/env python3
"""
05_analyze.py — full thesis analysis tables
============================================
Tables produced:
  1. summary_by_model_condition.csv  — main results (Model × Condition)
  2. model_leaderboard.csv           — overall model ranking
  3. by_condition_only.csv           — condition effect
  4. translation_impact.csv          — C2 vs C4 vs C5 (translation engines)
  5. by_ministry_condition.csv       — ministry difficulty
  6. c3_hallucination.csv            — hallucination baseline (no-doc)
  7. c1_vs_c6_rag.csv                — RAG improvement (if C6 data exists)

Run: python scripts/05_analyze.py
     python scripts/05_analyze.py --eval-csv outputs/evaluation/sample_results.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EVAL_CSV, ANALYSIS_DIR

METRICS = ["keyword_hit_rate", "rougeL_f1", "bertscore_f1", "hallucination_score"]
MLABELS = ["KW-Hit", "ROUGE-L", "BERTScore", "Judge"]

COND_LABELS = {
    "C1": "English doc (original)",
    "C2": "Sarvam machine-translated Hindi",
    "C3": "No document (hallucination floor)",
    "C4": "Official Hindi PDF",
    "C5": "IndicTrans2 Hindi",
    "C6": "RAG top-3 English chunks",
}


def hr(title: str):
    print(f"\n{'='*65}")
    print(title)
    print("="*65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", default=EVAL_CSV)
    args = parser.parse_args()

    if not os.path.exists(args.eval_csv):
        print(f"ERROR: {args.eval_csv} not found. Run 04_evaluate.py first.")
        sys.exit(1)

    df = pd.read_csv(args.eval_csv, encoding="utf-8-sig")
    print(f"Loaded {len(df)} evaluated rows")

    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    models     = sorted(df["model"].unique())
    conditions = sorted(df["condition"].unique())
    ministries = sorted(df["ministry"].unique())

    print(f"Models:     {models}")
    print(f"Conditions: {conditions}")
    print(f"Ministries: {ministries}")

    # ── Table 1: Model × Condition ─────────────────────────────────────────────
    hr("TABLE 1: Mean metrics by Model × Condition  (MAIN RESULTS)")
    t1 = (df.groupby(["model", "condition"])[METRICS]
          .mean().round(3).reset_index())
    t1.columns = ["Model", "Condition"] + MLABELS
    t1.to_csv(os.path.join(ANALYSIS_DIR, "summary_by_model_condition.csv"), index=False)
    print(t1.to_string(index=False))

    # ── Table 2: Model leaderboard ─────────────────────────────────────────────
    hr("TABLE 2: Overall Model Leaderboard (mean across all conditions)")
    t2 = (df.groupby("model")[METRICS]
          .mean().round(3).reset_index()
          .sort_values("hallucination_score", ascending=False))
    t2.columns = ["Model"] + MLABELS
    t2.to_csv(os.path.join(ANALYSIS_DIR, "model_leaderboard.csv"), index=False)
    print(t2.to_string(index=False))

    # ── Table 3: Condition effect ──────────────────────────────────────────────
    hr("TABLE 3: Condition Effect — Language and Document Supply")
    t3 = (df.groupby("condition")[METRICS]
          .mean().round(3).reset_index())
    t3["Description"] = t3["condition"].map(COND_LABELS).fillna("")
    t3 = t3[["condition", "Description"] + METRICS]
    t3.columns = ["Condition", "Description"] + MLABELS
    t3.to_csv(os.path.join(ANALYSIS_DIR, "by_condition_only.csv"), index=False)
    print(t3.to_string(index=False))

    # ── Table 4: Translation engine comparison ─────────────────────────────────
    hr("TABLE 4: Translation Engine Impact — C1 vs C2 vs C4 vs C5")
    print("(Does translating the document to Hindi help? Which translator is better?)")
    sub4 = df[df["condition"].isin(["C1", "C2", "C4", "C5"])]
    t4   = (sub4.groupby("condition")[METRICS]
            .mean().round(3).reset_index())
    t4["Description"] = t4["condition"].map(COND_LABELS).fillna("")

    # Add delta vs C1
    c1_mean = t4[t4["condition"] == "C1"][METRICS].values
    if len(c1_mean):
        for m, ml in zip(METRICS, MLABELS):
            t4[f"Δ vs C1 ({ml})"] = (t4[m] - float(c1_mean[0][METRICS.index(m)])).round(3)

    t4.to_csv(os.path.join(ANALYSIS_DIR, "translation_impact.csv"), index=False)
    print(t4[["condition", "Description"] + METRICS].to_string(index=False))

    # ── Table 5: Ministry difficulty ───────────────────────────────────────────
    hr("TABLE 5: Ministry Difficulty (hardest → easiest by Judge score)")
    t5 = (df.groupby(["ministry", "condition"])[METRICS]
          .mean().round(3).reset_index())
    t5.to_csv(os.path.join(ANALYSIS_DIR, "by_ministry_condition.csv"), index=False)

    t5_overall = (df.groupby("ministry")[METRICS]
                  .mean().round(3).reset_index()
                  .sort_values("hallucination_score"))
    t5_overall.columns = ["Ministry"] + MLABELS
    print(t5_overall.to_string(index=False))

    # ── Table 6: C3 hallucination baseline ────────────────────────────────────
    hr("TABLE 6: Hallucination Baseline — C3 (no document)")
    print("Non-zero score here = model is answering from memorised training knowledge.")
    c3 = df[df["condition"] == "C3"]
    if len(c3) > 0:
        t6 = (c3.groupby("model")[METRICS]
              .mean().round(3).reset_index()
              .sort_values("hallucination_score", ascending=False))
        t6.columns = ["Model"] + MLABELS
        t6["n"] = c3.groupby("model").size().values
        t6.to_csv(os.path.join(ANALYSIS_DIR, "c3_hallucination.csv"), index=False)
        print(t6.to_string(index=False))
    else:
        print("No C3 rows found.")

    # ── Table 7: RAG improvement C1 vs C6 ─────────────────────────────────────
    hr("TABLE 7: RAG Improvement — C1 (full doc) vs C6 (retrieved chunks)")
    rag_sub = df[df["condition"].isin(["C1", "C6"])]
    c6_rows = rag_sub[rag_sub["condition"] == "C6"]

    if len(c6_rows) > 0:
        t7 = (rag_sub.groupby(["model", "condition"])[METRICS]
              .mean().round(3).reset_index())

        # Delta
        c1_t = t7[t7["condition"] == "C1"].set_index("model")[METRICS]
        c6_t = t7[t7["condition"] == "C6"].set_index("model")[METRICS]
        common = c1_t.index.intersection(c6_t.index)
        delta  = (c6_t.loc[common] - c1_t.loc[common]).round(3).reset_index()
        delta.insert(1, "note", "C6−C1 (+ve = RAG helped)")
        delta.columns = ["Model", "Note"] + MLABELS

        t7.to_csv(os.path.join(ANALYSIS_DIR, "c1_vs_c6_rag.csv"), index=False)
        delta.to_csv(os.path.join(ANALYSIS_DIR, "rag_delta.csv"), index=False)

        print(t7.to_string(index=False))
        print()
        print("RAG DELTA (C6 − C1):")
        print(delta.to_string(index=False))
    else:
        print("No C6 rows yet. Run: python scripts/06_run_rag.py --model <name>")

    # ── Cross-model C1 comparison (clean single-table for thesis) ─────────────
    hr("THESIS TABLE: C1 comparison across all models")
    c1_only = df[df["condition"] == "C1"].groupby("model")[METRICS].mean().round(3)
    c1_only.columns = MLABELS
    c1_only = c1_only.sort_values("Judge", ascending=False)
    print(c1_only.to_string())

    print(f"\n✓ All tables saved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
