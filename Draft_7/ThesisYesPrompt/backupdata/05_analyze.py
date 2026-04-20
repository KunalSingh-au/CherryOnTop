import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EVAL_CSV, ANALYSIS_DIR

# REMOVED hallucination_score from METRICS
METRICS = ["keyword_hit_rate", "rougeL_f1", "bertscore_f1"]
MLABELS = ["KW-Hit", "ROUGE-L", "BERTScore"]

COND_LABELS = {
    "C1": "English doc (original)",
    "C2": "Sarvam machine-translated Hindi",
    "C3": "No document (hallucination floor)",
    "C4": "Official Hindi PDF",
    "C5": "IndicTrans2 Hindi",
    "C6": "RAG top-3 English chunks",
    "C7": "RAG Sarvam-MT Hindi",
    "C8": "RAG Official Hindi",
    "C9": "RAG IndicTrans2 Hindi"
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

    # Clean metrics
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0.0)
        else:
            print(f"Warning: Metric {m} not found in CSV. Filling with 0.")
            df[m] = 0.0

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # ── Table 1: Model × Condition ─────────────────────────────────────────────
    hr("TABLE 1: Mean metrics by Model × Condition (MAIN RESULTS)")
    t1 = (df.groupby(["model", "condition"])[METRICS]
          .mean().round(3).reset_index())
    t1.columns = ["Model", "Condition"] + MLABELS
    t1.to_csv(os.path.join(ANALYSIS_DIR, "summary_by_model_condition.csv"), index=False)
    print(t1.to_string(index=False))

    # ── Table 2: Model leaderboard ─────────────────────────────────────────────
    hr("TABLE 2: Overall Model Leaderboard (Sorted by BERTScore)")
    t2 = (df.groupby("model")[METRICS]
          .mean().round(3).reset_index()
          .sort_values("bertscore_f1", ascending=False))
    t2.columns = ["Model"] + MLABELS
    t2.to_csv(os.path.join(ANALYSIS_DIR, "model_leaderboard.csv"), index=False)
    print(t2.to_string(index=False))

    # ── Table 3: Condition effect ──────────────────────────────────────────────
    hr("TABLE 3: Condition Effect — Language and Document Supply")
    t3 = (df.groupby("condition")[METRICS]
          .mean().round(3).reset_index())
    t3["Description"] = t3["condition"].map(COND_LABELS).fillna("RAG Variation")
    t3 = t3[["condition", "Description"] + METRICS]
    t3.columns = ["Condition", "Description"] + MLABELS
    t3.to_csv(os.path.join(ANALYSIS_DIR, "by_condition_only.csv"), index=False)
    print(t3.to_string(index=False))

 # ── Table 4: Translation engine comparison ─────────────────────────────────
    hr("TABLE 4: Translation Engine Impact — C1 vs C2 vs C4 vs C5")
    sub4 = df[df["condition"].isin(["C1", "C2", "C4", "C5"])]
    if not sub4.empty:
        t4 = (sub4.groupby("condition")[METRICS].mean().round(3).reset_index())
        t4["Description"] = t4["condition"].map(COND_LABELS).fillna("")
        
        # Use METRICS (the original column names) to print, not MLABELS
        print(t4[["condition", "Description"] + METRICS].to_string(index=False))
        t4.to_csv(os.path.join(ANALYSIS_DIR, "translation_impact.csv"), index=False)

    # ── Table 7: RAG improvement ─────────────────────────────────────────────
    hr("TABLE 7: RAG Improvement — C1 vs C6 (English) and C2 vs C7 (Sarvam)")
    # English RAG comparison
    rag_sub = df[df["condition"].isin(["C1", "C6"])]
    if not rag_sub[rag_sub["condition"] == "C6"].empty:
        t7 = (rag_sub.groupby(["model", "condition"])[METRICS].mean().round(3).reset_index())
        t7.to_csv(os.path.join(ANALYSIS_DIR, "rag_comparison.csv"), index=False)
        print(t7.to_string(index=False))
    else:
        print("No RAG (C6) data found in the CSV.")

    print(f"\n✓ All tables saved to {ANALYSIS_DIR}/")

if __name__ == "__main__":
    main()