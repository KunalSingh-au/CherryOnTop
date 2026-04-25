"""
scripts/05_analyze.py — Generate all result tables for the thesis.

Reads all_results.csv (output of 04_evaluate.py) and produces 8 CSV tables
in outputs/analysis/.  All tables filter to valid responses before computing
means (so null/refusal rows do not distort the numbers).

Can optionally compare two evaluation files side-by-side (e.g., prompted vs
no-prompt) using --compare.

Usage:
    python scripts/05_analyze.py
    python scripts/05_analyze.py --input outputs/evaluation/all_results.csv
    python scripts/05_analyze.py --run-tag prompted
    python scripts/05_analyze.py --compare \\
        outputs/evaluation/all_results_prompted.csv \\
        outputs/evaluation/all_results_noprompt.csv
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EVAL_CSV, ANALYSIS_DIR, ALL_CONDITIONS

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY LABELS
# ─────────────────────────────────────────────────────────────────────────────
COND_LABELS = {
    "C1": "Full Doc (English)",
    "C2": "Full Doc (Sarvam MT)",
    "C3": "No Context (Closed-Book)",
    "C4": "Full Doc (Official Hindi)",
    "C5": "Full Doc (IndicTrans2)",
    "C6": "RAG (English Chunks)",
    "C7": "RAG (Sarvam MT Chunks)",
    "C8": "RAG (Official Hindi Chunks)",
    "C9": "RAG (IndicTrans2 Chunks)",
}

METRICS      = ["bertscore_f1", "rougeL_f1", "keyword_hit_rate"]
METRIC_LABELS= ["BERTScore",    "ROUGE-L",   "KW-Hit"]


def hr(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def save(df: pd.DataFrame, filename: str, out_dir: str):
    path = os.path.join(out_dir, filename)
    df.to_csv(path)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame, out_dir: str, tag: str = ""):
    """Run all 8 analysis tables on a filtered DataFrame."""
    prefix = f"{tag}_" if tag else ""

    # Filter to valid responses only
    if "response_type" in df.columns:
        valid = df[df["response_type"] == "valid"].copy()
    else:
        # Fallback: treat all non-empty rows as valid (old CSV format)
        valid = df[df["answer_hi"].fillna("").str.strip() != ""].copy()

    print(f"\nValid rows: {len(valid)} / {len(df)}")

    # ── TABLE 1: Model × Condition heatmap ────────────────────────────────────
    hr("TABLE 1: BERTScore by Model × Condition")
    t1 = (
        valid.groupby(["model", "condition"])[METRICS]
        .mean().round(3).reset_index()
    )
    pivot = t1.pivot(index="model", columns="condition", values="bertscore_f1")
    print(pivot.to_string())
    t1.to_csv(os.path.join(out_dir, f"{prefix}01_model_condition_matrix.csv"), index=False)

    # ── TABLE 2: Overall leaderboard ──────────────────────────────────────────
    hr("TABLE 2: Overall Model Leaderboard (valid rows, all conditions)")
    t2 = valid.groupby("model")[METRICS].mean().round(3)
    t2["combined_score"] = t2.mean(axis=1).round(3)
    t2 = t2.sort_values("combined_score", ascending=False)
    print(t2.to_string())
    save(t2, f"{prefix}02_model_leaderboard.csv", out_dir)

    # ── TABLE 3: Coverage (valid rate) per model × condition ──────────────────
    hr("TABLE 3: Coverage (valid response rate) by Model × Condition")
    if "response_type" in df.columns:
        t3 = (
            df.groupby(["model", "condition"])["response_type"]
            .apply(lambda x: (x == "valid").mean())
            .rename("valid_rate")
            .round(3)
            .reset_index()
        )
        pivot3 = t3.pivot(index="model", columns="condition", values="valid_rate")
        print(pivot3.to_string())
        save(t3, f"{prefix}03_coverage_by_model_condition.csv", out_dir)
    else:
        print("  [SKIP] response_type column not found — run updated 04_evaluate.py")

    # ── TABLE 4: Condition impact (averaged across all models) ────────────────
    hr("TABLE 4: Condition Impact (averaged across models, valid rows)")
    t4 = valid.groupby("condition")[METRICS].mean().round(3).reset_index()
    t4["description"] = t4["condition"].map(COND_LABELS)
    t4 = t4[["condition", "description"] + METRICS]
    t4 = t4.sort_values("bertscore_f1", ascending=False)
    print(t4.to_string(index=False))
    t4.to_csv(os.path.join(out_dir, f"{prefix}04_condition_impact.csv"), index=False)

    # ── TABLE 5: RAG delta (RAG minus full-doc, per model) ────────────────────
    hr("TABLE 5: RAG Effectiveness (RAG score − Full-Doc score)")
    rag_pairs = [("C1","C6","English"), ("C2","C7","Sarvam MT"),
                 ("C4","C8","Official Hindi"), ("C5","C9","IndicTrans2")]
    rows = []
    for full, rag, label in rag_pairs:
        f = valid[valid["condition"]==full].groupby("model")[METRICS].mean()
        r = valid[valid["condition"]==rag ].groupby("model")[METRICS].mean()
        delta = (r - f).round(3)
        delta["context_type"] = label
        rows.append(delta.reset_index())
    if rows:
        t5 = pd.concat(rows)
        print("Positive = RAG improved over full-document.")
        print(t5.groupby("context_type")[METRICS].mean().round(3).to_string())
        t5.to_csv(os.path.join(out_dir, f"{prefix}05_rag_delta.csv"), index=False)

    # ── TABLE 6: Translation engine head-to-head (C2 vs C5 vs C1 baseline) ───
    hr("TABLE 6: Translation Engine Comparison (ROUGE-L, valid rows)")
    t6_sub = valid[valid["condition"].isin(["C1", "C2", "C5"])]
    if not t6_sub.empty:
        t6 = t6_sub.groupby(["condition","model"])["rougeL_f1"].mean().round(3).unstack("condition")
        t6.columns.name = None
        print(t6.to_string())
        save(t6, f"{prefix}06_translation_comparison.csv", out_dir)

    # ── TABLE 7: Ministry breakdown ───────────────────────────────────────────
    hr("TABLE 7: Performance by Ministry")
    t7 = valid.groupby("ministry")[METRICS].mean().round(3).sort_values("bertscore_f1", ascending=False)
    print(t7.to_string())
    save(t7, f"{prefix}07_ministry_breakdown.csv", out_dir)

    # ── TABLE 8: Hallucination floor (C3 — closed-book) ──────────────────────
    hr("TABLE 8: Closed-Book Hallucination Floor (C3, valid rows)")
    t8 = valid[valid["condition"] == "C3"].groupby("model")[METRICS].mean().round(3)
    print(t8.to_string())
    save(t8, f"{prefix}08_hallucination_floor.csv", out_dir)

    print(f"\n✓ All tables saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARE MODE — prompted vs no-prompt side-by-side
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(file_a: str, file_b: str, out_dir: str):
    """Generate a comparison table between two evaluation CSVs."""
    hr("COMPARISON: Prompted vs No-Prompt")
    dfa = pd.read_csv(file_a)
    dfb = pd.read_csv(file_b)

    if "response_type" in dfa.columns:
        va = dfa[dfa["response_type"] == "valid"]
    else:
        va = dfa[dfa["answer_hi"].fillna("").str.strip() != ""]

    if "response_type" in dfb.columns:
        vb = dfb[dfb["response_type"] == "valid"]
    else:
        vb = dfb[dfb["answer_hi"].fillna("").str.strip() != ""]

    la = os.path.basename(file_a).replace(".csv","")
    lb = os.path.basename(file_b).replace(".csv","")

    a_scores = va.groupby("model")[METRICS].mean().round(4).add_prefix(f"{la}__")
    b_scores = vb.groupby("model")[METRICS].mean().round(4).add_prefix(f"{lb}__")
    cmp = a_scores.join(b_scores, how="outer")
    print(cmp.to_string())

    # Coverage comparison
    if "response_type" in dfa.columns and "response_type" in dfb.columns:
        cov_a = dfa.groupby("model")["response_type"].apply(lambda x: (x=="valid").mean()).rename(f"{la}__coverage")
        cov_b = dfb.groupby("model")["response_type"].apply(lambda x: (x=="valid").mean()).rename(f"{lb}__coverage")
        cov = cov_a.to_frame().join(cov_b.to_frame(), how="outer").round(3)
        hr("Coverage Comparison")
        print(cov.to_string())
        cov.to_csv(os.path.join(out_dir, "compare_coverage.csv"))

    cmp.to_csv(os.path.join(out_dir, "compare_scores.csv"))
    print(f"\n✓ Comparison saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate thesis analysis tables.")
    parser.add_argument("--input",    default=EVAL_CSV, help="Path to all_results.csv")
    parser.add_argument("--out-dir",  default=ANALYSIS_DIR, help="Output directory")
    parser.add_argument(
        "--run-tag", choices=["prompted","noprompt","all"], default="all",
        help="Filter to a specific run_tag before analysis",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE_A","FILE_B"),
        help="Compare two evaluation CSVs side by side",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.compare:
        run_comparison(args.compare[0], args.compare[1], args.out_dir)
        return

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.  Run 04_evaluate.py first.")
        sys.exit(1)

    df = pd.read_csv(args.input)

    if args.run_tag != "all" and "run_tag" in df.columns:
        df = df[df["run_tag"] == args.run_tag].copy()
        print(f"Filtered to run_tag='{args.run_tag}': {len(df)} rows")

    run_analysis(df, args.out_dir, tag=args.run_tag if args.run_tag != "all" else "")


if __name__ == "__main__":
    main()
