"""
scripts/04_evaluate.py — Evaluate LLM answers against gold QA.

What it does:
  1. Loads all_runs.csv (output of 03_run_inference.py)
  2. Optionally filters by run_tag (--run-tag prompted / noprompt)
  3. Back-translates Hindi answers to English using Sarvam-Translate
  4. Classifies each response as "valid", "refusal", or "null"
  5. Computes metrics ONLY on valid responses
  6. Reports coverage (valid rate) separately
  7. Saves all_results.csv

WHY classify before scoring:
  Null responses score ~0 on BERTScore, dragging model averages down.
  Refusals score ~0.85 (boilerplate text is semantically similar to gold),
  inflating averages.  Both distortions are removed by filtering to valid only.

Usage:
    python scripts/04_evaluate.py
    python scripts/04_evaluate.py --run-tag prompted
    python scripts/04_evaluate.py --run-tag noprompt
    python scripts/04_evaluate.py --input outputs/runs/demo_runs.csv \\
                                  --output outputs/evaluation/demo_results.csv
"""

import argparse
import os
import re
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RUNS_CSV, EVAL_CSV, BERTSCORE_BATCH, HINDI_CHAR_THRESHOLD
from utils.metrics import (
    classify_response, keyword_hit_rate, rouge_l_f1, bertscore_batch
)
from utils.llm import hi_to_en_local, extract_answer_from_output


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def is_hindi(text: str) -> bool:
    """
    Return True if more than HINDI_CHAR_THRESHOLD of the string is Devanagari.

    Used to decide whether to back-translate (Hindi) or use directly (English).
    """
    if not isinstance(text, str) or not text.strip():
        return False
    hi_count = len(re.findall(r"[\u0900-\u097F]", text))
    return (hi_count / len(text)) > HINDI_CHAR_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate HindiParl-QA runs.")
    parser.add_argument("--input",    default=RUNS_CSV,  help="Path to all_runs.csv")
    parser.add_argument("--output",   default=EVAL_CSV,  help="Path for output CSV")
    parser.add_argument(
        "--run-tag",
        choices=["prompted", "noprompt", "all"],
        default="all",
        help="Filter to a specific run_tag (default: all rows)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BERTSCORE_BATCH,
        help=f"BERTScore batch size (default: {BERTSCORE_BATCH}). "
             "Reduce to 16 if you hit OOM.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.  Run 03_run_inference.py first.")
        sys.exit(1)

    df = pd.read_csv(args.input)

    # ── Filter by run_tag if requested ────────────────────────────────────────
    if args.run_tag != "all" and "run_tag" in df.columns:
        df = df[df["run_tag"] == args.run_tag].copy()
        print(f"Filtered to run_tag='{args.run_tag}': {len(df)} rows")
    else:
        print(f"Loaded {len(df)} rows from {args.input}")

    df["answer_hi"] = df["answer_hi"].fillna("")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Clean answers + back-translate
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Cleaning and back-translating …")
    back_translations = []

    for text in tqdm(df["answer_hi"], desc="  Back-translating"):
        # Strip <think> reasoning blocks (Sarvam, Qwen chain-of-thought)
        clean = extract_answer_from_output(str(text))

        if not clean.strip():
            back_translations.append("")
        elif is_hindi(clean):
            # Hindi/Hinglish → translate to English for metric comparison
            back_translations.append(hi_to_en_local(clean))
        else:
            # Model answered in English — use directly (no translation noise)
            back_translations.append(clean)

    df["back_en"] = back_translations

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Classify responses
    # ─────────────────────────────────────────────────────────────────────────
    print("[2/4] Classifying responses …")
    df["response_type"] = [
        classify_response(a, b)
        for a, b in zip(df["answer_hi"], df["back_en"])
    ]

    null_n    = (df["response_type"] == "null").sum()
    refusal_n = (df["response_type"] == "refusal").sum()
    valid_n   = (df["response_type"] == "valid").sum()
    total     = len(df)

    print(f"  null={null_n}  refusal={refusal_n}  valid={valid_n}  "
          f"(coverage={valid_n/total:.1%})")

    # Per-model coverage report
    if "model" in df.columns:
        print("\n  Coverage by model:")
        cov = (
            df.groupby("model")["response_type"]
            .apply(lambda x: (x == "valid").mean())
            .rename("valid_rate")
            .round(3)
        )
        print(cov.to_string(header=True))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Compute metrics on valid rows only
    # ─────────────────────────────────────────────────────────────────────────
    # Initialise metric columns with NaN; filled in for valid rows below
    df["keyword_hit_rate"] = float("nan")
    df["rougeL_f1"]        = float("nan")
    df["bertscore_f1"]     = float("nan")

    valid_mask = df["response_type"] == "valid"
    valid_df   = df[valid_mask]

    if valid_df.empty:
        print("\nWARNING: No valid responses found — all metrics will be NaN.")
    else:
        print(f"\n[3/4] Computing metrics on {len(valid_df)} valid rows …")

        # ── Keyword Hit Rate ──────────────────────────────────────────────────
        print("  keyword_hit_rate … ", end="", flush=True)
        df.loc[valid_mask, "keyword_hit_rate"] = valid_df.apply(
            lambda r: keyword_hit_rate(str(r["keywords_en"]), str(r["back_en"])),
            axis=1,
        ).values
        print("done")

        # ── ROUGE-L ───────────────────────────────────────────────────────────
        print("  rougeL_f1 … ", end="", flush=True)
        df.loc[valid_mask, "rougeL_f1"] = valid_df.apply(
            lambda r: rouge_l_f1(str(r["gold_answer_en"]), str(r["back_en"])),
            axis=1,
        ).values
        print("done")

        # ── BERTScore ─────────────────────────────────────────────────────────
        # Uses xlm-roberta-large with lang=None for cross-lingual scoring.
        # See utils/metrics.py for why lang="hi" is WRONG here.
        print(f"  bertscore_f1 (batch={args.batch_size}) … ", end="", flush=True)
        refs  = valid_df["gold_answer_en"].astype(str).tolist()
        hyps  = valid_df["back_en"].astype(str).tolist()
        # Replace empty strings to prevent silent NaN bugs
        refs  = [r if r.strip() else "empty" for r in refs]
        hyps  = [h if h.strip() else "empty" for h in hyps]

        scores = bertscore_batch(refs, hyps, batch_size=args.batch_size)
        df.loc[valid_mask, "bertscore_f1"] = scores
        print("done")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Save
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/4] Saving results …")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"✓ Saved → {args.output}")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("QUALITY SUMMARY (valid rows only)")
    print("=" * 65)
    if valid_n > 0 and "model" in df.columns:
        summary = (
            df[valid_mask]
            .groupby("model")[["bertscore_f1", "rougeL_f1", "keyword_hit_rate"]]
            .mean()
            .round(4)
        )
        print(summary.to_string())
    print("\nCOVERAGE SUMMARY")
    print("=" * 65)
    if "model" in df.columns:
        print(cov.to_string())
    print("=" * 65)


if __name__ == "__main__":
    main()
