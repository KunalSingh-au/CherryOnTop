#!/usr/bin/env python3
"""
04_evaluate.py  — fixed version
================================
Changes vs original:
  - No Gemini API, no Sarvam API — uses local IndicTrans2 + local LLM judge
  - keyword_hit_rate now called on answer_hi directly (not back_en)
  - BERTScore uses xlm-roberta-large (cross-lingual, correct for Hindi→English comparison)
  - Resume support: already-evaluated rows are skipped
  - Deduplication: removes duplicate sarvam runs before evaluating
  - Thread count reduced to 1 (local inference is serial, not API calls)

Run: python scripts/04_evaluate.py
     python scripts/04_evaluate.py --skip-judge     (skip LLM judge, much faster)
     python scripts/04_evaluate.py --runs-csv outputs/runs/sample_runs.csv --out outputs/evaluation/sample_results.csv
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RUNS_CSV, EVAL_CSV, TRANSLATED_SARVAM, TRANSLATED_INDICTRANS, EXTRACTED_HI
from utils.jsonl import load_docs_jsonl
from utils.llm import sarvam_hi_to_en
from utils.metrics import (
    keyword_hit_rate, rouge_l_f1,
    bertscore_multilingual_batch,
    doc_fidelity_chrf,
    llm_judge_hallucination, hallucination_numeric,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip local LLM judge (saves time, no GPU judge model needed)")
    parser.add_argument("--runs-csv", default=RUNS_CSV)
    parser.add_argument("--out",      default=EVAL_CSV)
    args = parser.parse_args()

    if not os.path.exists(args.runs_csv):
        print(f"ERROR: {args.runs_csv} not found. Run 03_run_qa.py first.")
        sys.exit(1)

    # ── Load and deduplicate ───────────────────────────────────────────────────
    df = pd.read_csv(args.runs_csv, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {args.runs_csv}")

    before = len(df)
    df = df.drop_duplicates(
        subset=["doc_id", "question_id", "condition", "model"], keep="last"
    ).copy()
    if len(df) < before:
        print(f"Deduplicated: {before} → {len(df)} rows")

    # ── Resume: skip already-evaluated rows ────────────────────────────────────
    if os.path.exists(args.out):
        done_df = pd.read_csv(args.out, encoding="utf-8-sig")
        done_keys = set(
            f"{r.doc_id}_{r.question_id}_{r.condition}_{r.model}"
            for r in done_df.itertuples()
        )
        df["_key"] = (
            df["doc_id"].astype(str) + "_" + df["question_id"].astype(str)
            + "_" + df["condition"] + "_" + df["model"]
        )
        df = df[~df["_key"].isin(done_keys)].copy()
        print(f"Resuming: {len(done_keys)} already done, {len(df)} remaining")

    if df.empty:
        print("All rows already evaluated!")
        return

    # ── Load reference docs ────────────────────────────────────────────────────
    sarv_docs  = load_docs_jsonl(TRANSLATED_SARVAM,     "answer_hi")
    indic_docs = load_docs_jsonl(TRANSLATED_INDICTRANS, "answer_hi")
    hi_docs    = load_docs_jsonl(EXTRACTED_HI,          "answer_hi")
    print(f"Ref docs: Sarvam={len(sarv_docs)} IndicTrans2={len(indic_docs)} Hindi={len(hi_docs)}")

    # ── Step 1: Back-translate all Hindi answers → English (for ROUGE-L) ──────
    print(f"\nStep 1/4: Back-translating {len(df)} Hindi answers → English (local IndicTrans2)...")
    print("  (First call loads the model — ~30s)")
    back_ens = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Back-translating"):
        ans = str(row.get("answer_hi", ""))
        back_ens.append(sarvam_hi_to_en(ans) if ans.strip() else "")
    df["back_en"] = back_ens

    # ── Step 2: BERTScore batch ────────────────────────────────────────────────
    print("\nStep 2/4: BERTScore (xlm-roberta-large, cross-lingual)...")
    print("  refs=gold_answer_en (English), hyps=answer_hi (Hindi) — cross-lingual comparison")
    berts = bertscore_multilingual_batch(
        df["gold_answer_en"].fillna("").tolist(),
        df["answer_hi"].fillna("").tolist(),
    )
    df["bertscore_f1"] = berts

    # ── Step 3: Rule-based metrics (instant) ──────────────────────────────────
    print("\nStep 3/4: Rule-based metrics (keyword hit rate, ROUGE-L, chrF)...")
    df["keyword_hit_rate"] = df.apply(
        lambda r: keyword_hit_rate(
            str(r.get("keywords_en", "")),
            str(r.get("answer_hi", "")),   # ← on Hindi directly
        ),
        axis=1,
    )
    df["rougeL_f1"] = df.apply(
        lambda r: rouge_l_f1(
            str(r.get("gold_answer_en", "")),
            str(r.get("back_en", "")),
        ),
        axis=1,
    )
    df["doc_fidelity_chrf"] = df.apply(
        lambda r: (
            doc_fidelity_chrf(
                sarv_docs.get(r["doc_id"], ""),
                hi_docs.get(r["doc_id"], ""),
            ) if r["condition"] == "C2"
            else doc_fidelity_chrf(
                indic_docs.get(r["doc_id"], ""),
                hi_docs.get(r["doc_id"], ""),
            ) if r["condition"] == "C5"
            else float("nan")
        ),
        axis=1,
    )
    print(f"  Mean keyword_hit_rate: {df['keyword_hit_rate'].mean():.3f}")
    print(f"  Mean rougeL_f1:        {df['rougeL_f1'].mean():.3f}")
    print(f"  Mean bertscore_f1:     {df['bertscore_f1'].mean():.3f}")

    # ── Step 4: Local LLM judge ────────────────────────────────────────────────
    if args.skip_judge:
        print("\nStep 4/4: Judge SKIPPED (--skip-judge)")
        df["hallucination_label"]    = "skipped"
        df["hallucination_rationale"] = "skipped"
        df["hallucination_score"]    = float("nan")
    else:
        print(f"\nStep 4/4: Local LLM judge ({len(df)} rows)...")
        print("  (Loads Qwen-2.5-7B-Instruct once, then runs serially)")
        labels, rationales, scores = [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
            result = llm_judge_hallucination(
                str(row.get("gold_answer_en", "")),
                str(row.get("back_en", "")),
            )
            label = result.get("label", "unknown")
            labels.append(label)
            rationales.append(result.get("rationale", ""))
            scores.append(hallucination_numeric(label))

        df["hallucination_label"]     = labels
        df["hallucination_rationale"] = rationales
        df["hallucination_score"]     = scores

        valid = [s for s in scores if not pd.isna(s)]
        if valid:
            print(f"  Mean judge score: {sum(valid)/len(valid):.3f}")
            hall_rate = sum(1 for s in valid if s < 1.0) / len(valid)
            print(f"  Hallucination rate (minor+major): {hall_rate:.1%}")

    # ── Save ───────────────────────────────────────────────────────────────────
    # Drop internal key column if present
    df = df.drop(columns=["_key"], errors="ignore")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Append to existing file if resuming
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out, encoding="utf-8-sig")
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n✓ Saved {len(df)} evaluated rows → {args.out}")
    print("Next: python scripts/05_analyze.py")


if __name__ == "__main__":
    main()
