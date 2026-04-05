#!/usr/bin/env python3
"""
04_evaluate.py
For every pipeline result:
  1. Back-translate Hindi answer → English
  2. Compute chrF, BERTScore, NE preservation, numeric fidelity, acronym fidelity
  3. GPT-4o semantic score (0-5) vs gold answer
  4. Keyword match check
Output: outputs/evaluation/all_results.csv
"""

import os, csv, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OPENAI_KEY, EVAL_ALL
from utils.translate import sarvam_hi_to_en
from utils.metrics import (chrf_score, bertscore_batch, ne_preservation,
                            numeric_fidelity, acronym_fidelity,
                            semantic_score_gpt, keyword_match)

os.makedirs("outputs/evaluation", exist_ok=True)

INPUT_FILES = [
    ("outputs/pipeline_a/sarvam.csv", "A"),
    ("outputs/pipeline_a/gpt.csv",    "A"),
    ("outputs/pipeline_b/sarvam.csv", "B"),
    ("outputs/pipeline_b/gpt.csv",    "B"),
]

all_rows = []

for input_path, pipeline_label in INPUT_FILES:
    if not os.path.exists(input_path):
        print(f"SKIP (not found): {input_path}")
        continue

    print(f"\n{'─'*55}")
    print(f"Evaluating: {input_path}")

    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        continue

    total = len(rows)

    # Step 1: Back-translate all Hindi answers
    print(f"  Back-translating {total} answers...")
    for i, row in enumerate(rows, 1):
        answer_hi = row.get("answer_hi", "")
        if answer_hi and "ERROR" not in answer_hi:
            row["back_en"] = sarvam_hi_to_en(answer_hi)
        else:
            row["back_en"] = ""
        print(f"  {i}/{total}", end="\r")
    print()

    # Step 2: BERTScore batch (fast)
    golds = [r.get("gold_answer_en", "") for r in rows]
    backs = [r.get("back_en", "")        for r in rows]
    print(f"  Computing BERTScore...")
    bert_scores = bertscore_batch(backs, golds)

    # Step 3: All other metrics per row
    scored = []
    for i, row in enumerate(rows, 1):
        gold = row.get("gold_answer_en", "")
        back = row.get("back_en", "")
        keywords = row.get("keywords_en", "")

        print(f"  Scoring {i}/{total}...", end="\r")

        # Automated metrics
        metrics = {
            "pipeline":          pipeline_label,
            "back_en":           back,
            "chrf":              round(chrf_score(back, gold), 2) if back else 0,
            "bertscore_f1":      bert_scores[i-1],
            "ne_preservation":   ne_preservation(gold, back),
            "numeric_fidelity":  numeric_fidelity(gold, back),
            "acronym_fidelity":  acronym_fidelity(gold, back),
            "keyword_match":     keyword_match(gold, back, keywords) if gold else None,
        }

        # GPT semantic score
        if gold and back and OPENAI_KEY != "your_openai_key_here":
            sem = semantic_score_gpt(gold, back, OPENAI_KEY)
        else:
            sem = {"score": -1, "reasoning": "no gold answer or API key"}

        metrics["semantic_score"]     = sem.get("score", -1)
        metrics["semantic_reasoning"] = sem.get("reasoning", "")

        scored.append({**row, **metrics})
        all_rows.append({**row, **metrics})

    print()

# Save combined results
if all_rows:
    fieldnames = list(all_rows[0].keys())
    with open(EVAL_ALL, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nAll results → {EVAL_ALL} ({len(all_rows)} rows)")

print("Next: python scripts/05_results_table.py")
