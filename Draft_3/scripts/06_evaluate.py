#!/usr/bin/env python3
"""
06_evaluate.py
Compute all metrics for every pipeline result.

Reads:  outputs/pipeline_*/*_bt.csv
Writes: outputs/pipeline_*/*_scored.csv
        outputs/evaluation/all_scores.csv
"""

import os, csv, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OPENAI_KEY
from utils.metrics import (
    chrf_score, bertscore_batch,
    ne_preservation, numeric_fidelity, acronym_fidelity,
    semantic_score_gpt
)

os.makedirs("outputs/evaluation", exist_ok=True)

all_rows = []

patterns = [
    "outputs/pipeline_a/*_bt.csv",
    "outputs/pipeline_b/*_bt.csv",
    "outputs/pipeline_c/*_bt.csv",
]

for pattern in sorted(glob.glob(p) for p in patterns):
    for path in sorted(pattern):
        print(f"\nEvaluating: {path}")

        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        # Determine pipeline label from path
        pipeline = "A" if "pipeline_a" in path else \
                   "B" if "pipeline_b" in path else "C"

        # Collect hypotheses and references for batch BERTScore
        hypotheses = [r.get("back_translated_en", "") for r in rows]
        references = [r.get("gold_answer_en", "")      for r in rows]

        print(f"  Running BERTScore on {len(rows)} pairs...")
        bert_scores = bertscore_batch(
            [h if h else " " for h in hypotheses],
            [r if r else " " for r in references]
        )

        scored_rows = []
        for i, row in enumerate(rows):
            orig  = row.get("gold_answer_en", "")
            back  = row.get("back_translated_en", "")

            if not back:
                metrics = {
                    "chrf": 0, "bertscore_f1": 0,
                    "ne_preservation": None,
                    "numeric_fidelity": None,
                    "acronym_fidelity": None,
                    "semantic_score": -1,
                    "semantic_reasoning": "No back-translation available"
                }
            else:
                metrics = {
                    "chrf":             round(chrf_score(back, orig), 2),
                    "bertscore_f1":     round(bert_scores[i], 4),
                    "ne_preservation":  ne_preservation(orig, back),
                    "numeric_fidelity": numeric_fidelity(orig, back),
                    "acronym_fidelity": acronym_fidelity(orig, back),
                }
                # Round optional metrics
                for k in ["ne_preservation", "numeric_fidelity", "acronym_fidelity"]:
                    if metrics[k] is not None:
                        metrics[k] = round(metrics[k], 3)

                # GPT semantic score
                print(f"  [{i+1}/{len(rows)}] semantic scoring...", end="\r")
                sem = semantic_score_gpt(orig, back, OPENAI_KEY)
                metrics["semantic_score"]     = sem.get("score", -1)
                metrics["semantic_reasoning"] = sem.get("reasoning", "")

            scored = {**row, "pipeline": pipeline, **metrics}
            scored_rows.append(scored)
            all_rows.append(scored)

        print()

        # Save scored file
        out_path = path.replace("_bt.csv", "_scored.csv")
        if scored_rows:
            with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(scored_rows[0].keys()))
                w.writeheader()
                w.writerows(scored_rows)
            print(f"  → {out_path}")

# Save combined file
if all_rows:
    combined_path = "outputs/evaluation/all_scores.csv"
    with open(combined_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nAll scores combined → {combined_path}")

print("\nNext: python scripts/07_results_table.py")
