#!/usr/bin/env python3
"""
05_backtranslate.py
Back-translate all Hindi answers to English so we can compare against
the gold English answer using automated metrics.

Reads all CSV files from outputs/pipeline_*/
Adds column: back_translated_en
Saves to outputs/pipeline_*/  with _bt suffix
"""

import os, csv, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.translate import sarvam_hi_to_en

def backtranslate_file(input_path: str):
    output_path = input_path.replace(".csv", "_bt.csv")

    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    fieldnames = list(rows[0].keys()) + ["back_translated_en"]
    total = len(rows)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            answer_hi = row.get("answer_hi", "")
            if not answer_hi or "ERROR" in answer_hi:
                row["back_translated_en"] = ""
            else:
                print(f"  [{i}/{total}] back-translating...", end="\r")
                row["back_translated_en"] = sarvam_hi_to_en(answer_hi)

            writer.writerow(row)

    print(f"\n  Saved → {output_path}")


# Find all pipeline output CSVs (not already back-translated)
patterns = [
    "outputs/pipeline_a/*.csv",
    "outputs/pipeline_b/*.csv",
    "outputs/pipeline_c/*.csv",
]

for pattern in patterns:
    for path in sorted(glob.glob(pattern)):
        if path.endswith("_bt.csv"):
            continue
        print(f"\nBack-translating: {path}")
        backtranslate_file(path)

print("\nDone. Next: python scripts/06_evaluate.py")
