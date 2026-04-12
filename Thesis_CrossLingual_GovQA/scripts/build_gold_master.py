#!/usr/bin/env python3
"""
Merge Draft_2 gold_qa_complete.csv + gold_qa_official_hindi.csv into gold_qa_master.csv.
Excludes doc_id AU3868 (not part of the thesis set).

Columns:
  doc_id, ministry, question_id, question_hi_official,
  gold_answer_en, keywords_en, doc_available, c4_available

Only official Hindi question text is kept (from gold_qa_official_hindi.csv).
Rows without an official question are skipped.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import QA_GOLD_PATH

# Dropped from the thesis set (no source document / placeholder rows).
EXCLUDED_DOC_IDS = frozenset({"AU3868"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--complete",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "..",
            "Draft_2",
            "gold_qa_complete.csv",
        ),
    )
    ap.add_argument(
        "--official",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "..",
            "Draft_2",
            "gold_qa_official_hindi.csv",
        ),
    )
    ap.add_argument("--out", default=QA_GOLD_PATH)
    args = ap.parse_args()

    complete_path = os.path.abspath(args.complete)
    official_path = os.path.abspath(args.official)

    with open(complete_path, encoding="utf-8") as f:
        complete = list(csv.DictReader(f))
    with open(official_path, encoding="utf-8") as f:
        official = list(csv.DictReader(f))

    off_map = {}
    for r in official:
        k = (r["doc_id"], r["question_id"])
        off_map[k] = r

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        "doc_id",
        "ministry",
        "question_id",
        "question_hi_official",
        "gold_answer_en",
        "keywords_en",
        "doc_available",
        "c4_available",
    ]
    rows = []
    skipped_no_official = 0
    for r in complete:
        if r.get("doc_id") in EXCLUDED_DOC_IDS:
            continue
        k = (r["doc_id"], r["question_id"])
        o = off_map.get(k)
        q_off = ((o or {}).get("question_hi", "") or "").strip()
        if not q_off:
            skipped_no_official += 1
            continue
        doc_ok = str(r.get("doc_available", "True")).lower() == "true"
        c4_ok = bool(o) and doc_ok and "MISSING" not in (r.get("gold_answer_en") or "")
        rows.append(
            {
                "doc_id": r["doc_id"],
                "ministry": r["ministry"],
                "question_id": r["question_id"],
                "question_hi_official": q_off,
                "gold_answer_en": r["gold_answer_en"],
                "keywords_en": r["keywords_en"],
                "doc_available": doc_ok,
                "c4_available": c4_ok,
            }
        )

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_c4 = sum(1 for x in rows if str(x["c4_available"]).lower() == "true")
    print(f"Wrote {len(rows)} rows → {args.out}")
    print(f"C4 available for {n_c4} rows (non-missing gold + doc_available)")
    if skipped_no_official:
        print(f"Skipped {skipped_no_official} rows with no official Hindi question")


if __name__ == "__main__":
    main()
