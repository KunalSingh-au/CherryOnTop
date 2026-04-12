#!/usr/bin/env python3
"""
Sarvam EN→HI translation of extracted English answers (condition C2 context).
Input:  data/extracted/english_docs.jsonl
Output: data/translated/sarvam_hi.jsonl
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EXTRACTED_EN, TRANSLATED_SARVAM
from utils.llm import sarvam_en_to_hi, translate_long_text


def en_to_hi(text: str) -> str:
    return translate_long_text(text, sarvam_en_to_hi, max_chunk=900)


def main():
    if not os.path.exists(EXTRACTED_EN):
        print(f"Missing {EXTRACTED_EN}. Run scripts/01_extract_docs.py first.")
        sys.exit(1)

    with open(EXTRACTED_EN, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(TRANSLATED_SARVAM), exist_ok=True)
    results = []
    for doc in docs:
        print(f"  {doc['doc_id']} ({doc['ministry']})")
        answer_hi = en_to_hi(doc.get("answer_en", ""))
        results.append(
            {
                "doc_id": doc["doc_id"],
                "ministry": doc["ministry"],
                "model": "sarvam_translate",
                "answer_hi": answer_hi,
            }
        )
        print(f"    → {len(answer_hi)} chars")

    with open(TRANSLATED_SARVAM, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved → {TRANSLATED_SARVAM}")
    print("Next: python scripts/03_run_qa.py")


if __name__ == "__main__":
    main()
