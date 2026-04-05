#!/usr/bin/env python3
"""
02_translate_docs.py
Translate all extracted English answer texts to Hindi.
Uses Sarvam and GPT-4o.
Output: data/translated/sarvam_hi.jsonl
        data/translated/gpt_hi.jsonl
"""

import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EXTRACTED_PATH, TRANSLATED_SARVAM, TRANSLATED_GPT
from utils.translate import sarvam_en_to_hi, gpt_en_to_hi, translate_long_text

os.makedirs("data/translated", exist_ok=True)

with open(EXTRACTED_PATH, encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

print(f"Translating {len(docs)} documents\n")

for model_name, fn, out_path in [
    ("Sarvam",  sarvam_en_to_hi, TRANSLATED_SARVAM),
    ("GPT-4o",  gpt_en_to_hi,    TRANSLATED_GPT),
]:
    print(f"{'─'*45}")
    print(f"Model: {model_name}")

    results = []
    for doc in docs:
        print(f"  {doc['doc_id']} ({doc['ministry']})")
        answer_hi = translate_long_text(doc["answer_en"], fn)
        results.append({
            "doc_id":    doc["doc_id"],
            "ministry":  doc["ministry"],
            "model":     model_name,
            "answer_hi": answer_hi,
        })
        print(f"    → {len(answer_hi)} chars Hindi")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved → {out_path}\n")

print("Done. Next: python scripts/03_run_pipelines.py")
