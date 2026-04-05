#!/usr/bin/env python3
"""
01_extract_pdfs.py
Walk through data/raw_pdfs/1_Ayush/, 2_Education/, etc.
Match each PDF to its doc_id by AU-number prefix.
Output: data/extracted/english_docs.jsonl
"""

import os, json, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import PDF_DIR, EXTRACTED_PATH, MINISTRY_MAP
from utils.extract import read_pdf, parse_english_qa

os.makedirs(os.path.dirname(EXTRACTED_PATH), exist_ok=True)

records = []
found = []

# Walk each ministry subfolder
for subfolder, ministry_label in MINISTRY_MAP.items():
    folder_path = os.path.join(PDF_DIR, subfolder)
    if not os.path.isdir(folder_path):
        print(f"  WARNING: folder not found: {folder_path}")
        continue

    pdfs = sorted(f for f in os.listdir(folder_path) if f.endswith(".pdf"))
    print(f"\n{subfolder} ({len(pdfs)} PDFs)")

    for fname in pdfs:
        # Extract doc_id: AU2338_kTs52v.pdf → AU2338
        m = re.match(r"(AU\d+)", fname)
        if not m:
            print(f"  SKIP: cannot parse doc_id from {fname}")
            continue
        doc_id = m.group(1)
        path = os.path.join(folder_path, fname)

        text = read_pdf(path)
        if text.startswith("PDF_ERROR"):
            print(f"  ERROR: {fname} — {text}")
            continue

        question, answer = parse_english_qa(text)
        status = "✓" if answer else "⚠ no answer"
        print(f"  {doc_id}: Q={len(question)}c A={len(answer)}c {status}")

        records.append({
            "doc_id":      doc_id,
            "ministry":    ministry_label,
            "filename":    fname,
            "question_en": question,
            "answer_en":   answer,
        })
        found.append(doc_id)

with open(EXTRACTED_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nExtracted {len(records)} documents → {EXTRACTED_PATH}")
print(f"Doc IDs: {sorted(found)}")
print("\nNext: add gold_answer_en to data/qa_gold/gold_qa.csv")
print("Then: python scripts/02_translate_docs.py")
