#!/usr/bin/env python3
"""
01_extract_pdfs.py
Extract English text from all Lok Sabha PDFs in data/raw_pdfs/
Output: data/extracted/english_docs.jsonl

PDF naming convention: ministry_N_en.pdf
Example: education_1_en.pdf, disability_3_en.pdf
"""

import os, json, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import PDF_DIR, EXTRACTED_DIR
from utils.extract import read_pdf, parse_loksabha_english

os.makedirs(EXTRACTED_DIR, exist_ok=True)

output_path = os.path.join(EXTRACTED_DIR, "english_docs.jsonl")
records = []

pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.endswith("_en.pdf"))
if not pdf_files:
    print(f"No *_en.pdf files found in {PDF_DIR}")
    print("Name your PDFs like: education_1_en.pdf")
    sys.exit(1)

print(f"Found {len(pdf_files)} English PDFs\n")

for fname in pdf_files:
    # Parse filename → doc_id and ministry
    # education_1_en.pdf → doc_id=education_1, ministry=education
    doc_id   = fname.replace("_en.pdf", "")
    ministry = re.sub(r"_\d+$", "", doc_id)

    path = os.path.join(PDF_DIR, fname)
    print(f"  Extracting: {fname}")

    raw_text = read_pdf(path)
    question, answer = parse_loksabha_english(raw_text)

    if not answer:
        print(f"  WARNING: Could not parse answer from {fname}")

    record = {
        "doc_id":      doc_id,
        "ministry":    ministry,
        "filename":    fname,
        "question_en": question,
        "answer_en":   answer,
    }
    records.append(record)
    print(f"    Q: {len(question)} chars | A: {len(answer)} chars")

with open(output_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nSaved {len(records)} documents → {output_path}")
print("\nNext: fill in data/qa_gold/gold_qa.csv with your questions")
print("Then run: python scripts/02_translate_docs.py")
