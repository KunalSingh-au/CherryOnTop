#!/usr/bin/env python3
"""
Extract English and official Hindi answer text from PDFs.

English:  data/raw_pdfs/<MINISTRY_MAP key>/*.pdf  (AUxxxx_*.pdf)
Hindi:    data/raw_pdfs_hi/<same subfolders>/*.pdf
Output:   data/extracted/english_docs.jsonl, hindi_official_docs.jsonl
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    PDF_DIR_EN,
    PDF_DIR_HI,
    MINISTRY_MAP,
    EXTRACTED_EN,
    EXTRACTED_HI,
)
from utils.extract import read_pdf, parse_english_qa, parse_hindi_qa


def extract_tree(base_dir: str, ministry_map: dict, lang: str) -> list[dict]:
    records = []
    for subfolder, ministry_label in ministry_map.items():
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: missing folder {folder_path}")
            continue
        pdfs = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))
        print(f"\n{subfolder} ({len(pdfs)} PDFs) [{lang}]")
        for fname in pdfs:
            m = re.match(r"(AU\d+)", fname)
            if not m:
                print(f"  SKIP {fname}")
                continue
            doc_id = m.group(1)
            path = os.path.join(folder_path, fname)
            text = read_pdf(path)
            if lang == "en":
                q, a = parse_english_qa(text)
            else:
                q, a = parse_hindi_qa(text)
            status = "✓" if a else "⚠ no answer"
            print(f"  {doc_id}: Q={len(q)}c A={len(a)}c {status}")
            rec = {
                "doc_id": doc_id,
                "ministry": ministry_label,
                "filename": fname,
                "question_raw": q,
            }
            if lang == "en":
                rec["answer_en"] = a
            else:
                rec["answer_hi"] = a
            records.append(rec)
    return records


def main():
    os.makedirs(os.path.dirname(EXTRACTED_EN), exist_ok=True)

    print("=== English PDFs ===")
    en_recs = extract_tree(PDF_DIR_EN, MINISTRY_MAP, "en")
    with open(EXTRACTED_EN, "w", encoding="utf-8") as f:
        for r in en_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n→ {EXTRACTED_EN} ({len(en_recs)} docs)")

    print("\n=== Hindi official PDFs ===")
    hi_recs = extract_tree(PDF_DIR_HI, MINISTRY_MAP, "hi")
    with open(EXTRACTED_HI, "w", encoding="utf-8") as f:
        for r in hi_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n→ {EXTRACTED_HI} ({len(hi_recs)} docs)")

    print("\nNext: python scripts/02_translate_docs.py")


if __name__ == "__main__":
    main()
