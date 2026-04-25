"""
scripts/01_extract_docs.py — Extract text from English and Hindi PDFs.

What it does:
  - Walks data/raw_pdfs/english/ and data/raw_pdfs/hindi/
  - Extracts text from every PDF using pdfplumber
  - Parses out the answer section from each parliamentary document
  - Saves results as JSONL files in data/extracted/

Run once when you add new PDFs.  Safe to re-run (overwrites outputs).

Usage:
    python scripts/01_extract_docs.py
"""

import json
import os
import sys

# Add project root to path so config and utils are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import PDF_DIR_EN, PDF_DIR_HI, MINISTRY_MAP, EXTRACTED_EN, EXTRACTED_HI
from utils.extract import read_pdf, parse_english_qa, parse_hindi_qa


def extract_tree(base_dir: str, lang: str) -> list:
    """
    Walk a directory tree of PDFs and extract text from each.

    Expected structure:
      base_dir/
        1_Ayush/     ← subfolder name must match a key in MINISTRY_MAP
          doc1.pdf
        2_Education/
          doc2.pdf
        ...

    Parameters
    ----------
    base_dir : Root directory (PDF_DIR_EN or PDF_DIR_HI from config)
    lang     : "en" or "hi" — determines which parser is used

    Returns
    -------
    list of dicts: [{doc_id, ministry, answer_en|answer_hi}, ...]
    """
    records = []

    for subfolder, ministry_label in MINISTRY_MAP.items():
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(folder_path):
            print(f"  [WARN] Folder not found, skipping: {folder_path}")
            continue

        pdf_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".pdf"))
        for filename in pdf_files:
            doc_id = filename.replace(".pdf", "")
            full_path = os.path.join(folder_path, filename)

            raw_text = read_pdf(full_path)

            if raw_text.startswith("PDF_ERROR"):
                print(f"  [ERROR] Could not read {filename}: {raw_text}")
                continue

            # Parse out just the answer portion
            if lang == "en":
                _, answer = parse_english_qa(raw_text)
            else:
                _, answer = parse_hindi_qa(raw_text)

            records.append({
                "doc_id":   doc_id,
                "ministry": ministry_label,
                f"answer_{lang}": answer,
            })

    return records


def main():
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(EXTRACTED_EN), exist_ok=True)

    # ── Extract English PDFs ───────────────────────────────────────────────────
    print("Extracting English PDFs …")
    en_records = extract_tree(PDF_DIR_EN, "en")
    with open(EXTRACTED_EN, "w", encoding="utf-8") as f:
        for record in en_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(en_records)} English docs → {EXTRACTED_EN}")

    # ── Extract Hindi PDFs ─────────────────────────────────────────────────────
    print("Extracting Hindi PDFs …")
    hi_records = extract_tree(PDF_DIR_HI, "hi")
    with open(EXTRACTED_HI, "w", encoding="utf-8") as f:
        for record in hi_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(hi_records)} Hindi docs → {EXTRACTED_HI}")

    print("\n✓ Extraction complete.")


if __name__ == "__main__":
    main()
