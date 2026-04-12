#!/usr/bin/env python3
"""
03_translate_gpu.py
Translate all English documents to Hindi using IndicTrans2 on local GPU.
Output: data/translated/indictrans2_hi.jsonl

Run this on your GPU server:
  python scripts/03_translate_gpu.py

Requires:
  pip install transformers sentencepiece sacremoses
  (model downloads ~2GB on first run)
"""

import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EXTRACTED_DIR, TRANSLATED_DIR
from utils.translate import indictrans_en_to_hi
from utils.extract import split_sentences

os.makedirs(TRANSLATED_DIR, exist_ok=True)

docs_path = os.path.join(EXTRACTED_DIR, "english_docs.jsonl")
with open(docs_path, encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

print(f"Translating {len(docs)} documents with IndicTrans2 (GPU)...\n")


def translate_in_chunks(text: str, chunk_size: int = 400) -> str:
    """
    IndicTrans2 handles longer texts but we chunk at 400 chars
    to stay within the 512 token model limit reliably.
    """
    sentences = split_sentences(text, min_len=10)
    if not sentences:
        return indictrans_en_to_hi(text[:400])

    translated = []
    for i, sent in enumerate(sentences):
        # If sentence is very long, hard-truncate
        result = indictrans_en_to_hi(sent[:400])
        translated.append(result)
        print(f"    sentence {i+1}/{len(sentences)}", end="\r")
    print()
    return " ".join(translated)


results = []
for doc in docs:
    print(f"  {doc['doc_id']}")
    print(f"    Translating question...")
    q_hi = indictrans_en_to_hi(doc["question_en"][:400])

    print(f"    Translating answer...")
    a_hi = translate_in_chunks(doc["answer_en"])

    results.append({
        "doc_id":      doc["doc_id"],
        "ministry":    doc["ministry"],
        "model":       "IndicTrans2",
        "question_hi": q_hi,
        "answer_hi":   a_hi,
    })

out_path = os.path.join(TRANSLATED_DIR, "indictrans2_hi.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nSaved → {out_path}")
print("Next: python scripts/04_run_pipelines.py")
