#!/usr/bin/env python3
"""
02_translate_docs.py
Translate all English documents to Hindi using Sarvam and GPT-4o.
Output: data/translated/sarvam_hi.jsonl
        data/translated/gpt_hi.jsonl

Long answers are split into sentences, translated sentence by sentence,
then rejoined — this respects the 900 char API limit.
"""

import os, json, time, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EXTRACTED_DIR, TRANSLATED_DIR
from utils.translate import sarvam_en_to_hi, gpt_en_to_hi
from utils.extract import split_sentences

os.makedirs(TRANSLATED_DIR, exist_ok=True)

# Load extracted docs
docs_path = os.path.join(EXTRACTED_DIR, "english_docs.jsonl")
with open(docs_path, encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

print(f"Translating {len(docs)} documents...\n")


def translate_long_text(text: str, translate_fn) -> str:
    """
    Split text into sentences, translate each, rejoin.
    Handles the 900-char Sarvam limit gracefully.
    """
    sentences = split_sentences(text, min_len=10)
    if not sentences:
        return translate_fn(text[:900])

    translated = []
    for i, sent in enumerate(sentences):
        result = translate_fn(sent)
        translated.append(result)
        print(f"      sentence {i+1}/{len(sentences)}", end="\r")
    print()
    return " ".join(translated)


# Translate with both models
for model_name, translate_fn, out_file in [
    ("Sarvam",  sarvam_en_to_hi, "sarvam_hi.jsonl"),
    ("GPT-4o",  gpt_en_to_hi,    "gpt_hi.jsonl"),
]:
    out_path = os.path.join(TRANSLATED_DIR, out_file)
    print(f"\n{'─'*50}")
    print(f"Model: {model_name}")
    print(f"{'─'*50}")

    results = []
    for doc in docs:
        print(f"\n  {doc['doc_id']}")

        print(f"    Translating question...")
        q_hi = translate_fn(doc["question_en"][:900])

        print(f"    Translating answer ({len(doc['answer_en'])} chars)...")
        a_hi = translate_long_text(doc["answer_en"], translate_fn)

        results.append({
            "doc_id":      doc["doc_id"],
            "ministry":    doc["ministry"],
            "model":       model_name,
            "question_hi": q_hi,
            "answer_hi":   a_hi,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  Saved → {out_path}")

print("\nDone. Next: python scripts/03_translate_gpu.py")
print("(or skip to 04 if GPU is not available)")
