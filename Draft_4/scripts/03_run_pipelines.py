#!/usr/bin/env python3
"""
03_run_pipelines.py
Run Pipeline A and Pipeline B over all 100 QA pairs.

Pipeline A: English doc + Hindi question → Hindi answer   (no translation)
Pipeline B: Translated Hindi doc + Hindi question → Hindi answer

For each pipeline, both Sarvam and GPT-4o are used as the QA model.
Output: outputs/pipeline_a/sarvam.csv  etc.
"""

import os, json, csv, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (EXTRACTED_PATH, TRANSLATED_SARVAM, TRANSLATED_GPT,
                    QA_GOLD_PATH, OUTPUT_A_SARVAM, OUTPUT_A_GPT,
                    OUTPUT_B_SARVAM, OUTPUT_B_GPT)
from utils.translate import sarvam_qa, gpt_qa

os.makedirs("outputs/pipeline_a", exist_ok=True)
os.makedirs("outputs/pipeline_b", exist_ok=True)

# ── Load all data ──────────────────────────────────────────────────────────────

with open(EXTRACTED_PATH, encoding="utf-8") as f:
    eng_docs = {r["doc_id"]: r for r in (json.loads(l) for l in f)}

def load_translated(path):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["doc_id"]: r for r in (json.loads(l) for l in f)}

sarvam_docs = load_translated(TRANSLATED_SARVAM)
gpt_docs    = load_translated(TRANSLATED_GPT)

with open(QA_GOLD_PATH, encoding="utf-8") as f:
    qa_pairs = list(csv.DictReader(f))

print(f"Loaded {len(qa_pairs)} QA pairs")
print(f"Loaded {len(eng_docs)} English docs")
print(f"Loaded {len(sarvam_docs)} Sarvam-translated docs")
print(f"Loaded {len(gpt_docs)} GPT-translated docs\n")

COLS = ["doc_id", "ministry", "question_id", "question_hi",
        "gold_answer_en", "context_type", "qa_model", "answer_hi"]


def run(output_path, context_fn, context_label, qa_fn, qa_label):
    total = len(qa_pairs)
    rows = []
    for i, qa in enumerate(qa_pairs, 1):
        ctx = context_fn(qa["doc_id"])
        if not ctx:
            print(f"  SKIP {qa['doc_id']} — no context")
            continue
        print(f"  [{i:3d}/{total}] {qa['doc_id']} {qa['question_id']}", end="  ")
        answer_hi = qa_fn(ctx, qa["question_hi"])
        # Strip <think> blocks if model outputs them
        import re
        answer_hi = re.sub(r"<think>.*?</think>", "", answer_hi,
                           flags=re.DOTALL).strip()
        print("✓")
        rows.append({
            "doc_id":        qa["doc_id"],
            "ministry":      qa.get("ministry", ""),
            "question_id":   qa["question_id"],
            "question_hi":   qa["question_hi"],
            "gold_answer_en": qa.get("gold_answer_en", ""),
            "context_type":  context_label,
            "qa_model":      qa_label,
            "answer_hi":     answer_hi,
        })
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {len(rows)} rows saved to {output_path}\n")


# ── Pipeline A — English context ──────────────────────────────────────────────
print("=" * 55)
print("PIPELINE A — English doc + Hindi question")
print("=" * 55)

run(OUTPUT_A_SARVAM,
    context_fn=lambda did: eng_docs.get(did, {}).get("answer_en", ""),
    context_label="english",
    qa_fn=sarvam_qa, qa_label="sarvam")

run(OUTPUT_A_GPT,
    context_fn=lambda did: eng_docs.get(did, {}).get("answer_en", ""),
    context_label="english",
    qa_fn=gpt_qa, qa_label="gpt4o")

# ── Pipeline B — Sarvam-translated Hindi context ──────────────────────────────
print("=" * 55)
print("PIPELINE B — Sarvam-translated Hindi doc")
print("=" * 55)

run(OUTPUT_B_SARVAM,
    context_fn=lambda did: sarvam_docs.get(did, {}).get("answer_hi", ""),
    context_label="sarvam_translated",
    qa_fn=sarvam_qa, qa_label="sarvam")

# ── Pipeline B — GPT-translated Hindi context ─────────────────────────────────
print("=" * 55)
print("PIPELINE B — GPT-translated Hindi doc")
print("=" * 55)

run(OUTPUT_B_GPT,
    context_fn=lambda did: gpt_docs.get(did, {}).get("answer_hi", ""),
    context_label="gpt_translated",
    qa_fn=gpt_qa, qa_label="gpt4o")

print("All pipelines done.")
print("Next: python scripts/04_evaluate.py")
