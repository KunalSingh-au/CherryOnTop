#!/usr/bin/env python3
"""
04_run_pipelines.py
Run all three QA pipelines over all 100 questions.

Pipeline A: English doc + Hindi question → Hindi answer  (no translation)
Pipeline B: API-translated Hindi doc + Hindi question → Hindi answer
Pipeline C: IndicTrans2 Hindi doc + Hindi question → Hindi answer

For each pipeline, we run both Sarvam and GPT-4o as the QA model.
Output: one CSV per pipeline in outputs/pipeline_*/
"""

import os, json, csv, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (EXTRACTED_DIR, TRANSLATED_DIR, QA_GOLD_PATH,
                    OUTPUT_A, OUTPUT_B_SARVAM, OUTPUT_B_GPT, OUTPUT_C)
from utils.translate import sarvam_qa, gpt_qa

os.makedirs("outputs/pipeline_a", exist_ok=True)
os.makedirs("outputs/pipeline_b", exist_ok=True)
os.makedirs("outputs/pipeline_c", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

# English docs
with open(os.path.join(EXTRACTED_DIR, "english_docs.jsonl"), encoding="utf-8") as f:
    eng_docs = {r["doc_id"]: r for r in (json.loads(l) for l in f)}

# Translated docs (API)
def load_translated(filename):
    path = os.path.join(TRANSLATED_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping")
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["doc_id"]: r for r in (json.loads(l) for l in f)}

sarvam_docs    = load_translated("sarvam_hi.jsonl")
gpt_docs       = load_translated("gpt_hi.jsonl")
indictrans_docs = load_translated("indictrans2_hi.jsonl")

# Gold QA pairs
with open(QA_GOLD_PATH, encoding="utf-8") as f:
    qa_pairs = list(csv.DictReader(f))

print(f"Loaded {len(qa_pairs)} QA pairs from gold_qa.csv")

# ── Output schema ─────────────────────────────────────────────────────────────

COLS = [
    "doc_id", "ministry", "question_id",
    "question_hi", "gold_answer_en",
    "context_type",   # what was passed as context
    "qa_model",       # sarvam or gpt
    "answer_hi",      # raw Hindi answer from QA model
]


def run_pipeline(output_path: str, context_getter, context_label: str, qa_fn, qa_label: str):
    """
    Generic pipeline runner.
    context_getter(doc_id) → context string
    qa_fn(context, question_hi) → Hindi answer
    """
    rows = []
    total = len(qa_pairs)

    for i, qa in enumerate(qa_pairs, 1):
        doc_id = qa["doc_id"]
        context = context_getter(doc_id)

        if not context:
            print(f"  SKIP {doc_id} — no context available")
            continue

        print(f"  [{i}/{total}] {doc_id} | {qa['question_id']}", end="  ")
        answer_hi = qa_fn(context, qa["question_hi"])
        print("✓")

        rows.append({
            "doc_id":        doc_id,
            "ministry":      qa.get("ministry", ""),
            "question_id":   qa["question_id"],
            "question_hi":   qa["question_hi"],
            "gold_answer_en": qa["gold_answer_en"],
            "context_type":  context_label,
            "qa_model":      qa_label,
            "answer_hi":     answer_hi,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"  → Saved {len(rows)} rows to {output_path}\n")


# ── Pipeline A — English doc, no translation ──────────────────────────────────
print("\n" + "="*55)
print("PIPELINE A — English doc + Hindi question")
print("="*55)

# A1: Sarvam QA on English doc
run_pipeline(
    "outputs/pipeline_a/sarvam.csv",
    context_getter=lambda did: eng_docs.get(did, {}).get("answer_en", ""),
    context_label="english_original",
    qa_fn=sarvam_qa,
    qa_label="sarvam"
)

# A2: GPT-4o QA on English doc
run_pipeline(
    "outputs/pipeline_a/gpt.csv",
    context_getter=lambda did: eng_docs.get(did, {}).get("answer_en", ""),
    context_label="english_original",
    qa_fn=gpt_qa,
    qa_label="gpt4o"
)


# ── Pipeline B — Sarvam-translated Hindi doc ──────────────────────────────────
print("="*55)
print("PIPELINE B — Sarvam-translated Hindi doc")
print("="*55)

run_pipeline(
    "outputs/pipeline_b/sarvam_context_sarvam_qa.csv",
    context_getter=lambda did: sarvam_docs.get(did, {}).get("answer_hi", ""),
    context_label="sarvam_translated",
    qa_fn=sarvam_qa,
    qa_label="sarvam"
)

run_pipeline(
    "outputs/pipeline_b/sarvam_context_gpt_qa.csv",
    context_getter=lambda did: sarvam_docs.get(did, {}).get("answer_hi", ""),
    context_label="sarvam_translated",
    qa_fn=gpt_qa,
    qa_label="gpt4o"
)

# GPT-translated Hindi doc
run_pipeline(
    "outputs/pipeline_b/gpt_context_gpt_qa.csv",
    context_getter=lambda did: gpt_docs.get(did, {}).get("answer_hi", ""),
    context_label="gpt_translated",
    qa_fn=gpt_qa,
    qa_label="gpt4o"
)


# ── Pipeline C — IndicTrans2-translated Hindi doc (GPU) ───────────────────────
if indictrans_docs:
    print("="*55)
    print("PIPELINE C — IndicTrans2 translated Hindi doc")
    print("="*55)

    run_pipeline(
        "outputs/pipeline_c/indictrans2_context_sarvam_qa.csv",
        context_getter=lambda did: indictrans_docs.get(did, {}).get("answer_hi", ""),
        context_label="indictrans2_translated",
        qa_fn=sarvam_qa,
        qa_label="sarvam"
    )

    run_pipeline(
        "outputs/pipeline_c/indictrans2_context_gpt_qa.csv",
        context_getter=lambda did: indictrans_docs.get(did, {}).get("answer_hi", ""),
        context_label="indictrans2_translated",
        qa_fn=gpt_qa,
        qa_label="gpt4o"
    )
else:
    print("Pipeline C skipped — no IndicTrans2 translations found")
    print("Run python scripts/03_translate_gpu.py first")

print("\nAll pipelines complete.")
print("Next: python scripts/05_backtranslate.py")
