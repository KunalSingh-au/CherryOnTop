#!/usr/bin/env python3
"""
Run 100 questions × 4 conditions × 4 models = 1600 generations (with full gold + PDF coverage).

All conditions use the same official Hindi question text (question_hi_official).

C1: official Hindi Q + English document
C2: official Hindi Q + Sarvam machine-translated Hindi document
C3: official Hindi Q only
C4: official Hindi Q + official Hindi document (human-translated PDF text)

Resume: existing rows in outputs/runs/all_runs.csv are skipped.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    QA_GOLD_PATH,
    EXTRACTED_EN,
    TRANSLATED_SARVAM,
    EXTRACTED_HI,
    RUNS_CSV,
    CONDITIONS,
    MODELS,
)
from utils.llm import run_qa

COLS = [
    "doc_id",
    "ministry",
    "question_id",
    "condition",
    "model",
    "question_hi_used",
    "gold_answer_en",
    "keywords_en",
    "context_preview",
    "answer_hi",
]


def load_jsonl_map(path: str, key: str, field: str) -> dict:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r[key]] = r.get(field, "")
    return out


def load_done_keys(path: str) -> set[tuple]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {(r["doc_id"], r["question_id"], r["condition"], r["model"]) for r in rows}


def official_question(qa: dict) -> str:
    return (qa.get("question_hi_official") or "").strip()


def context_for(
    condition: str,
    qa: dict,
    q_hi: str,
    eng_by_doc: dict,
    sarvam_by_doc: dict,
    hi_by_doc: dict,
) -> str:
    """Returns context text for this condition (question is always q_hi from caller)."""
    did = qa["doc_id"]
    if condition == "C1":
        return eng_by_doc.get(did, "")
    if condition == "C2":
        return sarvam_by_doc.get(did, "")
    if condition == "C3":
        return ""
    if condition == "C4":
        return hi_by_doc.get(did, "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-condition", choices=list(CONDITIONS), default=None)
    ap.add_argument("--only-model", choices=list(MODELS), default=None)
    ap.add_argument("--limit", type=int, default=0, help="Max new runs (0 = no limit)")
    args = ap.parse_args()

    if not os.path.exists(QA_GOLD_PATH):
        print(f"Missing {QA_GOLD_PATH}. Run: python scripts/build_gold_master.py")
        sys.exit(1)

    with open(QA_GOLD_PATH, encoding="utf-8") as f:
        qa_rows = list(csv.DictReader(f))

    eng = load_jsonl_map(EXTRACTED_EN, "doc_id", "answer_en")
    sarv = load_jsonl_map(TRANSLATED_SARVAM, "doc_id", "answer_hi")
    hi = load_jsonl_map(EXTRACTED_HI, "doc_id", "answer_hi")

    done = load_done_keys(RUNS_CSV)
    os.makedirs(os.path.dirname(RUNS_CSV), exist_ok=True)
    new_count = 0
    file_exists = os.path.exists(RUNS_CSV) and os.path.getsize(RUNS_CSV) > 0
    write_header = not file_exists
    mode = "a" if file_exists else "w"

    conditions = [args.only_condition] if args.only_condition else list(CONDITIONS)
    models = [args.only_model] if args.only_model else list(MODELS)

    with open(RUNS_CSV, mode, newline="", encoding="utf-8-sig") as fout:
        w = csv.DictWriter(fout, fieldnames=COLS)
        if write_header:
            w.writeheader()

        for qa in qa_rows:
            q_hi = official_question(qa)
            if not q_hi:
                print(f"SKIP {qa.get('doc_id')} {qa.get('question_id')} — empty question_hi_official")
                continue
            for cond in conditions:
                if cond == "C4" and str(qa.get("c4_available", "false")).lower() != "true":
                    continue
                ctx = context_for(cond, qa, q_hi, eng, sarv, hi)
                if cond == "C1" and not (ctx or "").strip():
                    print(f"SKIP {qa['doc_id']} {qa['question_id']} {cond} — empty context")
                    continue
                if cond == "C2" and not (ctx or "").strip():
                    print(f"SKIP {qa['doc_id']} {qa['question_id']} {cond} — empty context")
                    continue
                if cond == "C4" and not (ctx or "").strip():
                    print(f"SKIP {qa['doc_id']} {qa['question_id']} {cond} — empty context")
                    continue
                for model in models:
                    key = (qa["doc_id"], qa["question_id"], cond, model)
                    if key in done:
                        continue
                    print(
                        f"RUN {qa['doc_id']} {qa['question_id']} {cond} {model}...",
                        flush=True,
                    )
                    ans = run_qa(model, ctx, q_hi, cond)
                    preview = (ctx or "")[:200].replace("\n", " ")
                    w.writerow(
                        {
                            "doc_id": qa["doc_id"],
                            "ministry": qa["ministry"],
                            "question_id": qa["question_id"],
                            "condition": cond,
                            "model": model,
                            "question_hi_used": q_hi,
                            "gold_answer_en": qa.get("gold_answer_en", ""),
                            "keywords_en": qa.get("keywords_en", ""),
                            "context_preview": preview,
                            "answer_hi": ans,
                        }
                    )
                    fout.flush()
                    done.add(key)
                    new_count += 1
                    if args.limit and new_count >= args.limit:
                        print(f"Stopped after --limit {args.limit}")
                        return

    print(f"\nDone. New rows this session: {new_count}. Total file: {RUNS_CSV}")


if __name__ == "__main__":
    main()
