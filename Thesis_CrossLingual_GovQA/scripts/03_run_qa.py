#!/usr/bin/env python3
"""
Run N questions × 4 conditions × 4 models (default N=100, full PDF coverage → 1600 calls).

All conditions use question_hi_official. Resume: skips keys already in the runs CSV.

See RUNBOOK.txt for setup and sample commands.
"""

import argparse
import csv
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
from utils.jsonl import load_docs_jsonl
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

CONDS_NEED_CONTEXT = frozenset({"C1", "C2", "C4"})


def load_done_keys(path: str) -> set[tuple]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {(r["doc_id"], r["question_id"], r["condition"], r["model"]) for r in rows}


def official_question(qa: dict) -> str:
    return (qa.get("question_hi_official") or "").strip()


def context_for(condition: str, qa: dict, eng: dict, sarv: dict, hi: dict) -> str:
    did = qa["doc_id"]
    if condition == "C1":
        return eng.get(did, "")
    if condition == "C2":
        return sarv.get(did, "")
    if condition == "C3":
        return ""
    if condition == "C4":
        return hi.get(did, "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-condition", choices=list(CONDITIONS), default=None)
    ap.add_argument("--only-model", choices=list(MODELS), default=None)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many new API rows (across all Q×condition×model). 0 = no cap.",
    )
    ap.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Only first K rows from gold CSV (after load). 0 = all rows.",
    )
    ap.add_argument(
        "--runs-csv",
        default=None,
        help=f"Output CSV path (default: {RUNS_CSV}). Use for sample runs.",
    )
    args = ap.parse_args()

    runs_path = args.runs_csv or RUNS_CSV

    if not os.path.exists(QA_GOLD_PATH):
        print(f"Missing {QA_GOLD_PATH}. Run: python scripts/build_gold_master.py")
        sys.exit(1)

    with open(QA_GOLD_PATH, encoding="utf-8") as f:
        qa_rows = list(csv.DictReader(f))
    if args.max_questions and args.max_questions > 0:
        qa_rows = qa_rows[: args.max_questions]

    eng = load_docs_jsonl(EXTRACTED_EN, "answer_en")
    sarv = load_docs_jsonl(TRANSLATED_SARVAM, "answer_hi")
    hi = load_docs_jsonl(EXTRACTED_HI, "answer_hi")

    done = load_done_keys(runs_path)
    os.makedirs(os.path.dirname(runs_path), exist_ok=True)
    new_count = 0
    file_exists = os.path.exists(runs_path) and os.path.getsize(runs_path) > 0
    write_header = not file_exists
    mode = "a" if file_exists else "w"

    conditions = [args.only_condition] if args.only_condition else list(CONDITIONS)
    models = [args.only_model] if args.only_model else list(MODELS)

    with open(runs_path, mode, newline="", encoding="utf-8-sig") as fout:
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
                ctx = context_for(cond, qa, eng, sarv, hi)
                if cond in CONDS_NEED_CONTEXT and not (ctx or "").strip():
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

    print(f"\nDone. New rows this session: {new_count}. Output: {runs_path}")


if __name__ == "__main__":
    main()
