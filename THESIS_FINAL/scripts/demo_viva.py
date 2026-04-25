"""
scripts/demo_viva.py — Lightweight demo for viva / presentation.

Runs 10 questions through C1 (English full-doc) and C3 (closed-book) using a
small 7B model that fits on a single GPU or even CPU (slow but runnable).

Produces:
  - outputs/runs/demo_runs.csv        (raw answers)
  - outputs/evaluation/demo_results.csv  (scored results)
  - A printed table showing side-by-side Gold vs Model answers

No vLLM required — uses HuggingFace transformers pipeline directly,
so this runs on any machine with 16 GB RAM (CPU) or a single 16 GB GPU.

Usage:
    # Default: 10 questions, Qwen-2.5-7B-Instruct, with prompts
    python scripts/demo_viva.py

    # Custom model and question count
    python scripts/demo_viva.py --model Qwen/Qwen2.5-7B-Instruct --n 5

    # No prompts (baseline)
    python scripts/demo_viva.py --no-prompts
"""

import argparse
import csv
import os
import re
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    QA_GOLD_PATH, EXTRACTED_EN, TRANSLATED_SARVAM,
    TRANSLATED_INDICTRANS, EXTRACTED_HI,
)
from utils.jsonl import load_docs_jsonl
from utils.llm import build_qa_prompt, extract_answer_from_output
from utils.metrics import classify_response, keyword_hit_rate, rouge_l_f1

# ── Demo defaults ────────────────────────────────────────────────────────────
# ── TUNABLE: Swap for a different small model if you prefer ──────────────────
DEMO_MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
DEMO_CONDITIONS    = ["C1", "C3"]   # English full-doc + closed-book
DEMO_OUTPUT_RUNS   = "outputs/runs/demo_runs.csv"
DEMO_OUTPUT_EVAL   = "outputs/evaluation/demo_results.csv"


def load_pipeline(model_id: str, device: str):
    """Load a HuggingFace text-generation pipeline (no vLLM needed)."""
    from transformers import pipeline, AutoTokenizer

    print(f"Loading {model_id} on {device} …")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tok,
        device=device,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    return pipe, tok


def run_demo(model_id: str, n: int, use_prompt: bool, device: str):
    """Run the demo pipeline end-to-end."""
    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(QA_GOLD_PATH):
        print(f"Error: {QA_GOLD_PATH} not found.")
        sys.exit(1)

    qa_df = pd.read_csv(QA_GOLD_PATH).head(n)
    contexts = {
        "C1": load_docs_jsonl(EXTRACTED_EN,          "answer_en"),
        "C2": load_docs_jsonl(TRANSLATED_SARVAM,     "answer_hi"),
        "C4": load_docs_jsonl(EXTRACTED_HI,          "answer_hi"),
        "C5": load_docs_jsonl(TRANSLATED_INDICTRANS, "answer_hi"),
    }

    run_tag = "prompted" if use_prompt else "noprompt"

    # ── Build tasks ───────────────────────────────────────────────────────────
    tasks = []
    for _, row in qa_df.iterrows():
        did = row["doc_id"]
        for cond in DEMO_CONDITIONS:
            ctx = contexts.get(cond, {}).get(did, "") if cond != "C3" else ""
            tasks.append({
                "doc_id":         did,
                "ministry":       row["ministry"],
                "question_id":    row["question_id"],
                "condition":      cond,
                "question_hi":    row["question_hi_official"],
                "gold_answer_en": row["gold_answer_en"],
                "keywords_en":    row.get("keywords_en", ""),
                "context":        ctx,
                "prompt":         build_qa_prompt(
                                      row["question_hi_official"],
                                      ctx, cond, use_prompt=use_prompt
                                  ),
            })

    # ── Load model ────────────────────────────────────────────────────────────
    pipe, tok = load_pipeline(model_id, device)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(tasks)} tasks …")
    results = []

    for task in tqdm(tasks, desc="Generating"):
        msgs = [{"role": "user", "content": task["prompt"]}]
        formatted = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        out = pipe(
            formatted,
            max_new_tokens=512,
            temperature=None,    # greedy
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"]

        clean = extract_answer_from_output(out.strip())

        results.append({
            **{k: task[k] for k in
               ["doc_id","ministry","question_id","condition",
                "question_hi","gold_answer_en","keywords_en"]},
            "model":   model_id,
            "run_tag": run_tag,
            "answer_hi": clean,
        })

    # ── Save raw answers ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(DEMO_OUTPUT_RUNS), exist_ok=True)
    df_runs = pd.DataFrame(results)
    df_runs.to_csv(DEMO_OUTPUT_RUNS, index=False, encoding="utf-8-sig")
    print(f"\nRaw answers saved → {DEMO_OUTPUT_RUNS}")

    # ── Quick scoring (no back-translation — compare directly for demo) ───────
    print("\nScoring (keyword hit rate and ROUGE-L only — no GPU back-translation) …")
    df_runs["keyword_hit_rate"] = df_runs.apply(
        lambda r: keyword_hit_rate(str(r["keywords_en"]), str(r["answer_hi"])),
        axis=1,
    )
    df_runs["rougeL_f1"] = df_runs.apply(
        lambda r: rouge_l_f1(str(r["gold_answer_en"]), str(r["answer_hi"])),
        axis=1,
    )

    os.makedirs(os.path.dirname(DEMO_OUTPUT_EVAL), exist_ok=True)
    df_runs.to_csv(DEMO_OUTPUT_EVAL, index=False, encoding="utf-8-sig")
    print(f"Scored results saved → {DEMO_OUTPUT_EVAL}")

    # ── Pretty-print table for viva ───────────────────────────────────────────
    print("\n" + "═" * 80)
    print(f"DEMO RESULTS — {model_id} | {run_tag} | {n} questions")
    print("═" * 80)

    for _, row in df_runs.iterrows():
        print(f"\n[{row['condition']}] Q: {row['question_hi'][:80]}")
        print(f"  GOLD  : {row['gold_answer_en'][:120]}")
        print(f"  MODEL : {str(row['answer_hi'])[:120]}")
        kw  = row['keyword_hit_rate']
        rl  = row['rougeL_f1']
        kw_str = f"{kw:.2f}" if pd.notna(kw) else "N/A"
        print(f"  KW-Hit: {kw_str}   ROUGE-L: {rl:.4f}")

    print("\n" + "═" * 80)
    print("SUMMARY BY CONDITION")
    print("─" * 40)
    summary = df_runs.groupby("condition")[["keyword_hit_rate", "rougeL_f1"]].mean().round(4)
    print(summary.to_string())
    print("═" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Viva/demo: 10-question run with a small model, no vLLM needed."
    )
    parser.add_argument("--model",  default=DEMO_MODEL_DEFAULT,
                        help=f"HuggingFace model ID (default: {DEMO_MODEL_DEFAULT})")
    parser.add_argument("--n",      type=int, default=10,
                        help="Number of questions to run (default: 10)")
    parser.add_argument("--device", default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)")

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompts",    dest="use_prompt", action="store_true",
                              default=True, help="Use Hindi task instructions (default)")
    prompt_group.add_argument("--no-prompts", dest="use_prompt", action="store_false",
                              help="Send bare context + question")

    args = parser.parse_args()

    # Resolve 'auto' device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Device: {device}")

    run_demo(args.model, args.n, args.use_prompt, device)


if __name__ == "__main__":
    main()
