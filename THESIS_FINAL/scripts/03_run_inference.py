"""
scripts/03_run_inference.py — Unified LLM inference for all conditions.

This single script replaces the old 03_run_qa.py and 06_run_rag.py.
It handles BOTH full-document conditions (C1–C5) AND RAG conditions (C6–C9),
controlled by the --conditions flag.

The key flag --prompts / --no-prompts controls whether Hindi task instructions
are included in every prompt.  Both experimental branches (Prompted / No-Prompt)
use exactly the same code — only the flag changes.

Results are appended to outputs/runs/all_runs.csv.
The 'run_tag' column records "prompted" or "noprompt" so both branches can
coexist in the same CSV and be filtered during evaluation.

Usage:
  # Full-doc conditions only, with prompts (Prompted branch)
  python scripts/03_run_inference.py --model llama --prompts

  # RAG conditions only, no prompts (No-Prompt branch)
  python scripts/03_run_inference.py --model qwen --no-prompts --conditions rag

  # All conditions (C1–C9), with prompts
  python scripts/03_run_inference.py --model sarvam --prompts --conditions all

  # Demo / viva mode: 10 questions, no GPU needed (uses a tiny model)
  python scripts/03_run_inference.py --model llama --prompts --demo
"""

import argparse
import csv
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    RUNS_CSV, QA_GOLD_PATH,
    EXTRACTED_EN, EXTRACTED_HI, TRANSLATED_SARVAM, TRANSLATED_INDICTRANS,
    MODELS_HF, FULL_DOC_CONDITIONS, RAG_CONDITIONS, ALL_CONDITIONS,
    RAG_SOURCE_MAP, RAG_EMBEDDER,
    TENSOR_PARALLEL_SIZE, MAX_MODEL_LEN, MAX_NEW_TOKENS,
    CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K,
)
from utils.jsonl import load_docs_jsonl
from utils.llm import build_qa_prompt, extract_answer_from_output


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_contexts() -> dict:
    """Load all four base document corpora into a dict keyed by condition."""
    return {
        "C1": load_docs_jsonl(EXTRACTED_EN,         "answer_en"),
        "C2": load_docs_jsonl(TRANSLATED_SARVAM,    "answer_hi"),
        "C4": load_docs_jsonl(EXTRACTED_HI,         "answer_hi"),
        "C5": load_docs_jsonl(TRANSLATED_INDICTRANS,"answer_hi"),
    }


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split text into overlapping word-based windows.

    ── TUNABLE: chunk_size and overlap are set in config.py ─────────────────
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def get_rag_context(query_hi: str, full_text: str, embedder) -> str:
    """
    Retrieve the top-K most relevant chunks for a Hindi query using
    multilingual sentence embeddings + cosine similarity.

    Returns concatenated chunks as a single context string.
    Returns empty string if the document has no usable text.
    """
    from sentence_transformers import util

    chunks = chunk_text(full_text)
    if not chunks:
        return ""

    q_emb = embedder.encode(query_hi, convert_to_tensor=True)
    c_emb = embedder.encode(chunks,   convert_to_tensor=True)
    hits  = util.semantic_search(q_emb, c_emb, top_k=RAG_TOP_K)[0]

    top_chunks = [chunks[h["corpus_id"]] for h in hits]
    return "\n\n".join(top_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# TASK BUILDER — creates the list of (condition, question, context, prompt) rows
# ─────────────────────────────────────────────────────────────────────────────

def build_tasks(
    qa_df: pd.DataFrame,
    contexts: dict,
    conditions: list,
    use_prompt: bool,
    embedder=None,
) -> list:
    """
    Build the full list of inference tasks.

    Each task is a dict with all metadata needed to write a CSV row after
    inference.  Prompts are built here so the LLM step is a pure batch call.
    """
    tasks = []

    # Determine which conditions need RAG retrieval
    rag_needed = [c for c in conditions if c in RAG_CONDITIONS]
    full_needed = [c for c in conditions if c in FULL_DOC_CONDITIONS]

    for _, row in qa_df.iterrows():
        did  = row["doc_id"]
        qid  = row["question_id"]
        qhi  = row["question_hi_official"]
        gold = row["gold_answer_en"]
        kw   = row.get("keywords_en", "")

        # ── Full-document conditions ───────────────────────────────────────────
        for cond in full_needed:
            if cond == "C3":
                ctx = ""
            else:
                ctx = contexts.get(cond, {}).get(did, "")
            prompt = build_qa_prompt(qhi, ctx, cond, use_prompt=use_prompt)
            tasks.append({
                "doc_id": did, "ministry": row["ministry"],
                "question_id": qid, "condition": cond,
                "question_hi_used": qhi, "gold_answer_en": gold,
                "keywords_en": kw, "context_preview": ctx[:150],
                "prompt": prompt,
            })

        # ── RAG conditions ─────────────────────────────────────────────────────
        if rag_needed and embedder is None:
            raise RuntimeError("RAG conditions requested but embedder is None.")

        for cond in rag_needed:
            source_cond = RAG_SOURCE_MAP[cond]
            full_text   = contexts.get(source_cond, {}).get(did, "")
            ctx         = get_rag_context(qhi, full_text, embedder)
            prompt      = build_qa_prompt(qhi, ctx, cond, use_prompt=use_prompt)
            tasks.append({
                "doc_id": did, "ministry": row["ministry"],
                "question_id": qid, "condition": cond,
                "question_hi_used": qhi, "gold_answer_en": gold,
                "keywords_en": kw, "context_preview": ctx[:150],
                "prompt": prompt,
            })

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM inference across all HindiParl-QA conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Model selection ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", required=True, choices=list(MODELS_HF.keys()),
        help="Which LLM to evaluate. Choices: " + ", ".join(MODELS_HF.keys()),
    )

    # ── Prompting mode — THE key flag ──────────────────────────────────────────
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompts", dest="use_prompt", action="store_true",
        help="Include Hindi task instructions in every prompt (Prompted branch).",
    )
    prompt_group.add_argument(
        "--no-prompts", dest="use_prompt", action="store_false",
        help="Send only context + question, no instructions (No-Prompt branch).",
    )

    # ── Condition selection ────────────────────────────────────────────────────
    parser.add_argument(
        "--conditions",
        choices=["fulldoc", "rag", "all"],
        default="all",
        help=(
            "fulldoc = C1–C5 only | "
            "rag = C6–C9 only | "
            "all = all 9 conditions (default)"
        ),
    )

    # ── Demo / viva mode ───────────────────────────────────────────────────────
    parser.add_argument(
        "--demo", action="store_true",
        help=(
            "Demo mode: run on first 10 questions, "
            "C1+C3 only, save to outputs/runs/demo_runs.csv."
        ),
    )

    # ── GPU configuration ──────────────────────────────────────────────────────
    parser.add_argument(
        "--tensor-parallel", type=int, default=TENSOR_PARALLEL_SIZE,
        help=f"Number of GPUs for tensor parallelism (default: {TENSOR_PARALLEL_SIZE}).",
    )

    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    # Load gold QA and document contexts
    # ─────────────────────────────────────────────────────────────────────────
    if not os.path.exists(QA_GOLD_PATH):
        print(f"Error: Gold QA file not found: {QA_GOLD_PATH}")
        sys.exit(1)

    qa_df    = pd.read_csv(QA_GOLD_PATH)
    contexts = load_all_contexts()

    # ─────────────────────────────────────────────────────────────────────────
    # Demo mode overrides
    # ─────────────────────────────────────────────────────────────────────────
    output_csv = RUNS_CSV
    if args.demo:
        qa_df      = qa_df.head(10)             # Only first 10 questions
        conditions = ["C1", "C3"]               # Two simplest conditions
        output_csv = RUNS_CSV.replace("all_runs.csv", "demo_runs.csv")
        print("── DEMO MODE: 10 questions, C1 + C3 only ──")
    else:
        if args.conditions == "fulldoc":
            conditions = FULL_DOC_CONDITIONS
        elif args.conditions == "rag":
            conditions = RAG_CONDITIONS
        else:
            conditions = ALL_CONDITIONS

    run_tag = "prompted" if args.use_prompt else "noprompt"
    print(f"Model     : {args.model} ({MODELS_HF[args.model]})")
    print(f"Mode      : {run_tag}")
    print(f"Conditions: {conditions}")
    print(f"Questions : {len(qa_df)}")
    print(f"Output    : {output_csv}")

    # ─────────────────────────────────────────────────────────────────────────
    # Load RAG embedder only if needed
    # ─────────────────────────────────────────────────────────────────────────
    rag_conditions_needed = [c for c in conditions if c in RAG_CONDITIONS]
    embedder = None
    if rag_conditions_needed:
        from sentence_transformers import SentenceTransformer
        print(f"\nLoading RAG embedder: {RAG_EMBEDDER} …")
        embedder = SentenceTransformer(RAG_EMBEDDER)
        print("  Embedder ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # Build all tasks (prompts assembled here, before model loads)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nBuilding inference tasks …")
    tasks = build_tasks(qa_df, contexts, conditions, args.use_prompt, embedder)
    print(f"  {len(tasks)} tasks ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # Load LLM
    # ─────────────────────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    hf_id = MODELS_HF[args.model]
    print(f"\nLoading {hf_id} on {args.tensor_parallel} GPU(s) …")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    llm = LLM(
        model=hf_id,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,   # ── TUNABLE: see config.py ────────────
    )
    print("  Model loaded.")

    # ─────────────────────────────────────────────────────────────────────────
    # Format prompts using model's own chat template
    # ─────────────────────────────────────────────────────────────────────────
    formatted_prompts = []
    for task in tasks:
        msgs = [{"role": "user", "content": task["prompt"]}]
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Run batch inference
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(formatted_prompts)} prompts …")
    outputs = llm.generate(
        formatted_prompts,
        SamplingParams(
            temperature=0.0,          # Greedy decode — reproducible results
            max_tokens=MAX_NEW_TOKENS, # ── TUNABLE: see config.py ─────────────
        ),
    )
    print("  Inference complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "doc_id", "ministry", "question_id", "condition", "model",
                "run_tag",           # "prompted" or "noprompt" — NEW column
                "question_hi_used", "gold_answer_en", "keywords_en",
                "context_preview",  "answer_hi",
            ])
        for task, out in zip(tasks, outputs):
            raw_ans = out.outputs[0].text.strip()
            # Strip <think> blocks from Sarvam / Qwen chain-of-thought output
            clean_ans = extract_answer_from_output(raw_ans)
            writer.writerow([
                task["doc_id"],    task["ministry"],  task["question_id"],
                task["condition"], args.model,        run_tag,
                task["question_hi_used"], task["gold_answer_en"],
                task["keywords_en"],      task["context_preview"],
                clean_ans,
            ])

    print(f"\n✓ Results saved → {output_csv}")
    print(f"  Rows written: {len(tasks)}")


if __name__ == "__main__":
    main()
