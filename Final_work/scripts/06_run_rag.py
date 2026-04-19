#!/usr/bin/env python3
"""
06_run_rag.py — Retrieval-Augmented Generation (RAG) module
============================================================
This is the algorithmic improvement for the thesis.

WHAT IT DOES:
  Instead of passing the full document (avg 5500 chars) as context,
  it splits each doc into overlapping chunks, embeds them with a
  multilingual model, and retrieves only the top-K most relevant
  chunks for each question.

  This creates a new condition C6:
    C6 = Hindi question + RAG-retrieved English chunks (top-3)

WHY THIS MATTERS FOR THE THESIS:
  1. Tests whether retrieval helps Sarvam more than other models
  2. Directly addresses prof's "algo improvement" requirement
  3. Shows whether Parliamentary Hindi questions can cross-lingually
     retrieve relevant English government document chunks
  4. Compares full-doc prompting (C1) vs RAG (C6) across all 4 models

MODEL USED:
  paraphrase-multilingual-MiniLM-L12-v2
  - 118MB, 50 languages including Hindi + English
  - Cross-lingual: Hindi question retrieves English chunks (and vice versa)
  - No GPU needed for embeddings (CPU is fast enough)

Run: python scripts/06_run_rag.py --model sarvam
     python scripts/06_run_rag.py --model llama
     python scripts/06_run_rag.py --model mixtral
     python scripts/06_run_rag.py --model qwen

Output: outputs/runs/all_runs.csv (appended with condition=C6)
"""

import os, sys, re, json, csv, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    QA_GOLD_PATH, RUNS_CSV, EXTRACTED_EN, MODELS_HF
)
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks at sentence boundaries.
    
    chunk_size: target chars per chunk
    overlap:    chars of overlap between consecutive chunks
                (helps when answer spans chunk boundary)
    """
    text = re.sub(r'\s+', ' ', text.strip())
    if len(text) <= chunk_size:
        return [text]

    # Split at sentence boundaries first
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            # Overlap: carry last `overlap` chars into next chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = (overlap_text + " " + sent).strip()
    if current:
        chunks.append(current)

    # Hard split any remaining long chunks
    final = []
    for chunk in chunks:
        if len(chunk) <= chunk_size * 1.5:
            final.append(chunk)
        else:
            for i in range(0, len(chunk), chunk_size - overlap):
                piece = chunk[i:i + chunk_size]
                if piece.strip():
                    final.append(piece.strip())
    
    return [c for c in final if len(c.strip()) > 20]


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

class MultilingualRetriever:
    """
    Embeds document chunks and retrieves top-K most relevant for a query.
    
    Uses paraphrase-multilingual-MiniLM-L12-v2:
    - Supports Hindi (Devanagari) + English in the same vector space
    - Cross-lingual: Hindi question can retrieve English chunks
    - 118MB download on first use
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"  Embedding model ready")
    
    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def retrieve(self, query: str, chunks: list[str], top_k: int = 3) -> list[str]:
        """
        Retrieve top_k most relevant chunks for the query.
        
        Works cross-lingually: Hindi query retrieves English chunks.
        Uses cosine similarity.
        """
        if not chunks:
            return []
        if len(chunks) <= top_k:
            return chunks
        
        q_emb = self.embed([query])          # shape (1, 384)
        c_emb = self.embed(chunks)           # shape (N, 384)
        
        # Cosine similarity
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        c_norm = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-9)
        scores = (q_norm @ c_norm.T).flatten()  # shape (N,)
        
        top_idx = np.argsort(scores)[::-1][:top_k]
        # Return in document order (not similarity order) for coherence
        top_idx_sorted = sorted(top_idx)
        return [chunks[i] for i in top_idx_sorted]


# ══════════════════════════════════════════════════════════════════════════════
# BUILD INDEX for all docs
# ══════════════════════════════════════════════════════════════════════════════

def build_doc_index(retriever: MultilingualRetriever) -> dict:
    """
    Pre-chunk and pre-embed all English documents.
    Returns dict: doc_id → {'chunks': [...], 'embeddings': np.array}
    """
    print("Building document index (chunking + embedding all docs)...")
    index = {}
    
    with open(EXTRACTED_EN, encoding='utf-8') as f:
        docs = [json.loads(l) for l in f]
    
    for doc in docs:
        doc_id = doc['doc_id']
        text = doc.get('answer_en', '')
        
        if not text.strip():
            print(f"  SKIP {doc_id}: empty")
            continue
        
        chunks = chunk_text(text, chunk_size=350, overlap=75)
        embeddings = retriever.embed(chunks)
        
        index[doc_id] = {
            'chunks':     chunks,
            'embeddings': embeddings,
        }
        print(f"  {doc_id}: {len(chunks)} chunks")
    
    print(f"Index built: {len(index)} docs")
    return index


def retrieve_for_question(
    retriever: MultilingualRetriever,
    index: dict,
    doc_id: str,
    question_hi: str,
    top_k: int = 3,
) -> str:
    """
    Retrieve top-K chunks for a question. Returns concatenated context string.
    """
    if doc_id not in index:
        return ""
    
    entry = index[doc_id]
    chunks = entry['chunks']
    doc_emb = entry['embeddings']
    
    if len(chunks) <= top_k:
        return "\n\n".join(chunks)
    
    # Embed question
    q_emb = retriever.embed([question_hi])
    q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    c_norm = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-9)
    scores = (q_norm @ c_norm.T).flatten()
    
    top_idx = sorted(np.argsort(scores)[::-1][:top_k])
    retrieved = [chunks[i] for i in top_idx]
    
    return "\n\n".join(retrieved)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT (same system prompt as 03_run_qa.py for fair comparison)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM = (
    "You are a government document assistant for Indian Parliamentary QA. "
    "Extract the specific factual answer from the document and answer in Hindi. "
    "Be direct and concise (1-3 sentences). Include specific numbers and names. "
    "Do NOT explain your reasoning. Answer ONLY in Hindi."
)

def build_prompt(question_hi: str, rag_context: str) -> str:
    return (
        f"{SYSTEM}\n\n"
        f"Relevant document passages:\n{rag_context}\n\n"
        f"Question: {question_hi}\n\n"
        f"Answer in Hindi:"
    )


def extract_answer(text: str) -> str:
    """Strip <think> blocks from Sarvam-30B output."""
    text = str(text).strip()
    if "</think>" in text:
        after = text.split("</think>")[-1].strip()
        if len(after) > 3:
            return after
    if "<think>" not in text:
        return text
    # Mine final answer from truncated think block
    for pat in [
        r"\*\*Final Answer[:\*\s]+\"?([^\n\"]{5,400})",
        r"Final Answer[:\s]+\"?([^\n\"]{5,400})",
        r"\*\*उत्तर[:\*\s]+([^\n]{5,400})",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().strip('"').strip("*")
            if len(ans) > 4:
                return ans
    hindi_lines = re.findall(r"[^\n]*[\u0900-\u097F][^\n]{5,}", text)
    meta = ["Constraint", "Instruction", "Task", "Persona"]
    for line in reversed(hindi_lines):
        line = line.strip().strip("*")
        if len(line) > 10 and not any(w in line for w in meta):
            return line
    return text[:300]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS_HF.keys()))
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--chunk-size", type=int, default=350,
                        help="Target chars per chunk (default: 350)")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    # Load gold QA
    qa_df = pd.read_csv(QA_GOLD_PATH)
    if args.max_questions:
        qa_df = qa_df.head(args.max_questions)
    print(f"Questions: {len(qa_df)}")

    # Check which rows already done
    already_done = set()
    if os.path.exists(RUNS_CSV):
        existing = pd.read_csv(RUNS_CSV)
        c6_done = existing[
            (existing['condition'] == 'C6') & (existing['model'] == args.model)
        ]
        for _, r in c6_done.iterrows():
            already_done.add(f"{r['doc_id']}_{r['question_id']}")
        print(f"Already done (C6 {args.model}): {len(already_done)}")

    # Build retrieval index
    retriever = MultilingualRetriever()
    index = build_doc_index(retriever)

    # Build tasks
    tasks = []
    for _, row in qa_df.iterrows():
        key = f"{row['doc_id']}_{row['question_id']}"
        if key in already_done:
            continue
        q = row.get('question_hi_official') or row.get('question_hi_used') or row.get('question_hi', '')
        rag_ctx = retrieve_for_question(retriever, index, row['doc_id'], q, top_k=args.top_k)
        tasks.append({
            'doc_id':          row['doc_id'],
            'ministry':        row['ministry'],
            'question_id':     row['question_id'],
            'condition':       'C6',
            'question_hi_used': q,
            'gold_answer_en':  row.get('gold_answer_en', ''),
            'keywords_en':     row.get('keywords_en', ''),
            'context_preview': rag_ctx[:200],
            'prompt':          build_prompt(q, rag_ctx),
            'rag_chunks':      args.top_k,
        })

    if not tasks:
        print("All C6 rows already done!")
        return

    print(f"\nRunning C6 (RAG) for {args.model}: {len(tasks)} questions")
    print(f"Config: top_k={args.top_k}, chunk_size={args.chunk_size}\n")

    # Load LLM
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    hf_id     = MODELS_HF[args.model]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    print(f"Loading {hf_id}...")
    llm = LLM(
        model=hf_id,
        tensor_parallel_size=4,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=16384,
    )

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t['prompt']}],
            tokenize=False, add_generation_prompt=True
        )
        for t in tasks
    ]

    print(f"Generating {len(formatted)} responses (max_tokens=2048)...")
    outputs = llm.generate(
        formatted,
        SamplingParams(temperature=0.0, max_tokens=2048)
    )

    # Save
    os.makedirs(os.path.dirname(RUNS_CSV), exist_ok=True)
    file_exists = os.path.exists(RUNS_CSV)

    with open(RUNS_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "doc_id", "ministry", "question_id", "condition", "model",
                "question_hi_used", "gold_answer_en", "keywords_en",
                "context_preview", "answer_hi"
            ])
        for task, out in zip(tasks, outputs):
            raw  = out.outputs[0].text.strip()
            ans  = extract_answer(raw)
            writer.writerow([
                task['doc_id'], task['ministry'], task['question_id'],
                'C6', args.model, task['question_hi_used'],
                task['gold_answer_en'], task['keywords_en'],
                task['context_preview'], ans,
            ])

    print(f"\n✓ C6 results saved → {RUNS_CSV}")
    del llm
    torch.cuda.empty_cache()
    print("Next: python scripts/04_evaluate.py")


if __name__ == "__main__":
    main()
