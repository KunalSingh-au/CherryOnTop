import argparse
import csv
import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RUNS_CSV, QA_GOLD_PATH, EXTRACTED_EN, TRANSLATED_SARVAM, EXTRACTED_HI, TRANSLATED_INDICTRANS, MODELS_HF
from utils.llm import build_qa_prompt, extract_answer_from_output
from utils.jsonl import load_docs_jsonl

def chunk_text(text: str, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def load_all_contexts():
    return {
        "C1": load_docs_jsonl(EXTRACTED_EN, "answer_en"),
        "C2": load_docs_jsonl(TRANSLATED_SARVAM, "answer_hi"),
        "C4": load_docs_jsonl(EXTRACTED_HI, "answer_hi"),
        "C5": load_docs_jsonl(TRANSLATED_INDICTRANS, "answer_hi")
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=MODELS_HF.keys(), required=True)
    args = parser.parse_args()

    # Load Questions and Contexts
    qa_df = pd.read_csv(QA_GOLD_PATH)
    contexts = load_all_contexts()
    
    # Map Source Condition -> New RAG Condition (C6-C9)
    rag_map = {"C1": "C6", "C2": "C7", "C4": "C8", "C5": "C9"}
    
    print("Loading SentenceTransformer for Multilingual RAG...")
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    tasks = []
    print("Starting Retrieval and Prompt Building...")
    for _, row in qa_df.iterrows():
        did = row["doc_id"]
        qid = row["question_id"]
        # FIXED: Using 'question_hi_official' to match your gold CSV
        qhi = row["question_hi_official"] 
        qen = row["gold_answer_en"]
        kw = row.get("keywords_en", "")
        
        query_embedding = embedder.encode(qhi, convert_to_tensor=True)
        
        for source_cond, rag_cond in rag_map.items():
            full_text = contexts.get(source_cond, {}).get(did, "")
            chunks = chunk_text(full_text)
            
            if chunks:
                chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=3)[0]
                top_chunks = [chunks[hit['corpus_id']] for hit in hits]
                retrieved_context = "\n\n".join(top_chunks)
            else:
                retrieved_context = ""
            
            # build_qa_prompt now uses the version with instructions from your new utils/llm.py
            prompt = build_qa_prompt(qhi, retrieved_context, rag_cond)
            
            tasks.append({
                "doc_id": did, "ministry": row["ministry"], "question_id": qid,
                "condition": rag_cond, "question_hi_used": qhi, 
                "gold_answer_en": qen, "keywords_en": kw,
                "context_preview": str(retrieved_context)[:200], "prompt": prompt
            })

    # Inference Section
    hf_id = MODELS_HF[args.model]
    print(f"Loading {hf_id} on 4 GPUs...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    llm = LLM(model=hf_id, tensor_parallel_size=4, dtype="bfloat16", trust_remote_code=True, max_model_len=16384)

    formatted = []
    for t in tasks:
        msgs = [{"role": "user", "content": t['prompt']}]
        formatted.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    print(f"Generating {len(formatted)} Prompted RAG responses...")
    outputs = llm.generate(formatted, SamplingParams(temperature=0.0, max_tokens=1024))

    # Save to all_runs.csv
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
            # extract_answer_from_output cleans <think> tags (essential for Sarvam)
            ans = extract_answer_from_output(out.outputs[0].text.strip())
            writer.writerow([
                task['doc_id'], task['ministry'], task['question_id'],
                task['condition'], args.model, task['question_hi_used'],
                task['gold_answer_en'], task['keywords_en'],
                task['context_preview'], ans
            ])

if __name__ == "__main__":
    main()