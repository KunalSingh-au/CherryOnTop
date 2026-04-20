import argparse
import csv
import os
import sys
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RUNS_CSV, CONDITIONS, QA_GOLD_PATH, EXTRACTED_EN, TRANSLATED_SARVAM, TRANSLATED_INDICTRANS, EXTRACTED_HI, MODELS_HF
from utils.jsonl import load_docs_jsonl
from utils.llm import build_qa_prompt, extract_answer_from_output

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

    qa_df = pd.read_csv(QA_GOLD_PATH)
    contexts = load_all_contexts()
    all_tasks = []
    
    for _, row in qa_df.iterrows():
        # FIXED: Using "question_hi_official" exactly as it appears in your CSV
        did = row["doc_id"]
        qid = row["question_id"]
        qhi = row["question_hi_official"]
        qen = row["gold_answer_en"]
        kw = row.get("keywords_en", "")
        
        for cond in CONDITIONS:
            ctx = "" if cond == "C3" else contexts.get(cond, {}).get(did, "")
            prompt = build_qa_prompt(qhi, ctx, cond)
            
            all_tasks.append({
                "doc_id": did, "ministry": row["ministry"], "question_id": qid,
                "condition": cond, "question_hi_used": qhi, 
                "gold_answer_en": qen, "keywords_en": kw,
                "context_preview": str(ctx)[:200], "prompt": prompt
            })

    hf_id = MODELS_HF[args.model]
    print(f"Loading {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    llm = LLM(model=hf_id, tensor_parallel_size=4, dtype="bfloat16", trust_remote_code=True, max_model_len=16384)
    
    formatted_prompts = []
    for task in all_tasks:
        msgs = [{"role": "user", "content": task["prompt"]}]
        formatted_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    print(f"Running unprompted inference on {len(formatted_prompts)} questions...")
    outputs = llm.generate(formatted_prompts, SamplingParams(temperature=0.0, max_tokens=1024))

    os.makedirs(os.path.dirname(RUNS_CSV), exist_ok=True)
    file_exists = os.path.exists(RUNS_CSV)
    
    with open(RUNS_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["doc_id", "ministry", "question_id", "condition", "model", "question_hi_used", "gold_answer_en", "keywords_en", "context_preview", "answer_hi"])
        for task, out in zip(all_tasks, outputs):
            ans = extract_answer_from_output(out.outputs[0].text.strip())
            writer.writerow([
                task['doc_id'], task['ministry'], task['question_id'],
                task['condition'], args.model, task['question_hi_used'],
                task['gold_answer_en'], task['keywords_en'],
                task['context_preview'], ans
            ])

if __name__ == "__main__":
    main()