import argparse
import csv
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RUNS_CSV, CONDITIONS, QA_GOLD_PATH, EXTRACTED_EN, TRANSLATED_SARVAM, TRANSLATED_INDICTRANS, EXTRACTED_HI, MODELS_HF
from utils.jsonl import load_docs_jsonl
from utils.llm import build_qa_prompt

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

    # Load Questions
    qa_df = pd.read_csv(QA_GOLD_PATH)
    contexts = load_all_contexts()
    
    all_tasks = []
    
    # Build prompt for every condition for every question
    for _, row in qa_df.iterrows():
        did, qid, q_hi = row["doc_id"], row["question_id"], row["question_hi_official"]
        for cond in CONDITIONS:
            ctx = contexts.get(cond, {}).get(did, "") if cond != "C3" else ""
            prompt_str = build_qa_prompt(q_hi, ctx, cond)
            all_tasks.append({
                "doc_id": did, "ministry": row["ministry"], "question_id": qid, 
                "condition": cond, "question_hi_used": q_hi, "gold_answer_en": row["gold_answer_en"],
                "keywords_en": row["keywords_en"], "prompt": prompt_str, "context_preview": ctx[:100]
            })

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    hf_id = MODELS_HF[args.model]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    
    # Run across 4 GPUs (48GB x 4 = 192GB VRAM, perfect for 70B models)
    print(f"Loading {hf_id} on 4 GPUs via Tensor Parallelism...")
    #llm = LLM(model=hf_id, tensor_parallel_size=4, dtype="bfloat16", trust_remote_code=True)
    llm = LLM(model=hf_id, tensor_parallel_size=4, dtype="bfloat16", trust_remote_code=True, max_model_len=16384)
    
    # Format prompts with specific model chat templates
    formatted_prompts = []
    for task in all_tasks:
        msgs = [{"role": "user", "content": task["prompt"]}]
        formatted_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    print(f"Running inference on {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, SamplingParams(temperature=0.0, max_tokens=1024))

    os.makedirs(os.path.dirname(RUNS_CSV), exist_ok=True)
    file_exists = os.path.exists(RUNS_CSV)
    
    with open(RUNS_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["doc_id", "ministry", "question_id", "condition", "model", "question_hi_used", "gold_answer_en", "keywords_en", "context_preview", "answer_hi"])
        for task, out in zip(all_tasks, outputs):
            ans = out.outputs[0].text.strip()
            writer.writerow([task["doc_id"], task["ministry"], task["question_id"], task["condition"], args.model, task["question_hi_used"], task["gold_answer_en"], task["keywords_en"], task["context_preview"], ans])

    print("Inference complete.")

if __name__ == "__main__": main()