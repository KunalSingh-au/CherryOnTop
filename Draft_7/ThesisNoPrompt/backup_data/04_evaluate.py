#!/usr/bin/env python3
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RUNS_CSV, EVAL_CSV, TRANSLATED_SARVAM, TRANSLATED_INDICTRANS, EXTRACTED_HI
from utils.jsonl import load_docs_jsonl
from utils.llm import hi_to_en_local_sarvam
from utils.metrics import (
    keyword_hit_rate, rouge_l_f1,
    bertscore_multilingual_batch, doc_fidelity_chrf
)

def main():
    if not os.path.exists(RUNS_CSV):
        print(f"Error: {RUNS_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(RUNS_CSV, encoding="utf-8-sig")
    
    sarv_docs = load_docs_jsonl(TRANSLATED_SARVAM, "answer_hi")
    indic_docs = load_docs_jsonl(TRANSLATED_INDICTRANS, "answer_hi")
    hi_docs = load_docs_jsonl(EXTRACTED_HI, "answer_hi")

    # Clean missing answers
    df["answer_hi"] = df["answer_hi"].fillna("")
    df["gold_answer_en"] = df["gold_answer_en"].fillna("")

    print("1. Running BERTScore on all rows...")
    df["bertscore_f1"] = bertscore_multilingual_batch(
        df["gold_answer_en"].tolist(), 
        df["answer_hi"].tolist()
    )

    print(f"2. Running Back-Translation and Keyword/ROUGE Eval on {len(df)} rows...")
    back_en_list = []
    kw_hits = []
    rouge_scores = []
    fidelity_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Local Eval"):
        ans_hi = str(row["answer_hi"])
        gold = str(row["gold_answer_en"])
        kw = str(row.get("keywords_en", ""))
        did = row["doc_id"]
        cond = row["condition"]
        
        # Local vLLM Back-translation
        back_en = hi_to_en_local_sarvam(ans_hi)
        back_en_list.append(back_en)
        
        kw_hits.append(keyword_hit_rate(kw, ans_hi))
        rouge_scores.append(rouge_l_f1(gold, back_en))
        
        # Fidelity matching against MT docs
        ref_doc = ""
        if cond in ["C2", "C7"]: ref_doc = sarv_docs.get(did, "")
        elif cond in ["C5", "C9"]: ref_doc = indic_docs.get(did, "")
        
            
        df_val = doc_fidelity_chrf(ref_doc, hi_docs.get(did, "")) if ref_doc else float("nan")
        fidelity_scores.append(df_val)

    df["back_en"] = back_en_list
    df["keyword_hit_rate"] = kw_hits
    df["rougeL_f1"] = rouge_scores
    df["doc_fidelity_chrf"] = fidelity_scores

    os.makedirs(os.path.dirname(EVAL_CSV), exist_ok=True)
    df.to_csv(EVAL_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved local evaluation to {EVAL_CSV}")

if __name__ == "__main__":
    main()