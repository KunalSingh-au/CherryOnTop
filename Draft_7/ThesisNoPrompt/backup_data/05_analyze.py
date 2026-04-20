#!/usr/bin/env python3
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EVAL_CSV, ANALYSIS_DIR

# Removed 'hallucination_score' and 'Judge' from labels
METRICS = ["keyword_hit_rate", "rougeL_f1", "bertscore_f1"]
MLABELS = ["KW-Hit", "ROUGE-L", "BERTScore"]

def main():
    if not os.path.exists(EVAL_CSV):
        print(f"Error: {EVAL_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(EVAL_CSV, encoding="utf-8-sig")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # 1. Summary by Model & Condition
    t1 = df.groupby(["model", "condition"])[METRICS].mean().round(3).reset_index()
    t1.columns = ["Model", "Condition"] + MLABELS
    t1.to_csv(os.path.join(ANALYSIS_DIR, "summary_by_model_condition.csv"), index=False)

    # 2. Overall Model Leaderboard
    t2 = df.groupby("model")[METRICS].mean().round(3).reset_index()
    t2.columns = ["Model"] + MLABELS
    t2.to_csv(os.path.join(ANALYSIS_DIR, "model_leaderboard.csv"), index=False)

    # 3. Translation Engine Impact (C2 vs C4 vs C5)
    mt_sub = df[df["condition"].isin(["C2", "C4", "C5", "C1"])]
    t3 = mt_sub.groupby("condition")[METRICS].mean().round(3).reset_index()
    t3.columns = ["Condition"] + MLABELS
    t3.to_csv(os.path.join(ANALYSIS_DIR, "translation_impact.csv"), index=False)

    # 4. Multilingual RAG Delta (Chunked vs Full Doc)
    rag_pairs = [("C1", "C6"), ("C2", "C7"), ("C4", "C8"), ("C5", "C9")]
    delta_frames = []
    
    t_rag = df.groupby(["model", "condition"])[METRICS].mean().round(3).reset_index()
    
    for full, chunked in rag_pairs:
        if full in df["condition"].values and chunked in df["condition"].values:
            full_t = t_rag[t_rag["condition"] == full].set_index("model")[METRICS]
            chunked_t = t_rag[t_rag["condition"] == chunked].set_index("model")[METRICS]
            
            common = full_t.index.intersection(chunked_t.index)
            delta = (chunked_t.loc[common] - full_t.loc[common]).round(3).reset_index()
            delta.insert(1, "Pair", f"{chunked} - {full}")
            delta_frames.append(delta)
            
    if delta_frames:
        final_delta = pd.concat(delta_frames, ignore_index=True)
        final_delta.columns = ["Model", "Comparison (+ve = RAG helped)"] + MLABELS
        final_delta.to_csv(os.path.join(ANALYSIS_DIR, "rag_delta_all.csv"), index=False)
    

    print(f"All analysis tables saved to {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()