import os
import pandas as pd
import numpy as np

# Configuration - Ensure these match your evaluation output
INPUT_CSV = "outputs/evaluation/all_results.csv"
OUTPUT_DIR = "outputs/analysis"
METRICS = ["keyword_hit_rate", "rougeL_f1", "bertscore_f1"]
MLABELS = ["KW-Hit", "ROUGE-L", "BERTScore"]

COND_LABELS = {
    "C1": "Full Doc (English)",
    "C2": "Full Doc (Sarvam MT)",
    "C3": "No Doc (Closed Book)",
    "C4": "Full Doc (Official Hindi)",
    "C5": "Full Doc (IndicTrans2 MT)",
    "C6": "RAG (English Chunks)",
    "C7": "RAG (Sarvam MT Chunks)",
    "C8": "RAG (Official Hindi Chunks)",
    "C9": "RAG (IndicTrans2 MT Chunks)"
}

def hr(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run evaluation first.")
        return

    df = pd.read_csv(INPUT_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Main Table: Model x Condition
    hr("TABLE 1: Main Performance Matrix (Model vs Condition)")
    t1 = df.groupby(["model", "condition"])[METRICS].mean().round(3).reset_index()
    t1.to_csv(os.path.join(OUTPUT_DIR, "01_model_condition_matrix.csv"), index=False)
    print(t1.pivot(index="model", columns="condition", values="bertscore_f1"))

    # 2. Leaderboard: Overall Model Ranking
    hr("TABLE 2: Overall Model Leaderboard (Averaged across all tasks)")
    t2 = df.groupby("model")[METRICS].mean().round(3)
    t2["Combined_Score"] = t2.mean(axis=1).round(3)
    t2 = t2.sort_values("Combined_Score", ascending=False)
    t2.to_csv(os.path.join(OUTPUT_DIR, "02_model_leaderboard.csv"))
    print(t2)

    # 3. Condition Impact: Which setup is most effective?
    hr("TABLE 3: Condition Impact (The 'Difficulty' of different document types)")
    t3 = df.groupby("condition")[METRICS].mean().round(3).reset_index()
    t3["Description"] = t3["condition"].map(COND_LABELS)
    t3 = t3[["condition", "Description"] + METRICS]
    t3.to_csv(os.path.join(OUTPUT_DIR, "03_condition_impact.csv"), index=False)
    print(t3)

    # 4. RAG vs. Full-Doc (The Delta Analysis)
    hr("TABLE 4: RAG Effectiveness (Score change when using RAG vs Full Document)")
    rag_pairs = [("C1", "C6", "English"), ("C2", "C7", "Sarvam MT"), 
                 ("C4", "C8", "Official"), ("C5", "C9", "IndicTrans2")]
    
    rag_deltas = []
    for full, rag, label in rag_pairs:
        f_scores = df[df["condition"] == full].groupby("model")[METRICS].mean()
        r_scores = df[df["condition"] == rag].groupby("model")[METRICS].mean()
        delta = (r_scores - f_scores).round(3)
        delta["Context_Type"] = label
        rag_deltas.append(delta.reset_index())
    
    t4 = pd.concat(rag_deltas)
    t4.to_csv(os.path.join(OUTPUT_DIR, "04_rag_delta_analysis.csv"), index=False)
    print("Positive values mean RAG performed BETTER than providing the full document.")
    print(t4.groupby("Context_Type")[METRICS].mean())

    # 5. Translation Engine Head-to-Head (C2 vs C5)
    hr("TABLE 5: Translation Engine Comparison (Sarvam-MT vs IndicTrans2)")
    t5_sub = df[df["condition"].isin(["C2", "C5"])]
    t5 = t5_sub.groupby(["condition", "model"])[METRICS].mean().round(3).unstack(0)
    t5.to_csv(os.path.join(OUTPUT_DIR, "05_translator_comparison.csv"))
    print(t5["bertscore_f1"])

    # 6. Ministry Performance: Domain Specificity
    hr("TABLE 6: Performance by Ministry (Ayush vs. Women & Child)")
    t6 = df.groupby("ministry")[METRICS].mean().round(3).sort_values("bertscore_f1", ascending=False)
    t6.to_csv(os.path.join(OUTPUT_DIR, "06_ministry_analysis.csv"))
    print(t6)

    # 7. Hallucination Baseline (C3)
    hr("TABLE 7: C3 Hallucination Baseline (Knowledge from model weights only)")
    t7 = df[df["condition"] == "C3"].groupby("model")[METRICS].mean().round(3)
    t7.to_csv(os.path.join(OUTPUT_DIR, "07_hallucination_floor.csv"))
    print(t7)

    print(f"\n✓ All 7 thesis tables saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()