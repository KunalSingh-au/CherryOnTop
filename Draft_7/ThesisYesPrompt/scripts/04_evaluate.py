import pandas as pd
import os
import sys
import re
from tqdm import tqdm

# Setup Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.metrics import keyword_hit_rate, rouge_l_f1, bertscore_multilingual_batch
from utils.llm import hi_to_en_local, extract_answer_from_output

def is_hindi(text):
    """
    Standard Language Detector.
    If >10% of text is Devanagari, we classify as Hindi/Hinglish.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    return (hindi_chars / len(text)) > 0.1

def main():
    # USAGE: Change these paths to point to either your No-Prompt or Prompted runs
    input_file = "outputs/runs/all_runs.csv"
    output_file = "outputs/evaluation/all_results.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    df['answer_hi'] = df['answer_hi'].fillna("")

    print(f"Starting Unified Evaluation on {len(df)} rows...")

    # --- STEP 1: CLEAN & TRANSLATE ---
    back_translations = []
    for text in tqdm(df['answer_hi'], desc="Cleaning & Translating"):
        # 1. Strip reasoning blocks (Essential for 'Yes-Prompt' models)
        clean_text = extract_answer_from_output(str(text))
        
        # 2. Logic: Only translate if it's actually Hindi/Hinglish
        if not clean_text.strip():
            back_translations.append("")
        elif is_hindi(clean_text):
            # Translate Hindi/Mix to English using Sarvam-Translate
            back_translations.append(hi_to_en_local(clean_text))
        else:
            # If model answered in English, use it directly (No translation noise)
            back_translations.append(clean_text)
            
    df['back_en'] = back_translations

    # --- STEP 2: KEYWORD HIT RATE ---
    # We check for English Keywords in the English Back-translation
    print("Calculating Keyword Hits...")
    df['keyword_hit_rate'] = df.apply(
        lambda r: keyword_hit_rate(str(r['keywords_en']), str(r['back_en'])), 
        axis=1
    )

    # --- STEP 3: ROUGE-L ---
    print("Calculating ROUGE-L...")
    df['rougeL_f1'] = df.apply(
        lambda r: rouge_l_f1(str(r['gold_answer_en']), str(r['back_en'])), 
        axis=1
    )

    # --- STEP 4: BERTSCORE ---
    print("Calculating BERTScore...")
    # Standardizing BERTScore to compare English Back-translation vs English Gold
    df['bertscore_f1'] = bertscore_multilingual_batch(
        df['gold_answer_en'].astype(str).tolist(), 
        df['back_en'].astype(str).tolist()
    )

    # Final Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Done! Results saved to {output_file}")

if __name__ == "__main__":
    main()