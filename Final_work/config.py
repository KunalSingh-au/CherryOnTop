

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# API Keys for Evaluation
SARVAM_KEY = os.getenv("SARVAM_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Directory layout
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PDF_DIR_EN = os.path.join(DATA_DIR, "raw_pdfs", "english")
PDF_DIR_HI = os.path.join(DATA_DIR, "raw_pdfs", "hindi")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
TRANSLATED_DIR = os.path.join(DATA_DIR, "translated")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# File paths
EXTRACTED_EN = os.path.join(EXTRACTED_DIR, "english_docs.jsonl")
EXTRACTED_HI = os.path.join(EXTRACTED_DIR, "hindi_official_docs.jsonl")
TRANSLATED_SARVAM = os.path.join(TRANSLATED_DIR, "sarvam_hi.jsonl")
TRANSLATED_INDICTRANS = os.path.join(TRANSLATED_DIR, "indictrans2_hi.jsonl")

QA_GOLD_PATH = os.path.join(DATA_DIR, "qa_gold/gold_qa_master.csv")
RUNS_CSV = os.path.join(OUTPUTS_DIR, "runs", "all_runs.csv")
EVAL_CSV = os.path.join(OUTPUTS_DIR, "evaluation", "all_results.csv")
ANALYSIS_DIR = os.path.join(OUTPUTS_DIR, "analysis")

# Hugging Face Models for vLLM
MODELS_HF = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen": "Qwen/Qwen2.5-72B-Instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "sarvam": "sarvamai/sarvam-30b" 
}

# Translation Models
SARVAM_TRANSLATE_HF = "sarvamai/sarvam-translate"
INDICTRANS2_HF = "ai4bharat/indictrans2-en-indic-1B"

# Conditions
CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]

# Data mappings
MINISTRY_MAP = {
    "1_Ayush": "ayush",
    "2_Education": "education",
    "3_Labour": "labour",
    "4_Women": "women",
}