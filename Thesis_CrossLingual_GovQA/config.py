# Thesis experiment configuration — set API keys via environment variables.

import os

# ── API keys (env preferred) ─────────────────────────────────────────────────
SARVAM_KEY = os.getenv("SARVAM_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # Gemini
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Llama via Groq

# ── Model IDs ────────────────────────────────────────────────────────────────
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-m")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
LLAMA_GROQ_MODEL = os.getenv("LLAMA_GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Rate limiting (seconds between calls) ────────────────────────────────────
SLEEP_SARVAM = float(os.getenv("SLEEP_SARVAM", "1.0"))
SLEEP_GEMINI = float(os.getenv("SLEEP_GEMINI", "0.4"))
SLEEP_MISTRAL = float(os.getenv("SLEEP_MISTRAL", "0.3"))
SLEEP_GROQ = float(os.getenv("SLEEP_GROQ", "0.3"))
MAX_CHARS_TRANSLATE = int(os.getenv("MAX_CHARS_TRANSLATE", "900"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3500"))
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "512"))

# ── PDF layout (same as Draft_4) ─────────────────────────────────────────────
MINISTRY_MAP = {
    "1_Ayush": "ayush",
    "2_Education": "education",
    "3_Labour": "labour",
    "4_Women": "women",
}

PDF_DIR_EN = "data/raw_pdfs"
PDF_DIR_HI = "data/raw_pdfs_hi"

EXTRACTED_EN = "data/extracted/english_docs.jsonl"
EXTRACTED_HI = "data/extracted/hindi_official_docs.jsonl"
TRANSLATED_SARVAM = "data/translated/sarvam_hi.jsonl"

QA_GOLD_PATH = "data/qa_gold/gold_qa_master.csv"

RUNS_CSV = "outputs/runs/all_runs.csv"
EVAL_CSV = "outputs/evaluation/all_results.csv"
SUMMARY_CSV = "outputs/evaluation/summary_by_model_condition.csv"
ANALYSIS_DIR = "outputs/analysis"

CONDITIONS = ("C1", "C2", "C3", "C4")
MODELS = ("sarvam", "gemini", "mistral", "llama")

BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "microsoft/mdeberta-v3-base-xnli")
