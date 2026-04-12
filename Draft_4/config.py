# config.py — fill in your API keys, everything else is set

import os

# ── API Keys ───────────────────────────────────────────────────────────────────
SARVAM_KEY  = os.getenv("SARVAM_KEY",  "")
OPENAI_KEY  = os.getenv("OPENAI_KEY",  "your_openai_key_here")

# ── Model names ────────────────────────────────────────────────────────────────
SARVAM_CHAT_MODEL = "sarvam-m"
OPENAI_CHAT_MODEL = "gpt-4o"

# ── Rate limiting ──────────────────────────────────────────────────────────────
SLEEP_SARVAM = 1.0    # seconds between Sarvam calls
SLEEP_OPENAI = 0.3    # seconds between OpenAI calls
MAX_CHARS    = 900    # Sarvam translate hard limit is 1000

# ── Your PDF folder structure ──────────────────────────────────────────────────
# PDFs live inside ministry subfolders, named with random suffixes:
#   data/raw_pdfs/1_Ayush/AU2338_kTs52v.pdf
#   data/raw_pdfs/2_Education/AU414_EvHGT6.pdf  etc.
#
# Ministry subfolder names → ministry label used in CSV:
MINISTRY_MAP = {
    "1_Ayush":     "ayush",
    "2_Education": "education",
    "3_Labour":    "labour",
    "4_Women":     "women",
}

# ── Paths ──────────────────────────────────────────────────────────────────────
PDF_DIR          = "data/raw_pdfs"
EXTRACTED_PATH   = "data/extracted/english_docs.jsonl"
QA_GOLD_PATH     = "data/qa_gold/gold_qa.csv"

TRANSLATED_SARVAM = "data/translated/sarvam_hi.jsonl"
TRANSLATED_GPT    = "data/translated/gpt_hi.jsonl"

OUTPUT_A_SARVAM  = "outputs/pipeline_a/sarvam.csv"
OUTPUT_A_GPT     = "outputs/pipeline_a/gpt.csv"
OUTPUT_B_SARVAM  = "outputs/pipeline_b/sarvam.csv"
OUTPUT_B_GPT     = "outputs/pipeline_b/gpt.csv"

EVAL_ALL         = "outputs/evaluation/all_results.csv"
EVAL_SUMMARY     = "outputs/evaluation/summary_table.csv"
