# config.py
# ── All API keys and settings in one place ────────────────────────────────────
# Copy this file and fill in your keys before running anything.

import os

# ── API Keys ──────────────────────────────────────────────────────────────────
SARVAM_KEY  = os.getenv("SARVAM_KEY",  "your_sarvam_key_here")
OPENAI_KEY  = os.getenv("OPENAI_KEY",  "your_openai_key_here")

# ── Models ────────────────────────────────────────────────────────────────────
SARVAM_CHAT_MODEL       = "sarvam-m"
OPENAI_CHAT_MODEL       = "gpt-4o"
INDICTRANS2_MODEL       = "ai4bharat/indictrans2-en-indic-1B"  # HuggingFace ID

# ── Translation settings ──────────────────────────────────────────────────────
MAX_CHARS       = 900       # Sarvam hard limit is 1000 — stay safe
SLEEP_SARVAM    = 1.0       # seconds between Sarvam API calls
SLEEP_OPENAI    = 0.3       # seconds between OpenAI calls

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_DIR         = "data/raw_pdfs"
EXTRACTED_DIR   = "data/extracted"
TRANSLATED_DIR  = "data/translated"
QA_GOLD_PATH    = "data/qa_gold/gold_qa.csv"

OUTPUT_A        = "outputs/pipeline_a/results.csv"
OUTPUT_B_SARVAM = "outputs/pipeline_b/results_sarvam.csv"
OUTPUT_B_GPT    = "outputs/pipeline_b/results_gpt.csv"
OUTPUT_C        = "outputs/pipeline_c/results_indictrans2.csv"
FINAL_TABLE     = "outputs/evaluation/final_results.csv"

# ── Ministry labels ───────────────────────────────────────────────────────────
# Keep these consistent with your PDF filenames.
# Name your PDFs like: education_1_en.pdf, education_2_en.pdf etc.
MINISTRIES = [
    "education",
    "disability",
    "minority",
    "women_child",
]
