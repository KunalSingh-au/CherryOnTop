"""
config.py — Central configuration for the HindiParl-QA benchmark pipeline.

ALL paths, model names, and tunable parameters live here.
Change values in this file rather than editing individual scripts.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
PDF_DIR_EN    = os.path.join(DATA_DIR, "raw_pdfs", "english")   # English PDFs
PDF_DIR_HI    = os.path.join(DATA_DIR, "raw_pdfs", "hindi")     # Official Hindi PDFs
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")             # Extracted text JSONL
TRANSLATED_DIR= os.path.join(DATA_DIR, "translated")            # Translated text JSONL
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
EXTRACTED_EN        = os.path.join(EXTRACTED_DIR,   "english_docs.jsonl")
EXTRACTED_HI        = os.path.join(EXTRACTED_DIR,   "hindi_official_docs.jsonl")
TRANSLATED_SARVAM   = os.path.join(TRANSLATED_DIR,  "sarvam_hi.jsonl")
TRANSLATED_INDICTRANS = os.path.join(TRANSLATED_DIR,"indictrans2_hi.jsonl")

QA_GOLD_PATH  = os.path.join(DATA_DIR, "qa_gold", "gold_qa_master.csv")

# Outputs — runs and evaluation results land here
RUNS_CSV      = os.path.join(OUTPUTS_DIR, "runs",       "all_runs.csv")
EVAL_CSV      = os.path.join(OUTPUTS_DIR, "evaluation", "all_results.csv")
ANALYSIS_DIR  = os.path.join(OUTPUTS_DIR, "analysis")

# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# Keys are short names used on the command line (--model llama, etc.)
# Values are HuggingFace repo IDs loaded by vLLM.
# ─────────────────────────────────────────────────────────────────────────────
MODELS_HF = {
    "llama":   "meta-llama/Llama-3.3-70B-Instruct",
    "qwen":    "Qwen/Qwen2.5-72B-Instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "sarvam":  "sarvamai/sarvam-30b",
}

# Translation / back-translation models
SARVAM_TRANSLATE_HF  = "sarvamai/sarvam-translate"
INDICTRANS2_HF       = "ai4bharat/indictrans2-en-indic-1B"

# Embedding model used for RAG chunked retrieval (C6–C9)
# ── TUNABLE: swap for a larger multilingual embedder for better chunk recall ──
RAG_EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTAL CONDITIONS
# C1–C5 : full-document context  |  C6–C9 : RAG-chunked context
# C3    : no context (closed-book hallucination floor)
# ─────────────────────────────────────────────────────────────────────────────
# Full-doc conditions handled by 03_run_inference.py
FULL_DOC_CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]
# RAG conditions — derived from their source full-doc counterparts
RAG_CONDITIONS      = ["C6", "C7", "C8", "C9"]
# Combined (used for evaluation / analysis)
ALL_CONDITIONS      = FULL_DOC_CONDITIONS + RAG_CONDITIONS

# Which source condition each RAG condition derives its corpus from
RAG_SOURCE_MAP = {"C6": "C1", "C7": "C2", "C8": "C4", "C9": "C5"}

# ─────────────────────────────────────────────────────────────────────────────
# MINISTRY FOLDER NAMES → short labels used in outputs
# ─────────────────────────────────────────────────────────────────────────────
MINISTRY_MAP = {
    "1_Ayush":     "ayush",
    "2_Education": "education",
    "3_Labour":    "labour",
    "4_Women":     "women",
}

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE TUNING PARAMETERS
# ── TUNABLE: Adjust these if you get OOM errors or want longer answers ────────
# ─────────────────────────────────────────────────────────────────────────────

# vLLM tensor parallelism — must equal the number of GPUs available
TENSOR_PARALLEL_SIZE = 4          # 4 × 48 GB = 192 GB total VRAM

# Maximum tokens in a single context window (model input + output together)
# 16384 works for all 70B models on 4×48GB.
# Increase to 32768 if you upgrade hardware and want longer contexts.
MAX_MODEL_LEN = 16384

# Maximum tokens the model may generate per answer
# 1024 is generous for a single QA answer; increase to 2048 for multi-part answers
MAX_NEW_TOKENS = 2048

# Context character limit sent to the model.
# ~3500 chars ≈ 850 Hindi tokens, leaving room for question + instructions.
# ── TUNABLE: Increase to 8000 if your GPU memory allows it ──────────────────
CONTEXT_CHAR_LIMIT = 3500

# RAG retrieval — number of top chunks returned per query
# ── TUNABLE: Increase to 5 for longer or multi-topic answers ─────────────────
RAG_TOP_K = 3

# RAG chunk size and overlap (in words)
# ── TUNABLE: Smaller chunks (150 words) give finer retrieval;
#            larger chunks (500 words) preserve more paragraph context ─────────
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# BERTScore backbone — do NOT change to a monolingual model (see metrics.py)
BERTSCORE_MODEL = "xlm-roberta-large"

# Batch size for BERTScore computation — reduce to 16 if you hit OOM during eval
BERTSCORE_BATCH = 32

# Minimum fraction of Devanagari characters for a string to be classified Hindi
HINDI_CHAR_THRESHOLD = 0.40
