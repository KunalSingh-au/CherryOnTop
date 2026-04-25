# HindiParl-QA

**Benchmarking LLMs for Hindi Parliamentary Document Question Answering**

A research pipeline for evaluating Sarvam-30B, Llama-3.3-70B, Qwen-2.5-72B, and Mixtral-8x7B on answering Hindi questions from Indian Lok Sabha parliamentary records, across nine document context conditions with and without structured prompting instructions.

---

## Project Structure

```
hindiparl_qa/
├── config.py                    ← All paths, model names, tunable parameters
├── requirements.txt
│
├── data/
│   ├── raw_pdfs/
│   │   ├── english/             ← English Lok Sabha PDFs (1_Ayush/, 2_Education/, …)
│   │   └── hindi/               ← Official Hindi PDFs  (same subfolder structure)
│   ├── extracted/               ← Output of Step 1
│   ├── translated/              ← Output of Step 2
│   └── qa_gold/
│       └── gold_qa_master.csv   ← Your hand-annotated gold QA file
│
├── scripts/
│   ├── 01_extract_docs.py       ← Step 1: Extract text from PDFs
│   ├── 02_translate_docs.py     ← Step 2: Translate English docs to Hindi
│   ├── 03_run_inference.py      ← Step 3: Run LLM inference (all conditions)
│   ├── 04_evaluate.py           ← Step 4: Score answers
│   ├── 05_analyze.py            ← Step 5: Generate thesis tables
│   └── demo_viva.py             ← Demo: 10-question run for viva / presentation
│
├── utils/
│   ├── extract.py               ← PDF reading and parliamentary document parsing
│   ├── jsonl.py                 ← JSONL file helpers
│   ├── llm.py                   ← Prompt builder, answer cleaner, back-translator
│   └── metrics.py               ← BERTScore, ROUGE-L, keyword hit rate
│
└── outputs/
    ├── runs/                    ← all_runs.csv (raw model answers)
    ├── evaluation/              ← all_results.csv (scored)
    └── analysis/                ← 8 thesis tables (CSV)
```

---

## Setup

### 1. Create and activate environment

```bash
conda create -n hindiparl python=3.10 -y
conda activate hindiparl
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install vLLM (GPU inference only)

```bash
pip install vllm
```

### 4. Install IndicTrans2 toolkit (only needed for Step 2 IndicTrans2 translation)

```bash
pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git
```

### 5. Set HuggingFace token (required to download gated models like Llama)

```bash
export HF_TOKEN=hf_your_token_here
huggingface-cli login
```

---

## Running the Full Pipeline

Run the steps in order. Each step produces files that the next step depends on.

---

### Step 1 — Extract text from PDFs

Place your PDFs in:
- `data/raw_pdfs/english/1_Ayush/`, `data/raw_pdfs/english/2_Education/`, etc.
- `data/raw_pdfs/hindi/1_Ayush/`,   `data/raw_pdfs/hindi/2_Education/`,   etc.

Then run:

```bash
python scripts/01_extract_docs.py
```

**Output:** `data/extracted/english_docs.jsonl`, `data/extracted/hindi_official_docs.jsonl`

---

### Step 2 — Translate English docs to Hindi

```bash
# Run both translation engines (C2 and C5 contexts)
python scripts/02_translate_docs.py --engine both

# Or run one at a time:
python scripts/02_translate_docs.py --engine sarvam
python scripts/02_translate_docs.py --engine indictrans2
```

**Output:** `data/translated/sarvam_hi.jsonl`, `data/translated/indictrans2_hi.jsonl`

---

### Step 3 — Run LLM inference

This is the main experiment step. Use `--prompts` or `--no-prompts` to control the prompting condition. Both write to the same `all_runs.csv` with a `run_tag` column so you can filter later.

```bash
# ── Prompted branch (with Hindi task instructions) ────────────────────────────

# All 9 conditions, one model at a time:
python scripts/03_run_inference.py --model llama   --prompts --conditions all
python scripts/03_run_inference.py --model qwen    --prompts --conditions all
python scripts/03_run_inference.py --model mixtral --prompts --conditions all
python scripts/03_run_inference.py --model sarvam  --prompts --conditions all

# ── No-Prompt branch (bare context + question) ────────────────────────────────

python scripts/03_run_inference.py --model llama   --no-prompts --conditions all
python scripts/03_run_inference.py --model qwen    --no-prompts --conditions all
python scripts/03_run_inference.py --model mixtral --no-prompts --conditions all
python scripts/03_run_inference.py --model sarvam  --no-prompts --conditions all

# ── Run only full-document conditions (faster, skip RAG) ─────────────────────
python scripts/03_run_inference.py --model llama --prompts --conditions fulldoc

# ── Run only RAG conditions ───────────────────────────────────────────────────
python scripts/03_run_inference.py --model llama --prompts --conditions rag
```

**Output:** `outputs/runs/all_runs.csv`  
Each row has a `run_tag` column: `"prompted"` or `"noprompt"`.

---

### Step 4 — Evaluate (score all answers)

```bash
# Score everything in all_runs.csv
python scripts/04_evaluate.py

# Score only the prompted branch
python scripts/04_evaluate.py --run-tag prompted

# Score only the no-prompt branch
python scripts/04_evaluate.py --run-tag noprompt

# Custom input/output paths
python scripts/04_evaluate.py \
    --input  outputs/runs/all_runs.csv \
    --output outputs/evaluation/all_results.csv
```

**Output:** `outputs/evaluation/all_results.csv`  
New columns: `back_en`, `response_type` (valid/refusal/null), `keyword_hit_rate`, `rougeL_f1`, `bertscore_f1`.

---

### Step 5 — Generate thesis tables

```bash
# All 8 tables from the full results
python scripts/05_analyze.py

# Only prompted branch
python scripts/05_analyze.py --run-tag prompted

# Only no-prompt branch
python scripts/05_analyze.py --run-tag noprompt

# Side-by-side comparison of prompted vs no-prompt
python scripts/05_analyze.py --compare \
    outputs/evaluation/all_results_prompted.csv \
    outputs/evaluation/all_results_noprompt.csv
```

**Output:** 8 CSV files in `outputs/analysis/`

---

## Demo Mode (Viva / Presentation)

Runs **10 questions** through **2 conditions** (C1 English full-doc + C3 closed-book) using a **small 7B model** via HuggingFace transformers. No vLLM or multi-GPU needed. Works on a laptop with 16 GB RAM (CPU, slow) or a single 16 GB GPU.

```bash
# Default: Qwen-2.5-7B, 10 questions, with prompts
python scripts/demo_viva.py

# Without prompts (no-prompt baseline)
python scripts/demo_viva.py --no-prompts

# Different model or question count
python scripts/demo_viva.py --model Qwen/Qwen2.5-7B-Instruct --n 5

# Force CPU (if no GPU available)
python scripts/demo_viva.py --device cpu
```

The demo prints a clean side-by-side table:

```
[C1] Q: आयुर्ज्ञान योजना के अंतर्गत कितनी अनुसंधान परियोजनाएं ...
  GOLD  : A total of 34 research projects were approved under ...
  MODEL : इस योजना के तहत कुल 34 अनुसंधान परियोजनाएं स्वीकृत ...
  KW-Hit: 0.67   ROUGE-L: 0.3124
```

---

## Key Configuration Options (config.py)

All tunable parameters are in `config.py`. You should not need to edit any script files.

| Parameter | Default | What it controls |
|---|---|---|
| `TENSOR_PARALLEL_SIZE` | 4 | Number of GPUs for vLLM tensor parallelism |
| `MAX_MODEL_LEN` | 16384 | Maximum context window tokens |
| `MAX_NEW_TOKENS` | 1024 | Max tokens generated per answer |
| `CONTEXT_CHAR_LIMIT` | 3500 | Characters of document fed to model per call |
| `CHUNK_SIZE` | 300 | Words per RAG chunk |
| `CHUNK_OVERLAP` | 50 | Word overlap between consecutive chunks |
| `RAG_TOP_K` | 3 | Chunks retrieved per query |
| `BERTSCORE_BATCH` | 32 | BERTScore batch size (reduce to 16 if OOM) |
| `RAG_EMBEDDER` | paraphrase-multilingual-MiniLM-L12-v2 | Embedding model for RAG |

---

## The `--prompts` / `--no-prompts` Flag Explained

This is the core experimental variable. The same `03_run_inference.py` script handles both conditions:

| Flag | What the model sees |
|---|---|
| `--prompts` | Hindi task instructions + context + question |
| `--no-prompts` | Only context + question (no instructions) |

The instructions tell the model to: (1) answer in Devanagari Hindi, (2) keep the answer under 100 words, (3) write *jankaari upalabdh nahin hai* if the answer is not in the document.

Both runs write to the same CSV with `run_tag = "prompted"` or `"noprompt"`, so you can analyse them together or separately.

---

## Understanding the 9 Conditions

| ID | Description | Source |
|---|---|---|
| C1 | Full document — original English PDF | English extraction |
| C2 | Full document — Sarvam-Translate → Hindi | Sarvam translation |
| C3 | No context — closed-book hallucination floor | No document |
| C4 | Full document — official Hindi PDF | Hindi extraction |
| C5 | Full document — IndicTrans2 → Hindi | IndicTrans2 translation |
| C6 | RAG chunks from C1 (English) | Retrieval from C1 |
| C7 | RAG chunks from C2 (Sarvam MT) | Retrieval from C2 |
| C8 | RAG chunks from C4 (Official Hindi) | Retrieval from C4 |
| C9 | RAG chunks from C5 (IndicTrans2) | Retrieval from C5 |

---

## Evaluation Validity Notes

Three issues from the original pipeline are corrected here:

1. **BERTScore**: Always uses `model_type="xlm-roberta-large"` with `lang=None`. The original no-prompt code used `lang="hi"`, which produced incomparable scores.

2. **Null responses**: Rows where `answer_hi` is empty are classified as `"null"` and excluded from metric averages. They are reported separately as part of coverage.

3. **Refusal responses**: Rows where the model explicitly declines to answer ("information not available") are classified as `"refusal"` and also excluded from quality metrics, but reported as part of coverage.

---

## Citation

If you use this pipeline or benchmark, please cite:

```bibtex
@thesis{kunal2026hindiparl,
  title  = {HindiParl-QA: Benchmarking Large Language Models for Hindi Parliamentary Document Question Answering},
  author = {Kunal Singh},
  year   = {2026},
  school = {Ashoka University},
}
```
