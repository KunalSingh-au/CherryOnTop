# Cross-lingual government QA thesis experiment

End-to-end pipeline for **100 Hindi QA pairs** (25 per ministry) with four **conditions** and four **LLMs**, plus evaluation and analysis aligned with a cross-lingual policy-QA thesis.

## Conditions

| ID | Setup |
|----|--------|
| **C1** | **Official Hindi** question (Lok Sabha wording) + **English** official document |
| **C2** | Same official Hindi question + **Sarvam** machine-translated Hindi document |
| **C3** | Same official Hindi question **only** (no document; hallucination baseline) |
| **C4** | Same official Hindi question + **official Hindi** PDF text |

## Models

- **Sarvam** (`SARVAM_KEY`) — chat API  
- **Gemini Flash** (`GOOGLE_API_KEY`, default `gemini-2.0-flash`)  
- **Mistral** (`MISTRAL_API_KEY`, default `mistral-small-latest`)  
- **LLaMA** via **Groq** (`GROQ_API_KEY`, default `llama-3.3-70b-versatile`)

## Dataset

- Gold merged from `Draft_2/gold_qa_complete.csv` + `gold_qa_official_hindi.csv` → `data/qa_gold/gold_qa_master.csv`.
- **Questions:** only **`question_hi_official`** from the official Hindi sheet is stored and used for **every** condition (the shorter “original” Hindi phrasing from `gold_qa_complete.csv` is not used).
- **`AU3868` is omitted** during merge. Rows without an official Hindi question are dropped.
- **C4** is enabled when `doc_available` is true and gold is not `MISSING`.

Expected generation count with full PDF coverage: **100 × 4 × 4 = 1600** LLM calls.

## Setup

```bash
cd Thesis_CrossLingual_GovQA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables:

| Variable | Used for |
|----------|-----------|
| `SARVAM_KEY` | Sarvam translate + QA + back-translation in eval |
| `GOOGLE_API_KEY` | Gemini QA + LLM-as-judge |
| `MISTRAL_API_KEY` | Mistral QA |
| `GROQ_API_KEY` | LLaMA QA |

Optional overrides: `GEMINI_MODEL`, `MISTRAL_MODEL`, `LLAMA_GROQ_MODEL`, `BERTSCORE_MODEL`.

## PDF layout

Mirror **Draft_4**:

- English: `data/raw_pdfs/1_Ayush/`, `2_Education/`, `3_Labour/`, `4_Women/` — files named `AUxxxx_….pdf`.
- Official Hindi: same ministry subfolders under **`data/raw_pdfs_hi/`**.

## Pipeline (order)

1. `python scripts/build_gold_master.py` — refresh `gold_qa_master.csv` from `Draft_2` (default paths).
2. `python scripts/01_extract_docs.py` — `english_docs.jsonl`, `hindi_official_docs.jsonl`.
3. `python scripts/02_translate_docs.py` — Sarvam EN→HI for **C2** → `sarvam_hi.jsonl`.
4. `python scripts/03_run_qa.py` — fills `outputs/runs/all_runs.csv` (resumable).
   - Smoke test: `python scripts/03_run_qa.py --only-model sarvam --only-condition C3 --limit 2`
5. `python scripts/04_evaluate.py` — back-translate answers, keyword hit rate, ROUGE-L, multilingual BERTScore, Gemini hallucination rubric.  
   - Fast pass: `python scripts/04_evaluate.py --skip-judge`
6. `python scripts/05_analyze.py` — model × condition tables, C2 vs C4, ministry difficulty, doc-fidelity correlations.

## Evaluation metrics (per answer)

- **Keyword hit rate** — fraction of curated phrases from `keywords_en` found in Sarvam back-translation to English.
- **ROUGE-L** F1 — back-English vs `gold_answer_en`.
- **BERTScore** F1 — multilingual encoder (`microsoft/mdeberta-v3-base-xnli` by default).
- **Hallucination** — Gemini judge: `grounded` / `minor` / `major` (+ numeric score 1.0 / 0.5 / 0.0).

**Doc fidelity** (for correlation): chrF between Sarvam-translated document (`sarvam_hi.jsonl`) and official Hindi PDF text (`hindi_official_docs.jsonl`) per `doc_id`.

## Outputs

- `outputs/runs/all_runs.csv` — raw generations.  
- `outputs/evaluation/all_results.csv` — full scored table.  
- `outputs/evaluation/summary_by_model_condition.csv` + `outputs/analysis/*.csv` — thesis tables.

## Note on parliamentary Hindi PDFs

`parse_hindi_qa` uses tolerant markers (`उत्तर`, `ANSWER`, etc.). If extraction returns empty answers, adjust patterns in `utils/extract.py` to match your PDF layout.
