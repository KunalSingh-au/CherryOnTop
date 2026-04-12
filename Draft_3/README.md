# Thesis: LLM Translation Fidelity on Indian Civic Text

## Research Question
Which multilingual QA pipeline best preserves factual fidelity when answering
Hindi questions about English Lok Sabha documents?

## Three Pipelines Compared
- **Pipeline A** — English doc + Hindi question → Hindi answer (no translation)
- **Pipeline B** — API-translated Hindi doc + Hindi question → Hindi answer
- **Pipeline C** — IndicTrans2 (GPU) translated Hindi doc + Hindi question → Hindi answer

## Dataset
- 4 ministries × 5 documents = 20 Lok Sabha PDFs
- 5 manually written Hindi questions per document = 100 QA pairs

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run order
```bash
# 1. Extract text from PDFs
python scripts/01_extract_pdfs.py

# 2. Translate docs (API models)
python scripts/02_translate_docs.py

# 3. Translate docs (GPU — IndicTrans2)
python scripts/03_translate_gpu.py

# 4. Run all three QA pipelines
python scripts/04_run_pipelines.py

# 5. Back-translate all answers
python scripts/05_backtranslate.py

# 6. Evaluate everything
python scripts/06_evaluate.py

# 7. Build results table
python scripts/07_results_table.py
```

## Directory structure
```
data/
  raw_pdfs/          ← your 20 Lok Sabha PDFs, named ministry_N_en.pdf
  extracted/         ← extracted English text (JSONL)
  translated/        ← Hindi translated docs (JSONL, one per model)
  qa_gold/           ← gold_qa.csv (you fill this manually)
outputs/
  pipeline_a/        ← Pipeline A QA results
  pipeline_b/        ← Pipeline B QA results
  pipeline_c/        ← Pipeline C QA results
  evaluation/        ← metric scores, final table
utils/
  translate.py       ← all translation functions
  metrics.py         ← chrF, BERTScore, NE, semantic scoring
  extract.py         ← PDF extraction
config.py            ← API keys and settings
```
