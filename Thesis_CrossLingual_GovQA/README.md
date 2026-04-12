# Cross-lingual government QA (thesis experiment)

**100** Hindi QA pairs (official Lok Sabha Hindi questions), **four** document conditions (**C1–C4**), **four** LLMs (Sarvam, Gemini, Mistral, Llama via Groq), then automatic scoring and analysis.

## Quick pointer

Step-by-step commands, API key links, and a **10-question sample run** are in **`RUNBOOK.txt`** (start there).

## Conditions

| ID | Setup |
|----|--------|
| **C1** | Official Hindi question + **English** document |
| **C2** | Same + **Sarvam** machine-translated Hindi document |
| **C3** | Same question, **no** document |
| **C4** | Same + **official Hindi** PDF text |

## Outputs (default paths)

- `outputs/runs/all_runs.csv` — generations  
- `outputs/evaluation/all_results.csv` — + keyword / ROUGE-L / BERTScore / judge  
- `outputs/analysis/*.csv` — aggregated tables  

## Metrics (short)

Keyword hit rate (vs gold keywords), ROUGE-L and multilingual BERTScore on Sarvam back-translation vs English gold, optional Gemini hallucination rubric, doc-level chrF fidelity for correlations.

## PDF layout

Default: **`data/raw_pdfs/english/<ministry>/`**, **`data/raw_pdfs/hindi/<ministry>/`** (e.g. `1_Ayush`, `2_Education`, …).

Legacy (Draft_4): set `PDF_DIR_EN=data/raw_pdfs` and `PDF_DIR_HI=data/raw_pdfs_hi` so PDFs live in `data/raw_pdfs/1_Ayush/` etc.

Hindi parsing is heuristic; adjust `utils/extract.py` if your PDFs differ.
