"""
verify_tables.py — Regenerates every table in the paper from all_results.csv.

Prints each table to stdout AND saves it as a CSV so you can open it in Excel.

Usage:
    python verify_tables.py --results path/to/all_results.csv

Outputs land in --out-dir (default: outputs/analysis/verified_tables/).
One CSV per table, named to match the LaTeX label in the paper.
"""

import argparse
import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--results",
    default="outputs/evaluation/all_results.csv",
    help="Path to all_results.csv (default: outputs/evaluation/all_results.csv)",
)
parser.add_argument(
    "--out-dir",
    default="outputs/verified_tables",
    help="Folder to write CSVs into (created if it doesn't exist)",
)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sep(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def explain(text):
    for line in text.strip().split("\n"):
        print("  " + line.strip())
    print()

def save(frame, filename):
    path = os.path.join(args.out_dir, filename)
    frame.to_csv(path)
    print(f"\n  [saved -> {path}]")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
sep("LOADING DATA")
cols = ["model", "ministry", "condition", "run_tag",
        "response_type", "keyword_hit_rate", "rougeL_f1", "bertscore_f1"]
df = pd.read_csv(args.results, usecols=cols)

print(f"  Loaded {len(df):,} rows from: {args.results}")
print(f"  run_tag values  : {sorted(df['run_tag'].dropna().unique().tolist())}")
print(f"  response_type   : {sorted(df['response_type'].dropna().unique().tolist())}")
print(f"  models          : {sorted(df['model'].unique().tolist())}")
print(f"  conditions      : {sorted(df['condition'].unique().tolist())}")
print(f"  ministries      : {sorted(df['ministry'].unique().tolist())}")
print(f"\n  Output folder   : {args.out_dir}")

# Subsets used throughout
pv  = df[(df["run_tag"] == "prompted")  & (df["response_type"] == "valid")]
npv = df[(df["run_tag"] == "noprompt")  & (df["response_type"] == "valid")]

METRICS = ["bertscore_f1", "rougeL_f1", "keyword_hit_rate"]

COND_LABEL = {
    "C1": "Full Doc - English",
    "C2": "Full Doc - Sarvam-MT",
    "C3": "No Context (closed-book)",
    "C4": "Full Doc - Official Hindi",
    "C5": "Full Doc - IndicTrans2",
    "C6": "RAG - English Chunks",
    "C7": "RAG - Sarvam-MT Chunks",
    "C8": "RAG - Official Hindi",
    "C9": "RAG - IndicTrans2",
}


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — tab:main_prompted
# ─────────────────────────────────────────────────────────────────────────────
sep("TABLE 1 — tab:main_prompted — Main Model Leaderboard")
explain("""
FILTER : run_tag == 'prompted'  AND  response_type == 'valid'
GROUP  : by model
COMPUTE: mean of bertscore_f1, rougeL_f1, keyword_hit_rate
         across all 9 conditions combined (up to 900 rows per model)

Coverage column is computed separately — see Table 2.

Rows with response_type == 'refusal' or NaN are excluded entirely.
Metrics are only measured on answers the model actually gave.
""")

t1 = pv.groupby("model")[METRICS].mean().round(3)
t1 = t1.sort_values("bertscore_f1", ascending=False)
print(t1.to_string())
save(t1, "tab1_main_leaderboard_prompted_valid.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — tab:coverage
# ─────────────────────────────────────────────────────────────────────────────
sep("TABLE 2 — tab:coverage — Response Coverage by Model and Condition")
explain("""
FILTER : all rows (no filter on response_type)
GROUP  : by (model, run_tag)
COMPUTE: for each bucket:
           valid   = response_type == 'valid'
           refusal = response_type == 'refusal'
           null    = response_type is NaN  (empty string from model)
         valid_pct = valid / total_rows * 100

Expected total per model per run_tag = 100 questions x 9 conditions = 900.

Sarvam's NP null rate tells you whether its failures are explicit refusals
(model knows it doesn't know) or silent empty outputs (serving/token issue).
""")

coverage_rows = []
for tag in ["prompted", "noprompt"]:
    sub = df[df["run_tag"] == tag]
    for model, grp in sub.groupby("model"):
        total   = len(grp)
        valid   = (grp["response_type"] == "valid").sum()
        refusal = (grp["response_type"] == "refusal").sum()
        null    = grp["response_type"].isna().sum()
        coverage_rows.append({
            "run_tag":     tag,
            "model":       model,
            "total_rows":  total,
            "valid":       valid,
            "valid_pct":   round(valid / total * 100, 1),
            "refusal":     refusal,
            "refusal_pct": round(refusal / total * 100, 1),
            "null_empty":  null,
            "null_pct":    round(null / total * 100, 1),
        })

t2 = pd.DataFrame(coverage_rows).set_index(["run_tag", "model"])
print(t2.to_string())
save(t2, "tab2_coverage_by_model_runtag.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — tab:conditions_results
# ─────────────────────────────────────────────────────────────────────────────
sep("TABLE 3 — tab:conditions_results — Performance by Context Condition")
explain("""
FILTER : run_tag == 'prompted'  AND  response_type == 'valid'
GROUP  : by condition
COMPUTE: mean of all three metrics, collapsed across all 4 models

Answers: ignoring which model was used, which context type produced the best answers?
Sorted by rougeL_f1 descending.
""")

t3 = pv.groupby("condition")[METRICS].mean().round(3)
t3.insert(0, "description", t3.index.map(COND_LABEL))
t3 = t3.sort_values("rougeL_f1", ascending=False)
print(t3.to_string())
save(t3, "tab3_condition_rankings_prompted_valid.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4 — tab:prompt_comparison
# ─────────────────────────────────────────────────────────────────────────────
sep("TABLE 4 — tab:prompt_comparison — Prompted vs No-Prompt Quality on Valid Responses")
explain("""
FILTER : response_type == 'valid'  (applied to both run_tags separately)
GROUP  : by model
COMPUTE: mean metrics for noprompt and prompted side by side
         rougeL_gain = prompted_rougeL - noprompt_rougeL

Shows how much prompting improves quality on the answers that are given.
Does NOT show the coverage cost - that is in Table 2.
""")

np_q = npv.groupby("model")[METRICS].mean().round(3).add_suffix("_NP")
p_q  = pv.groupby("model")[METRICS].mean().round(3).add_suffix("_P")
t4 = pd.concat([np_q, p_q], axis=1)
t4["rougeL_gain"] = (t4["rougeL_f1_P"] - t4["rougeL_f1_NP"]).round(3)
t4["bert_gain"]   = (t4["bertscore_f1_P"] - t4["bertscore_f1_NP"]).round(3)
print(t4.to_string())
save(t4, "tab4_prompted_vs_noprompt_quality.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 5 — tab:translation
# ─────────────────────────────────────────────────────────────────────────────
sep("TABLE 5 — tab:translation — Translation Engine Comparison (C1 vs C2 vs C5)")
explain("""
FILTER : run_tag == 'prompted', response_type == 'valid',
         condition in ['C1', 'C2', 'C5']
GROUP  : by (condition, model)
COMPUTE: mean rougeL_f1
PIVOT  : rows = conditions, columns = models

C1 = English (no translation)
C2 = Sarvam-MT Hindi translation
C5 = IndicTrans2 Hindi translation

Comparing these three conditions isolates the effect of translation engine.
""")

t5_sub = pv[pv["condition"].isin(["C1","C2","C5"])]
t5 = t5_sub.groupby(["condition","model"])["rougeL_f1"].mean().unstack().round(3)
t5.insert(0, "description", t5.index.map(COND_LABEL))
print(t5.to_string())
save(t5, "tab5_translation_comparison_rougeL.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 6 — tab:ministry_bert
# ─────────────────────────────────────────────────────────────────────────────
sep("TABLE 6 — tab:ministry_bert — BERTScore by Ministry x Model (Prompted Valid)")
explain("""
FILTER : run_tag == 'prompted', response_type == 'valid'
GROUP  : by (ministry, model)
COMPUTE: mean bertscore_f1
PIVOT  : rows = ministries, columns = models

BERTScore = semantic similarity between back-translated answer and English gold,
using xlm-roberta-large. High score = answer is semantically on-topic.
Does NOT require exact numbers or scheme names.
""")

t6 = pv.groupby(["ministry","model"])["bertscore_f1"].mean().unstack().round(3)
print(t6.to_string())
save(t6, "tab6_ministry_bertscore.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 7 — tab:ministry_kw
# ─────────────────────────────────────────────────────────────────────────────
sep("TABLE 7 — tab:ministry_kw — Keyword Hit Rate by Ministry x Model (Prompted Valid)")
explain("""
FILTER : run_tag == 'prompted', response_type == 'valid'
GROUP  : by (ministry, model)
COMPUTE: mean keyword_hit_rate
PIVOT  : rows = ministries, columns = models

keyword_hit_rate per row = fraction of gold keywords found verbatim
(case-insensitive substring match) in the back-translated answer.

Hardest metric: requires exact numbers, scheme names, acronyms.
The Ayush gap vs Labour/Women is clearly visible here.
""")

t7 = pv.groupby(["ministry","model"])["keyword_hit_rate"].mean().unstack().round(3)
print(t7.to_string())
save(t7, "tab7_ministry_keyword_hitrate.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 8 — tab:rag_delta
# ─────────────────────────────────────────────────────────────────────────────
sep("TABLE 8 — tab:rag_delta — RAG Delta (RAG minus Full Doc) per Model and Context Type")
explain("""
FILTER : run_tag == 'prompted', response_type == 'valid'
PAIRS  : (C6 vs C1), (C7 vs C2), (C8 vs C4), (C9 vs C5)
COMPUTE: delta = mean(RAG condition) - mean(Full Doc condition), per model

Positive delta = RAG helped vs full document.
Negative delta = full document was better than RAG chunks.
""")

rag_pairs = [
    ("C6", "C1", "English"),
    ("C7", "C2", "Sarvam-MT"),
    ("C8", "C4", "Official Hindi"),
    ("C9", "C5", "IndicTrans2"),
]

rows = []
for rag_c, full_c, label in rag_pairs:
    for model in sorted(df["model"].unique()):
        rag_vals  = pv[(pv["condition"] == rag_c)  & (pv["model"] == model)][METRICS].mean()
        full_vals = pv[(pv["condition"] == full_c) & (pv["model"] == model)][METRICS].mean()
        delta = (rag_vals - full_vals).round(3)
        rows.append({
            "model":      model,
            "context":    label,
            "rag_cond":   rag_c,
            "full_cond":  full_c,
            "delta_bert": delta["bertscore_f1"],
            "delta_rougeL": delta["rougeL_f1"],
            "delta_kw":   delta["keyword_hit_rate"],
        })

t8 = pd.DataFrame(rows).set_index(["model", "context"])
print(t8.to_string())
save(t8, "tab8_rag_delta.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK — row counts per cell
# ─────────────────────────────────────────────────────────────────────────────
sep("SANITY CHECK — Valid row counts per (model, condition) — Prompted")
explain("""
If any cell has very few rows (< 20), that average is unreliable.
Expected max = 100 (one per question).
Low count = model refused/null for most queries in that condition.
""")

counts = pv.groupby(["model","condition"]).size().unstack(fill_value=0)
print(counts.to_string())
save(counts, "sanity_valid_row_counts_prompted.csv")

print()
totals = pv.groupby("model").size().rename("total_valid_prompted")
print(totals.to_string())
print("\n  Expected max per model: 900 (100 questions x 9 conditions)")

print("\n" + "=" * 70)
print(f"  Done. All CSVs written to: {args.out_dir}")
print("=" * 70 + "\n")