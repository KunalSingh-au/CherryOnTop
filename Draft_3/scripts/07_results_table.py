#!/usr/bin/env python3
"""
07_results_table.py
Build the thesis results tables from all_scores.csv.

Produces:
  - Table 1: Overall metrics by pipeline × QA model
  - Table 2: Metrics by ministry
  - Table 3: Correlation — do automated metrics catch manual errors?
  - outputs/evaluation/results_summary.csv
"""

import os, csv, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

all_scores_path = "outputs/evaluation/all_scores.csv"
if not os.path.exists(all_scores_path):
    print(f"ERROR: {all_scores_path} not found. Run 06_evaluate.py first.")
    sys.exit(1)

with open(all_scores_path, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} scored rows\n")


def safe_float(val):
    try:
        return float(val) if val not in ("", "None", None) else None
    except:
        return None


# ── Table 1: Overall by pipeline × QA model ──────────────────────────────────

print("="*65)
print("TABLE 1: Mean metrics by pipeline and QA model")
print("="*65)
print(f"{'Pipeline':<12} {'QA Model':<10} {'chrF':>6} {'BERT':>6} {'NE':>6} {'Num':>6} {'Sem':>6} {'N':>4}")
print("-"*65)

groups = defaultdict(list)
for r in rows:
    key = (r.get("pipeline","?"), r.get("qa_model","?"))
    groups[key].append(r)

summary_rows = []
for (pipeline, model), group in sorted(groups.items()):
    def mean(col):
        vals = [safe_float(r.get(col)) for r in group]
        vals = [v for v in vals if v is not None and v >= 0]
        return round(sum(vals)/len(vals), 3) if vals else None

    row_out = {
        "pipeline": pipeline,
        "qa_model": model,
        "n": len(group),
        "mean_chrf":             mean("chrf"),
        "mean_bertscore_f1":     mean("bertscore_f1"),
        "mean_ne_preservation":  mean("ne_preservation"),
        "mean_numeric_fidelity": mean("numeric_fidelity"),
        "mean_semantic_score":   mean("semantic_score"),
    }
    summary_rows.append(row_out)

    def fmt(v):
        return f"{v:.3f}" if v is not None else "  —  "

    print(f"{pipeline:<12} {model:<10} "
          f"{fmt(row_out['mean_chrf']):>6} "
          f"{fmt(row_out['mean_bertscore_f1']):>6} "
          f"{fmt(row_out['mean_ne_preservation']):>6} "
          f"{fmt(row_out['mean_numeric_fidelity']):>6} "
          f"{fmt(row_out['mean_semantic_score']):>6} "
          f"{len(group):>4}")


# ── Table 2: By ministry ──────────────────────────────────────────────────────

print("\n" + "="*65)
print("TABLE 2: Mean semantic score by ministry × pipeline")
print("="*65)

ministry_groups = defaultdict(list)
for r in rows:
    key = (r.get("ministry","?"), r.get("pipeline","?"))
    ministry_groups[key].append(r)

print(f"{'Ministry':<18} {'Pipeline':<12} {'Sem score':>10} {'N':>4}")
print("-"*50)
for (ministry, pipeline), group in sorted(ministry_groups.items()):
    vals = [safe_float(r.get("semantic_score")) for r in group]
    vals = [v for v in vals if v is not None and v >= 0]
    mean_sem = round(sum(vals)/len(vals), 2) if vals else None
    print(f"{ministry:<18} {pipeline:<12} "
          f"{'—' if mean_sem is None else f'{mean_sem:.2f}':>10} {len(group):>4}")


# ── Table 3: Metric correlation with semantic score ───────────────────────────

print("\n" + "="*65)
print("TABLE 3: Pearson correlation with semantic score")
print("(Do automated metrics agree with GPT-4o semantic judgment?)")
print("="*65)

try:
    import numpy as np

    def pearson(col):
        pairs = []
        for r in rows:
            x = safe_float(r.get(col))
            y = safe_float(r.get("semantic_score"))
            if x is not None and y is not None and y >= 0:
                pairs.append((x, y))
        if len(pairs) < 5:
            return None
        xs = np.array([p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])
        return round(float(np.corrcoef(xs, ys)[0,1]), 3)

    for metric in ["chrf", "bertscore_f1", "ne_preservation",
                   "numeric_fidelity", "acronym_fidelity"]:
        r = pearson(metric)
        indicator = ""
        if r is not None:
            if abs(r) >= 0.7:   indicator = " ← strong"
            elif abs(r) >= 0.4: indicator = " ← moderate"
            else:               indicator = " ← weak"
        print(f"  {metric:<25} r = {r if r is not None else '—'}{indicator}")

    print("\n  Interpretation: metrics with r < 0.4 miss errors that GPT catches.")
    print("  This shows why human/GPT evaluation is needed for civic domain text.")

except ImportError:
    print("  (numpy not available — skipping correlation)")


# ── Save summary ──────────────────────────────────────────────────────────────

summary_path = "outputs/evaluation/results_summary.csv"
if summary_rows:
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary table saved → {summary_path}")

print("\nYour thesis results are ready.")
print("Import outputs/evaluation/results_summary.csv into your thesis tables.")
