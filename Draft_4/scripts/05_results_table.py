#!/usr/bin/env python3
"""
05_results_table.py
Print all thesis result tables from outputs/evaluation/all_results.csv

Table 1: Mean metrics by pipeline × QA model × context type
Table 2: Mean semantic score by ministry
Table 3: Pipeline A vs B comparison (the key finding)
Table 4: Metric correlation with semantic score
"""

import os, csv, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EVAL_ALL, EVAL_SUMMARY

if not os.path.exists(EVAL_ALL):
    print(f"ERROR: {EVAL_ALL} not found. Run 04_evaluate.py first.")
    sys.exit(1)

with open(EVAL_ALL, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} rows\n")

def sfloat(v):
    try: return float(v) if v not in ("", "None", None) else None
    except: return None

def mean(vals):
    vals = [v for v in vals if v is not None and v >= 0]
    return round(sum(vals)/len(vals), 3) if vals else None

def fmt(v, decimals=3):
    if v is None: return "  —  "
    return f"{v:.{decimals}f}"


# ── Table 1: By pipeline × context × QA model ────────────────────────────────
print("=" * 80)
print("TABLE 1: Mean metrics by configuration")
print("=" * 80)
print(f"{'Pipeline':<4} {'Context':<22} {'QA model':<8} "
      f"{'chrF':>6} {'BERT':>6} {'NE':>6} {'Num':>6} {'Sem':>5} {'N':>4}")
print("-" * 80)

groups = defaultdict(list)
for r in rows:
    key = (r.get("pipeline"), r.get("context_type"), r.get("qa_model"))
    groups[key].append(r)

summary_rows = []
for (pipe, ctx, model), group in sorted(groups.items()):
    row_out = {
        "pipeline": pipe, "context_type": ctx, "qa_model": model,
        "n": len(group),
        "mean_chrf":            mean([sfloat(r.get("chrf")) for r in group]),
        "mean_bertscore":       mean([sfloat(r.get("bertscore_f1")) for r in group]),
        "mean_ne":              mean([sfloat(r.get("ne_preservation")) for r in group]),
        "mean_numeric":         mean([sfloat(r.get("numeric_fidelity")) for r in group]),
        "mean_semantic":        mean([sfloat(r.get("semantic_score")) for r in group]),
    }
    summary_rows.append(row_out)
    print(f"{pipe:<4} {ctx:<22} {model:<8} "
          f"{fmt(row_out['mean_chrf']):>6} "
          f"{fmt(row_out['mean_bertscore']):>6} "
          f"{fmt(row_out['mean_ne']):>6} "
          f"{fmt(row_out['mean_numeric']):>6} "
          f"{fmt(row_out['mean_semantic'], 2):>5} "
          f"{len(group):>4}")


# ── Table 2: By ministry ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 2: Mean semantic score by ministry × pipeline")
print("=" * 60)
print(f"{'Ministry':<14} {'Pipeline':<4} {'Sem score':>10} {'N':>4}")
print("-" * 36)

min_groups = defaultdict(list)
for r in rows:
    key = (r.get("ministry", "?"), r.get("pipeline", "?"))
    min_groups[key].append(r)

for (ministry, pipe), group in sorted(min_groups.items()):
    sem = mean([sfloat(r.get("semantic_score")) for r in group])
    print(f"{ministry:<14} {pipe:<4} {fmt(sem, 2):>10} {len(group):>4}")


# ── Table 3: Key comparison — Pipeline A vs B ─────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 3: Pipeline A vs B — does translation degrade QA?")
print("(This is your main finding)")
print("=" * 60)

a_scores = [sfloat(r.get("semantic_score"))
            for r in rows if r.get("pipeline") == "A"]
b_scores = [sfloat(r.get("semantic_score"))
            for r in rows if r.get("pipeline") == "B"]

mean_a = mean(a_scores)
mean_b = mean(b_scores)
delta  = round(mean_a - mean_b, 3) if mean_a and mean_b else None

print(f"  Pipeline A (English doc, Hindi Q):      {fmt(mean_a, 3)}")
print(f"  Pipeline B (Translated doc, Hindi Q):   {fmt(mean_b, 3)}")
print(f"  Delta (A - B):                          {fmt(delta, 3)}")
if delta is not None:
    if delta > 0:
        print(f"\n  → Pipeline A scores {delta:.3f} higher than B.")
        print(f"    Translation-first DEGRADES QA fidelity by this margin.")
    else:
        print(f"\n  → Pipeline B performs similarly to or better than A.")

# Keyword match rates
a_kw = [r.get("keyword_match") for r in rows
        if r.get("pipeline") == "A" and r.get("keyword_match") is not None]
b_kw = [r.get("keyword_match") for r in rows
        if r.get("pipeline") == "B" and r.get("keyword_match") is not None]

if a_kw and b_kw:
    a_rate = sum(1 for v in a_kw if v in (True, "True")) / len(a_kw)
    b_rate = sum(1 for v in b_kw if v in (True, "True")) / len(b_kw)
    print(f"\n  Keyword match rate:")
    print(f"    Pipeline A: {a_rate:.1%} ({len(a_kw)} questions)")
    print(f"    Pipeline B: {b_rate:.1%} ({len(b_kw)} questions)")


# ── Table 4: Metric correlations ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 4: Pearson correlation with semantic score")
print("(Do automated metrics agree with GPT-4o judgment?)")
print("=" * 60)

try:
    import numpy as np
    for metric in ["chrf", "bertscore_f1", "ne_preservation",
                   "numeric_fidelity", "acronym_fidelity"]:
        pairs = [(sfloat(r.get(metric)), sfloat(r.get("semantic_score")))
                 for r in rows]
        pairs = [(x, y) for x, y in pairs
                 if x is not None and y is not None and y >= 0]
        if len(pairs) < 10:
            continue
        xs = np.array([p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])
        r_val = round(float(np.corrcoef(xs, ys)[0, 1]), 3)
        strength = ("strong" if abs(r_val) >= 0.6
                    else "moderate" if abs(r_val) >= 0.4
                    else "weak")
        print(f"  {metric:<25} r = {r_val:+.3f}  ({strength})")
    print("\n  NOTE: metrics with r < 0.4 miss errors GPT catches.")
    print("  This shows why GPT-as-judge is necessary for civic text.")
except ImportError:
    print("  Install numpy for correlations: pip install numpy")


# ── Save summary CSV ──────────────────────────────────────────────────────────
if summary_rows:
    with open(EVAL_SUMMARY, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary saved → {EVAL_SUMMARY}")

print("\nYour thesis tables are ready.")
