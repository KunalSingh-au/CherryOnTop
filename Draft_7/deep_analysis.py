#!/usr/bin/env python3
"""
deep_analysis.py
================
Drop this in your project root and run:
  python deep_analysis.py --wp outputs_with_prompts --np outputs_no_prompts

Produces every table and finding needed for the thesis.
"""

import os, sys, re, argparse
import pandas as pd
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--wp", default="/scratch/kunal_singh/ThesisYesPrompt/outputs")#, help="with-prompts outputs dir")
parser.add_argument("--np", default="/scratch/kunal_singh/ThesisNoPrompt/outputs")#,    help="no-prompts outputs dir")
parser.add_argument("--out", default="Final_outputs/deep_analysis", help="output directory")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

WP_RUNS = f"{args.wp}/runs/all_runs.csv"
WP_EVAL = f"{args.wp}/evaluation/all_results.csv"
NP_RUNS = f"{args.np}/runs/all_runs.csv"
NP_EVAL = f"{args.np}/evaluation/all_results.csv"

for p in [WP_RUNS, WP_EVAL, NP_RUNS, NP_EVAL]:
    if not os.path.exists(p):
        print(f"ERROR: {p} not found"); sys.exit(1)

wp_runs = pd.read_csv(WP_RUNS)
wp_ev   = pd.read_csv(WP_EVAL)
np_runs = pd.read_csv(NP_RUNS)
np_ev   = pd.read_csv(NP_EVAL)

METRICS  = ["keyword_hit_rate", "rougeL_f1", "bertscore_f1"]
MLABELS  = ["KW-Hit", "ROUGE-L", "BERTScore"]
MODELS   = ["llama", "mixtral", "qwen", "sarvam"]
ALL_CONDS = ["C1","C2","C3","C4","C5","C6","C7","C8","C9"]
COND_LABELS = {
    "C1": "Full Doc — English",        "C2": "Full Doc — Sarvam MT",
    "C3": "No Context (Closed-Book)",  "C4": "Full Doc — Official Hindi",
    "C5": "Full Doc — IndicTrans2",    "C6": "RAG — English Chunks",
    "C7": "RAG — Sarvam MT Chunks",    "C8": "RAG — Official Hindi Chunks",
    "C9": "RAG — IndicTrans2 Chunks",
}

def hr(title): print(f"\n{'='*72}\n  {title}\n{'='*72}")
def save(df, name): df.to_csv(f"{args.out}/{name}.csv", index=False); return df


# ═══════════════════════════════════════════════════════════════════════════════
# A: ANSWER QUALITY AUDIT — what did each model actually produce?
# ═══════════════════════════════════════════════════════════════════════════════
hr("A: ANSWER QUALITY AUDIT")

def hindi_ratio(t):
    t = str(t) if t else ""
    hi = len(re.findall(r'[\u0900-\u097F]', t))
    return hi / len(t) if len(t) > 0 else 0

def is_empty(t):
    return not isinstance(t, str) or len(t.strip()) < 5

audit_rows = []
for exp_label, runs in [("with_prompts", wp_runs), ("no_prompts", np_runs)]:
    for model in MODELS:
        for cond in ALL_CONDS:
            sub = runs[(runs.model==model)&(runs.condition==cond)]
            if len(sub)==0: continue
            hi_r    = sub.answer_hi.apply(hindi_ratio).mean()
            empty   = sub.answer_hi.apply(is_empty).sum()
            think_o = sub.answer_hi.str.contains('<think>', na=False).sum()
            think_c = sub.answer_hi.str.contains('</think>', na=False).sum()
            audit_rows.append({
                "experiment": exp_label, "model": model, "condition": cond,
                "n": len(sub), "empty_pct": round(empty/len(sub)*100,1),
                "hindi_ratio": round(hi_r,2),
                "think_open": think_o, "think_closed": think_c,
                "completion_rate": round((think_c if think_o>0 else len(sub)-empty)/len(sub),2)
            })

df_audit = pd.DataFrame(audit_rows)
save(df_audit, "A_answer_quality_audit")
print(df_audit[df_audit.experiment=="with_prompts"].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# B: FULL SCORE TABLES — both experiments
# ═══════════════════════════════════════════════════════════════════════════════
hr("B: FULL SCORE TABLES")

for exp_label, ev in [("with_prompts", wp_ev), ("no_prompts", np_ev)]:
    t = ev.groupby(["model","condition"])[METRICS].mean().round(3).reset_index()
    t.columns = ["Model","Condition"] + MLABELS
    t["Condition_Desc"] = t["Condition"].map(COND_LABELS)
    save(t, f"B_scores_{exp_label}")
    print(f"\n--- {exp_label} ---")
    # Pivot BERTScore for compact view
    piv = t.pivot(index="Model", columns="Condition", values="BERTScore")
    piv = piv[[c for c in ALL_CONDS if c in piv.columns]]
    print(piv.round(3).to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# C: CONDITION EFFECTIVENESS — what helps most?
# ═══════════════════════════════════════════════════════════════════════════════
hr("C: CONDITION EFFECTIVENESS")

for exp_label, ev in [("with_prompts", wp_ev), ("no_prompts", np_ev)]:
    t = ev.groupby("condition")[METRICS].mean().round(3).reset_index()
    t["Description"] = t["condition"].map(COND_LABELS)
    t = t.sort_values("bertscore_f1", ascending=False)
    t.columns = ["Condition","KW-Hit","ROUGE-L","BERTScore","Description"]
    save(t, f"C_condition_effect_{exp_label}")
    print(f"\n--- {exp_label} ---")
    print(t.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# D: RAG DELTA — chunked retrieval vs full document
# ═══════════════════════════════════════════════════════════════════════════════
hr("D: RAG DELTA ANALYSIS (chunked retrieval vs full document)")

rag_pairs = [("C1","C6","English"), ("C2","C7","Sarvam MT"),
             ("C4","C8","Official Hindi"), ("C5","C9","IndicTrans2")]

for exp_label, ev in [("with_prompts", wp_ev), ("no_prompts", np_ev)]:
    rows = []
    for full_c, rag_c, doc_type in rag_pairs:
        for model in MODELS:
            full_s = ev[(ev.model==model)&(ev.condition==full_c)][METRICS].mean()
            rag_s  = ev[(ev.model==model)&(ev.condition==rag_c)][METRICS].mean()
            if full_s.isna().all(): continue
            delta = (rag_s - full_s).round(3)
            rows.append({"model":model, "doc_type":doc_type,
                         "full_cond":full_c, "rag_cond":rag_c,
                         **{f"delta_{m}":delta[m] for m in METRICS}})
    df_rag = pd.DataFrame(rows)
    save(df_rag, f"D_rag_delta_{exp_label}")
    print(f"\n--- {exp_label} ---")
    print(df_rag.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# E: SARVAM DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
hr("E: SARVAM DEEP ANALYSIS")

rows = []
for exp_label, runs, ev in [("with_prompts", wp_runs, wp_ev), 
                              ("no_prompts",  np_runs, np_ev)]:
    sarvam_r = runs[runs.model=='sarvam']
    sarvam_e = ev[ev.model=='sarvam']
    for cond in ALL_CONDS:
        sub_r = sarvam_r[sarvam_r.condition==cond]
        sub_e = sarvam_e[sarvam_e.condition==cond]
        if len(sub_r)==0: continue
        think_o = sub_r.answer_hi.str.contains('<think>', na=False).sum()
        think_c = sub_r.answer_hi.str.contains('</think>', na=False).sum()
        empty   = sub_r.answer_hi.apply(is_empty).sum()
        says_unavail = sub_r.answer_hi.str.contains(
            'जानकारी उपलब्ध नहीं|सूचना उपलब्ध नहीं', na=False).sum()
        rows.append({
            "experiment": exp_label, "condition": cond,
            "n": len(sub_r),
            "think_open": think_o, "think_closed": think_c,
            "empty": empty,
            "says_unavailable": says_unavail,
            "completion_rate": round((len(sub_r)-empty)/len(sub_r),2),
            **{m: sub_e[m].mean().round(3) if m in sub_e.columns else None
               for m in METRICS}
        })

df_sarvam = pd.DataFrame(rows)
save(df_sarvam, "E_sarvam_analysis")
print(df_sarvam.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# F: PROMPTING EFFECT — WP vs NP side by side
# ═══════════════════════════════════════════════════════════════════════════════
hr("F: PROMPTING EFFECT (With Prompts vs No Prompts)")

rows = []
for model in MODELS:
    for cond in ["C1","C2","C3","C6"]:
        wp_s = wp_ev[(wp_ev.model==model)&(wp_ev.condition==cond)][METRICS].mean()
        np_s = np_ev[(np_ev.model==model)&(np_ev.condition==cond)][METRICS].mean()
        for m, ml in zip(METRICS, MLABELS):
            rows.append({"model":model,"condition":cond,"metric":ml,
                         "with_prompts":round(wp_s[m],3),
                         "no_prompts":round(np_s[m],3),
                         "delta_np_minus_wp":round(np_s[m]-wp_s[m],3)})

df_prompt = pd.DataFrame(rows)
save(df_prompt, "F_prompting_effect")
# Show BERTScore only
print(df_prompt[df_prompt.metric=="BERTScore"].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# G: MINISTRY DIFFICULTY
# ═══════════════════════════════════════════════════════════════════════════════
hr("G: MINISTRY DIFFICULTY ANALYSIS")

for exp_label, ev in [("with_prompts", wp_ev), ("no_prompts", np_ev)]:
    t = ev[ev.condition.isin(["C1","C2","C3","C6"])].groupby(
        ["ministry","condition"])[METRICS].mean().round(3).unstack()
    save(t.reset_index(), f"G_ministry_{exp_label}")
    print(f"\n--- {exp_label} BERTScore by ministry ---")
    print(ev.groupby("ministry")[METRICS].mean().round(3).sort_values(
        "bertscore_f1").to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# H: TRANSLATION QUALITY IMPACT (C1 vs C2 vs C4 vs C5)
# ═══════════════════════════════════════════════════════════════════════════════
hr("H: TRANSLATION ENGINE IMPACT")

for exp_label, ev in [("with_prompts", wp_ev), ("no_prompts", np_ev)]:
    trans = ev[ev.condition.isin(["C1","C2","C4","C5"])]
    t = trans.groupby(["condition","model"])[METRICS].mean().round(3).unstack("model")
    # Compute delta vs C1
    means = trans.groupby("condition")[METRICS].mean().round(3)
    c1 = means.loc["C1"]
    means["BERTScore_delta_vs_C1"] = (means.bertscore_f1 - c1.bertscore_f1).round(3)
    means["KWHit_delta_vs_C1"]     = (means.keyword_hit_rate - c1.keyword_hit_rate).round(3)
    save(means.reset_index(), f"H_translation_impact_{exp_label}")
    print(f"\n--- {exp_label} ---")
    print(means.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# I: HALLUCINATION FLOOR (C3) — how much do models know without a document?
# ═══════════════════════════════════════════════════════════════════════════════
hr("I: HALLUCINATION FLOOR (C3 — no document)")

for exp_label, ev in [("with_prompts", wp_ev), ("no_prompts", np_ev)]:
    c1 = ev[ev.condition=="C1"].groupby("model")[METRICS].mean()
    c3 = ev[ev.condition=="C3"].groupby("model")[METRICS].mean()
    delta = (c3 - c1).round(3)
    delta.columns = [f"delta_{m}" for m in METRICS]
    combined = pd.concat([c1.round(3), c3.round(3), delta], axis=1)
    combined.columns = ["C1_KW","C1_RL","C1_BS","C3_KW","C3_RL","C3_BS",
                        "Δ_KW","Δ_RL","Δ_BS"]
    save(combined.reset_index(), f"I_hallucination_{exp_label}")
    print(f"\n--- {exp_label} ---")
    print(combined.to_string())

print(f"\n\n✓ All analysis tables saved to: {args.out}/")
print("Files produced:")
for f in sorted(os.listdir(args.out)):
    print(f"  {f}")
