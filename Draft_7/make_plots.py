#!/usr/bin/env python3
"""
make_plots.py
=============
Generates all figures for the thesis.
Run from project root:  python make_plots.py

Produces 8 figures in outputs/plots/
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Config ─────────────────────────────────────────────────────────────────────
WP_EVAL = "/scratch/kunal_singh/ThesisYesPrompt/outputs/evaluation/all_results.csv"
NP_EVAL = "/scratch/kunal_singh/ThesisNoPrompt/outputs/evaluation/all_results.csv"
OUT_DIR = "Final_outputs/plots"

# Adjust paths if running from a different directory
for p in [WP_EVAL, NP_EVAL]:
    if not os.path.exists(p):
        print(f"ERROR: {p} not found. Adjust WP_EVAL / NP_EVAL paths at top of script.")
        sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

wp = pd.read_csv(WP_EVAL)
np_ev = pd.read_csv(NP_EVAL)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

MODEL_COLORS = {
    "llama":   "#2196F3",   # blue
    "mixtral": "#4CAF50",   # green
    "qwen":    "#FF9800",   # orange
    "sarvam":  "#F44336",   # red  — Indian model, stands out
}
MODEL_LABELS = {
    "llama": "Llama-3.3-70B", "mixtral": "Mixtral-8x7B",
    "qwen":  "Qwen-2.5-72B",  "sarvam": "Sarvam-30B",
}
COND_LABELS = {
    "C1":"Eng\n(Full)", "C2":"Sarvam\nMT(Full)", "C3":"No\nDoc",
    "C4":"Official\nHindi(Full)", "C5":"IndicTrans2\n(Full)",
    "C6":"RAG\nEng", "C7":"RAG\nSarvam", "C8":"RAG\nOfficial", "C9":"RAG\nIndicT2",
}
MODELS = ["llama","mixtral","qwen","sarvam"]
ALL_CONDS = ["C1","C2","C3","C4","C5","C6","C7","C8","C9"]


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Grouped bar: BERTScore by Model × Condition (With Prompts)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
x      = np.arange(len(ALL_CONDS))
width  = 0.2
for i, model in enumerate(MODELS):
    scores = [
        wp[(wp.model==model)&(wp.condition==c)].bertscore_f1.mean()
        for c in ALL_CONDS
    ]
    bars = ax.bar(x + i*width, scores, width, label=MODEL_LABELS[model],
                  color=MODEL_COLORS[model], alpha=0.85, edgecolor="white")

ax.set_xticks(x + width*1.5)
ax.set_xticklabels([COND_LABELS.get(c, c) for c in ALL_CONDS], fontsize=9)
ax.set_xlabel("Document Condition")
ax.set_ylabel("BERTScore F1")
ax.set_title("Fig 1: BERTScore by Model and Document Condition  (With Instruction Prompts)")
ax.set_ylim(0.75, 0.93)
ax.legend(loc="lower left", ncol=4, framealpha=0.9)
ax.axvspan(5.5, 8.8, alpha=0.06, color="green", label="_RAG region")
ax.text(7.1, 0.755, "← RAG conditions →", ha="center", fontsize=9, color="green")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_bertscore_model_condition.png", bbox_inches="tight")
plt.close()
print("Fig 1 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Sarvam vs Others: all 3 metrics side by side (C1 only, WP)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
metrics = ["keyword_hit_rate","rougeL_f1","bertscore_f1"]
mlabels = ["Keyword Hit Rate","ROUGE-L F1","BERTScore F1"]

for ax, metric, mlabel in zip(axes, metrics, mlabels):
    vals  = [wp[(wp.model==m)&(wp.condition=="C1")][metric].mean() for m in MODELS]
    colors= [MODEL_COLORS[m] for m in MODELS]
    bars  = ax.bar([MODEL_LABELS[m] for m in MODELS], vals, color=colors, 
                   alpha=0.85, edgecolor="white", width=0.5)
    ax.set_ylabel(mlabel)
    ax.set_title(f"C1 — {mlabel}")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals)*1.2)

plt.suptitle("Fig 2: Sarvam-30B vs Multilingual Models — C1 (English Full Doc, With Prompts)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_sarvam_vs_others_c1.png", bbox_inches="tight")
plt.close()
print("Fig 2 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Heatmap: BERTScore Model × Condition (With Prompts)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4))
matrix = wp.groupby(["model","condition"])["bertscore_f1"].mean().unstack()
matrix = matrix.reindex(index=MODELS, columns=ALL_CONDS)

sns.heatmap(
    matrix, annot=True, fmt=".3f", cmap="RdYlGn",
    vmin=0.79, vmax=0.91, ax=ax, linewidths=0.5,
    xticklabels=[COND_LABELS.get(c,c) for c in ALL_CONDS],
    yticklabels=[MODEL_LABELS[m] for m in MODELS],
    cbar_kws={"label": "BERTScore F1"}
)
ax.set_title("Fig 3: BERTScore Heatmap — All Models × All Conditions (With Prompts)")
ax.set_xlabel("Condition")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_bertscore_heatmap.png", bbox_inches="tight")
plt.close()
print("Fig 3 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Translation engine comparison: C1 vs C2 vs C4 vs C5
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
trans_conds = ["C1","C2","C4","C5"]
trans_labels = ["English\n(Original)","Sarvam MT\n(C2)","Official\nHindi (C4)","IndicTrans2\n(C5)"]
trans_colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]

for ax, metric, mlabel in zip(axes, metrics, mlabels):
    vals = [wp[wp.condition==c][metric].mean() for c in trans_conds]
    bars = ax.bar(trans_labels, vals, color=trans_colors, alpha=0.85, edgecolor="white", width=0.5)
    ax.set_title(f"{mlabel}")
    ax.set_ylabel(mlabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=10)

plt.suptitle("Fig 4: Translation Engine Impact on QA Quality (With Prompts, averaged across models)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_translation_engine_comparison.png", bbox_inches="tight")
plt.close()
print("Fig 4 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — RAG delta: BERTScore change (C6-C1) per model, WP vs NP
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(MODELS))
width = 0.3

wp_deltas = [
    wp[(wp.model==m)&(wp.condition=="C6")].bertscore_f1.mean() -
    wp[(wp.model==m)&(wp.condition=="C1")].bertscore_f1.mean()
    for m in MODELS
]
np_deltas = [
    np_ev[(np_ev.model==m)&(np_ev.condition=="C6")].bertscore_f1.mean() -
    np_ev[(np_ev.model==m)&(np_ev.condition=="C1")].bertscore_f1.mean()
    for m in MODELS
]

b1 = ax.bar(x-width/2, wp_deltas, width, label="With Instruction Prompts",
            color=["#2196F3","#4CAF50","#FF9800","#F44336"], alpha=0.85, edgecolor="white")
b2 = ax.bar(x+width/2, np_deltas, width, label="Without Instruction Prompts",
            color=["#2196F3","#4CAF50","#FF9800","#F44336"], alpha=0.4,
            edgecolor="black", linewidth=0.5, hatch="///")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
ax.set_ylabel("BERTScore Delta (RAG − Full Doc)")
ax.set_title("Fig 5: RAG Improvement — BERTScore Δ(C6−C1) With vs Without Prompts")
ax.legend()
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,
                h + (0.001 if h >= 0 else -0.003),
                f"{h:+.3f}", ha="center", va="bottom" if h>=0 else "top", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_rag_delta_wp_vs_np.png", bbox_inches="tight")
plt.close()
print("Fig 5 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Sarvam anatomy: <think> completion rate by condition
# ═══════════════════════════════════════════════════════════════════════════════
wp_runs = pd.read_csv(WP_EVAL.replace("evaluation/all_results","runs/all_runs"))
sarvam = wp_runs[wp_runs.model=="sarvam"].copy()
sarvam["think_open"]   = sarvam.answer_hi.str.contains("<think>",  na=False)
sarvam["think_closed"] = sarvam.answer_hi.str.contains("</think>", na=False)
sarvam["empty"]        = sarvam.answer_hi.apply(lambda t: not isinstance(t,str) or len(str(t).strip())<5)

cond_stats = sarvam.groupby("condition").agg(
    n         = ("answer_hi","count"),
    think_pct = ("think_open", lambda x: x.mean()*100),
    closed_pct= ("think_closed",lambda x: x.mean()*100),
    empty_pct = ("empty",      lambda x: x.mean()*100),
).reset_index()

fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(cond_stats))
ax.bar(x, cond_stats.think_pct,   label="Has <think> block",  color="#FF9800", alpha=0.8)
ax.bar(x, cond_stats.closed_pct,  label="Think block closed",  color="#4CAF50", alpha=0.9)
ax.bar(x, cond_stats.empty_pct,   label="Empty response",      color="#F44336", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([COND_LABELS.get(c,c) for c in cond_stats.condition], fontsize=9)
ax.set_ylabel("Percentage of responses (%)")
ax.set_title("Fig 6: Sarvam-30B Response Anatomy — <think> Block Completion Rate by Condition")
ax.legend(loc="upper right")
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig6_sarvam_think_anatomy.png", bbox_inches="tight")
plt.close()
print("Fig 6 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Ministry difficulty heatmap
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (ev, label) in zip(axes, [(wp,"With Prompts"), (np_ev,"No Prompts")]):
    mat = ev.groupby(["ministry","condition"])["bertscore_f1"].mean().unstack()
    key_conds = [c for c in ["C1","C2","C3","C6"] if c in mat.columns]
    mat = mat[key_conds].reindex(["ayush","education","labour","women"])
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="YlOrRd_r",
                ax=ax, linewidths=0.5,
                xticklabels=key_conds,
                cbar_kws={"label":"BERTScore"})
    ax.set_title(f"{label}")
    ax.set_ylabel("Ministry")

plt.suptitle("Fig 7: BERTScore by Ministry and Condition (C1/C2/C3/C6)", 
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig7_ministry_heatmap.png", bbox_inches="tight")
plt.close()
print("Fig 7 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Hallucination floor: C1 vs C3 per model
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (ev, label) in zip(axes, [(wp,"With Prompts"), (np_ev,"No Prompts")]):
    x = np.arange(len(MODELS))
    w = 0.3
    c1_bs = [ev[(ev.model==m)&(ev.condition=="C1")].bertscore_f1.mean() for m in MODELS]
    c3_bs = [ev[(ev.model==m)&(ev.condition=="C3")].bertscore_f1.mean() for m in MODELS]
    b1 = ax.bar(x-w/2, c1_bs, w, label="C1 (Full Doc)", alpha=0.85,
                color=[MODEL_COLORS[m] for m in MODELS], edgecolor="white")
    b2 = ax.bar(x+w/2, c3_bs, w, label="C3 (No Doc)",   alpha=0.5,
                color=[MODEL_COLORS[m] for m in MODELS], edgecolor="black", hatch="xxx")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9, rotation=10)
    ax.set_ylabel("BERTScore F1")
    ax.set_title(f"{label}")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    # Annotate delta
    for i, (c1, c3) in enumerate(zip(c1_bs, c3_bs)):
        ax.annotate(f"Δ={c1-c3:+.3f}", xy=(i, min(c1,c3)-0.005),
                    ha="center", fontsize=8, color="gray")

plt.suptitle("Fig 8: Hallucination Floor — C1 (With Document) vs C3 (No Document)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig8_hallucination_floor.png", bbox_inches="tight")
plt.close()
print("Fig 8 saved")

print(f"\n✓ All 8 figures saved to {OUT_DIR}/")
