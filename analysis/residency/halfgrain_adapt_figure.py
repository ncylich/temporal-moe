#!/usr/bin/env python3
"""Half-grain adaptation before/after figure. Two panels (gemma R48, qwen R96,
both 18.75% resident, 1 half-swap/token, thinking off): per task, base free /
adapted free / base constrained / adapted constrained, from
instruct_genbench_vllm.csv. MMLU omitted: strict extraction is a documented
format artifact for both (adapted qwen relaxed free = 0.930, capability
intact; see screening_genbench.csv mmlu_gptoss_relaxed rows)."""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/workspace/temporal-moe/results/ablations/instruct_genbench_vllm.csv"
MET = {"ifeval": "prompt_level_strict_acc,none",
       "gsm8k_cot_zeroshot": "exact_match,flexible-extract",
       "humaneval_instruct": "pass@1,create_test",
       "humaneval_gemma_fixed": "pass@1,channel-aware"}

V = {}
fh = open(CSV); fh.readline()
for r in csv.DictReader(fh):
    if r["metric"] == MET.get(r["task"]):
        V[(r["model"], r["arm"], r["task"])] = float(r["value"])  # last row wins

PANELS = [
    ("gemma4 26B half-grain, R48", "gemma4_halfgrain", "gemma4_halfgrain_ce", "R48",
     [("GSM8K", "gsm8k_cot_zeroshot"), ("IFEval", "ifeval"),
      ("HumanEval", "humaneval_gemma_fixed"),
      ("HumanEval @3k cap", "humaneval_gemma_fixed")]),
    ("Qwen3.5 35B half-grain, R96", "qwen35_halfgrain_off", "qwen35_halfgrain_ce_off", "R96",
     [("GSM8K", "gsm8k_cot_zeroshot"), ("IFEval", "ifeval"),
      ("HumanEval", "humaneval_instruct")]),
]
BARS = [("base free", "base", "free", "#b8bfc6"),
        ("adapted free", "ce", "free", "#9dc3e6"),
        ("base constrained", "base", "R", "#5a6570"),
        ("adapted constrained", "ce", "R", "#1e618d")]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [4, 3]})
for ax, (title, base, ce, R, tasks) in zip(axes, PANELS):
    for ti, (lab, task) in enumerate(tasks):
        cap3k = lab.endswith("@3k cap")
        for bi, (blab, who, arm, color) in enumerate(BARS):
            model = base if who == "base" else ce
            a = arm if arm == "free" else R
            if cap3k:
                if arm == "free":
                    continue  # cap-3k rerun exists for the constrained arm only
                model = {"gemma4_halfgrain": "gemma4_halfgrain_cap3k",
                         "gemma4_halfgrain_ce": "gemma4_halfgrain_ce_cap3k"}[model]
            v = V.get((model, a, task))
            if v is None:
                continue
            x = ti + (bi - 1.5) * 0.19
            ax.bar(x, v, 0.19, color=color, edgecolor="black", linewidth=0.4,
                   label=blab if ti == 0 else None)
            ax.annotate(f"{v:.2f}", (x, v + 0.012), ha="center", fontsize=7.5)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[0] for t in tasks], fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("score (accuracy / pass@1, higher better)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4,
           loc="upper center", bbox_to_anchor=(0.5, 0.97))
fig.suptitle("Half-grain adaptation, before vs after (18.75% resident, 1 half-swap/token, thinking off)",
             fontsize=12, y=1.03)
fig.text(0.99, 0.01,
         "MMLU omitted: strict answer extraction is a format artifact for both models "
         "(adapted qwen relaxed extraction = 0.93 free, capability intact). "
         "HumanEval @3k cap: constrained arms rerun with a 3072-token budget "
         "(standard 1536 cap truncates the constrained model's longer generations).",
         ha="right", fontsize=7, color="#666666")
plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/temporal-moe/results/ablations/figures/halfgrain_adapt.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
