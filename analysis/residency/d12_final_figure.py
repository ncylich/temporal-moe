#!/usr/bin/env python3
"""D12 final figure: per-dataset deltas vs BASE-FREE (authoritative instrument).
Bars: base R8 (what the constraint costs the unadapted model), D12 free, D12 R8.
MMLU = multi-run means (base 2-run, D12 3-run free / 3-run R8)."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AB = "/workspace/temporal-moe/results/ablations"
A = {}
for r in csv.reader(open(f"{AB}/instruct_genbench_vllm.csv")):
    if len(r) > 7 and r[0] in ("gemma4_instruct", "gemma4_ce_d12"):
        met = {"exact_match,flexible-extract": "GSM8K", "prompt_level_strict_acc,none": "IFEval",
               "pass@1,channel-aware": "HumanEval"}.get(r[6])
        if met: A[(r[0], r[3], met)] = 100 * float(r[7])
    if len(r) > 7 and r[6] == "acc,relaxed-extract" and r[0].startswith("gemma4_ce_d12_dual"):
        A.setdefault(("d12mmlu", r[3]), []).append(100 * float(r[7]))
S = {}
for r in csv.reader(open(f"{AB}/screening_genbench.csv")):
    if len(r) > 7 and r[6] == "acc,relaxed-extract":
        if r[0] in ("dual_base", "pair_base"): S.setdefault(r[3], []).append(100 * float(r[7]))
        if r[0] == "scr_d12_dual": A.setdefault(("d12mmlu", r[3]), []).append(100 * float(r[7]))
mean = lambda xs: sum(xs) / len(xs)

DATASETS = ["GSM8K", "IFEval", "HumanEval", "MMLU"]
def val(model, arm, ds):
    if ds == "MMLU":
        return mean(A[("d12mmlu", arm)]) if model == "gemma4_ce_d12" else mean(S[arm])
    return A[(model, arm, ds)]

basefree = {ds: val("gemma4_instruct", "free", ds) for ds in DATASETS}
series = [("base under R8 constraint", "gemma4_instruct", "R8", "#b0b0b0"),
          ("D12 adapted, free",        "gemma4_ce_d12",   "free", "#7fb3d5"),
          ("D12 adapted, under R8",    "gemma4_ce_d12",   "R8", "#1f618d")]

fig, ax = plt.subplots(figsize=(9.5, 5.2))
W = 0.26
for i, (label, model, arm, color) in enumerate(series):
    xs = [j + (i - 1) * W for j in range(len(DATASETS))]
    ys = [val(model, arm, ds) - basefree[ds] for ds in DATASETS]
    bars = ax.bar(xs, ys, W, label=label, color=color, edgecolor="black", linewidth=0.4)
    for b, y in zip(bars, ys):
        ax.annotate(f"{y:+.1f}", (b.get_x() + b.get_width() / 2, y),
                    ha="center", va="bottom" if y >= 0 else "top", fontsize=9,
                    xytext=(0, 2 if y >= 0 else -2), textcoords="offset points")
ax.axhline(0, color="black", linewidth=1)
ax.text(len(DATASETS) - 0.42, 0.15, "unconstrained base = 0", fontsize=8.5,
        ha="right", color="#333333")
ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels([f"{ds}\n(base free {basefree[ds]:.1f})" for ds in DATASETS])
ax.set_ylabel("accuracy delta vs unconstrained base (pts)")
ax.set_title("D12 adaptation vs the R8 rolling-residency constraint\n"
             "(authoritative 200-item instrument; MMLU = multi-run means; screening noise ±2 pts)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
ax.spines[["top", "right"]].set_visible(False)
ax.margins(y=0.12)
plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d12_final_v1.png"
plt.savefig(out, dpi=160)
print("wrote", out)
for ds in DATASETS:
    print(ds, {lbl: round(val(m, a, ds) - basefree[ds], 1) for lbl, m, a, _ in series})
