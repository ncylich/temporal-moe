#!/usr/bin/env python3
"""Thinking trade-off under decode-time residency: mean accuracy loss (free routing minus
constrained, points, over the five benchmarks) with thinking off / low effort vs on / high
effort, for the four released models that have both modes. Same damage cells as Fig. 7
(analysis/residency/plot_instruct.py: load_damage). Writes paper/figures/thinking_tradeoff.pdf."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
src = open(os.path.join(HERE, "plot_instruct.py")).read().split("\nif __name__")[0]
g = {"__file__": os.path.join(HERE, "plot_instruct.py"), "__name__": "plot_instruct"}
exec(compile(src, "plot_instruct", "exec"), g)
SPEC, TASKS, delta = g["load_damage"]()
NAMES = {"gpt-oss-20b": "GPT-OSS-20B", "gemma4-26B-IT": "Gemma4", "Qwen3.5-35B": "Qwen3.5", "gpt-oss-120b": "GPT-OSS-120B"}
COL = {"GPT-OSS-20B": "#2ca02c", "Gemma4": "#d62728", "Qwen3.5": "#9467bd", "GPT-OSS-120B": "#8c564b"}
def mean_loss(spec, mode, arm):
    d = [delta(spec, mode, arm, b) for b in TASKS]; d = [x for x in d if x is not None]
    return -np.mean(d)
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
PANELS = [(0, "only the active experts resident ($R=k$)"), (1, "12.5% of the experts resident")]
rows = []
for spec in SPEC:
    name = NAMES.get(spec[0])
    if name is None or len(spec[1]) < 2: continue
    rows.append((name, spec))
for ax, (arm, title) in zip(axes, PANELS):
    left, right = [], []
    for name, spec in rows:
        off, on = spec[1][0], spec[1][1]
        y = [mean_loss(spec, off, arm), mean_loss(spec, on, arm)]
        ax.plot([0, 1], y, "-o", color=COL[name], lw=2)
        left.append([y[0], f"{y[0]:.1f}", COL[name]]); right.append([y[1], f"{y[1]:.1f}  {name}", COL[name]])
    for side, x, dx, ha in ((left, 0, -0.06, "right"), (right, 1, 0.06, "left")):
        side.sort(key=lambda r: r[0])
        for k in range(1, len(side)):          # spread labels that would overlap (data units)
            if side[k][0] - side[k - 1][0] < 0.32: side[k][0] = side[k - 1][0] + 0.32
        for yl, txt, col in side:
            ax.text(x + dx, yl, txt, ha=ha, va="center", fontsize=7.5, color=col)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["thinking off /\nlow effort", "thinking on /\nhigh effort"])
    ax.set_xlim(-0.3, 1.75); ax.grid(axis="y", ls=":", alpha=0.4)
axes[0].set_ylabel("mean accuracy loss vs.\nfree routing (points)")
axes[0].invert_yaxis()
fig.tight_layout()
for d in (os.path.join(REPO, "results/ablations/figures"), os.path.join(REPO, "paper/figures")):
    fig.savefig(os.path.join(d, "thinking_tradeoff.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(d, "thinking_tradeoff.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote thinking_tradeoff")
