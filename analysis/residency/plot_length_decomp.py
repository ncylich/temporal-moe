#!/usr/bin/env python3
"""How much of the constrained lengthening is blow-up traffic? Per cell, the total mean
per-item generation-length change (constrained minus free, paired doc ids) is overlaid
with the part contributed by items at the generation cap in either arm (cap-hit =
gen_toks >= cap-8). The gap between the bars is broad lengthening (or shortening) among
items that never hit the cap. Reads genbench_samples/*.json only.
Also writes the flip-direction strip (mean length change of items whose correctness
flipped, thinking vs non-thinking modes, both directions), figures/length_flips.png.
Writes figures/length_blowup_decomp.png; --no-caption writes the paper variants."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

SAMP = os.path.join(ABLATIONS, "genbench_samples")
FIG = os.path.join(ABLATIONS, "figures")
PAPER = "--no-caption" in sys.argv
if PAPER:
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 10.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 10})

CAP = {"gsm8k_cot_zeroshot": 4096, "ifeval": 8192}
CAPOFF = {"gsm8k_cot_zeroshot": 2048, "ifeval": 2048}   # think-off budgets
CELLS = [("qwen35_instruct", "R8", "Qwen on", True),
         ("gemma4_think_on", "R8", "gemma on", True),
         ("lfm25_instruct", "R4", "LFM", True),
         ("gptoss_120b_high", "R4", "120b high", True),
         ("gptoss_20b_high", "R4", "20b high", True),
         ("qwen35_think_off", "R8", "Qwen off", False),
         ("gemma4_instruct", "R8", "gemma off", False)]
TASKS = [("gsm8k_cot_zeroshot", "GSM8K"), ("ifeval", "IFEval")]


def blob(rec, arm, task):
    p = f"{SAMP}/{rec}_{arm}_{task}.json"
    if not os.path.exists(p):
        return None
    return {i["doc_id"]: i for i in json.load(open(p))["items"]}


cells = []
for rec, arm, nm, think in CELLS:
    for task, tnm in TASKS:
        fr, cn = blob(rec, "free", task), blob(rec, arm, task)
        if not fr or not cn:
            continue
        cm = sorted(set(fr) & set(cn))
        cap = (CAP if think else CAPOFF)[task]
        tot = cap_part = 0.0
        for d in cm:
            dl = cn[d]["gen_toks"] - fr[d]["gen_toks"]
            tot += dl
            if fr[d]["gen_toks"] >= cap - 8 or cn[d]["gen_toks"] >= cap - 8:
                cap_part += dl
        cells.append((f"{nm}\n{tnm}", think, tot / len(cm), cap_part / len(cm)))

x = np.array([c[2] for c in cells])
y = np.array([c[3] for c in cells])
b = np.polyfit(x, y, 1)
r2 = 1 - np.sum((y - np.polyval(b, x)) ** 2) / np.sum((y - y.mean()) ** 2)

fig, ax = plt.subplots(figsize=(12.5, 4.8) if PAPER else (12.5, 5.2))
for i, (lab, think, tot, capd) in enumerate(cells):
    ax.bar(i, tot, width=0.7, color="#c3d6ea", edgecolor="black", lw=0.5,
           label="total mean length change" if i == 0 else None)
    ax.bar(i, capd, width=0.42, color="#b03434", edgecolor="black", lw=0.5,
           label="part from items at the cap\nin either arm" if i == 0 else None)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylim(min(0, x.min(), y.min()) * 1.25, max(x.max(), y.max()) * 1.18)
sep = max(i for i, c in enumerate(cells) if c[1]) + 0.5
ax.axvline(sep, color="grey", lw=0.8, ls=":")
ax.text((sep - 0.3) / len(cells), 0.9, "thinking on / high", ha="right",
        fontsize=10 if PAPER else 8.5, color="grey", transform=ax.transAxes)
ax.text((sep + 0.3) / len(cells), 0.9, "thinking off", ha="left",
        fontsize=10 if PAPER else 8.5, color="grey", transform=ax.transAxes)
ax.set_xticks(range(len(cells)))
ax.set_xticklabels([c[0] for c in cells], fontsize=9.5 if PAPER else 8, rotation=28,
                   ha="right")
ax.set_ylabel("mean tokens per item,\nconstrained − free")
ax.annotate(f"cap part vs total, across cells:\nslope {b[0]:.2f}, $R^2$ = {r2:.2f}",
            (0.985, 0.965), xycoords="axes fraction", ha="right", va="top",
            fontsize=10.5 if PAPER else 9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.6"))
if not PAPER:
    ax.set_title("Blow-up traffic explains part of the lengthening, not all of it: "
                 "gemma/LFM-GSM8K are cap-dominated,\nQwen-on IFEval is half broad "
                 "lengthening, 120b-high is entirely broad, 20b-high shortens its "
                 "normal items", fontsize=9.5)
ax.legend(fontsize=9.5 if PAPER else 8, loc="upper left", framealpha=0.95)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
out = f"{FIG}/length_blowup_decomp{'_nocaption' if PAPER else ''}.png"
fig.savefig(out, dpi=170)
print(f"wrote {out}")



# ---- wide scatter: cap-carried change vs total change, one point per configuration ----
figs, axs = plt.subplots(figsize=(8.8, 3.4) if PAPER else (8.8, 3.8))
for lab, think, tot, capd in cells:
    axs.scatter(tot, capd, s=52, color="#d1605e" if think else "#4878b0",
                edgecolor="black", lw=0.6, zorder=3)
lo_, hi_ = min(x.min(), y.min()) - 40, max(x.max(), y.max()) + 40
axs.plot([lo_, hi_], [lo_, hi_], "k--", lw=0.9, alpha=0.6)
axs.axhline(0, color="grey", lw=0.6, alpha=0.6)
axs.axvline(0, color="grey", lw=0.6, alpha=0.6)
ANNOT = {"Qwen on\nIFEval": (8, -4), "gemma on\nIFEval": (8, -4),
         "20b high\nIFEval": (8, 4), "120b high\nIFEval": (10, 8),
         "Qwen on\nGSM8K": (8, -12), "gemma on\nGSM8K": (8, 6)}
for lab, think, tot, capd in cells:
    if lab in ANNOT:
        dx, dy = ANNOT[lab]
        axs.annotate(lab.replace("\n", " "), (tot, capd), textcoords="offset points",
                     xytext=(dx, dy), fontsize=9.5 if PAPER else 8,
                     ha="left" if dx > 0 else "right")
axs.scatter([], [], s=52, color="#d1605e", edgecolor="black", lw=0.6,
            label="thinking on / high effort")
axs.scatter([], [], s=52, color="#4878b0", edgecolor="black", lw=0.6,
            label="thinking off / low effort")
axs.plot([], [], "k--", lw=0.9, label="fully cap-carried (y = x)")
axs.legend(loc="upper left", fontsize=10 if PAPER else 8.5, framealpha=0.95)
axs.set_xlim(lo_, hi_)
axs.set_ylim(min(y.min(), 0) - 60, max(y.max(), 0) + 60)
axs.set_xlabel("total mean length change, tokens per item (constrained − free)")
axs.set_ylabel("part from items at the cap\nunder either setting")
axs.annotate(f"slope {b[0]:.2f}, $R^2$ = {r2:.2f}", (0.985, 0.06),
             xycoords="axes fraction", ha="right", va="bottom",
             fontsize=10.5 if PAPER else 9,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6"))
if not PAPER:
    axs.set_title("Cap-carried vs total length change: on the diagonal the cap traffic "
                  "IS the change; below it, broad lengthening among never-capped items",
                  fontsize=9.5)
figs.tight_layout()
outs = f"{FIG}/length_blowup_scatter{'_nocaption' if PAPER else ''}.png"
figs.savefig(outs, dpi=170)
print(f"wrote {outs}")

# ---- flip-direction strip: wrongness rides the length extremes in both directions ----
def acc(i):
    for k in ("exact_match", "prompt_level_strict_acc", "pass@1"):
        if k in i:
            return bool(i[k])

FLIPC = CELLS + [("gptoss_120b_low", "R4", "120b low", False),
                 ("gptoss_20b_low", "R4", "20b low", False),
                 ("olmoe_instruct", "R8", "OLMoE", False)]
rng = np.random.default_rng(7)
fig2, ax2 = plt.subplots(figsize=(6.2, 4.4) if PAPER else (7.6, 5))
for rec, arm, nm, think in FLIPC:
    for task, tnm in TASKS:
        fr, cn = blob(rec, "free", task), blob(rec, arm, task)
        if not fr or not cn:
            continue
        cm = sorted(set(fr) & set(cn))
        a0 = np.array([acc(fr[d]) for d in cm], float)
        a1 = np.array([acc(cn[d]) for d in cm], float)
        dl = np.array([cn[d]["gen_toks"] - fr[d]["gen_toks"] for d in cm], float)
        for sel, base_x in (((a0 == 1) & (a1 == 0), 1 if think else 0),
                            ((a0 == 0) & (a1 == 1), 3.35 if think else 2.5)):
            if sel.sum() < 5:
                continue
            ax2.plot(base_x + rng.uniform(-0.13, 0.13), dl[sel].mean(), "o", ms=8,
                     color="#c23b3b" if think else "#4878b0", mec="black", mew=0.5,
                     alpha=0.9)
ax2.set_xlim(-0.6, 4.05)
ax2.set_xticks([0, 1, 2.5, 3.35])
ax2.set_xticklabels(["non-thinking\nflip to WRONG", "thinking\nflip to WRONG",
                     "non-thinking\nflip to RIGHT", "thinking\nflip to RIGHT"],
                    fontsize=10.5 if PAPER else 8)
ax2.axvline(1.85, color="grey", lw=0.8, ls=":")
ax2.set_yscale("symlog", linthresh=300)
ax2.set_ylim(-7000, 7000)
ax2.set_yticks([-5000, -1000, -300, 0, 300, 1000, 5000])
ax2.set_yticklabels(["-5k", "-1k", "-300", "0", "+300", "+1k", "+5k"])
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("mean length change of flipped items,\ntokens (constrained − free)")
if not PAPER:
    ax2.set_title("Items that flip to wrong mostly grew; flips to right were "
                  "free-arm blow-ups\nthe constrained run avoided; non-thinking flips "
                  "are length-silent", fontsize=9.5)
ax2.grid(alpha=0.25, axis="y")
fig2.tight_layout()
out2 = f"{FIG}/length_flips{'_nocaption' if PAPER else ''}.png"
fig2.savefig(out2, dpi=170)
print(f"wrote {out2}")


# ---- percentile heatmap: think-token ratio (tight/free) at each percentile, all
# thinking cells, the rigor detail behind the decomposition figure ----
import numpy as np  # noqa: E811  (kept local to this block for clarity)


def think_toks(rec, arm, task):
    p = f"{SAMP}/{rec}_{arm}_{task}.json"
    if not os.path.exists(p):
        return None
    import json as _json
    b = _json.load(open(p))
    d = {int(k): v for k, v in b.get("think_toks_by_doc", {}).items()}
    return np.array(list(d.values()), float) if d else None


HCELLS = [("qwen35_instruct", "R8", "Qwen on"), ("gemma4_think_on", "R8", "gemma on"),
          ("lfm25_instruct", "R4", "LFM"), ("gptoss_120b_low", "R4", "120b low"),
          ("gptoss_120b", "R4", "120b med"), ("gptoss_120b_high", "R4", "120b high"),
          ("gptoss_20b_low", "R4", "20b low"), ("gptoss_20b", "R4", "20b med"),
          ("gptoss_20b_high", "R4", "20b high")]
PCTS = [25, 50, 75, 90, 95]
rows, labels = [], []
for rec, arm, nm in HCELLS:
    for task, tnm in TASKS:
        vf, vk = think_toks(rec, "free", task), think_toks(rec, arm, task)
        if vf is None or vk is None:
            continue
        rows.append([np.percentile(vk, q) / max(np.percentile(vf, q), 1) for q in PCTS])
        labels.append(f"{nm} · {tnm} ({np.median(vf):.0f}t)")
fig3, ax3 = plt.subplots(figsize=(12.5, 3.4) if PAPER else (12.5, 3.8))
im = ax3.imshow(np.array(rows).T, cmap="RdBu_r", vmin=0.6, vmax=1.4, aspect="auto")
for j, r in enumerate(rows):
    for i, v in enumerate(r):
        ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=8 if PAPER else 6.8,
                 color="white" if abs(v - 1) > 0.25 else "black")
ax3.set_yticks(range(len(PCTS)))
ax3.set_yticklabels(["P25", "median", "P75", "P90", "P95"],
                    fontsize=10 if PAPER else 8)
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, fontsize=8.5 if PAPER else 6.8, rotation=35, ha="right")
if not PAPER:
    ax3.set_title("Think-length ratio (tight cache / free) at each percentile, all "
                  "thinking cells (label: free-arm median tokens; ratios on tiny "
                  "thinks are noisy)", fontsize=9.5)
fig3.colorbar(im, ax=ax3, shrink=0.85, pad=0.01)
fig3.tight_layout()
out3 = f"{FIG}/length_percentiles{'_nocaption' if PAPER else ''}.png"
fig3.savefig(out3, dpi=170)
print(f"wrote {out3}")
