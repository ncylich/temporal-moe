#!/usr/bin/env python3
"""Response-length analysis under decode-time residency: one graph per claim.

Reads ONLY committed data: genbench_samples/*.json (per-item lengths, raw-capture
think tokens, per-item scores) and instruct_genbench_vllm.csv (damage). Writes
figures/length_story.png. Claims and coverage limits are printed on the figure;
cells without per-item capture (MMLU, bespoke HumanEval) are absent, not zero.
"""
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
OUT = os.path.join(ABLATIONS, "figures")


def blob(rec, arm, task):
    p = f"{SAMP}/{rec}_{arm}_{task}.json"
    if not os.path.exists(p):
        return None, None
    b = json.load(open(p))
    return ({i["doc_id"]: i for i in b["items"]},
            {int(k): v for k, v in b.get("think_toks_by_doc", {}).items()})


def acc(i):
    for k in ("exact_match", "prompt_level_strict_acc", "pass@1"):
        if k in i:
            return bool(i[k])


# model, tight arm, relaxed arm, display name, text-thinker?
THINKERS = [("qwen35_instruct", "R8", "R32", "Qwen3.5", True),
            ("gemma4_think_on", "R8", "R16", "gemma4", True),
            ("lfm25_instruct", "R4", None, "LFM2.5", True),
            ("gptoss_120b_high", "R4", "R16", "120b\nhigh", False),
            ("gptoss_120b", "R4", "R16", "120b\nmed", False),
            ("gptoss_120b_low", "R4", "R16", "120b\nlow", False),
            ("gptoss_20b_high", "R4", None, "20b\nhigh", False),
            ("gptoss_20b", "R4", None, "20b\nmed", False),
            ("gptoss_20b_low", "R4", None, "20b\nlow", False)]
# low-effort modes think 24-60 tokens at the median -- % changes there are a few
# tokens of noise. Excluded from graphs 1 and 3; kept in the heatmap, flagged.
FLOOR = 100  # tokens, free-arm median
TWO = [("gsm8k_cot_zeroshot", "GSM8K"), ("ifeval", "IFEval")]

# damage per cell from live CSV (primary metric)
import csv
DMG = {}
MET = {"gsm8k_cot_zeroshot": "exact_match,flexible-extract",
       "ifeval": "prompt_level_strict_acc,none"}
VAL = {}
for r in csv.reader(open(os.path.join(ABLATIONS, "instruct_genbench_vllm.csv"))):
    if len(r) > 7 and not r[0].startswith("#") and r[0] != "model":
        if MET.get(r[5]) == r[6]:
            VAL[(r[0], r[3], r[5])] = float(r[7])
for rec, armk, _, nm, _ in THINKERS:
    for task, tnm in TWO:
        f, c = VAL.get((rec, "free", task)), VAL.get((rec, armk, task))
        if f is not None and c is not None:
            DMG[(rec, task)] = 100 * (c - f)

fig = plt.figure(figsize=(15.5, 16))
GRID = (4, 2)

# =========== GRAPH 1: thinking lengthens under the tight cache, reverts when
# relaxed (grouped bars per model, two task subpanels, P90 markers)
for ti, (task, tnm) in enumerate(TWO):
    ax = fig.add_subplot(*GRID, 1 + ti)
    keep = []
    for rec, armk, arm125, nm, text in THINKERS:
        _, tf = blob(rec, "free", task)
        if tf and np.median(list(tf.values())) >= FLOOR:
            keep.append((rec, armk, arm125, nm, text))
    xs = np.arange(len(keep))
    for i, (rec, armk, arm125, nm, text) in enumerate(keep):
        _, tf = blob(rec, "free", task)
        _, tk = blob(rec, armk, task)
        if not tf or not tk:
            continue
        vf, vk = np.array(list(tf.values())), np.array(list(tk.values()))
        med = 100 * (np.median(vk) / np.median(vf) - 1)
        p90 = 100 * (np.percentile(vk, 90) / np.percentile(vf, 90) - 1)
        ax.bar(i - 0.18, med, width=0.34, color="#b03434", edgecolor="black",
               lw=0.5, label="tight cache (R=k), median item" if i == 0 else None)
        ax.plot(i - 0.18, p90, "v", color="#5c0f0f", ms=7, zorder=4,
                label="tight cache, long items (P90)" if i == 0 else None)
        if arm125:
            _, t2 = blob(rec, arm125, task)
            v2 = np.array(list(t2.values()))
            m2 = 100 * (np.median(v2) / np.median(vf) - 1)
            p902 = 100 * (np.percentile(v2, 90) / np.percentile(vf, 90) - 1)
            ax.bar(i + 0.18, m2, width=0.34, color="#e8b6b6", edgecolor="black",
                   lw=0.5, label="relaxed cache (12.5%)" if i == 0 else None)
            ax.plot(i + 0.18, p902, "v", mfc="white", mec="#5c0f0f", ms=7,
                    zorder=4, label="relaxed cache, long items (P90)"
                    if i == 0 else None)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([t[3] for t in keep], fontsize=8)
    ax.set_ylabel("thinking length change vs free, %")
    ax.set_ylim(-22, 47)
    ax.set_title(f"1{'ab'[ti]}. {tnm}: " + (
        "Qwen/gemma/120b think LONGER under the tight cache and revert when it\n"
        "relaxes (LFM: mainly its long items); gpt-oss-20b SHORTENS. (modes with "
        "<100-token thinks excluded)" if ti == 0 else
        "Qwen/gemma/LFM lengthen and revert; 120b lengthens but does NOT\n"
        "revert (and loses nothing); 20b med/high shorten. (modes with <100-token "
        "thinks excluded)"), fontsize=9.5, loc="left")
    if ti == 0:
        ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25, axis="y")

# =========== GRAPH 2: flips blow up ONLY in thinking modes (two-group strip)
ax2 = fig.add_subplot(*GRID, 3)
ALLC = [("olmoe_instruct", "OLMoE", "R8", False),
        ("lfm25_instruct", "LFM2.5", "R4", True),
        ("qwen35_instruct", "Qwen on", "R8", True),
        ("qwen35_think_off", "Qwen off", "R8", False),
        ("gemma4_instruct", "gemma off", "R8", False),
        ("gemma4_think_on", "gemma on", "R8", True),
        ("gptoss_20b_low", "20b low", "R4", False),
        ("gptoss_20b", "20b med", "R4", False),
        ("gptoss_20b_high", "20b high", "R4", True),
        ("gptoss_120b_low", "120b low", "R4", False),
        ("gptoss_120b", "120b med", "R4", False),
        ("gptoss_120b_high", "120b high", "R4", True)]
TASKS3 = TWO + [("humaneval_instruct", "HumanEval")]
rng = np.random.default_rng(7)
labeled = []
for rec, nm, arm, think in ALLC:
    for task, tnm in TASKS3:
        fr, _ = blob(rec, "free", task)
        cn, _ = blob(rec, arm, task)
        if not fr or not cn:
            continue
        cm = sorted(set(fr) & set(cn))
        a0 = np.array([acc(fr[d]) for d in cm], float)
        a1 = np.array([acc(cn[d]) for d in cm], float)
        dl = np.array([cn[d]["gen_toks"] - fr[d]["gen_toks"] for d in cm], float)
        for dirn, base_x in (((a0 == 1) & (a1 == 0), 1 if think else 0),
                             ((a0 == 0) & (a1 == 1), 3 if think else 2.55)):
            if dirn.sum() < 5:
                continue
            x = base_x + rng.uniform(-0.13, 0.13)
            v = dl[dirn].mean()
            ax2.plot(x, v, "o", ms=8, color="#c23b3b" if think else "#4878b0",
                     mec="black", mew=0.5, alpha=0.9)
            if abs(v) > 1500:
                labeled.append((x, v, f"{nm} {tnm}"))
ax2.set_xlim(-0.6, 3.7)
ax2.set_xticks([0, 1, 2.55, 3])
ax2.set_xticklabels(["non-thinking\nflip to WRONG", "thinking\nflip to WRONG",
                     "non-thinking\nflip to RIGHT", "thinking\nflip to RIGHT"],
                    fontsize=8)
ax2.axvline(1.85, color="grey", lw=0.8, ls=":")
ax2.set_yscale("symlog", linthresh=300)
ax2.set_ylim(-7000, 7000)
ax2.set_yticks([-5000, -1000, -300, 0, 300, 1000, 5000])
ax2.set_yticklabels(["-5k", "-1k", "-300", "0", "+300", "+1k", "+5k"])
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("mean length change of items that\nflipped correct -> wrong, tokens")
ax2.set_title("2. Blow-up and wrongness travel together, in BOTH directions, only "
              "in thinking modes:\nitems flipping to WRONG grew (up to +5k); items "
              "flipping to RIGHT are free-arm blow-ups\nthe constrained run "
              "avoided (-2k to -4k). Non-thinking flips barely move (length-"
              "silent).", fontsize=9.5, loc="left")
ax2.grid(alpha=0.25, axis="y")

# =========== GRAPH 2b: flip-flow accounting (where net damage comes from)
axF = fig.add_subplot(*GRID, 4)
CAPV = {"gsm8k_cot_zeroshot": 4096, "ifeval": 8192}
FCELLS = [("qwen35_instruct", "R8", "Qwen3.5", "ifeval"),
          ("gemma4_think_on", "R8", "gemma4", "ifeval"),
          ("gptoss_120b_high", "R4", "120b high", "ifeval"),
          ("gptoss_20b_high", "R4", "20b high", "ifeval"),
          ("qwen35_instruct", "R8", "Qwen3.5", "gsm8k_cot_zeroshot"),
          ("gemma4_think_on", "R8", "gemma4", "gsm8k_cot_zeroshot"),
          ("lfm25_instruct", "R4", "LFM2.5", "gsm8k_cot_zeroshot"),
          ("gptoss_120b_high", "R4", "120b high", "gsm8k_cot_zeroshot")]
for i, (rec, arm, nm, task) in enumerate(FCELLS):
    fr, _ = blob(rec, "free", task)
    cn, _ = blob(rec, arm, task)
    cm = sorted(set(fr) & set(cn))
    cap = CAPV[task]
    f2w = [d for d in cm if acc(fr[d]) and not acc(cn[d])]
    f2r = [d for d in cm if not acc(fr[d]) and acc(cn[d])]
    blow_w = sum(cn[d]["gen_toks"] >= cap - 8 or
                 cn[d]["gen_toks"] > 2 * fr[d]["gen_toks"] for d in f2w)
    blow_r = sum(fr[d]["gen_toks"] >= cap - 8 or
                 fr[d]["gen_toks"] > 2 * cn[d]["gen_toks"] for d in f2r)
    axF.bar(i, len(f2w), width=0.62, color="#e8b6b6", edgecolor="black", lw=0.5,
            label="items flipping to WRONG" if i == 0 else None)
    axF.bar(i, blow_w, width=0.62, color="#b03434", edgecolor="black", lw=0.5,
            label="...of which blew up (at cap or >2x)" if i == 0 else None)
    axF.bar(i, -len(f2r), width=0.62, color="#c3d6ea", edgecolor="black", lw=0.5,
            label="items flipping to RIGHT" if i == 0 else None)
    axF.bar(i, -blow_r, width=0.62, color="#2f5f8f", edgecolor="black", lw=0.5,
            label="...that were free-arm blow-ups" if i == 0 else None)
    net = len(f2w) - len(f2r)
    axF.annotate(f"{net:+d}", (i, len(f2w) + 0.8), ha="center", fontsize=8,
                 fontweight="bold")
axF.axhline(0, color="black", lw=0.9)
axF.set_xticks(range(len(FCELLS)))
axF.set_xticklabels([f"{c[2]}\n{'IFEval' if c[3] == 'ifeval' else 'GSM8K'}"
                     for c in FCELLS], fontsize=7.5)
axF.set_ylabel("items flipping, of 200  (up = to wrong, down = to right)")
axF.set_ylim(-16, 41)
axF.text(0.01, 0.97, "bold number = net items lost", transform=axF.transAxes,
         fontsize=7.5, va="top")
axF.set_title("2b. Net damage = one-way traffic into blow-up. Qwen/gemma IFEval: "
              "many wrong-flips, mostly\nblow-ups, few rescues. gpt-oss high: "
              "balanced two-way churn -> tiny net. GSM8K flips are mostly\n"
              "normal-length (dark share small): that damage is quality loss, not "
              "the budget wall.", fontsize=9.5, loc="left")
axF.legend(fontsize=7, loc="lower right")
axF.grid(alpha=0.25, axis="y")

# =========== GRAPH 3: lengthen vs damage quadrants (absorb vs break vs opt out)
ax3 = fig.add_subplot(*GRID, 5)
MCOL = {"Qwen3.5": "#e07b39", "gemma4": "#4878b0", "LFM2.5": "#8a5fbf",
        "120b\nhigh": "#3f8f5f", "120b\nmed": "#6fae8a", "120b\nlow": "#a8cbb8",
        "20b\nhigh": "#666666", "20b\nmed": "#8f8f8f", "20b\nlow": "#b8b8b8"}
seen_m = set()
for rec, armk, _, nm, text in THINKERS:
    for task, tnm in TWO:
        _, tf = blob(rec, "free", task)
        _, tk = blob(rec, armk, task)
        if not tf or not tk or (rec, task) not in DMG:
            continue
        if np.median(list(tf.values())) < FLOOR:
            continue
        med = 100 * (np.median(list(tk.values())) / np.median(list(tf.values())) - 1)
        d = DMG[(rec, task)]
        pretty = nm.replace(chr(10), " ")
        ax3.plot(med, d, "o" if task.startswith("g") else "s", ms=9,
                 color=MCOL[nm], mec="black", mew=0.5,
                 label=pretty if nm not in seen_m else None)
        seen_m.add(nm)
ax3.axhline(0, color="black", lw=0.8)
ax3.axvline(0, color="black", lw=0.8)
ax3.axhspan(-3, 3, color="grey", alpha=0.12)
ax3.text(0.98, 0.97, "shaded: typical single-cell noise (±1 SE ~ 3 pts)",
         fontsize=6.5, color="grey", transform=ax3.transAxes, ha="right",
         va="top")
ax3.legend(fontsize=7, loc="lower left", ncol=2)
ax3.set_xlabel("median thinking length change under tight cache, %")
ax3.set_ylabel("score damage under tight cache, points")
ax3.set_title("3. Three responses to the constraint: gpt-oss-120b (green) thinks "
              "longer and KEEPS its score;\nQwen/gemma/LFM think longer and LOSE "
              "points (their tail hits the cap); gpt-oss-20b (grey)\nSHORTENS "
              "thinking, damage small-to-marginal. Circle = GSM8K, square = IFEval.",
              fontsize=9.5, loc="left")
ax3.grid(alpha=0.25)

# =========== GRAPH 4: rigor heatmap (unchanged from v5, compressed)
ax4 = fig.add_subplot(4, 1, 4)
PCTS = [25, 50, 75, 90, 95]
rows, labels = [], []
for rec, armk, _, nm, text in THINKERS:
    for task, tnm in TWO:
        _, tf = blob(rec, "free", task)
        _, tk = blob(rec, armk, task)
        if not tf or not tk:
            continue
        vf = np.array(list(tf.values()), float)
        vk = np.array(list(tk.values()), float)
        rows.append([np.percentile(vk, p) / max(np.percentile(vf, p), 1)
                     for p in PCTS])
        labels.append(f"{nm.replace(chr(10), ' ')} · {tnm} "
                      f"({np.median(vf):.0f}t)")
im = ax4.imshow(np.array(rows).T, cmap="RdBu_r", vmin=0.6, vmax=1.4, aspect="auto")
for j, r in enumerate(rows):
    for i, v in enumerate(r):
        ax4.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                 color="white" if abs(v - 1) > 0.25 else "black")
ax4.set_yticks(range(len(PCTS)))
ax4.set_yticklabels(["P25", "median", "P75", "P90", "P95"], fontsize=8)
ax4.set_xticks(range(len(labels)))
ax4.set_xticklabels(labels, fontsize=6.8, rotation=35, ha="right")
ax4.set_title("4. Rigor detail behind graphs 1-3: think-length ratio (tight/free) "
              "at each percentile, all 18 thinking cells. Red = longer. "
              "(label: free-arm median tokens; ratios on tiny thinks are noisy)",
              fontsize=9.5, loc="left")
fig.colorbar(im, ax=ax4, shrink=0.8, pad=0.01)

fig.suptitle("How the residency constraint changes response length -- per-item "
             "paired data, all measurable cells (MMLU + bespoke HumanEval have no "
             "per-item capture)", fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(f"{OUT}/length_story.png", dpi=130)
print("wrote length_story.png")
