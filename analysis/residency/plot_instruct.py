#!/usr/bin/env python3
"""Instruct-program figures: self-CE damage vs residency fraction, per-benchmark damage at
matched fraction, and per-model mean damage, both damage figures carrying the thinking axis
(blue = thinking off / low effort, red = on / high effort) and WritingBench (critic-point
deltas x10 onto the 100-point accuracy axis; WritingBench ran think-off, so its dots attach
to the off-mode bars, LFM's to its native mode). Reads instruct_selfce.csv,
instruct_genbench_vllm.csv, screening_genbench.csv, think_ablation_summary.csv and
writingbench/cell_stats.csv.
Writes results/ablations/figures/{instruct_selfce_damage,instruct_bench_damage,
instruct_model_damage}.png; --no-caption writes the paper variants (short labels, no titles).
Model order everywhere: total parameters ascending. Bar means use the four accuracy
benchmarks (uniform basis across modes and models); WritingBench appears as dots only.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

FIG = os.path.join(ABLATIONS, "figures")
PAPER = "--no-caption" in sys.argv
if PAPER:                       # paper mode: larger in-figure text survives column downscaling
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5, "legend.fontsize": 10})

MODE_COL = {"off": "#4878b0", "on": "#d1605e"}


def _save(fig, name):
    fig.savefig(f"{FIG}/{name}{'_nocaption' if PAPER else ''}.png", dpi=170)
    print(f"wrote {name}{'_nocaption' if PAPER else ''}.png")


# total-parameter ascending order, used for every figure in this file
NAMES = {"olmoe_instruct": "OLMoE-Instruct (64 experts)",
         "lfm25_instruct": "LFM2.5-A1B (32 experts)",
         "gemma4_instruct": "gemma4-26B-IT (128 experts)",
         "qwen35_instruct": "Qwen3.5-35B (256 experts)"}
SIZE_ORDER = ["olmoe_instruct", "lfm25_instruct", "gemma4_instruct", "qwen35_instruct"]


def selfce():
    rows = [r for r in csv.reader(open(f"{ABLATIONS}/instruct_selfce.csv"))
            if r and r[0] in NAMES]
    per = {}
    for r in rows:
        per.setdefault(r[0], {})[r[3]] = float(r[6])
    fig, ax = plt.subplots(figsize=(5.8, 4.2) if PAPER else (7, 5))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, m in enumerate([m for m in SIZE_ORDER if m in per]):
        d = per[m]
        E = int([r[1] for r in rows if r[0] == m][0])
        k = int([r[2] for r in rows if r[0] == m][0])
        free = d["free"]
        pts = sorted((100 * int(a[1:]) / E, d[a] - free) for a in d
                     if a.startswith("R") and not a.endswith("cold"))
        x, y = zip(*pts)
        ax.plot(x, y, "o-", color=cols[i], lw=2, ms=9,
                label=f"{NAMES[m]}, k={k}")
        ax.annotate("R=k", (x[0], y[0]), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=cols[i])
    ax.set_xscale("log")
    ax.set_xticks([3.125, 6.25, 12.5])
    ax.set_xticklabels(["3.1%", "6.25%", "12.5%"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("resident experts, % of total experts")
    if PAPER:
        ax.set_ylabel("self-CE damage, nats/token")
        ax.legend(fontsize=10.5)
    else:
        ax.set_ylabel("self-CE damage, nats/token (constrained − free)")
        ax.set_title("Instruct models: damage on their own responses\n"
                     "(prefill free, rule on generated tokens; lower is better)", fontsize=10)
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "instruct_selfce_damage")


def load_damage():
    """One accessor for every (model, mode-role, arm, surface) damage cell, in points."""
    # genbench values (OLMoE, the one model outside the thinking summary)
    METRIC = {"gsm8k_cot_zeroshot": ("exact_match,flexible-extract",),
              "ifeval": ("prompt_level_strict_acc,none",),
              "mmlu_flan_cot_fewshot": ("exact_match,get-answer",)}
    vals = {}
    for r in csv.reader(open(f"{ABLATIONS}/instruct_genbench_vllm.csv")):
        if len(r) < 10 or r[0].startswith("#") or r[0] == "model":
            continue
        rec, arm, task, met = r[0], r[3], r[5], r[6]
        ok = met in METRIC.get(task, ()) or \
            (task.startswith("humaneval") and met.startswith("pass@1"))
        if ok:
            vals[(rec, arm, task)] = float(r[7])
    # thinking-ablation damage cells, already in points (5 models, both modes where a
    # toggle exists), MMLU relaxed where a dual re-score exists
    tcells = {}
    for r in csv.reader(open(f"{ABLATIONS}/think_ablation_summary.csv")):
        if len(r) > 7 and r[0] != "model" and not r[0].startswith("#"):
            tcells[(r[0], r[1], r[2], r[3])] = float(r[6])
    wb = {r[0]: float(r[1]) for r in csv.reader(
        open(f"{ABLATIONS}/writingbench/cell_stats.csv")) if r and r[0] != "cell"}

    # (display, [(mode label, role, source)], (R=k arm, 12.5% arm), WB cells, WB role)
    # source: ("think", model key) uses tcells[(key, mode label, arm, task)];
    #         ("vals", record, humaneval task, mmlu task) uses genbench accuracies.
    SPEC = [
        ("OLMoE-Instruct 7B", [("none", "off",
          ("vals", "olmoe_instruct", "humaneval_instruct", "mmlu_flan_cot_fewshot"))],
         ("R8", "R8"), None, None),
        ("LFM2.5-8B-A1B", [("on", "on", ("think", "LFM2.5-A1B"))],
         ("R4", "R4"), ("lfm25_free", "lfm25_R4", "lfm25_R4"), "on"),
        ("gpt-oss-20b", [("low", "off", ("think", "gpt-oss-20b")),
                         ("high", "on", ("think", "gpt-oss-20b"))],
         ("R4", "R4"), ("oss20_free", "oss20_R4", "oss20_R4"), "off"),
        ("gemma4-26B-IT", [("off", "off", ("think", "gemma4-26B-IT")),
                           ("on", "on", ("think", "gemma4-26B-IT"))],
         ("R8", "R16"), ("gemma4_base_free", "gemma4_base_R8", "gemma4_base_R16"), "off"),
        ("Qwen3.5-35B", [("off", "off", ("think", "Qwen3.5-35B")),
                         ("on", "on", ("think", "Qwen3.5-35B"))],
         ("R8", "R32"), ("qwen35_base_free", "qwen35_base_R8", "qwen35_base_R32"), "off"),
        ("gpt-oss-120b", [("low", "off", ("think", "gpt-oss-120b")),
                          ("high", "on", ("think", "gpt-oss-120b"))],
         ("R4", "R16"), ("oss120_free", "oss120_R4", "oss120_R16"), "off"),
    ]
    TASKS = ["GSM8K", "IFEval", "HumanEval", "MMLU"]

    def delta(spec, mode, arm_idx, bench):
        name, modes, arms, wbcells, wbrole = spec
        label, role, src = mode
        if bench == "WB":
            if wbcells is None or role != wbrole:
                return None
            fr, cn = wb.get(wbcells[0]), wb.get(wbcells[1 + arm_idx])
            return None if fr is None or cn is None else 10 * (cn - fr)
        arm = arms[arm_idx]
        if src[0] == "think":
            return tcells.get((src[1], label, arm, bench))
        rec, he, mm = src[1], src[2], src[3]
        task = {"GSM8K": "gsm8k_cot_zeroshot", "IFEval": "ifeval",
                "HumanEval": he, "MMLU": mm}[bench]
        fr, cn = vals.get((rec, "free", task)), vals.get((rec, arm, task))
        return None if fr is None or cn is None else 100 * (cn - fr)

    return SPEC, TASKS, delta


def bench():
    SPEC, TASKS, delta = load_damage()
    DCOL = dict(zip(TASKS + ["WB"], plt.cm.Set2(np.linspace(0, 0.75, 5))))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))

    # ---- per-benchmark figure: color = model, hue overlay = mode via hatch, alpha = arm ----
    groups = TASKS + ["WB"]
    glabel = {"WB": "WritingBench\n(critic pts x10)"}
    fig, ax = plt.subplots(figsize=(12.5, 5) if not PAPER else (12.5, 4.6))
    w = 0.075
    GX = 1.9
    for j, b in enumerate(groups):
        pos = 0.0
        for i, spec in enumerate(SPEC):
            for mode in spec[1]:
                for arm_idx in (0, 1):
                    d = delta(spec, mode, arm_idx, b)
                    if d is None:
                        if b == "MMLU" and spec[0].startswith("LFM") and arm_idx == 0:
                            ax.annotate("floor", (j * GX - 0.75 + pos, 0.3), fontsize=7,
                                        rotation=90, ha="center", color="grey")
                        continue
                    x = j * GX - 0.75 + pos
                    pos += w
                    if abs(d) < 0.05:
                        ax.annotate("0", (x, 0.35), fontsize=7, ha="center",
                                    color=cols[i], fontweight="bold")
                    ax.bar(x, d, width=w * 0.94, color=cols[i],
                           alpha=1.0 if arm_idx == 0 else 0.55,
                           hatch="///" if mode[1] == "on" else None,
                           edgecolor="black", lw=0.4)
            pos += w * 0.5
    for j in range(len(groups) - 1):
        ax.axvline((j + 0.55) * GX - 0.11, color="grey", lw=0.6, alpha=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([j * GX for j in range(len(groups))])
    ax.set_xticklabels([glabel.get(b, b) for b in groups])
    ax.set_ylabel("damage under residency, points")
    handles = [Patch(facecolor=cols[i], edgecolor="black", label=s[0])
               for i, s in enumerate(SPEC)]
    handles += [Patch(facecolor="0.85", edgecolor="black", hatch="///",
                      label="hatched: thinking on / high"),
                Patch(facecolor="0.25", edgecolor="black", label="dark: R = k"),
                Patch(facecolor="0.85", edgecolor="black", label="light: R = 12.5%")]
    ax.legend(handles=handles, fontsize=8.5 if PAPER else 7.5, ncol=5,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    if not PAPER:
        ax.set_title("Benchmarks under decode-time residency, both thinking modes; "
                     "WritingBench critic deltas x10, think-off runs\n"
                     "(constrained − free, same items and stack per pair; single runs, "
                     "binomial SE 2-4 pts; OLMoE has no thinking mode or WritingBench cell)",
                     fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    _save(fig, "instruct_bench_damage")

    # ---- per-model figure, think_tax style: mode color, arm alpha, dots per surface ----
    figm, axm = plt.subplots(figsize=(11.8, 4.1) if PAPER else (11, 5))
    w = 0.19
    means = {}
    for i, spec in enumerate(SPEC):
        nbars = 2 * len(spec[1])
        for s, mode in enumerate(spec[1]):
            for arm_idx in (0, 1):
                ds = [delta(spec, mode, arm_idx, b) for b in TASKS]
                got = [d for d in ds if d is not None]
                if not got:
                    continue
                m = sum(got) / len(got)
                se = (sum((d - m) ** 2 for d in got) / max(1, len(got) - 1)) ** 0.5 \
                    / len(got) ** 0.5
                means[(spec[0], mode[0], arm_idx)] = round(m, 1)
                x = i + (2 * s + arm_idx - (nbars - 1) / 2) * w
                axm.bar(x, m, width=w * 0.9, yerr=se, capsize=3, color=MODE_COL[mode[1]],
                        alpha=1.0 if arm_idx == 0 else 0.5, edgecolor="black", lw=0.5,
                        zorder=2)
                for b, d in zip(TASKS, ds):
                    if d is not None:
                        axm.scatter(x, d, s=34, color=DCOL[b], edgecolor="black", lw=0.5,
                                    zorder=3)
                dwb = delta(spec, mode, arm_idx, "WB")
                if dwb is not None:
                    axm.scatter(x, dwb, s=80, marker="*", color=DCOL["WB"],
                                edgecolor="black", lw=0.5, zorder=3)
    for b in TASKS:
        axm.scatter([], [], s=42, color=DCOL[b], edgecolor="black", lw=0.5, label=b)
    axm.scatter([], [], s=85, marker="*", color=DCOL["WB"], edgecolor="black", lw=0.5,
                label="WritingBench (x10)")
    axm.axhline(0, color="black", lw=0.8)
    axm.set_xticks(range(len(SPEC)))
    _r = lambda a: a.replace("R", "R=")
    axm.set_xticklabels(
        [s[0] + ("\n(k: %s = 12.5%%)" % _r(s[2][0]) if s[2][0] == s[2][1] else
                 "\n(k: %s, 12.5%%: %s)" % (_r(s[2][0]), _r(s[2][1]))) for s in SPEC],
        fontsize=9 if PAPER else 8.5)
    axm.set_ylabel("damage under residency, points")
    h, l = axm.get_legend_handles_labels()
    h += [Patch(facecolor=MODE_COL["off"], edgecolor="black",
                label="thinking off / low effort"),
          Patch(facecolor=MODE_COL["on"], edgecolor="black",
                label="thinking on / high effort"),
          Patch(facecolor="0.25", edgecolor="black", label="dark: R = k"),
          Patch(facecolor="0.85", edgecolor="black", label="light: R = 12.5%")]
    axm.legend(handles=h, fontsize=9 if PAPER else 8, ncol=3, loc="lower right")
    if not PAPER:
        axm.set_title("Damage per model and thinking mode: bar = mean over the four "
                      "accuracy benchmarks (whisker = SE of mean from the surface spread)\n"
                      "dots = per-surface deltas, star = WritingBench x10 (think-off runs, "
                      "dots only, not in the mean); OLMoE has neither", fontsize=9)
    axm.grid(alpha=0.25, axis="y")
    figm.tight_layout()
    _save(figm, "instruct_model_damage")
    for k, v in sorted(means.items()):
        print("mean", k, v)




def combined_row():
    """Paper variant: per-model and per-benchmark damage side by side in ONE row.
    Right panel restyled like the left: mode-colored mean bars (over models) with
    per-model dots, instead of 20 per-model bars per group."""
    SPEC, TASKS, delta = load_damage()
    DCOL = dict(zip(TASKS + ["WB"], plt.cm.Set2(np.linspace(0, 0.75, 5))))
    MCOL = plt.cm.tab10(np.linspace(0, 1, 10))
    SHORT = ["OLMoE", "LFM2.5", "20b", "gemma4", "Qwen3.5", "120b"]
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 15.5,
                         "xtick.labelsize": 13.5, "ytick.labelsize": 13,
                         "legend.fontsize": 10.5})
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14.2, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1],
                                                "wspace": 0.14})
    # ---- left: per model ----
    w = 0.19
    for i, spec in enumerate(SPEC):
        nbars = 2 * len(spec[1])
        for s, mode in enumerate(spec[1]):
            for arm_idx in (0, 1):
                ds = [delta(spec, mode, arm_idx, b) for b in TASKS]
                got = [d for d in ds if d is not None]
                if not got:
                    continue
                m = sum(got) / len(got)
                se = (sum((d - m) ** 2 for d in got) / max(1, len(got) - 1)) ** 0.5 \
                    / len(got) ** 0.5
                x = i + (2 * s + arm_idx - (nbars - 1) / 2) * w
                axl.bar(x, m, width=w * 0.9, yerr=se, capsize=2,
                        color=MODE_COL[mode[1]], alpha=1.0 if arm_idx == 0 else 0.5,
                        edgecolor="black", lw=0.5, zorder=2)
                for b, d in zip(TASKS, ds):
                    if d is not None:
                        axl.scatter(x, d, s=26, color=DCOL[b], edgecolor="black",
                                    lw=0.4, zorder=3)
                dwb = delta(spec, mode, arm_idx, "WB")
                if dwb is not None:
                    axl.scatter(x, dwb, s=62, marker="*", color=DCOL["WB"],
                                edgecolor="black", lw=0.4, zorder=3)
    for b in TASKS:
        axl.scatter([], [], s=34, color=DCOL[b], edgecolor="black", lw=0.4, label=b)
    axl.scatter([], [], s=66, marker="*", color=DCOL["WB"], edgecolor="black",
                lw=0.4, label="WritingBench (x10)")
    axl.axhline(0, color="black", lw=0.8)
    axl.set_xticks(range(len(SPEC)))
    axl.set_xticklabels(SHORT)
    axl.set_ylabel("damage under residency, points")
    axl.legend(ncol=2, loc="lower right", framealpha=0.95)
    axl.grid(alpha=0.25, axis="y")
    # ---- right: per benchmark, mode-mean bars + per-model dots ----
    groups = TASKS + ["WB"]
    glabel = {"WB": "WritingB.\n(x10)"}
    w = 0.19
    for j, b in enumerate(groups):
        combos = [("off", 0), ("off", 1), ("on", 0), ("on", 1)]
        for c, (role, arm_idx) in enumerate(combos):
            vals, mods = [], []
            for i, spec in enumerate(SPEC):
                for mode in spec[1]:
                    if mode[1] != role:
                        continue
                    d = delta(spec, mode, arm_idx, b)
                    if d is not None:
                        vals.append(d)
                        mods.append(i)
            if not vals:
                continue
            m = sum(vals) / len(vals)
            se = (sum((d - m) ** 2 for d in vals) / max(1, len(vals) - 1)) ** 0.5 \
                / len(vals) ** 0.5
            x = j + (c - 1.5) * w
            axr.bar(x, m, width=w * 0.9, yerr=se, capsize=2, color=MODE_COL[role],
                    alpha=1.0 if arm_idx == 0 else 0.5, edgecolor="black", lw=0.5,
                    zorder=2)
            for d, i in zip(vals, mods):
                axr.scatter(x, d, s=26, color=MCOL[i], edgecolor="black", lw=0.4,
                            zorder=3)
    for i, nm in enumerate(SHORT):
        axr.scatter([], [], s=34, color=MCOL[i], edgecolor="black", lw=0.4, label=nm)
    from matplotlib.patches import Patch as _P
    axr.legend(handles=axr.get_legend_handles_labels()[0] +
               [_P(facecolor=MODE_COL["off"], edgecolor="black", label="think off/low"),
                _P(facecolor=MODE_COL["on"], edgecolor="black", label="think on/high"),
                _P(facecolor="0.25", edgecolor="black", label="dark: R = k"),
                _P(facecolor="0.85", edgecolor="black", label="light: R = 12.5%")],
               ncol=2, loc="upper right", framealpha=0.95, fontsize=9.5)
    axr.axhline(0, color="black", lw=0.8)
    axr.set_xticks(range(len(groups)))
    axr.set_xticklabels([glabel.get(b, b) for b in groups])
    axr.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    _save(fig, "instruct_damage_row")


if __name__ == "__main__":
    selfce()
    bench()
    combined_row()
