#!/usr/bin/env python3
"""Thinking-ablation analysis (INSTRUCT_ANALYSIS_PLAN.md analyses 1-3).

Reads instruct_genbench_vllm.csv (last row wins per cell) and genbench_samples dumps
(doc_id-deduped). Emits:
  1. damage x thinking-mode table (constrained - free within each mode)
  2. think-length shift (mean think tokens, free vs constrained arms)
  3. backtracks per 1k think tokens (dilution vs error-reaction)
plus figures think_damage.png / think_length_shift.png and
results/ablations/think_ablation_summary.csv. Partial data prints partial tables.
"""
import csv
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

# record-name per mode; "default" marks the grid's deployment-default side
PAIRS = {
    "gemma4-26B-IT": {"modes": {"off": "gemma4_instruct", "on": "gemma4_think_on"},
                      "default": "off", "arms": ["R8", "R16"]},
    "Qwen3.5-35B": {"modes": {"on": "qwen35_instruct", "off": "qwen35_think_off"},
                    "default": "on", "arms": ["R8", "R32"]},
    "gpt-oss-20b": {"modes": {"low": "gptoss_20b_low", "medium": "gptoss_20b",
                              "high": "gptoss_20b_high"},
                    "default": "medium", "arms": ["R4"]},
    "gpt-oss-120b": {"modes": {"low": "gptoss_120b_low", "medium": "gptoss_120b",
                               "high": "gptoss_120b_high"},
                     "default": "medium", "arms": ["R4", "R16"]},
    "LFM2.5-A1B": {"modes": {"on": "lfm25_instruct"},        # no toggle: lengths only
                   "default": "on", "arms": ["R4"]},
}
# per-record task->metric map; channel-native variants override per family
TASKS = {"GSM8K": ("gsm8k_cot_zeroshot", "exact_match,flexible-extract"),
         "IFEval": ("ifeval", "prompt_level_strict_acc,none"),
         "HumanEval": ("humaneval_instruct", "pass@1,create_test"),
         "MMLU": ("mmlu_flan_cot_fewshot", "exact_match,get-answer")}
OVERRIDES = {"gemma4": {"HumanEval": ("humaneval_gemma_fixed", "pass@1,channel-aware")},
             "lfm": {"HumanEval": ("humaneval_think", "pass@1,channel-aware")},
             "qwen35": {"HumanEval": ("humaneval_think", "pass@1,channel-aware")},
             "gptoss": {"HumanEval": ("humaneval_gptoss", "pass@1,channel-aware"),
                        "MMLU": ("mmlu_gptoss_relaxed", "acc,relaxed-extract")}}


def task_map(record):
    fam = "gemma4" if record.startswith("gemma4") else \
          "gptoss" if record.startswith("gptoss") else \
          "lfm" if record.startswith("lfm") else \
          "qwen35" if record.startswith("qwen35") and "think_off" not in record else None
    m = dict(TASKS)
    m.update(OVERRIDES.get(fam, {}))
    return m


def load_cells():
    cells = {}                                    # (record, arm, task, metric) -> value
    for r in csv.reader(open(f"{ABLATIONS}/instruct_genbench_vllm.csv")):
        if len(r) > 7 and not r[0].startswith(("#", "smoke", "model")):
            try:
                cells[(r[0], r[3], r[5], r[6])] = float(r[7])
            except ValueError:
                pass
    return cells


def load_lengths(record, arm, task):
    p = os.path.join(SAMP, f"{record}_{arm}_{task}.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    if isinstance(b, list):                      # pre-2026-08-12 dump format
        b = {"items": b}
    seen, rows = set(), []
    for i in b.get("items", []):
        if i["doc_id"] in seen:
            continue
        seen.add(i["doc_id"])
        rows.append(i)
    rows = [i for i in rows if "gen_toks" in i]  # pre-capture-era dumps lack lengths
    if not rows:
        return None
    at = b.get("analysis_toks") or []
    out = {"n": len(rows),
           "gen": np.mean([i["gen_toks"] for i in rows]),
           "think": np.mean(at) if at else np.mean([i.get("think_toks", 0)
                                                    for i in rows]),
           "backtracks": np.mean([i.get("backtracks", 0) for i in rows])}
    return out


def main():
    cells = load_cells()
    sm = open(os.path.join(ABLATIONS, "think_ablation_summary.csv"), "w", newline="")
    w = csv.writer(sm)
    sm.write('"# Thinking ablation summary: damage (constrained-free) and lengths per '
             'model/mode/arm/task. Producer: analysis/residency/think_analysis.py"\n')
    w.writerow(["model", "mode", "arm", "task", "free", "constrained", "damage",
                "gen_free", "gen_con", "think_free", "think_con",
                "backtracks_per_1k_free", "backtracks_per_1k_con"])

    print("=== 1. damage x thinking mode (points, constrained - free) ===")
    dam = {}
    for model, cfg in PAIRS.items():
        for mode, rec in cfg["modes"].items():
            tm = task_map(rec)
            for tname, (task, metric) in tm.items():
                free = cells.get((rec, "free", task, metric))
                for arm in cfg["arms"]:
                    con = cells.get((rec, arm, task, metric))
                    if free is None or con is None:
                        continue
                    d = 100 * (con - free)
                    dam[(model, mode, arm, tname)] = d
                    lf = load_lengths(rec, "free", task) or {}
                    lc = load_lengths(rec, arm, task) or {}
                    bf = 1000 * lf["backtracks"] / lf["think"] \
                        if lf.get("think") else ""
                    bc = 1000 * lc["backtracks"] / lc["think"] \
                        if lc.get("think") else ""
                    w.writerow([model, mode, arm, tname, f"{free:.4f}", f"{con:.4f}",
                                f"{d:+.1f}",
                                f"{lf.get('gen', ''):.0f}" if lf else "",
                                f"{lc.get('gen', ''):.0f}" if lc else "",
                                f"{lf.get('think', ''):.0f}" if lf else "",
                                f"{lc.get('think', ''):.0f}" if lc else "",
                                f"{bf:.1f}" if bf != "" else "",
                                f"{bc:.1f}" if bc != "" else ""])
                    print(f"  {model:14s} {mode:6s} {arm:4s} {tname:9s} "
                          f"{free:.3f} -> {con:.3f}  ({d:+.1f})")
    sm.close()

    # figure 1: damage by mode
    models = [m for m in PAIRS if any(k[0] == m for k in dam)]
    if models:
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4.4),
                                 squeeze=False)
        for ax, model in zip(axes[0], models):
            cfg = PAIRS[model]
            modes = [md for md in cfg["modes"] if any(
                k[:2] == (model, md) for k in dam)]
            tnames = sorted({k[3] for k in dam if k[0] == model})
            width = 0.8 / max(1, len(modes))
            for j, md in enumerate(modes):
                xs, ys = [], []
                for i, t in enumerate(tnames):
                    arm = cfg["arms"][0]
                    v = dam.get((model, md, arm, t))
                    if v is None:
                        continue
                    xs.append(i + j * width)
                    ys.append(v)
                ax.bar(xs, ys, width=width, label=f"think {md}"
                       if not model.startswith("gpt-oss") else f"effort {md}")
            ax.set_xticks(range(len(tnames)))
            ax.set_xticklabels(tnames, rotation=30, fontsize=8)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"{model} (R={PAIRS[model]['arms'][0].lstrip('R')})",
                         fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25, axis="y")
        axes[0][0].set_ylabel("accuracy change under residency, points")
        fig.suptitle("Damage with and without thinking (constrained − free, "
                     "same items and protocol per mode)", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{FIG}/think_damage.png", dpi=150)
        print(f"wrote {FIG}/think_damage.png")

    # figure 2 + table: think-length shift free -> constrained
    print("=== 2. think length: free vs constrained (mean tokens) ===")
    pts = []
    for model, cfg in PAIRS.items():
        for mode, rec in cfg["modes"].items():
            tm = task_map(rec)
            for tname, (task, _) in tm.items():
                lf = load_lengths(rec, "free", task)
                if not lf or not lf["think"]:
                    continue
                for arm in cfg["arms"]:
                    lc = load_lengths(rec, arm, task)
                    if not lc:
                        continue
                    pts.append((model, mode, arm, tname, lf["think"], lc["think"]))
                    print(f"  {model:14s} {mode:6s} {arm:4s} {tname:9s} "
                          f"{lf['think']:5.0f} -> {lc['think']:5.0f}  "
                          f"(x{lc['think']/max(1, lf['think']):.2f})")
    if pts:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        cols = {m: c for m, c in zip(PAIRS, plt.cm.tab10.colors)}
        for model, mode, arm, tname, f, c in pts:
            ax.scatter(f, c, color=cols[model], s=45)
        seen = set()
        handles = []
        for model, *_ in pts:
            if model in seen:
                continue
            seen.add(model)
            handles.append(plt.Line2D([], [], marker="o", ls="", color=cols[model],
                                      label=model))
        lim = max(max(p[4] for p in pts), max(p[5] for p in pts)) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
        ax.set_xlabel("mean think tokens, free")
        ax.set_ylabel("mean think tokens, constrained")
        ax.set_title("Does the residency constraint lengthen thinking?\n"
                     "(points above the diagonal think longer under the constraint)",
                     fontsize=10)
        ax.legend(handles=handles, fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{FIG}/think_length_shift.png", dpi=150)
        print(f"wrote {FIG}/think_length_shift.png")


if __name__ == "__main__":
    main()
