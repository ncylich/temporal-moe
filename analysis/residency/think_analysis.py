#!/usr/bin/env python3
"""Thinking-ablation analysis (INSTRUCT_ANALYSIS_PLAN.md analyses 1-3).

Reads instruct_genbench_vllm.csv (last row wins per cell) and genbench_samples dumps
(doc_id-deduped). Emits:
  1. damage x thinking-mode table (constrained - free within each mode)
  2. think-length shift (mean think tokens from the RAW doc-keyed capture
     think_toks_by_doc; the per-item think_toks field measured post-strip text and
     is a defect -- never read it)
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
    "OLMoE-1B-7B": {"modes": {"off": "olmoe_instruct"},      # no thinking mode
                    "default": "off", "arms": ["R8"]},
}
# fair-budget (8192-cap) resumed cells from the truncation-fix sweep (TODO.md).
# NOT merged into PAIRS: only used when --out targets a non-default file, so the
# default think_ablation_summary.csv output is byte-for-byte unchanged.
PAIRS_CAP8K = {
    "gemma4-26B-IT": {"modes": {"off_8k": "gemma4_instruct_cap8k",
                                "on_8k": "gemma4_think_on_cap8k"},
                      "default": "off_8k", "arms": ["R8", "R16"]},
    "Qwen3.5-35B": {"modes": {"on_8k": "qwen35_instruct_cap8k"},
                    "default": "on_8k", "arms": ["R8", "R32"]},
    "gpt-oss-20b": {"modes": {"medium_8k": "gptoss_20b_cap8k",
                              "high_8k": "gptoss_20b_high_cap8k"},
                    "default": "medium_8k", "arms": ["R4"]},
    "gpt-oss-120b": {"modes": {"medium_8k": "gptoss_120b_cap8k",
                               "high_8k": "gptoss_120b_high_cap8k"},
                     "default": "medium_8k", "arms": ["R4", "R16"]},
    "LFM2.5-A1B": {"modes": {"on_8k": "lfm25_instruct_cap8k"},
                   "default": "on_8k", "arms": ["R4"]},
}
# per-record task->metric candidates, first pair where BOTH arms have a row wins.
# MMLU: relaxed extraction (extractor v2, mmlu_gptoss.py) is the reported metric
# wherever the relaxed harness has run (gemma_adapt_RESULTS.md convention: strict
# measures few-shot format imitation, not knowledge); strict flan-CoT cells are
# the fallback for modes not regenerated under the relaxed harness.
MMLU_CAND = [("mmlu_gptoss_relaxed", "acc,relaxed-extract"),
             ("mmlu_flan_cot_fewshot", "exact_match,get-answer")]
TASKS = {"GSM8K": [("gsm8k_cot_zeroshot", "exact_match,flexible-extract")],
         "IFEval": [("ifeval", "prompt_level_strict_acc,none")],
         "HumanEval": [("humaneval_instruct", "pass@1,create_test")],
         "MMLU": MMLU_CAND}
OVERRIDES = {"gemma4": {"HumanEval": [("humaneval_gemma_fixed", "pass@1,channel-aware")]},
             "lfm": {"HumanEval": [("humaneval_think", "pass@1,channel-aware")]},
             "qwen35": {"HumanEval": [("humaneval_think", "pass@1,channel-aware")]},
             "gptoss": {"HumanEval": [("humaneval_gptoss", "pass@1,channel-aware")],
                        "MMLU": [("mmlu_gptoss_relaxed", "acc,relaxed-extract")]}}


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


def load_cap8k_cells():
    """Fair-budget (8192-cap) rows from screening_genbench.csv, record suffix
    _cap8k only -- screening rows are NOT protocol-comparable in general, but
    the _cap8k ones are a deliberate exception: same protocol, doubled budget,
    written for exactly this comparison (TODO.md)."""
    cells = {}
    for r in csv.reader(open(f"{ABLATIONS}/screening_genbench.csv")):
        if len(r) > 7 and not r[0].startswith(("#", "smoke", "model")) \
                and r[0].endswith("_cap8k"):
            try:
                cells[(r[0], r[3], r[5], r[6])] = float(r[7])
            except ValueError:
                pass
    return cells


def load_lengths(record, arm, task):
    p = os.path.join(SAMP, f"{record}_{arm}_{task}.json")
    if not os.path.exists(p) and task.startswith("mmlu"):
        # the relaxed harness writes its dual-scored dump as *_mmlu_dual.json
        p = os.path.join(SAMP, f"{record}_{arm}_mmlu_dual.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    if isinstance(b, list):                      # pre-2026-08-12 dump format
        b = {"items": b}
    seen, rows = set(), []
    for i in b.get("items", []):
        key = i.get("doc", i.get("doc_id"))      # new dumps carry doc (str) keys
        if key in seen:
            continue
        seen.add(key)
        rows.append(i)
    rows = [i for i in rows if "gen_toks" in i]  # pre-capture-era dumps lack lengths
    if not rows:
        return None
    # Valid think sources: (1) the doc-keyed raw capture think_toks_by_doc;
    # (2) per-item think_toks in NEW-format dumps (marker: "raw" present), which
    # are computed from the pre-strip raw text. The OLD per-item think_toks field
    # (no "raw" alongside) measured post-strip text -- a defect, never read it.
    bd = b.get("think_toks_by_doc") or {}
    tk = [bd[str(i.get("doc", i.get("doc_id")))] for i in rows
          if str(i.get("doc", i.get("doc_id"))) in bd]
    if not tk and all("raw" in i and "think_toks" in i for i in rows):
        tk = [i["think_toks"] for i in rows]
    out = {"n": len(rows),
           "gen": np.mean([i["gen_toks"] for i in rows]),
           "think": np.mean(tk) if len(tk) == len(rows) else None,
           "think_exact": len(tk) == len(rows)}
    return out


def main():
    out_name = "think_ablation_summary.csv"
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            out_name = a.split("=", 1)[1]
    cells = load_cells()
    pairs = {m: dict(cfg) for m, cfg in PAIRS.items()}   # shallow copy, never mutate PAIRS
    if out_name != "think_ablation_summary.csv":
        cells = {**cells, **load_cap8k_cells()}
        for m, cap_cfg in PAIRS_CAP8K.items():
            pairs[m]["modes"] = {**pairs[m]["modes"], **cap_cfg["modes"]}
    sm = open(os.path.join(ABLATIONS, out_name), "w", newline="")
    w = csv.writer(sm)
    sm.write('"# Thinking ablation summary: damage (constrained-free; damage_se = '
             'UNPAIRED binomial SE of the difference, n per task) and lengths per '
             'model/mode/arm/task. Producer: analysis/residency/think_analysis.py"\n')
    if out_name != "think_ablation_summary.csv":
        sm.write('"# LOWER CONFIDENCE than think_ablation_summary.csv (which this file '
                 'does NOT overwrite, by request). Regenerated 2026-08-24 by an agent '
                 'session that was not the original process/model that produced the '
                 'committed think_ablation_summary.csv, using the SAME unmodified '
                 'aggregation logic (task_map/PAIRS/candidate-task selection below) but '
                 'over a data mix that session did not independently author end to end: '
                 'the original grid rows, plus this session\\\'s Task-3 regeneration, plus '
                 'the fair-budget (8192-cap) resumed cells from the truncation-fix sweep '
                 '(see TODO.md). The new _cap8k rows specifically are single-run, use a '
                 'resume+retokenize+rescore pipeline that had one bug already found and '
                 'fixed mid-sweep (stale per-item pass -- see TODO.md section 1.8), and '
                 'have not had an independent review pass beyond the checks in TODO.md '
                 'section 4. Treat rows sourced from _cap8k records as provisional; cross-'
                 'check against genbench_samples/*_cap8k_*.json and screening_genbench.csv '
                 'before citing."\n')
    w.writerow(["model", "mode", "arm", "task", "free", "constrained", "damage",
                "damage_se",
                "gen_free", "gen_con", "think_free", "think_con"])

    print("=== 1. damage x thinking mode (points, constrained - free) ===")
    dam = {}
    for model, cfg in pairs.items():
        for mode, rec in cfg["modes"].items():
            tm = task_map(rec)
            for tname, cands in tm.items():
                for arm in cfg["arms"]:
                    # first candidate task with BOTH arms present (never mixes
                    # extraction protocols between free and constrained)
                    free = con = None
                    for task, metric in cands:
                        free = cells.get((rec, "free", task, metric))
                        con = cells.get((rec, arm, task, metric))
                        if free is not None and con is not None:
                            break
                    if free is None or con is None:
                        continue
                    d = 100 * (con - free)
                    dam[(model, mode, arm, tname)] = d
                    lf = load_lengths(rec, "free", task) or {}
                    lc = load_lengths(rec, arm, task) or {}
                    n_task = {"HumanEval": 164, "MMLU": 228}.get(tname, 200)
                    n_f = (lf or {}).get("n") or n_task
                    n_c = (lc or {}).get("n") or n_task
                    se = 100 * ((free * (1 - free) / n_f) +
                                (con * (1 - con) / n_c)) ** 0.5
                    fmt = lambda v: f"{v:.0f}" if v is not None else ""
                    w.writerow([model, mode, arm, tname, f"{free:.4f}", f"{con:.4f}",
                                f"{d:+.1f}", f"{se:.1f}",
                                fmt(lf.get("gen")), fmt(lc.get("gen")),
                                fmt(lf.get("think")), fmt(lc.get("think"))])
                    print(f"  {model:14s} {mode:6s} {arm:4s} {tname:9s} "
                          f"{free:.3f} -> {con:.3f}  ({d:+.1f})")
    sm.close()

    # figure 1: damage by mode
    models = [m for m in pairs if any(k[0] == m for k in dam)]
    if models:
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4.4),
                                 squeeze=False)
        for ax, model in zip(axes[0], models):
            cfg = pairs[model]
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
            ax.set_title(f"{model} (R={pairs[model]['arms'][0].lstrip('R')})",
                         fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25, axis="y")
        axes[0][0].set_ylabel("accuracy change under residency, points")
        fig.suptitle("Damage with and without thinking (constrained − free, "
                     "same items and protocol per mode)", fontsize=10)
        fig.tight_layout()
        fig_suffix = "" if out_name == "think_ablation_summary.csv" else "_" + out_name.rsplit(".", 1)[0]
        fig1_path = f"{FIG}/think_damage{fig_suffix}.png"
        fig.savefig(fig1_path, dpi=150)
        print(f"wrote {fig1_path}")

    # figure 2 + table: think-length shift free -> constrained
    print("=== 2. think length: free vs constrained (mean tokens) ===")
    pts = []
    for model, cfg in pairs.items():
        for mode, rec in cfg["modes"].items():
            tm = task_map(rec)
            for tname, cands in tm.items():
                lf = task = None
                for task, _ in cands:            # first candidate with a dump
                    lf = load_lengths(rec, "free", task)
                    if lf:
                        break
                if not lf or not lf["think"]:
                    continue
                for arm in cfg["arms"]:
                    lc = load_lengths(rec, arm, task)
                    if not lc:
                        continue
                    exact = lf.get("think_exact") and lc.get("think_exact")
                    pts.append((model, mode, arm, tname, lf["think"], lc["think"]))
                    print(f"  {model:14s} {mode:6s} {arm:4s} {tname:9s} "
                          f"{lf['think']:5.0f} -> {lc['think']:5.0f}  "
                          f"(x{lc['think']/max(1, lf['think']):.2f})"
                          f"{'' if exact else '  ~approx (retry-inclusive)'}")
    if pts:
        PAPER = "--no-caption" in sys.argv
        if PAPER:
            plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                                 "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                                 "legend.fontsize": 10})
        fig, ax = plt.subplots(figsize=(6.4, 5.4) if PAPER else (6.5, 5.5))
        cols = {m: c for m, c in zip(PAIRS, plt.cm.tab10.colors)}
        # filled = free-form thinking on / high effort (the amplified modes),
        # hollow = thinking off / low effort. The mode split IS the claim.
        for model, mode, arm, tname, f, c in pts:
            amp = mode not in ("off", "low")
            ax.scatter(f, c, s=52, color=cols[model] if amp else "none",
                       edgecolor=cols[model], lw=1.6, zorder=3)
        seen = set()
        handles = []
        for model, *_ in pts:
            if model in seen:
                continue
            seen.add(model)
            handles.append(plt.Line2D([], [], marker="o", ls="", color=cols[model],
                                      label=model))
        handles += [plt.Line2D([], [], marker="o", ls="", color="0.3",
                               label="thinking on / high effort"),
                    plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                               color="0.3", label="thinking off / low effort")]
        lo = min(min(p[4] for p in pts), min(p[5] for p in pts)) * 0.8
        lim = max(max(p[4] for p in pts), max(p[5] for p in pts)) * 1.2
        ax.plot([lo, lim], [lo, lim], "k--", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, lim)
        ax.set_ylim(lo, lim)
        ax.set_xlabel("mean think tokens, free")
        ax.set_ylabel("mean think tokens, constrained")
        if not PAPER:
            ax.set_title("Does the residency constraint lengthen thinking?\n"
                         "(points above the diagonal think longer under the constraint)",
                         fontsize=10)
        ax.legend(handles=handles, fontsize=9 if PAPER else 8, loc="upper left")
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig_suffix = "" if out_name == "think_ablation_summary.csv" else "_" + out_name.rsplit(".", 1)[0]
        out = f"{FIG}/think_length_shift{fig_suffix}{'_nocaption' if PAPER else ''}.png"
        fig.savefig(out, dpi=170)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
