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
# Fair-budget re-measurements (TRUNCATION_RERUN_PLAN.md). Each entry lists the
# re-run records for a mode's parent record, largest budget first. A re-run
# supersedes its parent for one task only when BOTH the free arm and the
# constrained arm were measured there, so a reported cell never mixes budgets
# across arms; where that fails the chain falls back to the parent record.
FAIR = {
    "gemma4_instruct":  ["gemma4_instruct_cap8k"],
    "gemma4_think_on":  ["gemma4_think_on_cap8k"],
    "qwen35_instruct":  ["qwen35_instruct_cap16k", "qwen35_instruct_cap8k"],
    "lfm25_instruct":   ["lfm25_instruct_cap8k"],
    "gptoss_20b":       ["gptoss_20b_cap8k"],
    "gptoss_20b_high":  ["gptoss_20b_high_cap16k", "gptoss_20b_high_cap8k"],
    "gptoss_120b":      ["gptoss_120b_cap8k"],
    "gptoss_120b_high": ["gptoss_120b_high_cap16k", "gptoss_120b_high_cap8k"],
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
    """(record, arm, task, metric) -> (value, budget). The authoritative grid,
    plus the fair-budget re-runs, which live in screening_genbench.csv because
    the budget is the variable under test and both budgets have to stay
    independently visible; only records named in FAIR are taken from there."""
    fair_recs = {r for chain in FAIR.values() for r in chain}
    cells = {}
    for f in ("instruct_genbench_vllm.csv", "screening_genbench.csv"):
        screening = f.startswith("screening")
        for r in csv.reader(open(f"{ABLATIONS}/{f}")):
            if len(r) > 9 and not r[0].startswith(("#", "smoke", "model")) \
                    and (not screening or r[0] in fair_recs):
                try:
                    cells[(r[0], r[3], r[5], r[6])] = (float(r[7]), int(r[9]))
                except ValueError:
                    pass
    return cells


CLEAN = 2.0     # cap-hit %, at or below which a budget increase cannot move a cell


def cap_hit(rec, arm, task, budget):
    """Percent of a cell's generations that ended within 8 tokens of its budget.
    Measured against the DECLARED budget, never the observed maximum: a single
    over-cap outlier shifts a max-based reference past the cap pile-up and hides
    it (qwen35 free IFEval reads 0.5% by observed max and 8.0% by declared)."""
    rows = _dump_rows(rec, arm, task)
    if not rows:
        return None
    return 100 * sum(1 for i in rows if i["gen_toks"] >= budget - 8) / len(rows)


def resolve(cells, rec, arm, cands):
    """Pick the record, task and metric for one cell at the largest budget that is
    fair to both arms, as (record, task, metric, free, con, budget_free, budget_con).

    A re-run supersedes its parent when the constrained arm was re-measured there.
    The free arm comes from the re-run too when it exists; when it does not, the
    parent's free arm is still admissible if it is budget-clean, since generations
    that all stopped on their own cannot move when the budget grows. A free arm
    that was itself hitting the cap is not admissible, so that cell falls back to
    the parent record and stays matched, if truncated, on both arms."""
    for r in FAIR.get(rec, []) + [rec]:
        for task, metric in cands:
            con = cells.get((r, arm, task, metric))
            if con is None:
                continue
            free = cells.get((r, "free", task, metric))
            if free is not None:
                return r, task, metric, free[0], con[0], free[1], con[1]
            if r != rec:
                pfree = cells.get((rec, "free", task, metric))
                if pfree is not None:
                    t = cap_hit(rec, "free", task, pfree[1])
                    if t is not None and t <= CLEAN:
                        return r, task, metric, pfree[0], con[0], pfree[1], con[1]
    return None


def _load_dump(record, arm, task):
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
    return (b, rows) if rows else None


def _dump_rows(record, arm, task):
    d = _load_dump(record, arm, task)
    return d[1] if d else None


def load_lengths(record, arm, task):
    d = _load_dump(record, arm, task)
    if d is None:
        return None
    b, rows = d
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
    sm = open(os.path.join(ABLATIONS, out_name), "w", newline="")
    w = csv.writer(sm)
    sm.write('"# Thinking ablation summary: damage (constrained-free; damage_se = '
             'UNPAIRED binomial SE of the difference, n per task) and lengths per '
             'model/mode/arm/task. Producer: analysis/residency/think_analysis.py"\n')
    sm.write('"# Every cell is reported at the largest generation budget fair to both of '
             'its arms (see FAIR and resolve). budget_free and budget_con are the budgets '
             'actually used, caphit_free and caphit_con the percent of that arms '
             'generations ending within 8 tokens of it, and record names the source '
             'record. A cap-hit above 2 percent means the cell is still budget-limited '
             'and its damage carries a truncation component."\n')
    w.writerow(["model", "mode", "arm", "task", "free", "constrained", "damage",
                "damage_se",
                "gen_free", "gen_con", "think_free", "think_con",
                "budget_free", "budget_con", "caphit_free", "caphit_con", "record"])

    print("=== 1. damage x thinking mode (points, constrained - free) ===")
    dam = {}
    for model, cfg in pairs.items():
        for mode, rec in cfg["modes"].items():
            tm = task_map(rec)
            for tname, cands in tm.items():
                for arm in cfg["arms"]:
                    # the reported record is the largest budget fair to both arms,
                    # and the first candidate task present there (never mixes
                    # extraction protocols between free and constrained)
                    got = resolve(cells, rec, arm, cands)
                    if got is None:
                        continue
                    src, task, metric, free, con, bud_f, bud_c = got
                    fsrc = src if (src, "free", task, metric) in cells else rec
                    d = 100 * (con - free)
                    dam[(model, mode, arm, tname)] = d
                    lf = load_lengths(fsrc, "free", task) or {}
                    lc = load_lengths(src, arm, task) or {}
                    n_task = {"HumanEval": 164, "MMLU": 228}.get(tname, 200)
                    n_f = (lf or {}).get("n") or n_task
                    n_c = (lc or {}).get("n") or n_task
                    se = 100 * ((free * (1 - free) / n_f) +
                                (con * (1 - con) / n_c)) ** 0.5
                    tf = cap_hit(fsrc, "free", task, bud_f)
                    tc = cap_hit(src, arm, task, bud_c)
                    fmt = lambda v: f"{v:.0f}" if v is not None else ""
                    pct = lambda v: f"{v:.1f}" if v is not None else ""
                    w.writerow([model, mode, arm, tname, f"{free:.4f}", f"{con:.4f}",
                                f"{d:+.1f}", f"{se:.1f}",
                                fmt(lf.get("gen")), fmt(lc.get("gen")),
                                fmt(lf.get("think")), fmt(lc.get("think")),
                                bud_f, bud_c, pct(tf), pct(tc), src])
                    flag = "" if max(tf or 0, tc or 0) <= CLEAN else \
                        f"   [cap-hit {max(tf or 0, tc or 0):.1f}%]"
                    print(f"  {model:14s} {mode:6s} {arm:4s} {tname:9s} "
                          f"{free:.3f} -> {con:.3f}  ({d:+.1f})  @{bud_c}{flag}")
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
