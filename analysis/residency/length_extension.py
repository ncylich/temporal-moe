#!/usr/bin/env python3
"""Length-mechanism analysis extended to the recovered surfaces (HumanEval, MMLU,
WritingBench) -- the datasets that were length-blind until the dump regeneration.

Definitions follow plot_length_decomp.py: cap-hit = gen_toks within 8 of the
cell's budget; blow-up = cap-hit in either arm OR length > 2x the paired
counterpart. Per cell (paired items, same doc across arms):
  - mean paired length delta (constrained - free) and the part from cap-hit items
  - flips by direction (right->wrong / wrong->right); WritingBench flips are
    per-item critic-score deltas beyond +-1 SD of that subset's deltas (stated
    operationalization -- no binary correctness exists there)
  - blow-up share of each flip direction and of all items
  - wrongness conditioned on blow-up in the constrained arm (wrong-rate among
    blown vs non-blown items; WritingBench: mean score delta)
Writes results/ablations/length_extension.csv and
figures/length_extension_decomp.png (+ _nocaption via --no-caption).
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

SAMP = os.path.join(ABLATIONS, "genbench_samples")
WB = os.path.join(ABLATIONS, "writingbench")
FIG = os.path.join(ABLATIONS, "figures")
PAPER = "--no-caption" in sys.argv

# FAIR-BUDGET cells: the same cells resumed to 8192 so no generation is scored as
# wrong merely for running out of room. Kept ALONGSIDE the original-budget cells,
# never replacing them -- the budget is the variable under test.
HUMANEVAL_CAP8K = [
    ("gemma off 8k", "gemma4_instruct_cap8k", "humaneval_gemma_fixed", "R8", 8192),
    ("gemma off R16 8k", "gemma4_instruct_cap8k", "humaneval_gemma_fixed", "R16", 8192),
    ("gemma on 8k", "gemma4_think_on_cap8k", "humaneval_gemma_fixed", "R8", 8192),
    ("gemma on R16 8k", "gemma4_think_on_cap8k", "humaneval_gemma_fixed", "R16", 8192),
    ("Qwen on 8k", "qwen35_instruct_cap8k", "humaneval_think", "R8", 8192),
    ("Qwen on R32 8k", "qwen35_instruct_cap8k", "humaneval_think", "R32", 8192),
    ("LFM 8k", "lfm25_instruct_cap8k", "humaneval_think", "R4", 8192),
    ("20b high 8k", "gptoss_20b_high_cap8k", "humaneval_gptoss", "R4", 8192),
    ("20b med 8k", "gptoss_20b_cap8k", "humaneval_gptoss", "R4", 8192),
    ("120b high 8k", "gptoss_120b_high_cap8k", "humaneval_gptoss", "R4", 8192),
    ("120b high R16 8k", "gptoss_120b_high_cap8k", "humaneval_gptoss", "R16", 8192),
    ("120b med 8k", "gptoss_120b_cap8k", "humaneval_gptoss", "R4", 8192),
    ("120b med R16 8k", "gptoss_120b_cap8k", "humaneval_gptoss", "R16", 8192),
]
MMLU_CAP8K = [
    ("Qwen on 8k", "qwen35_instruct_cap8k", "R8", 8192),
    ("Qwen on R32 8k", "qwen35_instruct_cap8k", "R32", 8192),
]

# (label, record, task-file, arm, cap)
HUMANEVAL = [
    ("gemma off", "gemma4_instruct", "humaneval_gemma_fixed", "R8", 1536),
    ("gemma off R16", "gemma4_instruct", "humaneval_gemma_fixed", "R16", 1536),
    ("gemma on", "gemma4_think_on", "humaneval_gemma_fixed", "R8", 3072),
    ("gemma on R16", "gemma4_think_on", "humaneval_gemma_fixed", "R16", 3072),
    ("Qwen on", "qwen35_instruct", "humaneval_think", "R8", 4096),
    ("Qwen on R32", "qwen35_instruct", "humaneval_think", "R32", 4096),
    ("LFM", "lfm25_instruct", "humaneval_think", "R4", 4096),
    ("20b low", "gptoss_20b_low", "humaneval_gptoss", "R4", 2048),
    ("20b med", "gptoss_20b", "humaneval_gptoss", "R4", 2048),
    ("20b high", "gptoss_20b_high", "humaneval_gptoss", "R4", 4096),
    ("120b low", "gptoss_120b_low", "humaneval_gptoss", "R4", 2048),
    ("120b med", "gptoss_120b", "humaneval_gptoss", "R4", 2048),
    ("120b high", "gptoss_120b_high", "humaneval_gptoss", "R4", 4096),
    ("120b low R16", "gptoss_120b_low", "humaneval_gptoss", "R16", 2048),
    ("120b med R16", "gptoss_120b", "humaneval_gptoss", "R16", 2048),
    ("120b high R16", "gptoss_120b_high", "humaneval_gptoss", "R16", 4096),
]
MMLU = [
    ("OLMoE", "olmoe_instruct", "R8", 2048),
    ("Qwen on", "qwen35_instruct", "R8", 4096),
    ("Qwen on R32", "qwen35_instruct", "R32", 4096),
    ("gemma on", "gemma4_think_on", "R8", 4096),
    ("gemma on R16", "gemma4_think_on", "R16", 4096),
    ("LFM", "lfm25_instruct", "R4", 4096),
    ("20b low", "gptoss_20b_low", "R4", 4096),
    ("20b med", "gptoss_20b", "R4", 4096),
    ("20b high", "gptoss_20b_high", "R4", 4096),
    ("120b low", "gptoss_120b_low", "R4", 4096),
    ("120b med", "gptoss_120b", "R4", 4096),
    ("120b high", "gptoss_120b_high", "R4", 4096),
    ("120b low R16", "gptoss_120b_low", "R16", 4096),
    ("120b med R16", "gptoss_120b", "R16", 4096),
    ("120b high R16", "gptoss_120b_high", "R16", 4096),
]
WB_CAP = 4096


def dump_items(rec, arm, task):
    p = f"{SAMP}/{rec}_{arm}_{task}.json"
    if not os.path.exists(p):
        return None
    return {i["doc"]: i for i in json.load(open(p))["items"]}


def pair_stats(fr, cn, cap, correct):
    """fr/cn: doc -> item; correct(item) -> bool|None. Returns cell stats dict."""
    cm = sorted(set(fr) & set(cn))
    if not cm:
        return None
    def L(i):
        return i["gen_toks"]
    o = dict(n=len(cm), dlen=0.0, dlen_cap=0.0, toW=0, toR=0, bw_toW=0, bw_toR=0,
             bw_all=0, wrong_bw=0, n_bw=0, wrong_nb=0, n_nb=0)
    for d in cm:
        f, c = fr[d], cn[d]
        dl = L(c) - L(f)
        caphit = L(f) >= cap - 8 or L(c) >= cap - 8
        blow = caphit or L(c) > 2 * L(f) or L(f) > 2 * L(c)
        o["dlen"] += dl
        if caphit:
            o["dlen_cap"] += dl
        o["bw_all"] += blow
        af, ac = correct(f), correct(c)
        if ac is not None:
            if blow:
                o["n_bw"] += 1
                o["wrong_bw"] += not ac
            else:
                o["n_nb"] += 1
                o["wrong_nb"] += not ac
        if af and not ac:
            o["toW"] += 1
            o["bw_toW"] += blow
        elif ac and not af:
            o["toR"] += 1
            o["bw_toR"] += blow
    o["dlen"] /= o["n"]
    o["dlen_cap"] /= o["n"]
    return o


def he_correct(i):
    return i.get("pass")


def mmlu_correct(i):
    g = i.get("gold")
    return None if g is None else i.get("pred_relaxed") == g


def wb_cells():
    """WritingBench: critic-score flips (delta beyond +-1 subset SD)."""
    scores, lens = {}, {}
    for f in sorted(glob.glob(f"{WB}/scores/*.jsonl")):
        rec = os.path.basename(f)[:-6]
        if rec.startswith("smoke"):
            continue
        base, subset = rec, "A"
        for sfx in ("_sB", "_sC"):
            if rec.endswith(sfx):
                base, subset = rec[:-3], sfx[2:]
        m = re.match(r"(.+)_(free|R\d+)$", base)
        per = {}
        for line in open(f):
            d = json.loads(line)
            if d["score"] is not None:
                per.setdefault(d["index"], []).append(d["score"])
        scores[(m.group(1), m.group(2), subset)] = {
            k: sum(v) / len(v) for k, v in per.items()}
    for r in csv.DictReader((line for line in open(f"{WB}/response_lengths.csv")
                             if not line.startswith('"#'))):
        lens.setdefault((r["record"], r["arm"], r["subset"]), {})[
            int(r["index"])] = int(r["gen_toks"])
    out = []
    for r0 in sorted({k[0] for k in scores}):
        for arm in sorted({k[1] for k in scores if k[0] == r0} - {"free"}):
            o = dict(n=0, dlen=0.0, dlen_cap=0.0, toW=0, toR=0, bw_toW=0,
                     bw_toR=0, bw_all=0, drop_bw=[], drop_nb=[])
            for sub in "ABC":
                sf = scores.get((r0, "free", sub), {})
                sc = scores.get((r0, arm, sub), {})
                lf = lens.get((r0, "free", sub), {})
                lc = lens.get((r0, arm, sub), {})
                cm = sorted(set(sf) & set(sc) & set(lf) & set(lc))
                dl = [sc[i] - sf[i] for i in cm]
                sd = st.stdev(dl) if len(dl) > 1 else 0
                for i in cm:
                    d = sc[i] - sf[i]
                    caphit = lf[i] >= WB_CAP - 8 or lc[i] >= WB_CAP - 8
                    blow = caphit or lc[i] > 2 * lf[i] or lf[i] > 2 * lc[i]
                    o["n"] += 1
                    o["dlen"] += lc[i] - lf[i]
                    if caphit:
                        o["dlen_cap"] += lc[i] - lf[i]
                    o["bw_all"] += blow
                    (o["drop_bw"] if blow else o["drop_nb"]).append(d)
                    if d < -sd:
                        o["toW"] += 1
                        o["bw_toW"] += blow
                    elif d > sd:
                        o["toR"] += 1
                        o["bw_toR"] += blow
            if o["n"]:
                o["dlen"] /= o["n"]
                o["dlen_cap"] /= o["n"]
                out.append((f"{r0} {arm}", o))
    return out


def main():
    rows = []                       # (surface, label, stats)
    for lab, rec, task, arm, cap in HUMANEVAL + [
            c for c in HUMANEVAL_CAP8K
            if os.path.exists(f"{SAMP}/{c[1]}_{c[3]}_{c[2]}.json")]:
        fr, cn = dump_items(rec, "free", task), dump_items(rec, arm, task)
        if fr and cn:
            s = pair_stats(fr, cn, cap, he_correct)
            if s:
                rows.append(("HumanEval-8k" if "8k" in lab else "HumanEval", lab, s))
    for lab, rec, arm, cap in MMLU + [
            c for c in MMLU_CAP8K
            if os.path.exists(f"{SAMP}/{c[1]}_{c[2]}_mmlu_dual.json")]:
        fr, cn = dump_items(rec, "free", "mmlu_dual"), dump_items(rec, arm, "mmlu_dual")
        if fr and cn:
            s = pair_stats(fr, cn, cap, mmlu_correct)
            if s:
                rows.append(("MMLU-8k" if "8k" in lab else "MMLU", lab, s))
    for lab, o in wb_cells():
        s = dict(o)
        s["wrong_bw"] = -sum(o["drop_bw"]) / max(1, len(o["drop_bw"]))  # mean drop
        s["n_bw"] = len(o["drop_bw"])
        s["wrong_nb"] = -sum(o["drop_nb"]) / max(1, len(o["drop_nb"]))
        s["n_nb"] = len(o["drop_nb"])
        rows.append(("WritingBench", lab, s))

    out = os.path.join(ABLATIONS, "length_extension.csv")
    with open(out, "w", newline="") as fh:
        fh.write('"# Length-mechanism analysis on the recovered surfaces (paired '
                 'items, constrained vs free). cap-hit = within 8 tokens of the '
                 'cell budget; blow-up = cap-hit either arm or >2x paired '
                 'counterpart. toW/toR = flips right->wrong / wrong->right '
                 '(WritingBench: critic-score delta beyond +-1 subset SD). '
                 'bw_toW/bw_toR = blow-up count within each flip direction. '
                 'wrong_bw/wrong_nb = constrained-arm wrong count among blown / '
                 'non-blown scoreable items over n_bw/n_nb (WritingBench: MEAN '
                 'critic-score DROP instead of counts). dlen = mean paired length '
                 'delta (tokens); dlen_cap = part from cap-hit items. Producer: '
                 'analysis/residency/length_extension.py"\n')
        w = csv.writer(fh)
        w.writerow(["surface", "cell", "n", "dlen", "dlen_cap", "toW", "toR",
                    "bw_toW", "bw_toR", "bw_all", "wrong_bw", "n_bw",
                    "wrong_nb", "n_nb"])
        for surf, lab, s in rows:
            w.writerow([surf, lab, s["n"], f"{s['dlen']:.1f}",
                        f"{s['dlen_cap']:.1f}", s["toW"], s["toR"], s["bw_toW"],
                        s["bw_toR"], s["bw_all"],
                        (f"{s['wrong_bw']:.3f}" if surf == "WritingBench"
                         else s["wrong_bw"]), s["n_bw"],
                        (f"{s['wrong_nb']:.3f}" if surf == "WritingBench"
                         else s["wrong_nb"]), s["n_nb"]])
    print("wrote", out, f"({len(rows)} cells)")
    for surf in ("HumanEval", "HumanEval-8k", "MMLU", "MMLU-8k", "WritingBench"):
        sub = [s for sf, _, s in rows if sf == surf]
        if not sub:
            continue
        toW, toR = sum(s["toW"] for s in sub), sum(s["toR"] for s in sub)
        bwW, bwR = sum(s["bw_toW"] for s in sub), sum(s["bw_toR"] for s in sub)
        print(f"  {surf}: {len(sub)} cells, flips {toW} toW ({bwW} blown) "
              f"vs {toR} toR ({bwR} blown)")

    # decomposition figure: total mean paired dlen with the cap-hit part overlaid
    if rows:
        if PAPER:
            plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                                 "xtick.labelsize": 9.5, "ytick.labelsize": 11.5,
                                 "legend.fontsize": 10})
        show = [(f"{lab}\n{surf[:2] if surf != 'WritingBench' else 'WB'}",
                 s["dlen"], s["dlen_cap"], surf) for surf, lab, s in rows]
        fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(show)), 5))
        for i, (lab, tot, capd, surf) in enumerate(show):
            ax.bar(i, tot, width=0.7, color="#c3d6ea", edgecolor="black", lw=0.5,
                   label="total mean length change" if i == 0 else None)
            ax.bar(i, capd, width=0.42, color="#b03434", edgecolor="black", lw=0.5,
                   label="part from items at the cap" if i == 0 else None)
        bounds = [i for i in range(1, len(show)) if show[i][3] != show[i - 1][3]]
        for b in bounds:
            ax.axvline(b - 0.5, color="grey", lw=0.8, ls=":")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels([s[0] for s in show], rotation=40, ha="right",
                           fontsize=8 if PAPER else 7)
        ax.set_ylabel("mean tokens per item,\nconstrained − free")
        if not PAPER:
            ax.set_title("Cap-traffic vs broad lengthening on the recovered "
                         "surfaces (HumanEval / MMLU / WritingBench)", fontsize=10)
        ax.legend(loc="upper left", fontsize=9 if PAPER else 8)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        p = f"{FIG}/length_extension_decomp{'_nocaption' if PAPER else ''}.png"
        fig.savefig(p, dpi=170)
        print("wrote", p)


if __name__ == "__main__":
    main()
