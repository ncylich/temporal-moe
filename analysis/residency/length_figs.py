#!/usr/bin/env python3
"""Generation-length figures, across every benchmark in the grid.

1. length_decomp: how a constrained generation ended, against how often it was
   wrong, on all five surfaces. Generations that reached their token budget
   emitted no usable answer and score wrong mechanically, so they form ONE group,
   kept apart from the ones that ran long and still finished, which are the only
   evidence that length itself costs accuracy.

   Two things this does that truncation_decomp.py does not. It covers GSM8K,
   IFEval and WritingBench as well as HumanEval and MMLU, which is possible only
   because merging the two budget-reaching groups drops the need for a
   thinking-block marker in the raw text (the older GSM8K and IFEval dumps saved
   scores and lengths but no raw text). And it is restricted to the cells the
   grid reports: truncation_decomp globs every dump on disk, which pools
   superseded original-budget records and off-paper screening runs in with the
   reported ones and inflates the budget-reaching group several-fold.

   WritingBench is scored 1 to 10 by a critic rather than marked right or wrong,
   so it carries its own axis, mean critic-score drop against the free-routing
   score on the same query.

2. adapt_length: generation and thinking length under residency, relative to free
   routing, for a released model against the same model after constraint-aware
   adaptation, with thinking both off and on. Thinking off is the control: no
   inflation on either side. Thinking on is where the effect lives.

Writes results/ablations/figures/{length_decomp,adapt_length}.png plus the
--no-caption paper variants. Labels avoid run-record shorthand throughout.
"""
import collections
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import think_analysis as ta                                          # noqa: E402

FIG = os.path.join(ABLATIONS, "figures")
SAMP = os.path.join(ABLATIONS, "genbench_samples")
WB = os.path.join(ABLATIONS, "writingbench")
PAPER = "--no-caption" in sys.argv
if PAPER:
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10.5,
                         "legend.fontsize": 10})


def _save(fig, name):
    fig.savefig(f"{FIG}/{name}{'_nocaption' if PAPER else ''}.png", dpi=170,
                bbox_inches="tight")
    print(f"wrote {name}{'_nocaption' if PAPER else ''}.png")


def _key(item):
    """Always a string. Older dumps key on an int doc_id and newer ones on a
    string doc, so pairing an old record against a new one silently produced an
    empty intersection and dropped the cell without warning."""
    k = item.get("doc", item.get("doc_id"))
    return None if k is None else str(k)


def _items(rec, arm, task):
    """doc -> item, only items carrying a length.

    GSM8K dumps hold TWO entries per item, one per lm-eval filter, with no field
    naming which is which. Flexible extraction is strictly more permissive than
    strict match, so the item's reported score is the max of the pair, which
    recovers the reported filter without depending on write order. Every other
    surface has one entry per item and keeps the first."""
    p = os.path.join(SAMP, f"{rec}_{arm}_{task}.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    out = {}
    for i in (b["items"] if isinstance(b, dict) else b):
        k = _key(i)
        if k is None or "gen_toks" not in i:
            continue
        prev = out.get(k)
        if prev is None:
            out[k] = i
        elif "exact_match" in i and "exact_match" in prev:
            if float(i["exact_match"]) > float(prev["exact_match"]):
                out[k] = i
    return out or None


def lengths(rec, arm, task):
    """doc -> (total, thinking, answer) tokens, or None.

    The dumps do NOT agree on what `gen_toks` means, so the route is decided per
    dump from which fields it carries, never assumed:

      raw_toks present     total = raw_toks,  answer = gen_toks
      raw text present     total = gen_toks (thinking included), think = think_toks.
                           Validated on all 92 such dumps: treating gen_toks as
                           the total puts len(raw)/gen_toks at 2.8 to 4.4 chars
                           per token, the normal band.
      think_toks_by_doc    total = gen_toks + think_by_doc, answer = gen_toks,
                           EXCEPT when think_by_doc equals gen_toks, which is the
                           producer's marker-absent case: it then stored the whole
                           generation there and stripping removed nothing, so the
                           two fields are one number and adding them would double
                           it. Validated against the 3800 items whose dumps carry
                           the authoritative raw_toks as well: 49% are the
                           marker-absent case and 51% satisfy gen + think = raw
                           within 2%, together 100%, median ratio 1.000.
      none of the above    no thinking on this surface, total = gen_toks.

    The per-item `think_toks` field is only trusted alongside `raw`; in older
    dumps it measured post-strip text and reads near zero, which is what made an
    earlier version of this analysis report a 3x thinking ratio that is really
    1.28x."""
    p = os.path.join(SAMP, f"{rec}_{arm}_{task}.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    bydoc = b.get("think_toks_by_doc") if isinstance(b, dict) else None
    items = _items(rec, arm, task)
    if not items:
        return None
    out = {}
    for k, i in items.items():
        g = i["gen_toks"]
        if "raw_toks" in i:
            total, ans = i["raw_toks"], g
            think = max(0, total - ans)
        elif "raw" in i:
            think = i.get("think_toks") or 0
            total, ans = g, max(0, g - think)
        elif bydoc and str(k) in bydoc:
            think = bydoc[str(k)]
            if think == g:
                # the thinking marker was absent, so the producer stored the whole
                # generation here and stripping removed nothing: one number, not two
                total, think, ans = g, 0, g
            else:
                total, ans = g + think, g
        else:
            total, think, ans = g, 0, g
        out[k] = (total, think, ans)
    return out


def _wrong(item, surface):
    """True, False, or None when the item cannot be scored."""
    if surface == "HumanEval":
        v = item.get("pass")
    elif surface == "GSM8K":
        v = item.get("exact_match")
    elif surface == "IFEval":
        v = item.get("prompt_level_strict_acc")
    else:                                        # MMLU, dual-scored dumps
        g = item.get("gold")
        v = None if g is None else item.get("pred_relaxed") == g
    return None if v is None else not bool(float(v) if isinstance(v, str) else v)


# ---------------------------------------------------------------- figure 1

def reported():
    """(surface, record-for-constrained, record-for-free, arm, dump task, budget)
    for every cell the grid reports, at the budget it is reported at."""
    cells = ta.load_cells()
    out = []
    for model, cfg in ta.PAIRS.items():
        for mode, rec in cfg["modes"].items():
            for tname, cands in ta.task_map(rec).items():
                for arm in cfg["arms"]:
                    got = ta.resolve(cells, rec, arm, cands)
                    if not got:
                        continue
                    src, task, metric, _, _, bud_f, bud_c = got
                    frec = src if (src, "free", task, metric) in cells else rec
                    dtask = "mmlu_dual" if task.startswith("mmlu") else task
                    out.append((tname, src, frec, arm, dtask, bud_f, bud_c))
    return out


GROUPS = [("normal", "normal\nlength", "0.72"),
          ("budget", "hit the budget,\nno usable answer", "#a83232"),
          ("long", "ran long,\nfinished anyway", "#4878b0"),
          ("freelong", "free routing\nran long too", "#9dbcd8")]


def decompose():
    """{surface: {group: [n, n_wrong]}} pooled over the reported cells."""
    g = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    ncell = 0
    for surf, crec, frec, arm, task, bud_f, bud_c in reported():
        fr, cn = _items(frec, "free", task), _items(crec, arm, task)
        lf_, lc = lengths(frec, "free", task), lengths(crec, arm, task)
        if not fr or not cn or not lf_ or not lc:
            continue
        ncell += 1
        for d in sorted(set(fr) & set(cn) & set(lf_) & set(lc)):
            c = cn[d]
            w = _wrong(c, surf)
            if w is None:
                continue
            # the budget applies to the WHOLE generation, so classify on total
            ftot, ctot = lf_[d][0], lc[d][0]
            if ctot >= bud_c - 8:
                key = "budget"
            elif ctot > 2 * max(1, ftot):
                key = "long"
            elif ftot >= bud_f - 8 or ftot > 2 * max(1, ctot):
                key = "freelong"
            else:
                key = "normal"
            g[surf][key][0] += 1
            g[surf][key][1] += w
    return g, ncell


# ---- WritingBench, scored 1 to 10 by a critic rather than right or wrong ----

WB_CELLS = [("gemma4_base_free", "gemma4_base_R8"), ("gemma4_base_free", "gemma4_base_R16"),
            ("qwen35_base_free", "qwen35_base_R8"), ("qwen35_base_free", "qwen35_base_R32"),
            ("oss20_free", "oss20_R4"), ("oss120_free", "oss120_R4"),
            ("oss120_free", "oss120_R16"), ("lfm25_free", "lfm25_R4")]
WB_BUDGET = 4096


def _wb_scores(cell):
    """(subset, index) -> mean critic score over that query's criteria."""
    acc = collections.defaultdict(list)
    for suf, sub in (("", "A"), ("_sB", "B"), ("_sC", "C")):
        p = os.path.join(WB, "scores", f"{cell}{suf}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            acc[(sub, str(r["index"]))].append(float(r["score"]))
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def _wb_lengths():
    """(record, arm) -> {(subset, index): gen_toks}"""
    out = collections.defaultdict(dict)
    for r in csv.DictReader(l for l in open(os.path.join(WB, "response_lengths.csv"))
                            if not l.startswith('"#')):
        out[f"{r['record']}_{r['arm']}"][(r["subset"], str(r["index"]))] = int(r["gen_toks"])
    return out


def wb_decompose():
    """{group: [n, total critic-score drop]} pooled over the WritingBench cells."""
    lens = _wb_lengths()
    g = collections.defaultdict(lambda: [0, 0.0])
    for fcell, ccell in WB_CELLS:
        fs, cs = _wb_scores(fcell), _wb_scores(ccell)
        fl, cl = lens.get(fcell), lens.get(ccell)
        if not fs or not cs or not fl or not cl:
            continue
        for k in sorted(set(fs) & set(cs) & set(fl) & set(cl)):
            if cl[k] >= WB_BUDGET - 8:
                key = "budget"
            elif cl[k] > 2 * max(1, fl[k]):
                key = "long"
            elif fl[k] >= WB_BUDGET - 8 or fl[k] > 2 * max(1, cl[k]):
                key = "freelong"
            else:
                key = "normal"
            g[key][0] += 1
            g[key][1] += fs[k] - cs[k]           # positive = worse under residency
    return g


SURFACES = ["GSM8K", "IFEval", "HumanEval", "MMLU"]


def length_decomp():
    groups, ncell = decompose()
    wb = wb_decompose()
    print(f"length_decomp: {ncell} reported cells, plus {len(WB_CELLS)} WritingBench cells")

    fig, axes = plt.subplots(1, 5, figsize=(15.4, 3.9) if PAPER else (16, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.05]})
    for ax, surf in zip(axes, SURFACES):
        g = groups[surf]
        total = sum(g[k][0] for k, _, _ in GROUPS)
        base = g["normal"][1] / max(1, g["normal"][0])
        for i, (key, lab, col) in enumerate(GROUPS):
            n, wrong = g[key]
            if not n:
                continue
            rate = wrong / n
            ax.bar(i, 100 * rate, color=col, edgecolor="black", lw=0.5, zorder=2)
            ax.annotate(f"n={n}\n{100*n/total:.1f}%", (i, 100 * rate),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=8)
            if key == "long" and base > 0:
                ax.annotate(f"{rate / base:.1f}x", (i, 100 * rate),
                            textcoords="offset points", xytext=(0, 28),
                            ha="center", fontsize=12, fontweight="bold",
                            color="#4878b0")
        ax.axhline(100 * base, color="black", lw=0.9, ls="--", zorder=1)
        ax.set_ylim(0, 118)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([x[1] for x in GROUPS], fontsize=7.6, rotation=32,
                           ha="right")
        ax.set_title(surf, fontsize=11)
        ax.grid(alpha=0.25, axis="y")
        ax.set_axisbelow(True)
    for ax in axes[1:4]:
        ax.set_yticklabels([])
    axes[0].set_ylabel("wrong, % of the group")

    ax = axes[4]
    total = sum(wb[k][0] for k, _, _ in GROUPS)
    for i, (key, lab, col) in enumerate(GROUPS):
        n, drop = wb[key]
        if not n:
            continue
        m = drop / n
        ax.bar(i, m, color=col, edgecolor="black", lw=0.5, zorder=2)
        ax.annotate(f"n={n}\n{100*n/total:.1f}%", (i, m), textcoords="offset points",
                    xytext=(0, 4 if m >= 0 else -16), ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.9, zorder=1)
    vals = [wb[k][1] / wb[k][0] for k, _, _ in GROUPS if wb[k][0]]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    ax.set_ylim(lo - 0.30 * (hi - lo), hi + 0.42 * (hi - lo))
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([x[1] for x in GROUPS], fontsize=7.6, rotation=32, ha="right")
    ax.set_title("WritingBench", fontsize=11)
    ax.set_ylabel("critic-score drop\n(1 to 10 scale)", fontsize=10)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(alpha=0.25, axis="y")
    ax.set_axisbelow(True)

    if not PAPER:
        fig.suptitle("How a constrained generation ended, against how well it scored "
                     "(reported cells only)\ndashed line: the normal-length rate. "
                     "WritingBench is critic-scored, so it carries its own axis",
                     fontsize=10)
    fig.tight_layout()
    _save(fig, "length_decomp")

    print(f"\n{'surface':11} {'group':10} {'n':>6} {'share':>7} {'wrong/drop':>11} {'vs normal':>10}")
    for surf in SURFACES:
        g = groups[surf]
        total = sum(g[k][0] for k, _, _ in GROUPS)
        base = g["normal"][1] / max(1, g["normal"][0])
        for key, _, _ in GROUPS:
            n, wrong = g[key]
            if not n:
                continue
            r = wrong / n
            print(f"{surf:11} {key:10} {n:6d} {100*n/total:6.1f}% {r:11.3f} "
                  f"{(r/base if base else 0):9.1f}x")
    total = sum(wb[k][0] for k, _, _ in GROUPS)
    for key, _, _ in GROUPS:
        n, drop = wb[key]
        if n:
            print(f"{'WritingBench':11} {key:10} {n:6d} {100*n/total:6.1f}% "
                  f"{drop/n:11.3f} {'(pts)':>10}")


# ---------------------------------------------------------------- figure 2

# Every bar is divided by the SAME reference, the released checkpoint under free
# routing, so the figure shows two things at once: how much the constraint
# lengthens each model, and whether adaptation moved the model's baseline length
# at all. Dividing each model by itself would hide the second.

def _wb_lengths_by_cell():
    out = collections.defaultdict(dict)
    for r in csv.DictReader(l for l in open(os.path.join(WB, "response_lengths.csv"))
                            if not l.startswith('"#')):
        out[f"{r['record']}_{r['arm']}"][(r["subset"], str(r["index"]))] = int(r["gen_toks"])
    return out


def _mean_total(rec, arm, task, keys=None):
    d = lengths(rec, arm, task)
    if not d:
        return None, None
    ks = sorted(d) if keys is None else sorted(set(d) & set(keys))
    if len(ks) < 50:
        return None, None
    return sum(d[k][0] for k in ks) / len(ks), set(ks)


TASK = {"GSM8K": "gsm8k_cot_zeroshot", "IFEval": "ifeval",
        "HumanEval": "humaneval_gemma_fixed", "MMLU": "mmlu_dual"}
ARMS = [("R8", "8 resident"), ("R16", "16 resident")]
PANELS = [("thinking on", "gemma4_think_on", "gemma4_ce_think3k",
           ["GSM8K", "IFEval"]),
          ("thinking off", "gemma4_instruct", "gemma4_ce_d12_freshregen",
           ["GSM8K", "IFEval", "HumanEval", "WritingBench"])]
WB_PAIR = ("gemma4_base", "gemma4_d12")

BARS = [("released, constrained", "#d1605e", 0),
        ("adapted, unconstrained", "#9dbcd8", 1),
        ("adapted, constrained", "#4878b0", 2)]


def _wb_ratio(arm):
    """(released constrained, adapted free, adapted constrained), each over the
    released free mean, on the shared queries."""
    L = _wb_lengths_by_cell()
    bf, bc = L.get(f"{WB_PAIR[0]}_free"), L.get(f"{WB_PAIR[0]}_{arm}")
    af, ac = L.get(f"{WB_PAIR[1]}_free"), L.get(f"{WB_PAIR[1]}_{arm}")
    if not all((bf, bc, af, ac)):
        return None
    ks = sorted(set(bf) & set(bc) & set(af) & set(ac))
    if len(ks) < 50:
        return None
    ref = sum(bf[k] for k in ks)
    return (sum(bc[k] for k in ks) / ref, sum(af[k] for k in ks) / ref,
            sum(ac[k] for k in ks) / ref)


def adapt_length():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.0) if PAPER else (13.2, 4.7),
                             sharey=True, gridspec_kw={"width_ratios": [1, 2]})
    for ax, (title, brec, arec, surfs) in zip(axes, PANELS):
        cols = [(s, a) for a, _ in ARMS for s in surfs]
        vals = {i: [] for i in range(3)}
        xs = {i: [] for i in range(3)}
        for ci, (surf, arm) in enumerate(cols):
            if surf == "WritingBench":
                r = _wb_ratio(arm)
            else:
                t = TASK[surf]
                bfm, ks = _mean_total(brec, "free", t)
                if bfm is None:
                    continue
                bcm, _ = _mean_total(brec, arm, t, ks)
                afm, _ = _mean_total(arec, "free", t, ks)
                acm, _ = _mean_total(arec, arm, t, ks)
                r = None if None in (bcm, afm, acm) else (bcm / bfm, afm / bfm, acm / bfm)
            if r is None:
                continue
            for i in range(3):
                xs[i].append(ci + (i - 1) * 0.27)
                vals[i].append(r[i])
        for i, (lab, col, _) in enumerate(BARS):
            ax.bar(xs[i], vals[i], width=0.25, color=col, edgecolor="black", lw=0.5,
                   label=lab if ax is axes[0] else None, zorder=2)
            for x, y in zip(xs[i], vals[i]):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8)
        ax.axhline(1.0, color="black", lw=1.1, zorder=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{s}\n{dict(ARMS)[a]}" for s, a in cols], fontsize=8.5)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("total generation length,\nvs released model unconstrained")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].set_ylim(0, 1.62)
    if not PAPER:
        fig.suptitle("Total generation length (thinking + answer), gemma4-26B, same "
                     "items\nblack line = the released checkpoint under free routing, "
                     "the shared reference for every bar", fontsize=10)
    fig.tight_layout()
    _save(fig, "adapt_length")
    print(f"\n{'panel':13} {'surface':13} {'arm':5} | {'rel.constr':>10} "
          f"{'adpt.free':>10} {'adpt.constr':>12}")
    for title, brec, arec, surfs in PANELS:
        for arm, _ in ARMS:
            for surf in surfs:
                if surf == "WritingBench":
                    r = _wb_ratio(arm)
                else:
                    t = TASK[surf]
                    bfm, ks = _mean_total(brec, "free", t)
                    if bfm is None:
                        continue
                    bcm, _ = _mean_total(brec, arm, t, ks)
                    afm, _ = _mean_total(arec, "free", t, ks)
                    acm, _ = _mean_total(arec, arm, t, ks)
                    r = None if None in (bcm, afm, acm) else (bcm/bfm, afm/bfm, acm/bfm)
                if r is None:
                    continue
                print(f"{title:13} {surf:13} {arm:5} | {r[0]:10.2f} {r[1]:10.2f} "
                      f"{r[2]:12.2f}")


if __name__ == "__main__":
    length_decomp()
    adapt_length()
