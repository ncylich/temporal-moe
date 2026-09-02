#!/usr/bin/env python3
"""Substitution tolerance: aggregate results/ablations/substitution/<run>.npz into
results/ablations/substitution_tolerance.csv and draw results/ablations/figures/substitution_depth.png.

Each .npz (from analysis/probes/substitution_eval.py) holds, for one checkpoint, the reference
per-document CE sums over a fixed set of test micro-batches and the same sums under every
perturbation arm. Deltas are per-token CE (arm minus reference) averaged over loss-masked tokens,
converted to BPB with the divisor of the run's tokenizer (CE / divisor, divisor includes ln 2).
Confidence intervals are 95% percentile bootstraps over documents (segments between EOD tokens),
2000 resamples, RNG seeded from a hash of the resampled values so the file regenerates identically.

Pair rows (regime "pair") give temporal minus full-MoE delta for matched checkpoints scored on the
same cached batches (the aggregator asserts the token streams are identical), with the bootstrap
taken jointly over documents so the difference is paired.

    $PY analysis/residency/substitution_tolerance.py [--no-caption]
"""
import csv
import glob
import hashlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS  # noqa: E402

SRC = os.path.join(ABLATIONS, "substitution")
CSV = os.path.join(ABLATIONS, "substitution_tolerance.csv")
FIG = os.path.join(ABLATIONS, "figures")
BOOT = 2000
DIVISOR = {"pythia": 2.9780, "tok16k": 2.7568}
FULL, TEMPORAL = "#2F6DB5", "#2E8B57"


def meta(run):
    """budget, grain, seed, corpus from the run name."""
    if run.startswith("flame38m_"):
        g = int(re.search(r"_g(\d)_", run).group(1))
        s = re.search(r"_s(\d)$", run)
        return "1e18", g, (int(s.group(1)) if s else 1), "pythia"
    if run.endswith("_1e17"):
        return "1e17", 3, 1, "tok16k"
    if run.endswith("_1e19"):
        return "1e19", (3 if "fine" in run or "g3" in run else 1), 1, "pythia"
    raise ValueError(run)


def _rng_for(vals):
    h = hashlib.blake2b(np.ascontiguousarray(vals, dtype=np.float64).tobytes(), digest_size=8)
    return np.random.default_rng(int(h.hexdigest(), 16))


def boot_ratio(num, den):
    """Bootstrap over documents of sum(num)/sum(den). Returns (point, lo95, hi95)."""
    n = len(num)
    rng = _rng_for(np.concatenate([num, den]))
    idx = rng.integers(0, n, size=(BOOT, n))
    stats = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def rows_for(d, run):
    budget, grain, seed, corpus = meta(run)
    div = DIVISOR[corpus]
    ntok = d["doc_ntok"].astype(np.float64)
    ref = d["ref_doc_sum"].astype(np.float64)
    out = []
    for i, arm in enumerate(d["arm"]):
        cond, gate, frac, layer = str(arm).split("|")
        num = d["doc_sum"][i].astype(np.float64) - ref
        pt, lo, hi = boot_ratio(num, ntok)
        out.append({
            "run": run, "budget": budget, "regime": str(d["regime"]), "grain": grain,
            "experts": int(d["E"]), "topk": int(d["k"]), "seed": seed, "condition": cond,
            "gate": gate, "fraction": frac, "layer": layer, "n_tokens": int(ntok.sum()),
            "n_docs": len(ntok), "ref_ce": f"{float(d['ref_mean']):.6f}",
            "ref_bpb": f"{float(d['ref_mean']) / div:.4f}",
            "delta_ce": f"{pt:.6f}", "delta_ce_lo95": f"{lo:.6f}", "delta_ce_hi95": f"{hi:.6f}",
            "delta_bpb": f"{pt / div:.4f}", "delta_bpb_lo95": f"{lo / div:.4f}",
            "delta_bpb_hi95": f"{hi / div:.4f}", "hook_calls": int(d["hook_calls"][i]),
            "subs_per_token_layer": f"{float(d['subs_per_token'][i]):.3f}",
            "replay_drift": f"{float(d['replay_drift']):.2e}",
            "_num": num, "_den": ntok, "_div": div,
        })
    return out


PAIRS = [  # temporal, full
    ("flame38m_g1_temporal", "flame38m_g1_moe"), ("flame38m_g1_temporal_s2", "flame38m_g1_moe_s2"),
    ("flame38m_g1_temporal_s3", "flame38m_g1_moe_s3"),
    ("flame38m_g3_temporal", "flame38m_g3_moe"), ("flame38m_g3_temporal_s2", "flame38m_g3_moe_s2"),
    ("flame38m_g3_temporal_s3", "flame38m_g3_moe_s3"),
    ("g3_tmoe_s2_1e17", "g3_moe_s2_1e17"), ("g1_tmoe_coarse_1e19", "moe_coarse_1e19"),
]


def pair_rows(rows, data):
    by = {}
    for r in rows:
        by.setdefault(r["run"], {})[(r["condition"], r["gate"], r["fraction"], r["layer"])] = r
    out = []
    for t, f in PAIRS:
        if t not in by or f not in by:
            continue
        if str(data[t]["tokens_sha256"]) != str(data[f]["tokens_sha256"]):
            raise RuntimeError(f"pair {t}/{f} scored different token streams; pairing is invalid")
        for key, rt in by[t].items():
            rf = by[f].get(key)
            if rf is None:
                continue
            num = rt["_num"] - rf["_num"]
            pt, lo, hi = boot_ratio(num, rt["_den"])
            div = rt["_div"]
            out.append({**{k: rt[k] for k in ("budget", "grain", "experts", "topk", "seed",
                                                "condition", "gate", "fraction", "layer",
                                                "n_tokens", "n_docs")},
                        "run": f"{t}-minus-{f}", "regime": "pair", "ref_ce": "", "ref_bpb": "",
                        "delta_ce": f"{pt:.6f}", "delta_ce_lo95": f"{lo:.6f}",
                        "delta_ce_hi95": f"{hi:.6f}", "delta_bpb": f"{pt / div:.4f}",
                        "delta_bpb_lo95": f"{lo / div:.4f}", "delta_bpb_hi95": f"{hi / div:.4f}",
                        "hook_calls": "", "subs_per_token_layer": "", "replay_drift": ""})
    return out


COLS = ["run", "budget", "regime", "grain", "experts", "topk", "seed", "condition", "gate",
        "fraction", "layer", "n_tokens", "n_docs", "ref_ce", "ref_bpb", "delta_ce", "delta_ce_lo95",
        "delta_ce_hi95", "delta_bpb", "delta_bpb_lo95", "delta_bpb_hi95", "hook_calls",
        "subs_per_token_layer", "replay_drift"]


def write_csv(rows):
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, "w", newline="") as fh:
        fh.write("# substitution tolerance: per-token CE delta (arm minus unperturbed) on cached "
                 "test micro-batches; BPB = CE/2.9780 (pythia-50k runs) or CE/2.7568 (tok16k 1e17); "
                 "95% percentile bootstrap over documents, 2000 resamples; "
                 "producer analysis/residency/substitution_tolerance.py from "
                 "analysis/probes/substitution_eval.py .npz records\n")
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {CSV} ({len(rows)} rows)")


def figure(rows, paper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if paper:
        plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 10,
                             "ytick.labelsize": 10.5, "legend.fontsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=False)
    marks = {"1e17": ("s", ":"), "1e18": ("o", "-"), "1e19": ("^", "--")}
    for ax, grain, title in ((axes[0], 1, "coarse, 6 of 64"), (axes[1], 3, "fine, 18 of 192")):
        for regime, color in (("full", FULL), ("temporal", TEMPORAL)):
            for budget, (mk, ls) in marks.items():
                sel = [r for r in rows if r["regime"] == regime and r["grain"] == grain
                       and r["budget"] == budget and r["layer"].startswith("L")
                       and r["condition"] == "random" and r["gate"] == "own"
                       and r["fraction"] == "matched"]
                if not sel:
                    continue
                layers = sorted({int(r["layer"][1:]) for r in sel})
                L = max(layers)
                ys = []
                for ln in layers:
                    v = [float(r["delta_bpb"]) for r in sel if int(r["layer"][1:]) == ln]
                    ys.append((ln / L, np.mean(v), np.min(v), np.max(v), len(v)))
                x = [y[0] for y in ys]
                ax.plot(x, [y[1] for y in ys], marker=mk, ls=ls, color=color, ms=4.5,
                        label=f"{'full MoE' if regime == 'full' else 'temporal'}, {budget}"
                              + (f" ({ys[0][4]} seeds)" if ys[0][4] > 1 else ""))
                if ys[0][4] > 1:
                    ax.fill_between(x, [y[2] for y in ys], [y[3] for y in ys], color=color, alpha=0.15,
                                    lw=0)
        ax.set_title(title)
        ax.set_xlabel("relative depth of the perturbed layer")
        ax.grid(alpha=0.25); ax.set_axisbelow(True)
    axes[0].set_ylabel("BPB increase from one perturbed layer")
    handles, labels = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h; labels.append(l)
    order = sorted(labels, key=lambda l: (l.startswith("temporal"), l))
    fig.legend([handles[l] for l in order], order, frameon=False, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    if not paper:
        fig.suptitle("Substitution tolerance by depth: at one layer, replace one of six active experts\n"
                     "(three of eighteen) with a random other expert at its own router weight. "
                     "Bands span the seeds.", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    name = f"substitution_depth{'_nocaption' if paper else ''}.png"
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight")
    print(f"wrote {name}")


def main():
    paper = "--no-caption" in sys.argv
    files = sorted(glob.glob(os.path.join(SRC, "*.npz")))
    if not files:
        sys.exit(f"no records in {SRC}")
    data, rows = {}, []
    for p in files:
        run = os.path.basename(p)[:-4]
        d = load(p)
        data[run] = d
        rows += rows_for(d, run)
    rows += pair_rows(rows, data)
    write_csv(rows)
    figure(rows, paper)


if __name__ == "__main__":
    main()
