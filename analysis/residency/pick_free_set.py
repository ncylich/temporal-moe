#!/usr/bin/env python3
"""Choose which free-set cell to extend, from the cells on disk.

Prints one line: `tag free_set bpb rationale`, for a shell `read` to split.

The first two rules were stated before any of the cells they rank had run. The third replaced an
inherited constant after that constant produced a wrong answer, which is worth being explicit about:
it is a change made with the results visible.

**Only data-seed-0 cells compete.** A replicate trains on a different draw of the corpus. Ranking it
alongside the others would let the luckier draw win and then be reported as a better configuration,
which is the error the replicate exists to detect.

**A tie inside the noise bar goes to the cheaper cell.** Sorting on BPB alone would hand the win to
+175% resident memory over +131% for a difference that is not an effect. Resident expert slots are
the currency the residency thesis is denominated in, so they are the tie-break, not an afterthought
to mention in the write-up.

**The bar is measured, not inherited.** Run with the published 2 sigma = 0.012 this script called a
0.0115 gap a tie and sent the 200M extension to the second-best cell. That bar is wrong for this
comparison and the program knew it: ple_RESULTS.md §6 records that sigma was estimated by scoring the
BASE model on DISJOINT data subsamples, i.e. data-slice noise, while every arm is scored on the same
fixed 256-pack subset with a bitwise-deterministic eval -- so subsample variance contributes nothing
to a difference between arms. §6 called it "conservative by roughly 2.4x" and retained it anyway.

Two estimators are available on disk and they disagree by two orders of magnitude, so the larger is
taken:

  replicate spread   ce_free_0_1_15 vs ce_free_0_1_15_ds1, same configuration on different corpus
                     permutations: 0.000004 at 50M. One pair at one point, and evidently fortunate.

  upward step        the largest INCREASE along any cell's eval curve. Training does not get worse
                     in expectation, so once a curve has flattened an increase is noise and nothing
                     else -- it needs no convergence test, no held-out replicate and no assumption
                     about what the eval slice measures. ce_free_0_1_14_15_250M rises 0.000199
                     between 130M and 140M, which puts the real precision on a cell's BPB near 2e-4,
                     roughly 60x below the published bar rather than 3000x.

Both are reported. The argument passed on the command line is only the fallback when neither is
computable.

    pick_free_set.py <data-dir> <fallback-bar-in-bpb>
"""
import glob
import json
import os
import sys


def _measured_bar(all_cells, fallback):
    """Spread between same-configuration cells trained on different corpus permutations.

    Returns (bar, description). Each pair contributes one absolute difference; the largest is used
    rather than the mean, because one pair is not a distribution and taking the maximum is the
    conservative reading of a tiny sample.
    """
    by_config = {}
    for c in all_cells:
        by_config.setdefault((c["free_set"], c["tokens"]), []).append(c)
    diffs = []
    for (fs, tok), group in sorted(by_config.items()):
        seeds = {c["data_seed"] for c in group}
        if len(group) < 2 or len(seeds) < 2:
            continue
        lo, hi = min(c["bpb"] for c in group), max(c["bpb"] for c in group)
        diffs.append((hi - lo, fs, tok, len(group)))
    rep = max(diffs)[0] if diffs else None

    # Largest increase along any eval curve. A cell's BPB should fall monotonically in expectation,
    # so an increase is noise -- and unlike the replicate spread this needs no second run, so it is
    # available for every cell rather than the one pair that happens to exist.
    up, up_where = 0.0, None
    for c in all_cells:
        curve = c.get("curve") or []
        for prev, nxt in zip(curve, curve[1:]):
            d = nxt["bpb"] - prev["bpb"]
            if d > up:
                up, up_where = d, f"{c['tag']} {prev['tok'] // 10**6}M->{nxt['tok'] // 10**6}M"

    cands = [(v, n) for v, n in ((rep, f"replicate spread ({len(diffs)} pair(s))"),
                                 (up or None, f"upward step [{up_where}]")) if v]
    if not cands:
        return fallback, f"published 2sigma {fallback} (no replicate pair, no curve reversal)"
    best = max(cands)
    other = ", ".join(f"{v:.6f} {n}" for v, n in cands if (v, n) != best)
    return best[0], (f"{best[0]:.6f} from {best[1]}"
                     + (f"; also saw {other}" if other else "")
                     + f"; published bar was {fallback}")


def main(data, bar):
    cells, every = [], []
    for p in glob.glob(os.path.join(data, "ple_*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if "final_bpb" not in r or str(r.get("rank")) != "off":
            continue
        # Two spellings of the same thing: --free-set names the layers, --free-layers N means the
        # first N. ce_free2, the first cell of this line, used the second, so a filter that reads
        # only free_set drops it -- and it drops it silently, leaving a one-candidate field that
        # still looks like a comparison.
        layers = [x.strip() for x in (r.get("free_set") or "").split(",") if x.strip()] \
            or [str(i) for i in range(r.get("free_layers", 0))]
        if not layers:
            continue
        n = len(layers)
        c = {"bpb": r["final_bpb"], "slots": (16 - n) * 8 + n * 64, "tag": r["tag"],
             "free_set": ",".join(layers), "tokens": r["train_tokens"],
             "data_seed": r.get("data_seed", 0), "curve": r.get("curve") or []}
        every.append(c)
        # Only seed-0 cells compete; the replicates still count toward the bar below.
        if c["data_seed"] == 0:
            cells.append(c)
    if not cells:
        print("NONE - - no-free-set-cells-found")
        return 1

    bar, bar_note = _measured_bar(every, bar)
    print(f"# bar: {bar_note}", file=sys.stderr)

    cells.sort(key=lambda c: c["bpb"])
    best = cells[0]
    tied = [c for c in cells if c["bpb"] - best["bpb"] <= bar]
    pick = min(tied, key=lambda c: (c["slots"], c["bpb"]))
    why = (f"{len(cells)}cells,best={best['tag']}@{best['bpb']:.6f},"
           f"tied={len(tied)},bar={bar:.6f},on=" + ("memory" if pick is not best else "bpb"))
    for c in cells:
        mark = "<-" if c is pick else "  "
        print(f"# {mark} {c['tag']:24s} bpb={c['bpb']:.6f} slots={c['slots']:3d} "
              f"(+{c['slots'] / 128 * 100 - 100:5.1f}%) tokens={c['tokens'] / 1e6:.0f}M",
              file=sys.stderr)
    print(f"{pick['tag']} {pick['free_set']} {pick['bpb']:.6f} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], float(sys.argv[2])))
