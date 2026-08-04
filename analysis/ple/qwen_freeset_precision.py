#!/usr/bin/env python3
"""Is the tail-only free set really better than the OLMoE recipe, or is 0.0025 BPB inside the noise?

Section 5B claims last-4 beats {0,1,L-2,L-1} at matched budget, on a gap of 0.0025 BPB out of a
0.056 constraint. That gap is small enough that it needs an error bar before anyone acts on it, and
a single 24-sequence measurement does not provide one.

Each free set is scored on THREE DISJOINT blocks of the held-out slice. The blocks give a direct
estimate of sampling spread, and because every set sees the same three blocks, the paired difference
between two sets can be computed per block -- which is far more sensitive than comparing two means,
since block-to-block difficulty is common to both arms and cancels.

Reported per set: mean damage, the spread across blocks, and for the two sets in contention the
paired per-block difference, which is the quantity the claim actually depends on.

    qwen_freeset_precision.py --block 32 --blocks 3
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_qwen as RQ                                        # noqa: E402
from qwen_sweep import score                                       # noqa: E402

DATA = "/workspace/qwen35-adapt/data"
OUT = "/workspace/qwen35-adapt/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--R", type=int, default=8)
    A = ap.parse_args()
    meta = json.load(open(f"{DATA}/bpb_slice_meta_qwen.json"))
    D = meta["divisor_D"]
    model, tok = RQ.load_model()
    L = model.config.num_hidden_layers
    ALL = list(range(L))

    ids = torch.load(f"{DATA}/bpb_slice_ids_qwen.pt", weights_only=False)
    blocks = [[ids[i:i + 1].long() for i in range(b * A.block, (b + 1) * A.block)]
              for b in range(A.blocks)]
    print(f"  {A.blocks} disjoint blocks x {A.block} sequences, R={A.R}", flush=True)

    SETS = {"free_none": None, "free_first2": [0, 1], "free_last2": [L - 2, L - 1],
            "free_first2_last2": [0, 1, L - 2, L - 1], "free_last4": list(range(L - 4, L)),
            "free_last8": list(range(L - 8, L))}
    # per-block free baseline, so damage is measured against the same text the cell was scored on
    base = [score(model, bl, D, ALL, A.R)[0] for bl in blocks]
    print(f"  free baseline per block: {' '.join(f'{b:.6f}' for b in base)}", flush=True)

    dmg = {}
    for name, fs in SETS.items():
        t0 = time.time()
        vals = [score(model, bl, D, fs, A.R)[0] - base[i] for i, bl in enumerate(blocks)]
        dmg[name] = vals
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {name:20} damage {m:+.6f} +- {sd:.6f}  "
              f"[{' '.join(f'{v:+.6f}' for v in vals)}]  ({time.time()-t0:.0f}s)", flush=True)

    print("\n  === paired per-block differences (the claim under test) ===", flush=True)
    pairs = [("free_last4", "free_first2_last2", "4 freed: tail-only vs OLMoE recipe"),
             ("free_last2", "free_first2", "2 freed: tail vs head")]
    verdicts = []
    for a, b, lbl in pairs:
        d = [dmg[a][i] - dmg[b][i] for i in range(A.blocks)]
        m = statistics.mean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        allsame = all(x < 0 for x in d) or all(x > 0 for x in d)
        print(f"  {lbl}")
        print(f"    per-block {a} - {b}: {' '.join(f'{x:+.6f}' for x in d)}")
        print(f"    mean {m:+.6f} +- {sd:.6f}   sign consistent across blocks: {allsame}")
        print(f"    -> {a} is {'BETTER' if m < 0 else 'WORSE'} by {abs(m):.6f} BPB"
              f"{'' if allsame else '  (INCONSISTENT -- do not rely on this)'}")
        verdicts.append((lbl, a, b, m, sd, allsame))

    path = os.path.join(OUT, "qwen35_freeset_precision.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"# Free-set damage on {A.blocks} disjoint blocks of {A.block} sequences each, "
                    f"R={A.R}, Qwen3.5-35B-A3B-Base. Damage is measured against a per-block free "
                    f"baseline, so block difficulty cancels. The paired rows are the test that "
                    f"matters: block-to-block difficulty is common to both arms, so the per-block "
                    f"difference is far more sensitive than comparing two means. "
                    f"Producer: analysis/ple/qwen_freeset_precision.py"])
        w.writerow(["free_set", "mean_damage", "sd_across_blocks"] +
                   [f"block{i}" for i in range(A.blocks)])
        for name, vals in dmg.items():
            w.writerow([name, f"{statistics.mean(vals):.6f}",
                        f"{statistics.stdev(vals) if len(vals) > 1 else 0:.6f}"] +
                       [f"{v:.6f}" for v in vals])
        w.writerow([])
        w.writerow(["paired_comparison", "mean_diff", "sd_diff", "sign_consistent"])
        for lbl, a, b, m, sd, ok in verdicts:
            w.writerow([f"{a} minus {b}", f"{m:.6f}", f"{sd:.6f}", ok])
    print(f"\n[write] {path}", flush=True)
    print("=== PRECISION COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
