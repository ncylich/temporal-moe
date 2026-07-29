#!/usr/bin/env python3
"""Collect the Phase 0 parity result into results/ablations/ple_parity.csv.

Reads the three run JSONs produced by the parity script and reports, without a verdict:

  floor  = |A1 - A2|, two identical flag-off runs of the new trainer. This is the run-to-run
           non-determinism of the whole stack (atomics in the expert index_add_, the triton
           residency scan), not an error bar on anything the code changed.
  delta  = |mean(A1, A2) - B|, the new trainer with the flag off against the adaptation program's
           train_bakeoff.py arm C, run unmodified at the same budget.

Both numbers are printed. Whether delta sits inside the floor is the reader's call, which is what
PLE_PLAN.md §4 item 3 asks for ("report both numbers, not a verdict").
"""

import csv, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe_paths import DATA_DIR            # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                 # noqa: E402


def load(path, key="final_bpb"):
    with open(path) as f:
        return json.load(f)


def main():
    a1 = load(os.path.join(DATA_DIR, "ple_parity_off_1.json"))
    a2 = load(os.path.join(DATA_DIR, "ple_parity_off_2.json"))
    b = load(os.path.join(DATA_DIR, "bakeoff_parity_bakeC.json"))
    b2_path = os.path.join(DATA_DIR, "bakeoff_parity_bakeC2.json")
    b2 = load(b2_path) if os.path.exists(b2_path) else None

    f1, f2, fb = a1["final_bpb"], a2["final_bpb"], b["final_bpb"]
    floor = abs(f1 - f2)
    mean_a = (f1 + f2) / 2
    # The A-pair floor is one binary run twice; the A-vs-B comparison is across two programs, where
    # allocator state and cuBLAS kernel selection can differ. |B1-B2| is the reference's own spread,
    # which is the comparable quantity, and mean(B) is the comparable centre.
    if b2 is not None:
        fb2 = b2["final_bpb"]
        ref_floor = abs(fb - fb2)
        mean_b = (fb + fb2) / 2
        delta = abs(mean_a - mean_b)
    else:
        fb2 = ref_floor = mean_b = None
        delta = abs(mean_a - fb)

    rows = [
        {"arm": "A1", "what": "new trainer, PLE flag off", "impl": "analysis/ple/train_ple.py",
         "tokens": a1["train_tokens"], "bpb": f"{f1:.6f}", "swap": f"{a1['final_swap']:.6f}",
         "entropy": f"{a1['final_entropy']:.6f}", "divisor": a1["divisor"]},
        {"arm": "A2", "what": "new trainer, PLE flag off, identical to A1",
         "impl": "analysis/ple/train_ple.py",
         "tokens": a2["train_tokens"], "bpb": f"{f2:.6f}", "swap": f"{a2['final_swap']:.6f}",
         "entropy": f"{a2['final_entropy']:.6f}", "divisor": a2["divisor"]},
        {"arm": "B", "what": "adaptation program arm C, unmodified",
         "impl": "olmoe-adapt/scripts/train_bakeoff.py",
         "tokens": b["train_tokens"], "bpb": f"{fb:.6f}", "swap": f"{b['final_swap']:.6f}",
         "entropy": f"{b['final_entropy']:.6f}", "divisor": b["divisor"]},
        {"arm": "floor", "what": "|A1 - A2| run-to-run non-determinism", "impl": "",
         "tokens": "", "bpb": f"{floor:.6f}", "swap": "", "entropy": "", "divisor": ""},
        {"arm": "delta", "what": "|mean(A1,A2) - B| new-code-off vs reference", "impl": "",
         "tokens": "", "bpb": f"{delta:.6f}", "swap": "", "entropy": "", "divisor": ""},
    ]
    path = os.path.join(ABLATIONS, "ple_parity.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"A1 (new, flag off)      BPB = {f1:.6f}")
    print(f"A2 (new, flag off)      BPB = {f2:.6f}")
    print(f"B  (reference arm C)    BPB = {fb:.6f}")
    print(f"floor |A1-A2|           = {floor:.6f}")
    print(f"delta |mean(A)-B|       = {delta:.6f}")
    print(f"delta / floor           = {delta/floor:.2f}x" if floor else "floor is exactly 0")
    print("wrote", path)


if __name__ == "__main__":
    main()
