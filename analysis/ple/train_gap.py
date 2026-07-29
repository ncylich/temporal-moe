#!/usr/bin/env python3
"""Recover the train/eval gap at each eval point from a cell's log.

Cells launched after train_ple.py gained per-eval train-loss logging emit `train_lm` directly. The
full-rank cell was already running when that was added, so its gap has to come from the log: the
mean of the `[step]` training-loss lines that fall between one eval and the next. That is a coarser
estimate than the trainer's own accumulator -- it samples every 20th step rather than every step --
and it is labelled as such rather than presented as equivalent.

Both are converted to BPB by the same divisor so the gap is in the units the gates use.

    train_gap.py /tmp/cell_full.log
"""

import re, sys, json, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe_paths import DATA_DIR   # noqa: E402

STEP = re.compile(r"\[step (\d+)\] tok=([\d.]+)M lm=([\d.]+)")
EVAL = re.compile(r"\[eval\].*?tok=(\d+)M BPB=([\d.]+)")
EVAL_NEW = re.compile(r"train_lm=([\d.]+)")


def main(path):
    D = json.load(open(os.path.join(DATA_DIR, "bpb_slice_meta.json")))["divisor_D"]
    steps, evals = [], []
    for line in open(path, errors="ignore"):
        m = STEP.search(line)
        if m:
            steps.append((float(m.group(2)), float(m.group(3))))
        m = EVAL.search(line)
        if m:
            n = EVAL_NEW.search(line)
            evals.append((int(m.group(1)), float(m.group(2)),
                          float(n.group(1)) if n else None))
    if not evals:
        print("no eval lines yet"); return
    print(f"{'tok':>6s} {'eval_bpb':>10s} {'train_lm':>10s} {'train_bpb':>10s} {'gap':>10s}  source")
    prev = 0.0
    for tok, bpb, tl in evals:
        if tl is None:
            win = [l for t, l in steps if prev < t <= tok]
            tl_use = sum(win) / len(win) if win else float("nan")
            src = f"log mean of {len(win)} step lines"
        else:
            tl_use, src = tl, "trainer accumulator (every step)"
        print(f"{tok:>5d}M {bpb:>10.6f} {tl_use:>10.6f} {tl_use/D:>10.6f} {tl_use/D - bpb:>+10.6f}  {src}")
        prev = tok
    print("\ngap = train_bpb - eval_bpb. Falling train with flat or rising eval is the "
          "pre-registered memorization falsifier; report it at once, not at the cell boundary.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cell_full.log")
