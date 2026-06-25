#!/usr/bin/env python3
"""Build the @1e17 and @1e16 BPB frontiers from the sweep runs and check acceptance criteria 1 & 2.

Each shape's 1e17 run yields BOTH points: final val = 1e17; val at iters/10 = 1e16 (per plan, read
from the early checkpoint of the 1e17 run). Reports per-shape BPB and pass/fail vs the bars.

Usage: summarize.py <run_name_prefix>   e.g. summarize.py v16k_sweep
  expects runs named <prefix>_s1_1e17 ... <prefix>_s6_1e17
"""
import sys, os, re, json, math

BPB_DIVISOR = float(os.environ.get("BPB_DIVISOR", "2.7568"))
BAR_1e17 = 1.645   # CE 4.9 @50k -> BPB
BAR_1e16 = 2.149   # CE 6.4 @50k -> BPB
RUNS = "/workspace/FLAME-MoE/results/phase0/runs"
SHAPES = ["s1", "s2", "s3", "s4", "s5", "s6"]

def parse(run_dir):
    log = os.path.join(run_dir, "train.log")
    meta = open(os.path.join(run_dir, "run.meta")).read() if os.path.exists(os.path.join(run_dir,"run.meta")) else ""
    m = re.search(r"iters=(\d+)", meta); total = int(m.group(1)) if m else None
    it_1e16 = round(total/10) if total else None
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    vals = []
    for line in txt.splitlines():
        if "validation loss at" in line and "lm loss value:" in line:
            it = re.search(r"iteration\s+(\d+)", line); lm = re.search(r"lm loss value:\s*([0-9.Ee+\-]+)", line)
            if lm: vals.append((int(it.group(1)) if it else None, float(lm.group(1))))
    final = vals[-1][1] if vals else None
    v1e16 = None
    if it_1e16 and vals:
        cand = sorted([(abs((i or 0)-it_1e16), l, i) for i,l in vals if i is not None])
        if cand: v1e16 = (cand[0][1], cand[0][2])
    return final, v1e16

def bpb(ce): return ce/BPB_DIVISOR if ce else None

def main():
    prefix = sys.argv[1]
    print(f"{'shape':5} {'1e17 CE':>8} {'1e17 BPB':>9} {'1e16 CE':>8} {'1e16 BPB':>9}")
    f17, f16 = {}, {}
    for s in SHAPES:
        d = os.path.join(RUNS, f"{prefix}_{s}_1e17")
        if not os.path.isdir(d): print(f"{s:5} (missing)"); continue
        final, v16 = parse(d)
        b17 = bpb(final); b16 = bpb(v16[0]) if v16 else None
        f17[s] = b17; f16[s] = b16
        print(f"{s:5} {final or 0:>8.4f} {b17 or 0:>9.4f} "
              f"{(v16[0] if v16 else 0):>8.4f} {b16 or 0:>9.4f}")
    # criterion 1
    best17 = min((v for v in f17.values() if v), default=None)
    best16 = min((v for v in f16.values() if v), default=None)
    print(f"\nCriterion 1: best @1e17 BPB={best17} (bar<= {BAR_1e17})  ->", "PASS" if best17 and best17<=BAR_1e17 else "FAIL")
    print(f"            best @1e16 BPB={best16} (bar<= {BAR_1e16})  ->", "PASS" if best16 and best16<=BAR_1e16 else "FAIL")
    # criterion 2: parabola @1e17 (min in s1-s3), monotone-ish @1e16
    ok17 = [s for s in SHAPES if f17.get(s)]
    if len(ok17) >= 3:
        mins = min(ok17, key=lambda s: f17[s])
        par = mins in ("s1","s2","s3")
        print(f"Criterion 2: @1e17 min at {mins} (want s1-s3) ->", "PASS" if par else "FAIL")
    print(json.dumps({"f1e17": f17, "f1e16": f16, "bar17": BAR_1e17, "bar16": BAR_1e16}))

if __name__ == "__main__":
    main()
