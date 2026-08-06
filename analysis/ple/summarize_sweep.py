#!/usr/bin/env python3
"""Tabulate an LR sweep and apply the pre-registered selection rules.

Written before the grid finished, so the rules are fixed in advance rather than fitted to whatever
came out. The programme has three separate cases on record of a number being read the way it was
hoped to read; a decision procedure committed ahead of the data is the cheapest guard against a
fourth.

Selection, in order:

  1. PRUNE diverged points -- training loss rising over the run, or any checkpoint's BPB above the
     untrained constrained baseline. Above the usable ceiling.
  2. PRUNE dead points -- final BPB within `--noise` of the untrained constrained baseline. Below the
     usable floor; the LR is not moving the model.
  3. RANK survivors on final BPB, lower better.
  4. TIE-BREAK inside `--noise` on the 5M checkpoint: prefer whichever got there soonest. Cheaper to
     run and less likely to be riding a lucky final eval.

Reports effective expert count alongside, because it is the mechanism BPB only implies: a run that
improves BPB while its minimum eff_load collapses is buying loss with routing degeneracy and should
be looked at rather than selected.

    summarize_sweep.py --glob '/tmp/sweep_olmoe_lr*.log' --baseline 0.842848
"""
import argparse
import glob
import re


def parse(path):
    txt = open(path, errors="ignore").read()
    evals = re.findall(r"tok=(\d+)M BPB=([0-9.]+).*?eff_load\[min/med/max\]=([0-9.]+)/([0-9.]+)/([0-9.]+)",
                       txt)
    lms = [float(x) for x in re.findall(r"lm=([0-9.]+)", txt)]
    lr = re.search(r"_lr([0-9e.\-]+?)\.log", path)
    return {
        "lr": lr.group(1) if lr else path,
        "curve": [(int(t), float(b)) for t, b, *_ in evals],
        "eff": [(float(a), float(m), float(x)) for _, _, a, m, x in evals],
        "lm_first": sum(lms[:5]) / max(1, len(lms[:5])),
        "lm_last": sum(lms[-5:]) / max(1, len(lms[-5:])),
        "done": "[DONE]" in txt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/tmp/sweep_olmoe_lr*.log")
    ap.add_argument("--baseline", type=float, required=True,
                    help="untrained CONSTRAINED BPB for this model (the floor test)")
    ap.add_argument("--free", type=float, default=None,
                    help="untrained UNCONSTRAINED BPB; enables a real recovery %%")
    ap.add_argument("--noise", type=float, default=0.003)
    A = ap.parse_args()

    runs = [parse(p) for p in sorted(glob.glob(A.glob))]
    runs = [r for r in runs if r["curve"]]
    if not runs:
        print("  no completed checkpoints yet")
        return

    print(f"  untrained constrained baseline {A.baseline:.6f}   noise band {A.noise}\n")
    print(f"  {'lr':>8} {'5M':>10} {'10M':>10} {'15M':>10}  {'eff min/med':>12}  {'lm drift':>9}  verdict")
    survivors = []
    for r in runs:
        c = dict(r["curve"])
        cells = "".join(f"{c.get(t, float('nan')):>10.6f}" if t in c else f"{'-':>10}"
                        for t in (5, 10, 15))
        final = r["curve"][-1][1]
        drift = r["lm_last"] - r["lm_first"]
        eff = r["eff"][-1] if r["eff"] else (0, 0, 0)
        if final > A.baseline or drift > 0.05:
            v = "PRUNE: diverged/above baseline"
        elif abs(final - A.baseline) < A.noise:
            v = "PRUNE: dead, within noise of untrained"
        else:
            v = "keep"
            survivors.append((final, r["curve"][0][1], r["lr"], eff))
        print(f"  {r['lr']:>8}{cells}  {eff[0]:>5.1f}/{eff[1]:<6.1f}  {drift:>+9.4f}  {v}")

    if not survivors:
        print("\n  ALL POINTS PRUNED -- the grid is in the wrong decade. Shift it, do not interpolate.")
        return
    survivors.sort()
    best = survivors[0]
    close = [s for s in survivors if s[0] - best[0] < A.noise]
    if len(close) > 1:
        close.sort(key=lambda s: s[1])          # tie-break on the 5M checkpoint
        best = close[0]
        print(f"\n  {len(close)} points tie within {A.noise}; tie-broken on the 5M checkpoint")
    print(f"\n  WINNER lr={best[2]}  final BPB {best[0]:.6f}  "
          f"(5M {best[1]:.6f}, eff_load min {best[3][0]:.1f})")
    if A.free is not None:
        # recovery = share of the constraint's damage removed, not a share of BPB
        print(f"  recovery = (untrained_constrained - trained)/(untrained_constrained - free)"
              f" = {(A.baseline-best[0])/(A.baseline-A.free)*100:.1f}%")
        print(f"  NOTE: measured against the UNTRAINED free model, so this is a LOWER BOUND --\n"
              f"  a matched null moves the ceiling down and the true figure up.")
    else:
        print(f"  BPB improvement over untrained constrained: {A.baseline-best[0]:+.6f}")
    print("=== SWEEP SUMMARY COMPLETE ===")


if __name__ == "__main__":
    main()
