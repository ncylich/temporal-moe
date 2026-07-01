#!/usr/bin/env python3
"""Plot the G=3 fine-grained IsoFLOP parabolas (MoE + temporal) vs the G=1 baseline + dense floor.

Reads measured BPB for the g3_* runs by calling parse_run.py on each run dir (so BPB_DIVISOR is
honored). Overlays the already-measured G=1 reference numbers (from results/phase0/*.md). Plots one
panel per budget (1e16, 1e17): BPB vs active non-embedding params N. Lower BPB is better.

Usage: BPB_DIVISOR=2.7600 .venv/bin/python scripts/phase0/plot_g3.py
"""
import os, sys, json, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/workspace/FLAME-MoE"
RUNS = f"{ROOT}/results/phase0/runs"
DIV = os.environ.get("BPB_DIVISOR", "2.7600")

# active non-embedding params (M). G1 from shapes.py (baseline); G3 from GRAIN=3 shapes.py.
N_G1 = {"sm1": 0.77, "s0": 1.36, "s1": 3.81, "s2": 8.12, "s3": 14.77}
N_G3 = {"sm1": 0.81, "s0": 1.42, "s1": 3.91, "s2": 8.23, "s3": 15.09}

# G=1 baseline reference BPB (measured; results/phase0/RESULTS.md, TEMPORAL_RESULTS.md, DENSE_BASELINES.md)
BASE = {
    "1e16": {
        "dense":    {"sm1": 1.534, "s0": 1.519, "s1": 1.591, "s2": 1.848},
        "moe":      {"sm1": 1.478, "s0": 1.447, "s1": 1.540, "s2": 1.819},
        "temporal": {"sm1": 1.4891, "s0": 1.4599, "s1": 1.5488, "s2": 1.8260},
    },
    "1e17": {
        "dense":    {"s1": 1.361, "s2": 1.341, "s3": 1.408},
        "moe":      {"s1": 1.284, "s2": 1.269, "s3": 1.289},
        "temporal": {"s1": 1.3039, "s2": 1.2821, "s3": 1.3073},
    },
}

# which shapes are in each budget's bracket for the G3 sweep
CELLS = {
    ("moe", "1e16"):      ["sm1", "s0", "s1"],
    ("moe", "1e17"):      ["s1", "s2", "s3"],
    ("temporal", "1e16"): ["sm1", "s0", "s1"],
    ("temporal", "1e17"): ["s1", "s2", "s3"],
}

def run_bpb(name):
    d = f"{RUNS}/{name}"
    if not os.path.isdir(d):
        return None
    env = dict(os.environ, BPB_DIVISOR=DIV)
    try:
        out = subprocess.run([f"{ROOT}/.venv/bin/python", f"{ROOT}/scripts/phase0/parse_run.py", d],
                             capture_output=True, text=True, env=env).stdout
        for line in out.splitlines():
            if line.startswith("{"):
                o = json.loads(line)
                if o.get("final_val_bpb") and not o.get("nan"):
                    return o["final_val_bpb"]
    except Exception as e:
        print(f"parse {name}: {e}")
    return None

def collect():
    res = {}
    for (mt, budget), shapes in CELLS.items():
        res[(mt, budget)] = {}
        for s in shapes:
            name = f"g3_{'tmoe' if mt=='temporal' else 'moe'}_{s}_{budget}"
            b = run_bpb(name)
            if b is not None:
                res[(mt, budget)][s] = b
    return res

def main():
    g3 = collect()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, budget in zip(axes, ["1e16", "1e17"]):
        # baseline reference (G1) — dashed
        for mt, color, label in [("dense", "tab:gray", "dense floor (G1)"),
                                 ("moe", "tab:blue", "MoE (G1)"),
                                 ("temporal", "tab:green", "temporal (G1)")]:
            d = BASE[budget][mt]
            xs = [N_G1[s] for s in d]; ys = [d[s] for s in d]
            pts = sorted(zip(xs, ys))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "--", color=color, alpha=0.45,
                    marker="o", ms=3, label=label)
        # G3 measured — solid bold
        for mt, color, label in [("moe", "tab:blue", "MoE (G3, 18/192)"),
                                  ("temporal", "tab:green", "temporal (G3, K=18)")]:
            d = g3.get((mt, budget), {})
            if not d:
                continue
            pts = sorted((N_G3[s], d[s], s) for s in d)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=color, lw=2.2,
                    marker="s", ms=6, label=label)
            for x, y, s in pts:
                ax.annotate(s, (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("active non-embedding params N (M)")
        ax.set_ylabel("validation BPB (lower better)")
        ax.set_title(f"IsoFLOP @ {budget}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper center")
    fig.suptitle("G=3 fine-grained (18 active routed of 192) vs G=1 baseline — FLAME-MoE Phase-0", fontsize=12)
    fig.tight_layout()
    out = f"{ROOT}/results/phase0/g3_isoflop.png"
    fig.savefig(out, dpi=130)
    print("saved", out)

    # text table
    print("\n=== G3 measured BPB (divisor", DIV, ") ===")
    for budget in ["1e16", "1e17"]:
        print(f"\n@{budget}:  shape   N(M)   dense(G1)  MoE(G1)  MoE(G3)  tmp(G1)  tmp(G3)")
        shapes = CELLS[("moe", budget)]
        for s in shapes:
            dg1 = BASE[budget]["dense"].get(s)
            mg1 = BASE[budget]["moe"].get(s)
            mg3 = g3.get(("moe", budget), {}).get(s)
            tg1 = BASE[budget]["temporal"].get(s)
            tg3 = g3.get(("temporal", budget), {}).get(s)
            f = lambda v: f"{v:.4f}" if isinstance(v, float) else "  -  "
            print(f"        {s:5} {N_G3[s]:6.2f}   {f(dg1):>8}  {f(mg1):>7}  {f(mg3):>7}  {f(tg1):>7}  {f(tg3):>7}")

if __name__ == "__main__":
    main()
