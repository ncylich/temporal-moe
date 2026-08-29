#!/usr/bin/env python3
"""Sequential one-knob sweep of the from-scratch on-policy recipe, with a KL-based early stop.

Each cell launches tmoe_gemma_online.sh scratch with its env, watches the reverse-KL trace, and
kills the run if the estimate at step 100 has not decreased at all from step 50 (the objective is
not moving: do not wait 80 minutes). Finished cells are scored on GSM8K n=1319 against --best
(paired fixed/broken, z) and appended to results/ablations/online_sweep.md.

    sweep_online.py --aux-loss revkl --best gemma4_ce_online_scratch_e16_n1319 [--cells anchor0,lr1e-4,...]
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CELLS = {   # name: (env, tokens, every, n); every cell has anchor 0 -- "baseline" is the reference the rest pair against
    "baseline":     ({"TMOE_ANCHOR_W": "0", "TMOE_BUDGET_ON": "sampled"}, 3400000, 16, 256),
    "lr1e-4":       ({"TMOE_ANCHOR_W": "0", "TMOE_BUDGET_ON": "sampled", "TMOE_LR": "1e-4"}, 3400000, 16, 256),
    "temp1.0":      ({"TMOE_ANCHOR_W": "0", "TMOE_BUDGET_ON": "sampled", "TMOE_ONLINE_TEMP": "1.0"}, 3400000, 16, 256),
    "refresh8x128": ({"TMOE_ANCHOR_W": "0", "TMOE_BUDGET_ON": "sampled"}, 3400000, 8, 128),
    "budget6.8M":   ({"TMOE_ANCHOR_W": "0", "TMOE_BUDGET_ON": "sampled"}, 6800000, 16, 256),
}


def score(rec, best):
    from failure_filter import load_arm
    out = {}
    for arm in ("free", "R8", "R16"):
        base = load_arm("gemma4_instruct_n1319", arm); b = load_arm(best, arm); n_ = load_arm(rec, arm)
        n = len(base); acc = lambda d: 100 * sum(v["correct"] for v in d.values()) / n
        c = sum(n_[d]["correct"] and not b[d]["correct"] for d in b); k = sum(b[d]["correct"] and not n_[d]["correct"] for d in b)
        out[arm] = (acc(base), acc(b), acc(n_), c, k, (c - k) / math.sqrt(c + k) if c + k else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux-loss", default="revkl")
    ap.add_argument("--best", default="gemma4_ce_online_scratch_e16_n1319")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--prio", default="3")
    A = ap.parse_args()
    best = A.best
    table = "results/ablations/online_sweep.md"
    if not os.path.exists(table):
        open(table, "w").write("# From-scratch on-policy sweep (one knob per cell; GSM8K n=1319; paired vs the running best)\n\n"
                               "| cell | env | KL@50 -> @100 | free | R8 | R16 | R8 vs best (fixed/broken, z) | verdict |\n|---|---|---|---|---|---|---|---|\n")
    for name in A.cells.split(","):
        env, tokens, every, n = CELLS[name]
        suffix = f"_{name}"
        full_env = dict(os.environ, TMOE_PRIO=A.prio, TMOE_AUX_LOSS=A.aux_loss, TMOE_NAME_SUFFIX=suffix, **env)
        log = f"/workspace/rerun-logs/sweep_{name}.out"
        print(f"[sweep] {name}: env {env} tokens {tokens} every {every} n {n} -> {log}", flush=True)
        proc = subprocess.Popen(["bash", "/workspace/tmoe_gemma_online.sh", "scratch", str(tokens), str(every), str(n)],
                                env=full_env, stdout=open(log, "a"), stderr=subprocess.STDOUT)
        kl = {}; verdict = "done"; t0 = time.time()
        while proc.poll() is None:
            time.sleep(60)
            txt = open(log).read() if os.path.exists(log) else ""
            for st, val in re.findall(r"\[gce\] step (\d+) seen.*?\n\[gce\] aux-\S+ loss [-\d.]+; reverse-KL estimate ([-\d.]+)", txt):
                kl[int(st)] = float(val)
            if 50 in kl and 100 in kl and kl[100] >= kl[50]:      # no decrease at all by step 100 (the working run dropped 4%)
                verdict = f"STALLED (KL {kl[50]:.3f}->{kl[100]:.3f})"
                # kill the chain: driver, its lease wrapper, the trainer
                subprocess.run(["bash", "-c", f"for p in $(pgrep -P {proc.pid}); do kill $(pgrep -P $p) $p 2>/dev/null; done; kill {proc.pid}"])
                break
            if time.time() - t0 > 4 * 3600:
                verdict = "TIMEOUT"; subprocess.run(["bash", "-c", f"for p in $(pgrep -P {proc.pid}); do kill $(pgrep -P $p) $p 2>/dev/null; done; kill {proc.pid}"]); break
        rec = f"gemma4_ce_online_scratch_e{every}{suffix}_n1319"
        row = f"| {name} | {env} | {kl.get(50, float('nan')):.3f} -> {kl.get(100, float('nan')):.3f} | "
        if verdict == "done":
            try:
                sc = score(rec, best)
                row += f"{sc['free'][2]:.1f} | {sc['R8'][2]:.1f} | {sc['R16'][2]:.1f} | {sc['R8'][2]-sc['R8'][1]:+.1f} ({sc['R8'][3]}/{sc['R8'][4]}, z={sc['R8'][5]:+.1f}) | "
                if sc["R8"][5] >= 2.0:
                    verdict = f"KEEP (new best)"; best = rec
                else:
                    verdict = "no gain"
            except Exception as e:
                verdict = f"score failed: {e}"; row += "| | | | "
        else:
            row += "| | | | "
        row += f"{verdict} |\n"
        open(table, "a").write(row); print("[sweep] " + row.strip(), flush=True)
        subprocess.run(["bash", "-c", f"git add {table} results/ablations/instruct_genbench_vllm.csv results/ablations/genbench_samples/{rec}_* 2>/dev/null; git commit -q -m 'online sweep: {name} -> {verdict}' && git push -q origin layer-lexicality"])
    print(f"[sweep] done; best = {best}", flush=True)


if __name__ == "__main__":
    main()
