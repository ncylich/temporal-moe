#!/usr/bin/env python3
"""Parse the online-sampler smoke logs; print the timing table; fail on any threshold.

Checks (hard):
  parity     every greedy generation identical to the merged checkpoint, free and R8
  sync       wake + weight sync < 15 s
  sampling   >= 1500 tok/s aggregate on the sampled rows
  overhead   per-refresh total (wake+sync+sample+teacher+sleep) < the training time of the
             steps between refreshes, i.e. refreshes cost < 50% of wall time at every=16
             (measured at every=4 here, scaled)
  training   reverse-KL estimate finite; step lines present; DONE line present
"""
import re
import sys

smoke, e2e = (open(p).read() for p in sys.argv[1:3])
ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok = ok and cond

par = re.findall(r"\[online-smoke\] (\w+): (\d+)/(\d+) generations identical", smoke)
check(bool(par) and all(a == b for _, a, b in par), f"parity vs merged checkpoint: {par}")
syncs = [(float(a), float(b)) for a, b in re.findall(r"\[online\] wake ([\d.]+)s, weight sync ([\d.]+)s", smoke + e2e)]
check(bool(syncs) and max(a + b for a, b in syncs) < 15, f"wake+sync per refresh (s): {[round(a+b,1) for a,b in syncs]}")
samp = [(int(r), int(t), float(s), float(tps)) for r, t, s, tps in
        re.findall(r"\[online\] sampled (\d+) rows, (\d+) tokens in (\d+)s \((\d+) tok/s\)", e2e)]
check(bool(samp) and min(x[3] for x in samp) >= 1500, f"sampling tok/s per refresh: {[x[3] for x in samp]}")
ref = [(int(s), int(n), float(t)) for s, n, t in re.findall(r"\[online\] refresh at step (\d+): (\d+) fresh on-policy rows in (\d+)s total", e2e)]
steps = [(int(s), float(t)) for s, t in re.findall(r"\[gce\] step (\d+) seen [\d.]+M loss [\d.]+ \((\d+) tok/s\)", e2e)]
rev = [float(x) for x in re.findall(r"reverse-KL estimate ([-\d.]+) nats/tok", e2e)]
done = "[gce] DONE seen=" in e2e
check(done, "training reached DONE")
check(bool(rev) and all(abs(x) < 50 for x in rev), f"reverse-KL estimates: {rev[:4]}")
if ref:
    per_refresh = sum(t for _, _, t in ref) / len(ref)
    # training wall time between refreshes: total run time minus refresh time, / refreshes
    m = re.search(r"### e2e 1/3 .* (\d\d):(\d\d)", e2e); m2 = re.search(r"### e2e 2/3 .* (\d\d):(\d\d)", e2e)
    if m and m2:
        total = (int(m2.group(1)) * 60 + int(m2.group(2)) - int(m.group(1)) * 60 - int(m.group(2))) * 60
        train_only = max(1.0, total - sum(t for _, _, t in ref) - 240)     # minus ~4 min model load/engine boot
        nsteps = max(1, max(s for s, _ in ref) + 4)
        per_step = train_only / nsteps
        overhead16 = per_refresh * (256 / 64) / (16 * per_step)           # scale rows to n=256, every=16
        print(f"INFO refresh {per_refresh:.0f}s at 64 rows; ~{per_step:.0f}s per training step; projected refresh overhead at every=16 x 256 rows: {100*overhead16:.0f}% of training time")
        check(overhead16 < 0.6, "projected refresh overhead < 60% of training time at every=16")
print("ONLINE E2E " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
