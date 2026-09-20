#!/usr/bin/env python3
"""Randomized equivalence: the slotted walker must reproduce the dict walker EXACTLY.

The dict walker is the path every committed constrained result was produced on,
and it is the one `test_vllm_walker.py` checks against the reference scan. So the
gate for the slotted rewrite is equality with it, on schedules far nastier than
the hand-written one: dozens of requests over many steps and several layers, with
requests joining, finishing (freeing their slot for a later arrival to REUSE),
being preempted and replaying, and with `DEC["state"].clear()` between arms.

Slot reuse is the specific hazard: row N belongs to request X, X finishes, row N
is handed to request Y, and Y must start from its own prefill -- never from X's
leftover resident set.

Run: python analysis/residency/test_walker_slots.py
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_residency as VR                                          # noqa: E402
from decode_state import DEC                                         # noqa: E402

E, R, LAYERS = 64, 8, 4


def run(schedule, logits, walker):
    """Replay one schedule through one walker; return every decode row's mask."""
    os.environ["TEMPORAL_WALKER"] = walker
    DEC.update(on=True, R=R, swaps=1)
    DEC["state"].clear()
    VR.SL.update(rows={}, free=[], next=0, res={}, cap=0, epoch=None, seeded=set())
    out = []
    for step_i, step in enumerate(schedule):
        spans = [(req, n, pf) for req, n, pf in step]
        VR.set_step(spans)
        for layer in range(LAYERS):
            flat = logits[(step_i, layer)]
            masked = VR.apply(layer, flat.clone())
            o = 0
            for req, n, pf in step:
                if not pf:
                    for j in range(n):
                        out.append(torch.isfinite(masked[o + j]).clone())
                o += n
    return out


def make_schedule(seed, n_steps=40):
    """Random continuous-batching schedule with joins, finishes, and preemptions."""
    rng = random.Random(seed)
    sched, live, nxt = [], {}, 0
    for _ in range(n_steps):
        step = []
        for req in list(live):
            if rng.random() < 0.15:              # request finishes: slot is freed
                del live[req]
            elif rng.random() < 0.08:            # preempted: replay a chunk
                step.append((req, rng.randint(2, 5), False))
            else:
                step.append((req, 1, False))
        while len(live) < 6 and rng.random() < 0.6:   # new arrivals prefill
            req = f"r{nxt}"; nxt += 1
            live[req] = True
            step.append((req, rng.randint(1, 7), True))
        if step:
            sched.append(step)
    return sched


def main():
    ok = True
    for seed in range(6):
        sched = make_schedule(seed)
        logits = {}
        g = torch.Generator().manual_seed(seed)
        for i, step in enumerate(sched):
            n = sum(s[1] for s in step)
            for layer in range(LAYERS):
                logits[(i, layer)] = torch.randn(n, E, generator=g)
        a = run(sched, logits, "dict")
        b = run(sched, logits, "slots")
        if len(a) != len(b):
            print(f"FAIL seed {seed}: {len(a)} vs {len(b)} decode rows")
            ok = False
            continue
        bad = sum(1 for x, y in zip(a, b) if not torch.equal(x, y))
        reqs = len({s[0] for st in sched for s in st})
        print(f"  {'FAIL' if bad else 'ok  '} seed {seed}: {len(a)} decode rows over "
              f"{len(sched)} steps, {reqs} requests, {LAYERS} layers"
              + (f" — {bad} MISMATCH" if bad else ""))
        ok &= not bad
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
