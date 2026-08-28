#!/usr/bin/env python3
"""Fast walker vs slots walker on one long randomised continuous-batching schedule.

Hundreds of requests with random prompt/generation lengths join over time, prefill in
random-sized chunks, decode, finish, and a fraction get preempted (vanish, then replay
prompt + generated so far as one prefill chunk, then continue). Both walkers see the
same flattened logits every step; every decode position's mask must be identical, and
both must equal the per-request reference scan. GPU only (the fast walker is Triton).
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_residency as VR                                          # noqa: E402
from decode_state import DEC, compute_resident_mask                  # noqa: E402

assert torch.cuda.is_available()
DEV = "cuda"
rng = random.Random(int(os.environ.get("SEED", "3")))
R, E, NREQ, STEPS = 8, 128, 300, 400
g = torch.Generator().manual_seed(5)
reqs = {}
for i in range(NREQ):
    p, gen = rng.randint(1, 40), rng.randint(1, 40)
    reqs[f"r{i}"] = {"p": p, "t": p + gen, "join": rng.randint(0, STEPS // 2),
                     "stream": torch.randn(p + gen, 1, E, generator=g).to(torch.bfloat16),
                     "computed": 0, "done": False, "gone": 0}
# schedule: each step, every joined+unfinished request gets a chunk (prefill: random size)
# or one decode token; a few get preempted (vanish for some steps, then replay everything)
schedule = []
for s in range(STEPS):
    step = []
    for rid, r in reqs.items():
        if r["join"] > s or r["done"]:
            continue
        if r["gone"] > 0:
            r["gone"] -= 1
            continue
        if r["computed"] > 0 and r["computed"] < r["t"] and rng.random() < 0.02:   # preempt
            r["gone"] = rng.randint(1, 4); r["replay_from"] = r["computed"]; r["computed"] = 0; r["was_preempted"] = True
            continue
        c = r["computed"]
        if c < r["p"]:                                     # prefill chunk
            n = rng.randint(1, r["p"] - c)
            if "replay_from" in r and c + n >= r["p"]:      # replay: prompt + generated as chunks
                n = r["p"] - c
            step.append((rid, c, c + n, True))
        elif "replay_from" in r and c < r["replay_from"]:  # generated-so-far recomputed as a chunk
            step.append((rid, c, r["replay_from"], False)); n = r["replay_from"] - c
            del r["replay_from"]
        else:
            step.append((rid, c, c + 1, False)); n = 1
        r["computed"] += n
        if r["computed"] >= r["t"]:
            r["done"] = True
    if step:
        schedule.append(step)


def run(walker):
    VR._WALKER = walker
    os.environ["TEMPORAL_WALKER"] = walker
    DEC.update(on=True, R=R, swaps=1)
    DEC["state"].clear()
    VR.SL.update(rows={}, free=[], next=0, res={}, cap=0, epoch=None, seeded=set())
    got = {}
    for step in schedule:
        spans = [(rid, e - s, pf) for rid, s, e, pf in step]
        VR.set_step(spans)
        flat = torch.cat([reqs[rid]["stream"][s:e, 0] for rid, s, e, _ in step]).to(DEV)
        for layer in (0, 1):                                # two layers: banks must not mix
            out = VR.apply(layer, flat.clone())
            o = 0
            for rid, s, e, pf in step:
                if not pf:
                    for j in range(e - s):
                        got[(layer, rid, s + j)] = torch.isfinite(out[o + j]).cpu()
                o += e - s
    return got


fast = run("fast")
slots = run("slots")
assert fast.keys() == slots.keys()
bad = [k for k in fast if not torch.equal(fast[k], slots[k])]
print(f"decode positions compared: {len(fast)} over {len(schedule)} steps, {NREQ} requests; "
      f"fast != slots at {len(bad)}")
assert not bad, bad[:5]
# and both equal the isolated reference scan (rule on positions >= prompt)
ref_bad = 0
for rid, r in reqs.items():
    full = compute_resident_mask(r["stream"].float().to(DEV), R, evict="min_logit", swaps=1)[:, 0].cpu()   # GPU path: the CPU path differs on ~3% of positions (bf16 ties)
    for pos in range(r["p"], r["t"]):
        for layer in (0, 1):
            k = (layer, rid, pos)
            if k in fast and not torch.equal(fast[k], full[pos]):
                ref_bad += 1
                if ref_bad <= 6:
                    print(f"  ref mismatch: {rid} p={r['p']} t={r['t']} pos={pos} layer={layer} "
                          f"preempted={'replay_from' in r or r.get('was_preempted')} "
                          f"fast_resident={int(fast[k].sum())} ref_resident={int(full[pos].sum())}")
print(f"vs reference scan: {ref_bad} mismatches")
if os.environ.get("DEBUG_RID"):
    import residency_kernels as RK
    rid = os.environ["DEBUG_RID"]; r = reqs[rid]
    spans = [(si, s_, e_, pf) for si, step in enumerate(schedule) for (q, s_, e_, pf) in step if q == rid]
    print(f"{rid}: p={r['p']} t={r['t']} join={r['join']} spans(step,start,end,prefill)={spans}")
    full = compute_resident_mask(r["stream"].float().to(DEV), R, evict="min_logit", swaps=1)[:, 0].cpu()   # GPU path: the CPU path differs on ~3% of positions (bf16 ties)
    lg = r["stream"][:, 0].to(DEV); st = None; cnt = torch.zeros(2, dtype=torch.int64, device=DEV)
    for si, s_, e_, pf in spans:
        m = RK.chunk_scan(lg[s_:e_], st, R, 0.0, 1, cnt); st = m[-1].clone()
        ok = all(torch.equal(m[j].bool().cpu(), full[s_ + j]) for j in range(e_ - s_))
        print(f"  chunk step{si} [{s_},{e_}) prefill={pf}: chunked-scan == continuous-ref: {ok}")
assert ref_bad == 0
print("WALKER EQUIVALENCE PASS (fast == slots == reference on a randomised schedule with "
      "chunked prefill, joins, finishes, slot reuse and preemption replay)")
