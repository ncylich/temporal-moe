#!/usr/bin/env python3
"""Per-step routing check for Skliar (C = K) on an engine dump (LLAMA_TEMPORAL_ROUTE_DUMP2).

gpubench/released/route_check.py replays the whole sequence through cache_bias.py with the reference's own
cache, so one exact logit tie (two equally valid selections) makes every later step differ until the two
caches meet again. This check isolates each step: the reference's `select` (imported from
analysis/residency/cache_bias.py, unmodified) is given the engine's cache from the previous step (with C = K
that is the engine's previous selection) and the running mean of max-min computed by cache_bias._delta on the
same logits. A step that differs is an exact tie if the experts the two sides disagree on have equal biased
logits; anything else is a real difference.

    python3 pixel_skliar_stepcheck.py route.bin [--K 18 --E 192 --lam 0.7207]
"""
import argparse, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpubench", "released")); sys.path.insert(0, os.path.join(HERE, "..", "analysis", "residency"))
sys.path.insert(0, os.path.join(HERE, ".."))
from route_check import read_dump
import cache_bias as CB

ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--K", type=int, default=18); ap.add_argument("--E", type=int, default=192)
ap.add_argument("--lam", type=float, default=0.7207); ap.add_argument("--J", type=int, default=1)
a = ap.parse_args()
hdr, ids, lg = read_dump(a.dump, a.K, a.E)
prev = {}                                                   # layer -> the engine's previous selection (its cache)
st = dict(steps=0, equal=0, exact_tie=0, real=0, loads_engine=0, loads_ref=0, loads_differ=0, first_real=None)
i, n = 0, len(hdr)
NEG = float("-inf")
while i < n:
    L, nt = int(hdr[i, 0]), int(hdr[i, 1])
    if L == -1:
        prev = {}; i += 1; continue
    x = torch.from_numpy(lg[i:i + nt].copy())
    if nt > 1 or L not in prev:                             # prompt batch or cold token: warms the cache
        CB._delta(L, x)
        prev[L] = set(ids[i + nt - 1].tolist())
    else:
        delta = CB._delta(L, x)
        last = torch.full((1, a.E), NEG); last[0, sorted(prev[L])] = 0.0
        ref = set(torch.nonzero(CB.select(x, last, delta, a.lam, a.J, a.K)[0]).flatten().tolist())
        got = set(ids[i].tolist())
        st["steps"] += 1
        le, lr = len(got - prev[L]), len(ref - prev[L])
        st["loads_engine"] += le; st["loads_ref"] += lr; st["loads_differ"] += le != lr
        assert le == int(hdr[i, 3]), (le, int(hdr[i, 3]))
        if ref == got:
            st["equal"] += 1
        else:
            zb = x[0] + a.lam * delta * (last[0] > NEG).float()
            za, zb_ = sorted(float(zb[e]) for e in ref - got), sorted(float(zb[e]) for e in got - ref)
            if za == zb_:
                st["exact_tie"] += 1
            else:
                st["real"] += 1
                if st["first_real"] is None:
                    st["first_real"] = dict(record=i, layer=L, ref_only=sorted(ref - got), engine_only=sorted(got - ref), zb_ref=za, zb_engine=zb_)
        prev[L] = got
    i += nt
print(json.dumps(st))
