#!/usr/bin/env python3
"""Gate: ref_model's teacher-forced `skliar` routing equals analysis/residency/cache_bias.apply fed the
same router logits one sequence at a time (token 0 as a one-token prefill, then one decode step per
token), on every MoE layer, for every row of a batch. Also checks that the window-transition load
count equals cache_bias's own count when C = k.

    python gpubench/pretrained/gate_skliar_ref.py --weights W.safetensors --tokens test_tokens.npz \
        --n 4 --seq 256 --lam 0.5 [--device cuda]
"""
import argparse, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "scripts", "serving_convert"), os.path.join(ROOT, "analysis", "residency")]
from ref_model import RefModel  # noqa: E402
import cache_bias  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--tokens", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--C", type=int, default=18)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    m = RefModel(a.weights, 18, device=a.device, cb_C=a.C, cb_lam=a.lam)
    seen = []                                                   # (layer, logits [B,S,E], mask)
    orig = m.route
    layer = [0]

    def spy(logits, routing):
        mask, gates = orig(logits, routing)
        layer[0] += 1
        seen.append((layer[0], logits.detach().float().cpu(), mask.cpu()))
        return mask, gates
    m.route = spy
    toks = torch.from_numpy(np.load(a.tokens)["tokens"][:a.n, :a.seq].astype(np.int64)).to(a.device)
    m.forward(toks, "skliar")
    cache_bias.CFG.update(C=a.C, lam=a.lam, J=1, k=18)
    bad = 0; rows = 0
    for L, lg, mk in seen:
        for b in range(lg.shape[0]):
            cache_bias.STATE.clear(); cache_bias.DELTA.clear()
            key = f"r{b}"
            cache_bias.apply(L, lg[b, :1], [(key, 1, True)], True)          # token 0: prefill (unbiased, warms the cache)
            for t in range(1, lg.shape[1]):
                out = cache_bias.apply(L, lg[b, t:t + 1], [(key, 1, False)], True)
                sel = out[0] > float("-inf")
                rows += 1
                if not torch.equal(sel, mk[b, t]):
                    bad += 1
    sw, tk, _ = cache_bias.stats()
    print(f"layers {len(seen)} token-layers {rows} mismatches {bad}")
    print(f"loads: ref {m.loads[0]}/{m.loads[1]}  cache_bias {sw}/{tk}")
    print("GATE PASS" if bad == 0 and m.loads[0] == sw and m.loads[1] == tk else "GATE FAIL")


if __name__ == "__main__":
    main()
