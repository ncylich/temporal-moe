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
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts", "serving_convert"), os.path.join(ROOT, "analysis", "residency")]
from ref_model import RefModel  # noqa: E402
import cache_bias  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--tokens", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--C", type=int, default=18)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--prefill", type=int, default=0)
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
    P = a.prefill or 1
    m.prefill = torch.full((a.n,), a.prefill, device=a.device) if a.prefill else None
    m.forward(toks, "skliar")
    cache_bias.CFG.update(C=a.C, lam=a.lam, J=1, k=18)
    bad = 0; rows = 0; ties = 0; tied_seq = set()
    for L, lg, mk in seen:
        for b in range(lg.shape[0]):
            cache_bias.STATE.clear(); cache_bias.DELTA.clear()
            key = f"r{b}"
            pre = cache_bias.apply(L, lg[b, :P], [(key, P, True)], True)    # prompt: one prefill batch (unbiased, warms the cache)
            for t in range(P):                                              # prompt tokens: stock top-k
                rows += 1
                if not torch.equal(lg[b, t].topk(18).indices.sort().values, mk[b, t].nonzero()[:, 0]):
                    v = lg[b, t].sort(descending=True).values; tie = bool(v[17] == v[18])
                    ties += tie; bad += not tie
            for t in range(P, lg.shape[1]):
                out = cache_bias.apply(L, lg[b, t:t + 1], [(key, 1, False)], True)
                sel = out[0] > float("-inf")
                rows += 1
                if not torch.equal(sel, mk[b, t]):                          # an exact tie at the cut is not a mismatch
                    d = sel ^ mk[b, t]
                    tie = bool(d.sum() == 2 and lg[b, t][d][0] == lg[b, t][d][1])
                    ties += tie; bad += not tie
                    if tie: tied_seq.add((L, b))
    sw, tk, _ = cache_bias.stats()
    print(f"layers {len(seen)} token-layers {rows} mismatches {bad} (plus {ties} exact logit ties at the cut, "
          f"in {len(tied_seq)} layer-sequences whose later steps may then differ)")
    print(f"loads: ref {m.loads[0]}/{m.loads[1]}  cache_bias {sw}/{tk}")
    sk_loads = list(m.loads)
    # our rule: the GPU scan used on CUDA equals the CPU reference scan on these same logits
    from temporal.temporal_router import compute_resident_mask, compute_resident_mask_accel
    tbad = 0
    if a.prefill:                                  # our rule with a free prompt equals decode_state.py prefill + steps
        import decode_state as DS
        DS.DEC.update(on=True, R=18, swaps=1)
        seen.clear(); layer[0] = 0; m.loads = [0, 0]
        m.forward(toks, "temporal")
        dbad = 0
        for L, lg, mk in seen:
            DS.reset()
            DS.prefill(L, lg[:, :P].transpose(0, 1))
            dbad += int(not torch.equal(lg[:, :P].topk(18).indices.sort().values, mk[:, :P].nonzero()[:, 2].view(-1, P, 18)))
            for t in range(P, lg.shape[1]):
                dbad += int(not torch.equal(DS.step(L, lg[:, t]), mk[:, t]))
        print(f"temporal with free prefill vs decode_state: {dbad} mismatching steps; loads {m.loads}")
        tbad += dbad
    if a.device == "cuda":
        for L, lg, _ in seen:
            x = lg.transpose(0, 1).contiguous()
            tbad += int(not torch.equal(compute_resident_mask(x, 18, "min_logit"),
                                        compute_resident_mask_accel(x.cuda(), 18, "min_logit").cpu()))
        print(f"temporal GPU scan vs CPU scan: {tbad} of {len(seen)} layers differ")
    print("GATE PASS" if tbad == 0 and bad == 0 and sk_loads == [sw, tk] else "GATE FAIL")


if __name__ == "__main__":
    main()
