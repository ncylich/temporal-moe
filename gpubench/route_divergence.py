#!/usr/bin/env python3
"""Where and why the engine's resident windows first leave the PyTorch reference.

For each test row and MoE layer, finds the first token where the engine window
(LLAMA_TEMPORAL_ROUTE_DUMP) differs from the reference mask, keeps the ones no earlier layer
can have caused (no lower layer has diverged at that position or before), and prints the
reference's own decision margins at that token: gap between the two best non-resident logits,
gap between the two worst resident logits, and best non-resident minus worst resident. A window
that leaves the reference at a near-tie is rounding; one that leaves it at a wide margin is a bug.

    python gpubench/route_divergence.py --weights fine_1e19.safetensors --tokens test_tokens.npz \
        --rows 4 --dump route_gpu.bin
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "serving_convert"))
from ref_model import RefModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--tokens", required=True)
    ap.add_argument("--rows", type=int, default=4); ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=18); ap.add_argument("--dump", required=True)
    a = ap.parse_args()
    K, S = a.topk, a.seq
    rec = np.fromfile(a.dump, dtype=np.int32).reshape(-1, K + 1)
    layers = sorted(set(rec[:, 0].tolist()))
    per_layer = {L: rec[rec[:, 0] == L, 1:] for L in layers}
    model = RefModel(a.weights, K)
    grabbed = []
    orig = model.route
    model.route = lambda logits, routing: (grabbed.append(logits[0].clone()), orig(logits, routing))[1]
    toks = np.load(a.tokens)["tokens"][:a.rows, :S + 1].astype(np.int64)
    margins = []
    for r in range(a.rows):
        grabbed.clear()
        _, masks = model.forward(torch.from_numpy(toks[r:r + 1, :-1]), "temporal", return_masks=True)
        first = {}
        for li, L in enumerate(layers):
            ref = masks[li][0].numpy()
            eng = np.zeros_like(ref)
            np.put_along_axis(eng, per_layer[L][r * S:(r + 1) * S].astype(np.int64), True, axis=1)
            d = (eng != ref).any(1)
            first[L] = int(d.argmax()) if d.any() else None
        upstream = None                                   # earliest divergence among lower layers
        for li, L in enumerate(layers):
            t = first[L]
            if t is not None and (upstream is None or t < upstream):
                x = grabbed[li][t].double().numpy()
                prev = masks[li][0][t - 1].numpy() if t > 0 else None
                if prev is None:                          # cold fill: gap at the top-K boundary
                    s = np.sort(x)[::-1]; g = s[K - 1] - s[K]
                    print(f"row {r} layer {L:2d} first at {t:4d} (cold fill): top-K boundary gap {g:.2e}")
                    margins.append(g)
                else:
                    nr = np.sort(x[~prev])[::-1]; rs = np.sort(x[prev])
                    g_nom, g_wst, g_dec = nr[0] - nr[1], rs[1] - rs[0], nr[0] - rs[0]
                    eng_set = set(per_layer[L][r * S + t].tolist()); ref_set = set(np.nonzero(masks[li][0][t].numpy())[0].tolist())
                    print(f"row {r} layer {L:2d} first at {t:4d}: best-2 non-resident gap {g_nom:.2e}  worst-2 resident gap {g_wst:.2e}  "
                          f"decision margin {g_dec:+.2e}  engine-only {sorted(eng_set - ref_set)} reference-only {sorted(ref_set - eng_set)}")
                    margins.append(min(abs(g_nom), abs(g_wst), abs(g_dec)))
            if t is not None:
                upstream = t if upstream is None else min(upstream, t)
    if margins:
        print(f"independent first divergences: {len(margins)}; smallest relevant margin per case: max {max(margins):.2e}, median {np.median(margins):.2e}")
    else:
        print("no divergence")


if __name__ == "__main__":
    main()
