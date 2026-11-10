#!/usr/bin/env python3
"""Engine trained routing versus the PyTorch reference, token by token.

Reads the per-layer windows the engine wrote with LLAMA_TEMPORAL_ROUTE_DUMP (records of
int32 layer + K int32 expert ids, in execution order) for rows scored by
`gguf_logits nll`, recomputes the reference masks with ref_model.py on the same rows, and
reports, per layer, the first token where the resident sets differ and how many tokens differ.
Optionally compares per-token losses with a reference loss file.

    python scripts/serving_convert/check_engine_routing.py --weights fine_1e19.safetensors \
        --tokens test_tokens.npz --rows 4 --dump route.bin --nll nll.f32 --ref-losses ref_temporal_512.npy
"""
import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_model import RefModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=18)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--nll", help="engine per-token NLL float32 [rows, seq]")
    ap.add_argument("--ref-losses", help="reference per-token losses .npy [N, seq]")
    ap.add_argument("--no-masks", action="store_true", help="skip the reference forward")
    a = ap.parse_args()
    K, S = a.topk, a.seq
    rec = np.fromfile(a.dump, dtype=np.int32).reshape(-1, K + 1)
    layers = sorted(set(rec[:, 0].tolist()))
    per_layer = {L: rec[rec[:, 0] == L, 1:] for L in layers}
    n_rows = a.rows
    for L in layers:
        assert per_layer[L].shape[0] == n_rows * S, (L, per_layer[L].shape)
    print(f"dump: {rec.shape[0]} records, layers {layers[0]}..{layers[-1]} ({len(layers)})")
    if not a.no_masks:
        model = RefModel(a.weights, K)
        toks = np.load(a.tokens)["tokens"][:n_rows, :S + 1].astype(np.int64)
        tot_diff = tot = 0
        for r in range(n_rows):
            _, masks = model.forward(torch.from_numpy(toks[r:r + 1, :-1]), "temporal", return_masks=True)
            for li, L in enumerate(layers):
                ref = masks[li][0].numpy()                          # [S, E] bool
                eng = np.zeros_like(ref)
                ids = per_layer[L][r * S:(r + 1) * S]
                np.put_along_axis(eng, ids.astype(np.int64), True, axis=1)
                assert (eng.sum(1) == K).all(), "engine window has duplicate experts"
                d = (eng != ref).any(1)
                tot_diff += int(d.sum()); tot += S
                if d.any():
                    print(f"row {r} layer {L}: {int(d.sum())} of {S} tokens differ, first at position {int(d.argmax())}")
            swaps = [int((masks[li][0][1:] != masks[li][0][:-1]).any(-1).sum()) for li in range(len(layers))]
            print(f"row {r}: reference swaps per layer-token {np.mean(swaps) / (S - 1):.4f}", flush=True)
        print(f"MASKS: {tot - tot_diff} of {tot} layer-tokens identical ({tot_diff} differ)")
    if a.nll and a.ref_losses:
        eng = np.fromfile(a.nll, dtype=np.float32).reshape(-1, S)[:n_rows].astype(np.float64)
        ref = np.load(a.ref_losses)[:n_rows].astype(np.float64)
        d = eng - ref
        print(f"LOSS rows {n_rows} tokens {eng.size}: engine {eng.mean():.5f}  reference {ref.mean():.5f}  "
              f"diff {d.mean():+.5f}  max|d| {np.abs(d).max():.4f}  median|d| {np.median(np.abs(d)):.2e}")


if __name__ == "__main__":
    main()
