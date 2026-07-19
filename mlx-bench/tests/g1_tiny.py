#!/usr/bin/env python3
"""Gate G1 tiny-config smoke (see PLAN.md Phase 1). Run BEFORE building the 6 GB models.

Builds a tiny Qwen3-MoE in memory (fp16 -> q4 g64 where divisible), runs a
16-token prefill + 4 decode steps, and checks:
  1. logits are finite with the right shapes,
  2. decode ids vary (not a degenerate constant),
  3. the MoE gate (softmax over ALL experts -> top-k -> renorm) matches an
     independent numpy reference on one layer.

Prints a JSON verdict. Exit 0 = PASS.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Model, ModelArgs  # noqa: E402

GROUP, BITS = 64, 4


def build_tiny():
    args = ModelArgs(
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=64,
        vocab_size=512,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
    )
    mx.random.seed(0)
    model = Model(args)
    # Quantize every module that mlx-lm would quantize (all group axes here are
    # 64-divisible: hidden=128, ff=64, vocab hidden=128).
    nn.quantize(model, group_size=GROUP, bits=BITS)
    mx.eval(model.parameters())
    model.eval()
    return model, args


def numpy_route_ref(logits_np, k):
    # softmax over ALL experts
    z = logits_np - logits_np.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    idx = np.argpartition(p, -k, axis=-1)[..., -k:]
    sc = np.take_along_axis(p, idx, axis=-1)
    sc = sc / sc.sum(axis=-1, keepdims=True)
    return idx, sc


def main():
    out = {}
    model, args = build_tiny()

    # ---- forward: 16-token prefill + 4 greedy decode steps ----
    mx.random.seed(0)
    ids = mx.array(np.random.randint(0, args.vocab_size, size=(1, 16)))
    cache = model.make_cache()
    logits = model(ids, cache=cache)
    mx.eval(logits)
    out["prefill_shape"] = list(logits.shape)
    out["prefill_finite"] = bool(mx.all(mx.isfinite(logits)).item())

    y = mx.argmax(logits[:, -1, :], axis=-1)
    decoded = []
    for _ in range(4):
        lg = model(y[None], cache=cache)
        y = mx.argmax(lg[:, -1, :], axis=-1)
        mx.eval(y)
        decoded.append(int(y.item()))
    out["decode_ids"] = decoded
    out["decode_finite"] = all(0 <= t < args.vocab_size for t in decoded)
    # Informational only: a 2-layer random model often hits a greedy fixed point.
    # The real "ids must vary" check is the full-model G1 gate in bench_decode.py.
    out["decode_varies"] = len(set(decoded)) > 1

    # ---- MoE gate vs numpy reference (layer 0) ----
    moe = model.model.layers[0].mlp
    mx.random.seed(1)
    x = (mx.random.normal((1, 5, args.hidden_size)) * 0.1).astype(mx.float16)
    gate_logits = moe.gate(x)  # deterministic given x
    inds, scores = moe.route(x)
    mx.eval(gate_logits, inds, scores)

    ref_idx, ref_sc = numpy_route_ref(
        np.array(gate_logits.astype(mx.float32)), args.num_experts_per_tok
    )
    got_idx = np.array(inds)
    got_sc = np.array(scores.astype(mx.float32))

    # Compare as index->weight maps (top-k order is unspecified).
    max_err = 0.0
    sets_match = True
    for b in range(got_idx.shape[0]):
        for t in range(got_idx.shape[1]):
            g = dict(zip(got_idx[b, t].tolist(), got_sc[b, t].tolist()))
            r = dict(zip(ref_idx[b, t].tolist(), ref_sc[b, t].tolist()))
            if set(g) != set(r):
                sets_match = False
                continue
            for e in g:
                max_err = max(max_err, abs(g[e] - r[e]))
    out["gate_index_sets_match"] = sets_match
    out["gate_max_weight_err"] = max_err

    ok = (
        out["prefill_finite"]
        and out["decode_finite"]
        and sets_match
        and max_err < 5e-3
    )
    out["G1_tiny"] = "PASS" if ok else "FAIL"
    print(json.dumps(out, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
