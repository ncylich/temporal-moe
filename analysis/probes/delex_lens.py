#!/usr/bin/env python3
"""A4 / A5 / C5 -- the output logit lens, at every MoE layer.

The locus probes ask what makes an expert fire. This asks what an expert *writes*: take its
routed-token-averaged output vector v_e from the capture, project through the final RMSNorm and the
unembedding, softmax, and report the effective vocabulary exp(H(p_e)) -- from 1 (a single word) to the
vocabulary size. A narrow effective vocabulary means the expert promotes a coherent lexical cluster.

  variant 'weighted'  v_e = mean expert output over the tokens actually routed to it (data-conditioned)
  variant 'static'    v_e = uniform mean of the expert's fc2 weight columns, no data at all

A5 is the reason 'static' exists and it is not optional: mid-network projections are rotated relative
to the output basis, and averaging unactivated columns cancels their directions, so this method reads
near the vocabulary size for every expert whether or not there is signal. It is the method's own
failure mode, measured, and every comparison must be weighted-versus-static *within a layer* -- the
static reference itself moves with depth (32.5k at layer 2 to 28.4k at layer 14 on the 1e19 coarse
baseline), so a weighted number that falls with depth is not evidence of sharpening until the static
reference is subtracted.

Changes from the version that produced the published numbers: layers come from the capture rather than
`LAYERS = [2,3,4]`, cells come from the registry, and rows carry run/budget/regime. Pooling the three
shallowest layers of a 14-layer model into one median, as the published table did, hid a real depth
effect -- at 1e16 the unconstrained baseline's effective vocabulary falls from 15,359 at layer 2 to
11,646 at layer 4 while its static reference is flat, and the temporal model's barely moves.

Needs the run's checkpoint for the unembedding and expert weights, so `megatron` must be importable:

    . scripts/env.sh
    PYTHONPATH="$ROOT/Megatron-LM:$ROOT" "$PY" analysis/probes/delex_lens.py
"""
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

OUT = os.path.join(ABLATIONS, "mechinterp_lens_1e19.csv")
EPS = 1e-6
HEADER = ["label", "run", "budget", "regime", "layer", "expert", "variant", "n_tokens",
          "eff_vocab", "dispersion"]


def rmsnorm(v, g):
    import torch
    return v / torch.sqrt((v * v).mean(-1, keepdim=True) + EPS) * g


def eff_vocab(v, gain, U):
    """v [E,H] -> (eff_vocab[E], dispersion[E]) via softmax(U @ rmsnorm(v)). dispersion = H(p)/ln V."""
    import torch
    x = rmsnorm(v.double(), gain.double())
    logits = x @ U.double().T
    logits = logits - logits.max(-1, keepdim=True).values
    p = torch.softmax(logits, -1)
    H = -(p * p.clamp(min=1e-20).log()).sum(-1)
    return torch.exp(H), H / np.log(U.shape[0])


def main():
    import torch
    import ckpt_read
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    cells = [r for r in registry.runs(capture=True)
             if (not only or r.name in only) and os.path.exists(r.path("delex_capture.pt"))]
    if not cells:
        sys.exit("no captures on disk")
    rows, skipped = [], []
    for r in cells:
        d = torch.load(r.path("delex_capture.pt"), map_location="cpu", weights_only=False)
        layers = registry.moe_layers(d)
        ck = os.path.join(registry.RUNS, r.name, "ckpt")
        if not os.path.isdir(ck):
            skipped.append(f"{r.name}: no checkpoint on disk "
                           f"(scripts/artifacts.py pull --run {r.name})")
            print(f"[skip] {skipped[-1]}")
            continue
        ip = ckpt_read.iter_dir(ck)
        want = ["output_layer.weight", "decoder.final_layernorm.weight"]
        # Capture layer keys are TopKRouter.layer_number, 1-based; checkpoint module paths are 0-based,
        # so the weights for layer L live at decoder.layers.{L-1}. Verified against the checkpoint: for
        # an 8-MoE-layer model the expert modules are at indices 1..8 and the capture keys are 2..9.
        # Using L directly read the next layer's weights and left the deepest layer with no static
        # reference at all -- the same off-by-one the capture itself had (see delex_probe.py).
        fc2 = {L: f"decoder.layers.{L - 1}.mlp.experts.experts.linear_fc2.weight" for L in layers}
        have = set(ckpt_read.weight_keys(ckpt_read.FileSystemReader(ip)))
        missing = [L for L in layers if fc2[L] not in have]
        if missing:
            print(f"[warn] {r.name}: layers {missing} have no fc2 weights in the checkpoint; "
                  f"their static reference (A5) will be blank, weighted still reported")
        w = ckpt_read.load(ip, want + [fc2[L] for L in layers if L not in missing])
        U = w["output_layer.weight"].float()
        gain = w["decoder.final_layernorm.weight"].float()
        print(f"[run] {r.name} ({r.regime}, {r.grain_label}, {r.budget}) layers "
              f"{layers[0]}-{layers[-1]}, vocab {U.shape[0]}", flush=True)
        for L in layers:
            Ld = d["layers"][L]
            cnt = Ld["out_cnt"]
            v_w = (Ld["out_sum"] / cnt.clamp(min=1).unsqueeze(1)).float()
            ev_w, disp_w = eff_vocab(v_w, gain, U)
            ev_s = disp_s = None
            if L not in missing:
                v_s = w[fc2[L]].float().mean(-1)
                ev_s, disp_s = eff_vocab(v_s, gain, U)
            for e in range(v_w.shape[0]):
                rows.append([r.name, r.name, r.budget, r.regime, L, e, "weighted", int(cnt[e]),
                             round(float(ev_w[e]), 1), round(float(disp_w[e]), 4)])
                if ev_s is not None:
                    rows.append([r.name, r.name, r.budget, r.regime, L, e, "static", int(cnt[e]),
                                 round(float(ev_s[e]), 1), round(float(disp_s[e]), 4)])
            mw = float(np.median(ev_w.numpy()))
            ms = float(np.median(ev_s.numpy())) if ev_s is not None else float("nan")
            print(f"    L{L:<3} weighted median eff_vocab={mw:8.0f}  static reference={ms:8.0f}  "
                  f"gap={mw - ms:+9.0f}", flush=True)

    os.makedirs(ABLATIONS, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(HEADER)
        wr.writerows(rows)
    print(f"\n[write] {OUT}: {len(rows)} rows")
    for s in skipped:
        print(f"omitted — {s}")


if __name__ == "__main__":
    main()
