#!/usr/bin/env python3
"""Gate G3 exactness proofs for the xl disk-expert path (bigger-than-RAM MoE).

Builds a TRUNCATED xl config (L=3, E=1024) whose full expert pool fits RAM
(~2 GB), in two forms from the SAME deterministic generators:
  * disk-experts   (nonexpert.safetensors + experts_flat.bin, disk_experts=True)
  * ordinary in-RAM (full E-expert switch_mlp stacks)

Then:
  bytes    disk file bytes for (layer0, expert0) and (layer0, expert1023) ==
           the in-RAM model's expert bytes, bytewise.
  (i)      lazy_full disk path == in-RAM full-MoE forward logits (8 decode
           tokens after 64 prefill), gate <= 1e-5 (bitwise expected).
  (ii)     deploy disk path == temporal.py deploy_ref on the in-RAM model
           (identical argmax ids AND ~0 logit delta).
  (iii)    floor bytes audit: a 4-token L=3 run at N=2 fetches exactly
           N * n_moe_layers * per_expert_bytes per token.

Run this BEFORE generating the 30.6 GB full model. Prints a JSON verdict;
exit 0 = all PASS.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import load                                        # noqa: E402
from temporal import TemporalController                       # noqa: E402
from xl import XLController                                   # noqa: E402
from gen_xl_model import (EXPERT_BYTES, PROJ, SUB_OFFSETS,    # noqa: E402
                          write_disk_model, write_inram_model)

GATE = 1e-5
L, E = 3, 1024


def logits_path(model, ids_prefill, decode_ids):
    """Teacher-forced: prefill then feed each fixed decode id; stack step logits."""
    cache = model.make_cache()
    out = [model(ids_prefill, cache=cache)[:, -1, :]]
    for t in decode_ids:
        out.append(model(mx.array([[int(t)]]), cache=cache)[:, -1, :])
    Lo = mx.stack(out, axis=0).astype(mx.float32)
    mx.eval(Lo)
    return Lo


def inram_expert_bytes(model, li, e):
    """Reconstruct expert e's on-disk byte layout from the in-RAM switch_mlp."""
    sm = model.model.layers[li].mlp.switch_mlp
    cols = []
    for name in PROJ:
        lin = getattr(sm, name)
        for tensor in (lin.weight, lin.scales, lin.biases):
            cols.append(np.array(mx.view(tensor[e:e + 1].reshape(1, -1), dtype=mx.uint8)))
    return np.concatenate(cols, axis=1).tobytes()


def main():
    results = []
    all_ok = True
    tmp = Path(tempfile.mkdtemp(prefix="g3_xl_"))
    disk_dir, inram_dir = tmp / "disk", tmp / "inram"

    write_disk_model(disk_dir, L=L, E=E)
    write_inram_model(inram_dir, L=L, E=E)

    # ---- bytewise: disk file vs in-RAM model, layer0 experts 0 and 1023 ----
    model_r, cfg_r = load(inram_dir)
    with open(disk_dir / "experts_flat.bin", "rb") as f:
        byte_ok = True
        for e in (0, E - 1):
            f.seek((0 * E + e) * EXPERT_BYTES)
            disk_b = f.read(EXPERT_BYTES)
            ram_b = inram_expert_bytes(model_r, 0, e)
            byte_ok &= (disk_b == ram_b)
    results.append(dict(name="bytewise_disk_eq_inram_L0_e0_e1023",
                        verdict="PASS" if byte_ok else "FAIL"))
    all_ok &= byte_ok

    # ---- (i) lazy_full disk == in-RAM full MoE ----
    rng = np.random.default_rng(1)
    ids_prefill = mx.array(rng.integers(0, cfg_r["vocab_size"], size=(1, 64)))
    decode = rng.integers(0, cfg_r["vocab_size"], size=8)

    Lc = logits_path(model_r, ids_prefill, decode)            # in-RAM full MoE ceiling

    model_d, cfg_d = load(disk_dir)
    ctrl = XLController(model_d, cfg_d, disk_dir, "lazy_full")
    Li = logits_path(model_d, ids_prefill, decode)
    ctrl.disable()
    di = float(mx.abs(Li - Lc).max())
    ok_i = di <= GATE
    results.append(dict(name="G3i_lazy_full_disk_vs_inram_fullmoe",
                        max_abs_logit_delta=di, bitwise=(di == 0.0),
                        verdict="PASS" if ok_i else "FAIL"))
    all_ok &= ok_i

    # ---- (ii) deploy disk == deploy_ref in-RAM ----
    ctrl = XLController(model_d, cfg_d, disk_dir, "deploy", exact_prefill=True)
    Ld = logits_path(model_d, ids_prefill, decode)
    ctrl.disable()

    ctrl_r = TemporalController(model_r, "deploy_ref")
    Lr = logits_path(model_r, ids_prefill, decode)
    ctrl_r.disable()
    dii = float(mx.abs(Ld - Lr).max())
    argmax_ok = bool(mx.all(mx.argmax(Ld, -1) == mx.argmax(Lr, -1)).item())
    ok_ii = dii <= GATE and argmax_ok
    results.append(dict(name="G3ii_deploy_disk_vs_deploy_ref",
                        max_abs_logit_delta=dii, bitwise=(dii == 0.0),
                        argmax_match=argmax_ok,
                        verdict="PASS" if ok_ii else "FAIL"))
    all_ok &= ok_ii

    # ---- (iii) floor bytes audit ----
    N = 2
    ctrl = XLController(model_d, cfg_d, disk_dir, "floor", N=N)
    rng = np.random.default_rng(2)
    cache = model_d.make_cache()
    model_d(mx.array(rng.integers(0, cfg_d["vocab_size"], size=(1, 16))), cache=cache)
    mx.eval(*[c.state for c in cache])
    for _ in range(4):
        mx.eval(model_d(mx.array(rng.integers(0, cfg_d["vocab_size"], size=(1, 1))),
                        cache=cache))
    copied = ctrl.copied_bytes
    ctrl.disable()
    exp_per_token = N * L * EXPERT_BYTES
    exp_total = 4 * exp_per_token
    ok_iii = copied == exp_total
    results.append(dict(name="G3iii_floor_n2_bytes_audit", N=N, n_moe_layers=L,
                        per_expert_bytes=EXPERT_BYTES,
                        copied_bytes_total=copied, expected_total=exp_total,
                        copied_bytes_per_token=copied // 4,
                        expected_per_token=exp_per_token,
                        verdict="PASS" if ok_iii else "FAIL"))
    all_ok &= ok_iii

    print(json.dumps({"results": results, "G3": "PASS" if all_ok else "FAIL",
                      "tmp": str(tmp)}, indent=2))
    # clean up the ~2 GB truncated models
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
