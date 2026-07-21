#!/usr/bin/env python3
"""Gate G2 exactness proofs for the temporal machinery (PLAN.md Phase 2).

  G2a  (tiny, seconds): noswap temporal path with R=E + identity remap, zero
        swaps == stock forward. Expect bitwise-equal logits.
  G2b-i (full fine, 8 decode after 64 prefill): lazy_full temporal path (every
        selected non-resident expert really copied into a slot, budget=R, GEMM
        reads slots) == ceiling logits. The fork's NOFORCE1 bit-identical proof
        that the load/swap/remap/GEMM infra is exact.
  G2b-ii (full fine, 8 decode): deploy_sync (hot slots + real copies) == a
        no-slots reference emulator computing the SAME masked-routing semantics
        straight from the cold pool. Identical argmax ids AND max logit delta.
  floor bytes audit: 4-token run at N=2 (fine) -> copied_bytes/token ==
        N * n_moe_layers * per_expert_bytes exactly.

Teacher-forced decode ids (fixed, identical across paths) isolate the MoE
computation from greedy divergence. Prints a JSON verdict; exit 0 = all PASS.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import load  # noqa: E402
from temporal import TemporalController  # noqa: E402
from tests.g1_tiny import build_tiny  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FINE = ROOT / "models" / "qwen3moe-rand-fine-q4"
PER_EXPERT_BYTES = 663552  # measured q4 g64 fine expert (config _build.per_expert_bytes)
GATE = 1e-5


def logits_path(model, ids_prefill, decode_ids):
    """Teacher-forced: prefill then feed each fixed decode id; stack step logits.

    Returns [1+T, 1, V] (prefill-last logit + one per decode step)."""
    cache = model.make_cache()
    out = []
    lg = model(ids_prefill, cache=cache)
    out.append(lg[:, -1, :])
    for t in decode_ids:
        lg = model(mx.array([[int(t)]]), cache=cache)
        out.append(lg[:, -1, :])
    L = mx.stack(out, axis=0).astype(mx.float32)
    mx.eval(L)
    return L


def g2a():
    model, args = build_tiny()
    E, V = args.num_experts, args.vocab_size
    rng = np.random.default_rng(0)
    ids_prefill = mx.array(rng.integers(0, V, size=(1, 12)))
    decode = rng.integers(0, V, size=8)

    Lc = logits_path(model, ids_prefill, decode)          # stock ceiling
    ctrl = TemporalController(model, "noswap", R=E)        # R=E, identity remap
    Lt = logits_path(model, ids_prefill, decode)
    ctrl.disable()

    delta = float(mx.abs(Lt - Lc).max())
    argmax_ok = bool(mx.all(mx.argmax(Lt, -1) == mx.argmax(Lc, -1)).item())
    ok = delta <= GATE and argmax_ok
    return dict(name="G2a_noswap_R=E_vs_stock", max_abs_logit_delta=delta,
                bitwise=(delta == 0.0), argmax_match=argmax_ok,
                verdict="PASS" if ok else "FAIL"), ok


def g2b(model, config):
    V = config["vocab_size"]
    rng = np.random.default_rng(1)
    ids_prefill = mx.array(rng.integers(0, V, size=(1, 64)))
    decode = rng.integers(0, V, size=8)

    Lc = logits_path(model, ids_prefill, decode)          # ceiling

    ctrl = TemporalController(model, "lazy_full")         # R=k, budget=R
    Li = logits_path(model, ids_prefill, decode)
    ctrl.disable()
    di = float(mx.abs(Li - Lc).max())
    ok_i = di <= GATE
    r_i = dict(name="G2b-i_lazy_full_vs_ceiling", max_abs_logit_delta=di,
               bitwise=(di == 0.0), verdict="PASS" if ok_i else "FAIL")

    ctrl = TemporalController(model, "deploy")            # hot slots + copies
    Ld = logits_path(model, ids_prefill, decode)
    ctrl.disable()
    ctrl = TemporalController(model, "deploy_ref")        # cold-pool reference
    Lr = logits_path(model, ids_prefill, decode)
    ctrl.disable()
    dii = float(mx.abs(Ld - Lr).max())
    argmax_ok = bool(mx.all(mx.argmax(Ld, -1) == mx.argmax(Lr, -1)).item())
    ok_ii = dii <= GATE and argmax_ok
    r_ii = dict(name="G2b-ii_deploy_vs_reference", max_abs_logit_delta=dii,
                argmax_match=argmax_ok, verdict="PASS" if ok_ii else "FAIL")

    # G2b-iii: masked SPLIT op order (fork TEMPORAL_UNIFIED_OVERLAP analog,
    # the structure the DISK deploy/floor rows use): fetched-expert
    # contribution moved last and summed separately ((k-1)-sum + 1-sum).
    # Splitting the weighted sum changes the float reduction order, so it is
    # gated against the reference emulator running the SAME split order --
    # both sides same math, same order, expected bitwise.
    ctrl = TemporalController(model, "deploy")
    ctrl.split_order = True
    Lds = logits_path(model, ids_prefill, decode)
    ctrl.disable()
    ctrl = TemporalController(model, "deploy_ref")
    ctrl.split_order = True
    Lrs = logits_path(model, ids_prefill, decode)
    ctrl.disable()
    diii = float(mx.abs(Lds - Lrs).max())
    argmax_ok3 = bool(mx.all(mx.argmax(Lds, -1) == mx.argmax(Lrs, -1)).item())
    ok_iii = diii <= GATE and argmax_ok3
    r_iii = dict(name="G2b-iii_deploy_splitorder_vs_reference",
                 max_abs_logit_delta=diii, argmax_match=argmax_ok3,
                 verdict="PASS" if ok_iii else "FAIL")
    return [r_i, r_ii, r_iii], (ok_i and ok_ii and ok_iii)


def floor_bytes_audit(model, config):
    n_layers = config["num_hidden_layers"]
    N = 2
    ctrl = TemporalController(model, "floor", N=N)
    rng = np.random.default_rng(2)
    cache = model.make_cache()
    model(mx.array(rng.integers(0, config["vocab_size"], size=(1, 16))), cache=cache)
    mx.eval(*[c.state for c in cache])
    for _ in range(4):                                    # 4 decode tokens
        lg = model(mx.array(rng.integers(0, config["vocab_size"], size=(1, 1))),
                   cache=cache)
        mx.eval(lg)
    ctrl.disable()

    expected_total = 4 * N * n_layers * PER_EXPERT_BYTES
    per_token = ctrl.copied_bytes // 4
    exp_per_token = N * n_layers * PER_EXPERT_BYTES
    ok = ctrl.copied_bytes == expected_total
    return dict(name="floor_n2_bytes_audit", N=N, n_moe_layers=n_layers,
                per_expert_bytes=PER_EXPERT_BYTES,
                copied_bytes_total=ctrl.copied_bytes,
                expected_total=expected_total,
                copied_bytes_per_token=per_token,
                expected_per_token=exp_per_token,
                verdict="PASS" if ok else "FAIL"), ok


def main():
    results = []
    all_ok = True

    ra, ok = g2a()
    results.append(ra)
    all_ok &= ok

    model, config = load(FINE)
    rb, ok = g2b(model, config)
    results += rb
    all_ok &= ok
    rf, ok = floor_bytes_audit(model, config)
    results.append(rf)
    all_ok &= ok

    print(json.dumps({"results": results,
                      "G2": "PASS" if all_ok else "FAIL"}, indent=2))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
