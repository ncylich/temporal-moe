#!/usr/bin/env python3
"""Gate G4 exactness proofs for the ROUTER-EARLY variant (deploy_early).

Mirrors the A6000 fork's "bit-identical overlap" standard: overlapping the
expert fetch with attention may change timing, never math.

  G4-i  (bit-identical overlap): deploy_early vs deploy_early +
        TEMPORAL_EARLY_NOOVERLAP=1 -> identical greedy token ids AND
        max |delta logit| == 0 over 8 decode tokens after a 64-token prefill
        (fine model, regime-2 disk pool env set). The overlapped fetch and the
        blocking control read the same disk bytes into a discarded buffer; the
        expert GEMM reads the cold pool through effective ids in both, so the
        two paths are the same graph -- any nonzero delta is a bug.
  G4-ii (bytes audit): deploy_early copied_bytes/token ==
        n_moe_layers * per_expert_bytes exactly (one single-expert fetch per
        layer per token).

Requires the regime-2 disk tier: TEMPORAL_DISK_POOL must point at the >RAM
pool file (28.5 GiB = 45 x 1024 x 663552 B). Prints a JSON verdict; exit 0 =
all PASS.
"""
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import load  # noqa: E402
from temporal import TemporalController  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FINE = ROOT / "models" / "qwen3moe-rand-fine-q4"
PER_EXPERT_BYTES = 663552  # measured q4 g64 fine expert
GATE = 1e-5


def greedy_logits(model, ids_prefill, n):
    """Greedy decode n tokens from a prefill; return (ids, stacked step logits).

    Deterministic argmax feeds the next step, so two math-identical paths yield
    identical ids and logits."""
    cache = model.make_cache()
    lg = model(ids_prefill, cache=cache)
    y = mx.argmax(lg[:, -1, :], axis=-1)
    mx.eval(y, *[c.state for c in cache])
    ids, outs = [], []
    for _ in range(n):
        lg = model(y[None], cache=cache)          # [1,1,V]
        outs.append(lg[:, -1, :])
        y = mx.argmax(lg[:, -1, :], axis=-1)      # [1]
        mx.eval(y)
        ids.append(int(y.item()))
    L = mx.stack(outs, axis=0).astype(mx.float32)
    mx.eval(L)
    return ids, L


def run_early(model, config, nooverlap):
    """One deploy_early run (overlap or the NOOVERLAP control)."""
    os.environ["TEMPORAL_EARLY_NOOVERLAP"] = "1" if nooverlap else "0"
    ctrl = TemporalController(model, "deploy", N=0)
    ctrl.router_early = True
    ctrl.reset()
    rng = np.random.default_rng(1)
    ids_prefill = mx.array(rng.integers(0, config["vocab_size"], size=(1, 64)))
    ids, L = greedy_logits(model, ids_prefill, 8)
    copied = ctrl.copied_bytes
    ctrl.disable()
    return ids, L, copied


def main():
    if not os.environ.get("TEMPORAL_DISK_POOL"):
        print(json.dumps({"error": "set TEMPORAL_DISK_POOL to the >RAM pool file"}))
        sys.exit(2)

    model, config = load(FINE)
    n_layers = config["num_hidden_layers"]

    ids_ov, L_ov, copied_ov = run_early(model, config, nooverlap=False)
    ids_no, L_no, _ = run_early(model, config, nooverlap=True)

    delta = float(mx.abs(L_ov - L_no).max())
    ids_match = ids_ov == ids_no
    ok_i = (delta == 0.0) and ids_match
    r_i = dict(name="G4-i_bit_identical_overlap",
               max_abs_logit_delta=delta, bitwise=(delta == 0.0),
               ids_match=ids_match, ids_overlap=ids_ov, ids_nooverlap=ids_no,
               verdict="PASS" if ok_i else "FAIL")

    # bytes audit over the overlap run: 8 decode tokens, one fetch/layer/token
    per_token = copied_ov // 8
    exp_per_token = n_layers * PER_EXPERT_BYTES
    ok_ii = (copied_ov == 8 * exp_per_token)
    r_ii = dict(name="G4-ii_bytes_audit", n_moe_layers=n_layers,
                per_expert_bytes=PER_EXPERT_BYTES,
                copied_bytes_total=copied_ov,
                copied_bytes_per_token=per_token,
                expected_per_token=exp_per_token,
                verdict="PASS" if ok_ii else "FAIL")

    all_ok = ok_i and ok_ii
    print(json.dumps({"results": [r_i, r_ii],
                      "G4": "PASS" if all_ok else "FAIL"}, indent=2))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
