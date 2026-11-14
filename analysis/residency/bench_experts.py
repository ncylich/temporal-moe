#!/usr/bin/env python3
"""Which expert kernel should we be running? We have been on `eager` -- the Python loop -- by default.

transformers ships an `ExpertsInterface` registry (`batched_mm`, `grouped_mm`, `deepgemm`,
`sonicmoe`) selected by `config._experts_implementation`, and Qwen's config leaves it unset, so every
measurement in this program has used the naive per-expert loop. At batch 64 that loop reaches ~7% of
the H100's bf16 peak: 32768 tokens x top-8 over 128 experts is ~39 TFLOP of expert GEMM per forward,
which should take ~100 ms at 400 TFLOP/s and takes 1.39 s.

Raising batch size, which is what we did first, is not optimisation -- it amortises launch overhead
and nothing else. This measures the actual alternatives.

Correctness is checked, not assumed: each candidate's logits are compared against `eager` on the same
input. A kernel that is faster and wrong is worse than the loop, and grouped/batched paths reorder
accumulation, so bitwise equality is not expected -- the tolerance is reported so the reader can
judge whether the difference is reassociation or a bug.

    bench_experts.py --model /dev/shm/qwen3-30b --family qwen3
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_qwen as RQ                                        # noqa: E402


@torch.no_grad()
def timed(model, ids, reps=3):
    torch.cuda.synchronize()
    out = model(ids).logits.float()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model(ids)
    torch.cuda.synchronize()
    return ids.numel() / ((time.time() - t0) / reps), out


def set_impl(model, name):
    """Set the implementation on every config object the experts modules might consult."""
    seen = []
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None and id(cfg) not in [id(x) for x in seen]:
            cfg._experts_implementation = name
            seen.append(cfg)
    for m in model.modules():
        if hasattr(m, "config") and m.config is not None:
            try:
                m.config._experts_implementation = name
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/qwen3-30b")
    ap.add_argument("--family", default="qwen3")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--impls", default="eager,grouped_mm,batched_mm,deepgemm,sonicmoe")
    A = ap.parse_args()

    model, tok = RQ.load_model(path=A.model, family=A.family)
    # stock forwards: this benchmark is about the expert kernel, not about residency
    blk, rtr = RQ.FAMILIES[A.family]
    blk.forward, rtr.forward = RQ._ORIG[A.family]
    V = model.config.vocab_size
    ids = torch.randint(0, V, (A.batch, A.seq), device="cuda")
    print(f"  E={model.config.num_experts} layers={model.config.num_hidden_layers} "
          f"batch={A.batch} seq={A.seq}\n", flush=True)

    ref = None
    print(f"  {'implementation':16}{'tok/s':>12}{'vs eager':>10}{'max|dlogit| vs eager':>24}", flush=True)
    base = None
    for name in A.impls.split(","):
        set_impl(model, name if name != "eager" else None)
        try:
            tps, out = timed(model, ids, A.reps)
        except Exception as e:
            msg = str(e).split("\n")[0][:60]
            print(f"  {name:16}{'unavailable':>12}   {msg}", flush=True)
            torch.cuda.empty_cache()
            continue
        if ref is None:
            ref, base = out, tps
            d = 0.0
        else:
            d = float((out - ref).abs().max())
        print(f"  {name:16}{tps:>12,.0f}{tps/base:>9.2f}x{d:>24.3e}", flush=True)
        del out
        torch.cuda.empty_cache()

    print("\n  A faster kernel that changes logits materially is not a win. grouped/batched paths"
          "\n  reorder accumulation, so expect ~1e-3 in bf16 from reassociation and treat anything"
          "\n  larger as a correctness problem to investigate before adopting.", flush=True)
    print("=== EXPERTS BENCH COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
