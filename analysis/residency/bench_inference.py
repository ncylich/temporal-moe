#!/usr/bin/env python3
"""Where does inference time actually go, and what does residency cost on top of a fast baseline?

Everything measured so far has been of the *hooked* model, so there is no reference point: a slow
number could be HuggingFace's MoE implementation, our residency machinery, or the scan. Optimising
without that decomposition is guesswork, and it produced two rounds of it.

Four variants, one model load, identical inputs:

    stock            HF forwards exactly as shipped. The baseline everything else is judged against.
    stock+fast       only the expert-loop sync fix. Isolates how much of the baseline's cost is
                     `for e in expert_hit:` iterating a CUDA tensor -- one device->host copy per
                     expert per layer, ~6k stalls per forward at 128 experts over 48 layers.
    hook_free        our router hook installed, every layer free. Residency is inert here, so any
                     gap to stock+fast is pure machinery overhead: an extra F.linear for the router
                     logits plus the capture/branch bookkeeping.
    hook_R8          residency active. The gap to hook_free is the resident-set scan, which is a
                     sequential scan over the sequence dimension and is the part that could plausibly
                     dominate at long context.

Reported as tokens/sec at several batch sizes, because the sync cost is per forward call and
therefore amortises with batch -- which is exactly why batch size looked like the lever earlier when
the real problem was the syncs.

    bench_inference.py --model /workspace/qwen3moe-adapt/model --family qwen3
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_qwen as RQ                                        # noqa: E402


def restore_stock(family):
    """Put back the shipped forwards, so 'stock' means stock."""
    blk, rtr = RQ.FAMILIES[family]
    blk.forward, rtr.forward = RQ._ORIG[family]
    for mod, cls in (("qwen3_5_moe", "Qwen3_5MoeExperts"), ("qwen3_moe", "Qwen3MoeExperts")):
        m = __import__(f"transformers.models.{mod}.modeling_{mod}", fromlist=[cls])
        if cls in _ORIG_EXPERTS:
            getattr(m, cls).forward = _ORIG_EXPERTS[cls]


_ORIG_EXPERTS = {}


@torch.no_grad()
def timed(model, ids, reps=3):
    torch.cuda.synchronize()
    model(ids)                                                     # warm
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model(ids)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    return ids.numel() / dt, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/qwen3moe-adapt/model")
    ap.add_argument("--family", default="qwen3")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batches", default="1,4,8,16")
    ap.add_argument("--reps", type=int, default=3)
    A = ap.parse_args()

    # capture the shipped expert forwards before anything patches them
    for mod, cls in (("qwen3_5_moe", "Qwen3_5MoeExperts"), ("qwen3_moe", "Qwen3MoeExperts")):
        m = __import__(f"transformers.models.{mod}.modeling_{mod}", fromlist=[cls])
        _ORIG_EXPERTS[cls] = getattr(m, cls).forward

    model, tok = RQ.load_model(path=A.model, family=A.family)
    E, L = model.config.num_experts, model.config.num_hidden_layers
    V = model.config.vocab_size
    print(f"  E={E} layers={L} vocab={V} seq={A.seq}\n", flush=True)

    variants = [
        ("stock", lambda: restore_stock(A.family)),
        ("hook_free", lambda: (RQ.install(A.family, fast_experts=False),
                               RES._CFG.update(on=True, R=8, collect_telem=False),
                               RES.set_free_layers(list(range(L))))),
        ("hook_R8", lambda: (RQ.install(A.family, fast_experts=False),
                             RES._CFG.update(on=True, R=8, collect_telem=False),
                             RES.set_free_layers(None))),
    ]
    print(f"  {'variant':14}" + "".join(f"{'bs='+b:>16}" for b in A.batches.split(",")), flush=True)
    base = {}
    for name, setup in variants:
        setup()
        row = []
        for b in [int(x) for x in A.batches.split(",")]:
            ids = torch.randint(0, V, (b, A.seq), device="cuda")
            try:
                tps, dt = timed(model, ids, A.reps)
                row.append(f"{tps:,.0f} tok/s")
                base.setdefault(b, {})[name] = tps
            except torch.OutOfMemoryError:
                row.append("OOM")
                torch.cuda.empty_cache()
            del ids
        print(f"  {name:14}" + "".join(f"{v:>16}" for v in row), flush=True)

    print("\n  === decomposition (higher is better; ratios vs stock) ===", flush=True)
    for b, d in sorted(base.items()):
        if "stock" not in d:
            continue
        s = d["stock"]
        parts = "  ".join(f"{k} {d[k]/s:.2f}x" for k in ("hook_free", "hook_R8") if k in d)
        print(f"  bs={b:<4} stock {s:,.0f} tok/s   {parts}", flush=True)
    print("\n  stock+fast/stock  = value of removing the per-expert GPU syncs")
    print("  hook_free/stock+fast = cost of the residency machinery when it is INERT")
    print("  hook_R8/hook_free    = cost of the resident-set scan itself", flush=True)
    print("=== BENCH COMPLETE ===", flush=True)


def _patch_fast():
    for mod, cls in (("qwen3_5_moe", "Qwen3_5MoeExperts"), ("qwen3_moe", "Qwen3MoeExperts")):
        m = __import__(f"transformers.models.{mod}.modeling_{mod}", fromlist=[cls])
        getattr(m, cls).forward = RQ._experts_forward_fast


if __name__ == "__main__":
    main()
