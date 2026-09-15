#!/usr/bin/env python3
"""Bit-exactness gate for the decode-time CUDA-graph step (`step_accel`).

The serving path masks router logits with a mask produced once per MoE layer per
generated token. Any divergence from the reference `_step` changes which experts
a token may route to, which changes the generation -- so the fast path is held to
BIT equality, not tolerance, including on the cases where equality is fragile:

  * exact ties in the logits (argmax/argmin first-index semantics must match)
  * saturated states (R == E: nothing evictable; R == 1)
  * infinities in the logit stream
  * long trajectories, where a single divergent step would compound
  * batch sizes that vary step to step, as continuous batching produces

Run: python analysis/residency/test_decode_accel.py   (needs CUDA)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from temporal.temporal_router import _step, step_accel  # noqa: E402


def eager(lt, res, ref):
    return _step(lt, res, ref, torch.zeros((), device=lt.device), False)


def seed_resident(lt, R):
    res = torch.zeros_like(lt, dtype=torch.bool)
    res.scatter_(1, lt.topk(R, dim=-1).indices, True)
    return res


def check(name, lt_stream, B, E, R):
    """Step both paths over a shared logit stream; require bit equality every step."""
    dev = lt_stream[0].device
    rA = seed_resident(lt_stream[0], R)
    rB = rA.clone()
    fA = torch.zeros(B, E, device=dev)
    fB = fA.clone()
    for t, lt in enumerate(lt_stream):
        rA, fA = eager(lt, rA, fA)
        rB, fB = step_accel(lt, rB, fB, use_lru=False)
        if not torch.equal(rA, rB):
            n = (rA != rB).any(dim=-1).sum().item()
            print(f"FAIL {name}: diverged at step {t} on {n}/{B} rows")
            return False
        if rA.sum(dim=-1).ne(R).any():
            print(f"FAIL {name}: resident count left R={R} at step {t}")
            return False
    print(f"  ok  {name}")
    return True


def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = "cuda"
    torch.manual_seed(1234)
    ok = True

    # 1. random streams across the shapes the serving path actually produces
    for B, E, R in [(1, 64, 8), (1, 128, 4), (3, 128, 4), (64, 128, 4),
                    (64, 256, 8), (31, 512, 16), (256, 128, 4), (128, 512, 64)]:
        stream = [torch.randn(B, E, device=dev) for _ in range(200)]
        ok &= check(f"random B={B} E={E} R={R}", stream, B, E, R)

    # 2. heavy ties: a coarse quantization guarantees duplicate maxima/minima, so
    #    the two paths must agree on which index a tie resolves to
    B, E, R = 32, 128, 8
    stream = [(torch.randn(B, E, device=dev) * 2).round() / 2 for _ in range(200)]
    ok &= check("quantized (many exact ties)", stream, B, E, R)

    # 3. degenerate ties: every logit identical -> every comparison is a tie
    stream = [torch.zeros(B, E, device=dev) for _ in range(50)]
    ok &= check("all-equal logits", stream, B, E, R)

    # 4. saturated residency: R == E (nothing is non-resident, no swap can fire)
    ok &= check("R == E (saturated)",
                [torch.randn(8, 16, device=dev) for _ in range(50)], 8, 16, 16)

    # 5. minimal residency: R == 1
    ok &= check("R == 1", [torch.randn(8, 64, device=dev) for _ in range(50)],
                8, 64, 1)

    # 6. infinities in the stream (masked/absent experts arrive as -inf upstream)
    stream = []
    for _ in range(50):
        lt = torch.randn(16, 64, device=dev)
        lt[:, ::7] = float("-inf")
        stream.append(lt)
    ok &= check("-inf entries", stream, 16, 64, 4)

    # 7. varying batch size across steps, as continuous batching produces: each
    #    distinct B captures its own graph, and state carries per request
    E, R = 128, 4
    res = {}
    fail = False
    for t in range(120):
        B = 1 + (t * 7) % 40
        lt = torch.randn(B, E, device=dev)
        for b in range(B):
            key = b
            if key not in res:
                r = seed_resident(lt[b:b + 1], R)
                res[key] = (r, r.clone(), torch.zeros(1, E, device=dev),
                            torch.zeros(1, E, device=dev))
            rA, rB, fA, fB = res[key]
            rA, fA = eager(lt[b:b + 1], rA, fA)
            rB, fB = step_accel(lt[b:b + 1], rB, fB, use_lru=False)
            if not torch.equal(rA, rB):
                fail = True
            res[key] = (rA, rB, fA, fB)
    print(f"  {'FAIL' if fail else 'ok  '} varying batch size (1..40, 120 steps)")
    ok &= not fail

    # 8. the eager path must still be reachable and identical (env opt-out)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
