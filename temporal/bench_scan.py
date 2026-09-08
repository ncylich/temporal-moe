#!/usr/bin/env python3
"""Benchmark for the rolling-residency scan fast path (`_triton_scan` / `_scan_kernel`).

The scan is called once per MoE layer (40-48x per forward), so its cost has two parts that must be
reported separately or the numbers are meaningless:

  * per-call FIXED overhead (us/call) -- torch cold fill (topk+scatter), tensor allocs, kernel
    launch, kernel prologue. Paid 40-48x per step regardless of sequence length. Measured two ways:
    (a) the intercept of a least-squares line through t(S) over the linear region, and
    (b) the directly measured t(S=1), where the scan loop body never executes.
  * per-token MARGINAL cost (ns/token) -- the slope of that line, i.e. what one more sequence
    position costs. This is the sequential recurrence itself.

Total for a given S is then approximately  fixed + slope * S, and both terms are reported.

Timing uses back-to-back calls between two torch.cuda.synchronize() barriers after a warmup that
covers Triton JIT compilation. Wall time per call is the number the caller actually pays; the
CUDA-event time over the same loop is reported alongside so a CPU-launch-bound regime (event time
well below wall time) is visible rather than hidden.

Usage:
    python -m temporal.bench_scan                      # all shapes, lru
    python -m temporal.bench_scan --evict min_logit
    python -m temporal.bench_scan --impl triton --json out.json
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal import temporal_router as tr  # noqa: E402


# (name, E, k, B, S_target) -- S_target is the shape this configuration is actually run at.
SHAPES = [
    ("Qwen3.5      (target)", 256, 8, 1, 2048),
    ("Qwen3-30B",             128, 8, 2, 2048),
    ("OLMoE",                  64, 8, 4, 4096),
    ("FLAME-MoE",             192, 18, 8, 2048),
]

# Sequence lengths used for the linear fit. S=1 is measured separately as the direct fixed-cost
# probe (the t=1..S-1 loop body never runs there).
FIT_S = [256, 512, 1024, 2048]


def _make_logits(S, B, E, seed=0, device="cuda", dtype=torch.float32):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(S, B, E, generator=g, device=device, dtype=dtype)


def _run(impl, logits, k, use_lru):
    if impl == "triton":
        return tr._triton_scan(logits, k, use_lru)
    if impl == "eager":
        return tr.compute_resident_mask(logits, k, "lru" if use_lru else "min_logit")
    raise ValueError(impl)


def time_call(impl, logits, k, use_lru, min_ms=60.0, warmup=5):
    """Return (wall_us_per_call, cpu_dispatch_us_per_call). Correct sync, Triton JIT warmed.

    wall_us is measured with the synchronize() INSIDE the timed region -- this kernel is
    asynchronous and the CPU runs far ahead of it, so timing that stops before the barrier measures
    only python/dispatch cost and understates the real cost by >10x. cpu_us is that dispatch cost,
    reported separately: when cpu_us << wall_us the shape is GPU-bound, and when they are equal the
    launch path is the bottleneck.
    """
    for _ in range(warmup):
        _run(impl, logits, k, use_lru)
    torch.cuda.synchronize()

    # calibrate iteration count so the timed region is >= min_ms
    t0 = time.perf_counter()
    _run(impl, logits, k, use_lru)
    torch.cuda.synchronize()
    one = (time.perf_counter() - t0) * 1e3
    n = max(3, min(2000, int(min_ms / max(one, 1e-3)) + 1))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _run(impl, logits, k, use_lru)
    t_disp = time.perf_counter()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / n, (t_disp - t0) * 1e6 / n


def _fit(xs, ys):
    """Least-squares y = a + b*x -> (a intercept, b slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def bench_shape(impl, name, E, k, B, S_target, evict, seed=0):
    use_lru = evict == "lru"
    big = _make_logits(max(max(FIT_S), S_target), B, E, seed=seed)

    walls, cpus = [], []
    for S in FIT_S:
        w, c = time_call(impl, big[:S].contiguous(), k, use_lru)
        walls.append(w)
        cpus.append(c)
    fixed_fit, slope = _fit([float(s) for s in FIT_S], walls)          # us/call, us/token

    w1, c1 = time_call(impl, big[:1].contiguous(), k, use_lru)         # direct fixed-cost probe
    wt, ct = time_call(impl, big[:S_target].contiguous(), k, use_lru)  # cost at the real S

    return {
        "name": name, "E": E, "k": k, "B": B, "S_target": S_target, "evict": evict, "impl": impl,
        "fixed_fit_us": fixed_fit, "fixed_s1_us": w1, "cpu_dispatch_us": c1,
        "slope_ns_per_token": slope * 1e3,
        "total_us_at_S": wt, "cpu_us_at_S": ct,
        "tok_per_s": S_target * 1e6 / wt,
        "fit_S": list(FIT_S), "fit_wall_us": walls, "fit_cpu_us": cpus,
    }


def check_correctness(evict, shapes=((257, 3, 128, 8), (129, 2, 256, 8), (200, 2, 192, 18))):
    """Cheap guard so a benchmark run can never report numbers for a broken kernel."""
    for S, B, E, k in shapes:
        for mk in (lambda: _make_logits(S, B, E, seed=5),
                   lambda: torch.randint(0, 4, (S, B, E), device="cuda").float()):
            lg = mk()
            ref = tr.compute_resident_mask(lg, k, evict)
            got = tr._triton_scan(lg, k, evict == "lru")
            if not torch.equal(got, ref):
                raise SystemExit(f"BENCH ABORT: kernel != reference at S={S} B={B} E={E} k={k} "
                                 f"evict={evict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evict", default="lru", choices=["lru", "min_logit"])
    ap.add_argument("--impl", default="triton", choices=["triton", "eager"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    torch.cuda.init()
    print(f"# device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"triton {getattr(__import__('triton'), '__version__', '?')}")
    print(f"# impl={args.impl} evict={args.evict} {args.tag}")

    if not args.skip_check and args.impl == "triton":
        check_correctness(args.evict)
        print("# correctness vs eager reference: OK")

    rows = [bench_shape(args.impl, *s, evict=args.evict) for s in SHAPES]

    hdr = (f"{'shape':<24} {'E':>4} {'k':>3} {'B':>3} {'S':>5} | {'fixed(fit)':>10} {'fixed(S=1)':>10}"
           f" | {'ns/token':>9} | {'total us':>9} {'cpu us':>7} {'Mtok/s':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<24} {r['E']:>4} {r['k']:>3} {r['B']:>3} {r['S_target']:>5} | "
              f"{r['fixed_fit_us']:>9.2f}u {r['fixed_s1_us']:>9.2f}u | "
              f"{r['slope_ns_per_token']:>9.1f} | "
              f"{r['total_us_at_S']:>9.1f} {r['cpu_us_at_S']:>7.1f} {r['tok_per_s']/1e6:>7.3f}")
    print("\nfixed  = per-call overhead in microseconds (lower better): cold fill + allocs + launch.")
    print("ns/token = marginal cost of one more sequence position (lower better), slope of t(S).")
    print("total  = wall microseconds for one full call at the stated S, GPU work included.")
    print("cpu us = python/dispatch time only; total-cpu is the GPU-bound part.")
    print("Mtok/s = S/total (sequence positions per second, higher better); B is folded in, not "
          "multiplied out.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"# wrote {args.json}")


if __name__ == "__main__":
    main()
