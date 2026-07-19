#!/usr/bin/env python3
"""Gate G0 physics probes for the Mac MLX serving benchmark (see mlx-bench/PLAN.md).

Probes:
  1. q4 g64 quantize + gather_qmm correctness vs dequantized matmul.
  2. Cold->hot expert copy bandwidth at the two real expert sizes
     (fine 648 KiB, coarse 1.944 MiB), GPU stream and CPU stream,
     per-expert scatter vs batched-gather styles.
  3. Elision audit: copy time must scale ~linearly with bytes.
  4. Two-stream overlap smoke: copies on a second stream while GEMMs run.

Prints a JSON verdict. Exit 0 = G0 PASS.
"""
import json
import sys
import time

import mlx.core as mx

GROUP, BITS = 64, 4
H = 1024

def sync():
    mx.synchronize()

def timeit(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t0) / reps

def expert_bytes(ff):
    # 3 matrices (gate/up/down) of [ff, H] / [H, ff]; q4 g64 = 4.5 bpw exactly
    per_mat = ff * H  # params
    return int(3 * per_mat * 4.5 / 8)

# ---------------- probe 1: gather_qmm correctness ----------------
def probe_gather_qmm():
    mx.random.seed(0)
    E, ff, k = 8, 384, 3
    w = mx.random.normal((E, ff, H)) * 0.02  # E experts, [ff, H] each (transpose=True convention)
    wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=BITS)
    x = mx.random.normal((1, 1, 1, H))  # one token
    idx = mx.array([[2, 5, 7]], dtype=mx.uint32).reshape(1, 1, k)
    out = mx.gather_qmm(x, wq, sc, bi, rhs_indices=idx, transpose=True,
                        group_size=GROUP, bits=BITS)
    wd = mx.dequantize(wq, sc, bi, group_size=GROUP, bits=BITS)
    ref = mx.stack([x[0, 0, 0] @ wd[i].T for i in [2, 5, 7]])
    err = float(mx.abs(out.reshape(k, ff) - ref).max())
    mx.eval(err)
    return {"max_abs_err": err, "pass": err < 1e-3}

# ---------------- probe 2/3: copy bandwidth + elision audit ----------------
def make_pool(E, ff):
    w = mx.random.normal((E, ff, H))
    wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=BITS)
    mx.eval(wq, sc, bi)
    return wq, sc, bi

N_POOLS = 8  # distinct pools cycled across layers to defeat SLC caching (~1 GB fine)

def copy_bench(E, ff, k, n_layers=45, n_swaps=16, stream=None, pools_cache={}):
    """Emulate one token of floor swaps: n_layers x n_swaps expert copies
    (3 tensors each) cold pool -> hot slots, batched-gather style per layer."""
    key = (E, ff)
    if key not in pools_cache:
        pools_cache[key] = [make_pool(E, ff) for _ in range(N_POOLS)]
    pool_set = pools_cache[key]
    # one hot buffer set PER LAYER (like the real model): writes must not all land
    # in the same SLC-resident 12 MB or write traffic is unrealistically cached
    hots = [[mx.array(p[:k]) for p in pool_set[li % N_POOLS]]
            for li in range(n_layers)]
    for hh in hots:
        mx.eval(*hh)
    step = {"i": 0}
    n = min(n_swaps, k)

    def one_token():
        for li in range(n_layers):
            pools = pool_set[li % N_POOLS]
            # cycle sources to defeat caching, like TEMPORAL_SWAP_N
            s = step["i"] % (E - n)
            step["i"] += 3
            srcs = mx.arange(s, s + n, dtype=mx.uint32)
            new = []
            for h, p in zip(hots[li], pools):
                if stream is None:
                    h[:n] = mx.take(p, srcs, axis=0)
                else:
                    with mx.stream(stream):
                        h[:n] = mx.take(p, srcs, axis=0)
                new.append(h)
            hots[li] = new
        mx.eval(*[a for hh in hots for a in hh])

    t = timeit(one_token, reps=5, warmup=2)
    nb = expert_bytes(ff) * n * n_layers
    return {"ms_per_token": t * 1e3, "bytes_per_token": nb, "GBps": nb / t / 1e9}

# ---------------- probe 4: overlap smoke ----------------
def probe_overlap():
    s2 = mx.new_stream(mx.gpu)
    a = mx.random.normal((2048, 2048))
    b = mx.random.normal((2048, 2048))
    E, ff, k = 192, 384, 18
    wq, sc, bi = make_pool(E, ff)
    hot = mx.array(wq[:k])
    mx.eval(a, b, hot)

    def gemms_only():
        c = a
        for _ in range(20):
            c = c @ b
        mx.eval(c)

    def copies_only():
        h = hot
        for i in range(45):
            with mx.stream(s2):
                h[:k] = mx.take(wq, mx.arange(i, i + k, dtype=mx.uint32), axis=0)
        mx.eval(h)

    def both():
        c = a
        h = hot
        for i in range(45):
            with mx.stream(s2):
                h[:k] = mx.take(wq, mx.arange(i, i + k, dtype=mx.uint32), axis=0)
        for _ in range(20):
            c = c @ b
        mx.eval(c, h)

    tg = timeit(gemms_only)
    tc = timeit(copies_only)
    tb = timeit(both)
    # overlap efficiency: 1.0 = perfectly hidden, 0.0 = fully serialized
    denom = min(tg, tc)
    eff = (tg + tc - tb) / denom if denom > 0 else 0.0
    return {"gemm_ms": tg * 1e3, "copy_ms": tc * 1e3, "both_ms": tb * 1e3,
            "overlap_efficiency": eff}

def main():
    out = {"mlx_version": mx.__version__}
    out["gather_qmm"] = probe_gather_qmm()

    # bandwidth at both granularities, GPU + CPU streams, floor-like swap counts
    out["copy_fine_n16_gpu"] = copy_bench(192, 384, k=18, n_swaps=16)
    out["copy_fine_n16_cpu"] = copy_bench(192, 384, k=18, n_swaps=16, stream=mx.cpu)
    out["copy_coarse_n5_gpu"] = copy_bench(64, 1152, k=6, n_swaps=5)
    out["copy_fine_n1_gpu"] = copy_bench(192, 384, k=18, n_swaps=1)

    # elision audit via MARGINAL bandwidth: fixed dispatch overhead (~2 ms/token for
    # 135 lazy ops) dominates small-N times, so audit (t16-t4)/(b16-b4) instead.
    # Real memory movement => marginal bw in a sane band; elision would look absurd.
    n4 = copy_bench(192, 384, k=18, n_swaps=4)
    out["copy_fine_n4_gpu"] = n4
    dt = (out["copy_fine_n16_gpu"]["ms_per_token"] - n4["ms_per_token"]) / 1e3
    db = out["copy_fine_n16_gpu"]["bytes_per_token"] - n4["bytes_per_token"]
    marg = db / dt / 1e9 if dt > 0 else float("inf")
    out["elision_audit"] = {"marginal_GBps_n4_to_n16": marg,
                            "pass": 20.0 < marg < 300.0}

    out["overlap"] = probe_overlap()  # informational only in G0; refined in Phase 4

    ok = out["gather_qmm"]["pass"] and out["elision_audit"]["pass"]
    out["G0"] = "PASS" if ok else "FAIL"
    print(json.dumps(out, indent=2))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
