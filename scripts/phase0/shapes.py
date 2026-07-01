#!/usr/bin/env python3
"""Compute active (non-embedding) params N and token budget D per shape so that C=6ND.

N = active non-embedding params (matches FLAME law intent: s2~7.8M near 1e17 optimum).
Architecture (fixed): swiglu FFN, num_experts=64, top-k=6, 1 shared expert
(intermediate=2*moe_ffn), moe_layer_freq=[0]+[1]*(L-1) (dense layer-0, MoE rest),
no biases, RoPE, RMSNorm. Norm params negligible but included.
"""
SEQ = 2048
SHAPES = {
    "sm1": dict(h=96,  L=4, ffn=512,  moe_ffn=66),   # s_-1: ~0.77M, left arm of 1e16 parabola
    "s0":  dict(h=128, L=4, ffn=684,  moe_ffn=88),   # ~1.36M, near the 1e16 optimum (min)
    "s1": dict(h=192, L=5,  ffn=1026, moe_ffn=132),
    "s2": dict(h=256, L=6,  ffn=1368, moe_ffn=176),
    "s3": dict(h=320, L=7,  ffn=1710, moe_ffn=220),
    "s4": dict(h=384, L=8,  ffn=2052, moe_ffn=264),
    "s5": dict(h=448, L=9,  ffn=2394, moe_ffn=308),
    "s6": dict(h=512, L=10, ffn=2736, moe_ffn=352),
}
import os
# GRAIN>1: fine-grain the ROUTED experts (see run.sh). num_experts and top-k scale by GRAIN; each
# routed expert's moe_ffn shrinks to even-rounded moe_ffn/GRAIN; the shared expert is unchanged.
# Active routed FLOPs (topk*moe_ffn) stay ~fixed, so N moves only by the router term (h*NEXP, x GRAIN)
# plus sub-% even-rounding drift. Must match run.sh exactly so C=6ND hits the budget.
GRAIN = int(os.environ.get("GRAIN", "1"))
TOPK, SHARED_MULT, NEXP = 6 * GRAIN, 2, 64 * GRAIN

def active_nonembed(h, L, ffn, moe_ffn):
    moe_ffn_routed = 2 * round((moe_ffn / GRAIN) / 2) if GRAIN != 1 else moe_ffn
    attn = 4 * h * h                      # qkv (3h^2) + out (h^2), no bias
    norm = 2 * h                          # 2 RMSNorms/layer (gamma only)
    dense_ffn = 3 * h * ffn               # swiglu: fc1=2*h*ffn, fc2=h*ffn
    router = h * NEXP
    shared = 3 * h * (SHARED_MULT * moe_ffn)          # shared from ORIGINAL moe_ffn (not grained)
    routed_active = TOPK * (3 * h * moe_ffn_routed)
    moe = router + shared + routed_active
    n = 0
    for layer in range(L):
        n += attn + norm
        n += dense_ffn if layer == 0 else moe
    return n

def fmt(x):
    return f"{x/1e6:.2f}M"

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "iters":
        # iters <shape> <flops> <global_batch>  -> prints "N ITERS"
        _, _, shape, flops, gb = sys.argv[:5]
        N = active_nonembed(**SHAPES[shape])
        D = float(flops) / (6 * N)
        iters = round(D / (int(gb) * SEQ))
        print(f"{N} {iters}")
        sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "N":
        print(active_nonembed(**SHAPES[sys.argv[2]]))
        sys.exit(0)
    print(f"{'shape':5} {'N_active':>10} | "
          f"{'D@1e16':>9} {'it@1e16(gb256)':>15} | "
          f"{'D@1e17':>9} {'it@1e17(gb256)':>15}")
    for name, s in SHAPES.items():
        N = active_nonembed(**s)
        row = f"{name:5} {fmt(N):>10} |"
        for C in (1e16, 1e17):
            D = C / (6 * N)
            it = D / (256 * SEQ)
            row += f" {D/1e9:>7.3f}B {it:>14.0f} |"
        print(row)
    print("\nMax tokens needed (s1@1e17):",
          f"{1e17/(6*active_nonembed(**SHAPES['s1']))/1e9:.2f}B")
