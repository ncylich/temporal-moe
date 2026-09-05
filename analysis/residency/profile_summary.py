#!/usr/bin/env python3
"""Summarise a torch.profiler trace (json or json.gz, tensorboard_trace_handler output) by CUDA
kernel: total kernel time per kernel name (top N) and per coarse group (gemm, attention, moe
dispatch, softmax/cross-entropy, residency scan, optimizer, other), with the wall span covered.

    $PY analysis/residency/profile_summary.py <trace.json[.gz]> [--top 25]
"""
import gzip, json, re, sys
from collections import defaultdict

GROUPS = [("gemm", r"gemm|cutlass|nvjet|Cijk|sm90_xmma|ampere_|cublas"), ("attention", r"attn|flash|fmha|cudnn.*(fused|attention)"),
          ("moe dispatch", r"permute|unpermute|sort|scatter|gather|index_|topk|Topk|bincount|cumsum|group|repeat_interleave"),
          ("softmax / cross-entropy", r"softmax|cross_entropy|logsumexp|nll|log_softmax|exp_kernel"),
          ("residency scan", r"scan|resident|triton"), ("norm / elementwise", r"norm|elementwise|vectorized|fill|copy|cast|add_|mul_|silu|swiglu|gelu|rotary|rope"),
          ("optimizer", r"adam|multi_tensor|lamb|zero_"), ("reduce", r"reduce|nccl|allreduce|all_gather")]

def main():
    path = sys.argv[1]; top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    op = gzip.open if path.endswith(".gz") else open
    ev = json.load(op(path, "rt"))["traceEvents"]
    kern = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
    if not kern:
        kern = [e for e in ev if e.get("cat") in ("Kernel", "gpu_op", "gpu_memcpy", "gpu_memset") and "dur" in e]
    by = defaultdict(float)
    for e in kern:
        by[e["name"]] += e["dur"]
    total = sum(by.values()); span = (max(e["ts"] + e["dur"] for e in kern) - min(e["ts"] for e in kern)) if kern else 0
    print(f"kernels {len(kern)}, kernel time {total / 1e3:.0f} ms, wall span {span / 1e3:.0f} ms, GPU busy {100 * total / max(span, 1):.0f}%")
    groups = defaultdict(float)
    for name, t in by.items():
        g = next((gn for gn, pat in GROUPS if re.search(pat, name, re.I)), "other")
        groups[g] += t
    print("\nby group (ms, share of kernel time):")
    for g, t in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g:26s} {t / 1e3:8.0f}  {100 * t / total:5.1f}%")
    print(f"\ntop {top} kernels (ms, share):")
    for name, t in sorted(by.items(), key=lambda x: -x[1])[:top]:
        print(f"  {t / 1e3:8.1f}  {100 * t / total:5.1f}%  {name[:110]}")

if __name__ == "__main__":
    main()
