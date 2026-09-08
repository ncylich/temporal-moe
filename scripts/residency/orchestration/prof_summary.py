#!/usr/bin/env python3
"""Summarise a torch-profiler Chrome trace: top ops by CPU self time and by CUDA time,
plus GPU busy fraction over the captured window. Usage: prof_summary.py <trace.json[.gz]>"""
import sys, json, gzip, collections
p=sys.argv[1]; f=gzip.open(p,"rt") if p.endswith(".gz") else open(p)
ev=json.load(f)["traceEvents"]
cpu=collections.Counter(); rt=collections.Counter(); rtn=collections.Counter(); gpu=collections.Counter(); gpu_busy=0; t0=1e30; t1=0
for e in ev:
    if e.get("ph")!="X": continue
    name=e.get("name","?"); dur=e.get("dur",0); cat=e.get("cat","")
    if cat in ("kernel","gpu_memcpy","gpu_memset"): gpu[name]+=dur; gpu_busy+=dur; t0=min(t0,e["ts"]); t1=max(t1,e["ts"]+dur)
    elif cat in ("cpu_op","user_annotation"): cpu[name]+=dur
    elif cat in ("cuda_runtime","cuda_driver"): rt[name]+=dur; rtn[name]+=1
win=(t1-t0) if t1>t0 else 1
print(f"window {win/1e6:.1f}s  GPU busy {gpu_busy/win*100:.0f}% (kernels summed, may exceed 100% w/ streams)")
print("== top CPU ops (s)"); [print(f"{v/1e6:8.2f}  {k[:90]}") for k,v in cpu.most_common(15)]
print("== CUDA runtime/driver API (s, calls)"); [print(f"{v/1e6:8.2f} {rtn[k]:8d}  {k[:70]}") for k,v in rt.most_common(12)]
print("== top GPU kernels (s)"); [print(f"{v/1e6:8.2f}  {k[:90]}") for k,v in gpu.most_common(12)]
cats=collections.Counter()
for k,v in gpu.items():
    kl=k.lower()
    c=("attention" if "sdpa" in kl or "flash" in kl or "fmha" in kl else "gemm" if "gemm" in kl or "nvjet" in kl or "cutlass" in kl or "wgmma" in kl or "sm90_xmma" in kl
       else "sort/permute" if "sort" in kl or "permute" in kl or "chunk" in kl or "scatter" in kl or "gather" in kl or "index" in kl
       else "elementwise" if "elementwise" in kl or "vectorized" in kl or "unrolled" in kl or "fused" in kl or "triton" in kl
       else "memcpy" if "memcpy" in kl or "memset" in kl or "copy" in kl else "other")
    cats[c]+=v
print("== GPU time by category (s):", ", ".join(f"{c} {v/1e6:.1f}" for c,v in cats.most_common()))
