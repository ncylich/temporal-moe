#!/usr/bin/env python3
"""The paper's RTX A6000 decode rows on the fused build, under the paper's own protocol (gpubench/run_decode.py,
llamacpp-bench/run_serving_bench.sh): batch 1, the context filled token by token (-b 1 -ub 1 -d <ctx>), then
100 tokens generated and timed, four trials, the first discarded. One llama-bench process per (arm, context);
peak VRAM is the largest nvidia-smi memory.used during the process, polled every 0.2 s. Passes alternate the
arm order so drift does not decide a difference.

Arms whose name starts with old_ or stock use the kernels the paper's current numbers were taken with, on the
same pod, so a row can be compared with the paper before it is replaced.

    paper_rows.py --bin <dir> --model <gguf> --kind small|big|coarse --ctx 1024 2048 4096 --passes 3 \
        --out rows.jsonl --tag T [--arms ...] [--prefix "taskset -c 0-31"]
    paper_rows.py --summary rows.jsonl
"""
import argparse, json, os, subprocess, sys, threading, time

G = {"TEMPORAL_QKV": "2", "TEMPORAL_ROPE_PAIR": "2", "TEMPORAL_ATTN1": "1"}          # no fused expert op needed
ALL = {**G, "TEMPORAL_FUSED": "3", "TEMPORAL_FUSED_ADD": "1"}
U = {"TEMPORAL_UNIFIED": "1"}


def arms(kind):
    if kind == "small":                                   # trained 1.26B model, 13 MoE layers, no flash attention
        tr = {"LLAMA_TEMPORAL_ROUTE": "trained"}; f = {**tr, **ALL, "TEMPORAL_SHEXP_FIRST": "1"}; nc = ["-ncmoe", "14"]
        return ["-fa", "0", "-t", "8"], {
            "old_resident": (tr, []),                     # the paper's "resident" column today
            "old_stream":   ({**tr, **U}, nc),            # the paper's "active-only" column today
            "stock_topk":   ({}, []),                     # stock engine, stock top-k router
            "topk_fused":   (G, []),                      # stock top-k router, attention-side launches fused
            "resident":     (f, []),
            "nobytes":      ({**f, **U, "TEMPORAL_ROUTE_NOCOPY": "1"}, nc),
            "stream":       ({**f, **U}, nc),
        }
    nc = ["-ncmoe", "48"]
    old = {**U, "TEMPORAL_UNIFIED_OVERLAP": "1"}
    if kind == "big":                                     # 11B random weights, 18 of 192, flash attention
        dec = {"LLAMA_TEMPORAL_ROUTE": "decode", **ALL}; one = {"TEMPORAL_SWAP_HIST": "0,1"}
        sixteen = {"TEMPORAL_SWAP_HIST": ",".join(["0"] * 16 + ["1"])}
        return ["-fa", "1"], {
            "stock":        ({}, []),                     # paper row a
            "stock_fused":  (G, []),
            "old_nobytes":  ({**old, "TEMPORAL_SWAP_PROB": "0"}, nc),      # paper row b
            "old_stream":   ({**old, "TEMPORAL_SWAP_PROB": "1.0"}, nc),    # paper row c
            "old_floor":    ({**U, "TEMPORAL_SWAP_N": "16"}, nc),          # paper row d
            "resident":     (dec, []),
            # same kernels, no load: the histogram path ignores TEMPORAL_ROUTE_NOCOPY, so the load count is pinned to 0
            "nobytes":      ({**dec, **U, "TEMPORAL_SWAP_HIST": "1,0"}, nc),
            "stream":       ({**dec, **U, **one}, nc),
            "floor":        ({**dec, **U, **sixteen}, nc),                 # 16 loads per layer-token through the fused op
        }
    if kind == "coarse":                                  # 11B random weights, 6 of 64: the fused op is not built for this shape
        return ["-fa", "1"], {
            "stock":        ({}, []),
            "stock_fused":  (G, []),
            "old_nobytes":  ({**old, "TEMPORAL_SWAP_PROB": "0"}, nc),
            "old_stream":   ({**old, "TEMPORAL_SWAP_PROB": "1.0"}, nc),
            "old_floor":    ({**U, "TEMPORAL_SWAP_N": "5"}, nc),
            "nobytes":      ({**old, **G, "TEMPORAL_SWAP_PROB": "0"}, nc),
            "stream":       ({**old, **G, "TEMPORAL_SWAP_PROB": "1.0"}, nc),
            "floor":        ({**U, **G, "TEMPORAL_SWAP_N": "5"}, nc),
        }
    raise SystemExit("unknown kind")


def gpu_used():
    out = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else -1


def loads(lines):
    """Loads per layer-token: the imposed histogram's count where there is one, else the router's own swaps."""
    for l in lines:
        if l.startswith("temporal-swap-hist:"):
            f = dict(x.split("=") for x in l.split()[1:3]); return int(f["loads"]) / max(1, int(f["layer_tokens"]))
    for l in lines:
        if "swaps_per_layer_token=" in l and "layer_tokens=0 " not in l:
            return float(l.split("swaps_per_layer_token=")[1].split()[0])
    return None


def summary(path):
    rows = [json.loads(l) for l in open(path)]
    keys = []
    for r in rows:
        k = (r["kind"], r["ctx"], r["arm"])
        if k not in keys: keys.append(k)
    print("kind ctx arm | tok/s pooled (per pass) | ms | peak VRAM MiB | loads per layer-token | notes")
    for k in keys:
        rs = [r for r in rows if (r["kind"], r["ctx"], r["arm"]) == k]
        ok = [r for r in rs if "tok_s" in r]
        if not ok:
            print(*k, "| FAILED", rs[-1].get("error", ""), rs[-1].get("stderr_tail", "")[-300:].replace("\n", " / ")); continue
        tr = [t for r in ok for t in r["samples_ts"][1:]]; m = sum(tr) / len(tr)
        ld = sorted({x for x in (loads(r.get("route_lines", [])) for r in ok) if x is not None})
        said = sorted({s for r in ok for s in r.get("fused_said", [])})
        print(f"{k[0]} {k[1]:>5} {k[2]:13s} | {m:8.1f} ({' '.join('%.1f' % r['tok_s'] for r in ok)}) | {1000/m:.4f} | "
              f"{max(r['vram_peak_mib'] for r in ok)} | {' '.join('%.4f' % x for x in ld)} | {' '.join(said)}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--summary": return summary(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--kind", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024]); ap.add_argument("--arms", nargs="+", default=["all"])
    ap.add_argument("--passes", type=int, default=3); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=4); ap.add_argument("--prefix", default="")
    ap.add_argument("--env", nargs="*", default=[])
    a = ap.parse_args()
    base, A = arms(a.kind)
    names = list(A) if a.arms == ["all"] else a.arms
    for p in range(1, a.passes + 1):
        for ctx in a.ctx:
            for arm in (names if p % 2 else names[::-1]):
                env_add, extra = A[arm]
                env_add = {**env_add, **dict(kv.split("=", 1) for kv in a.env)}
                cmd = a.prefix.split() + [os.path.join(a.bin, "llama-bench"), "-m", a.model, "-ngl", "99"] + base + \
                      ["-ub", "1", "-b", "1", "-p", "0", "-n", str(a.n), "-d", str(ctx), "-r", str(a.reps), "-o", "json"] + extra
                idle = gpu_used(); peak = [idle]; stop = threading.Event()

                def poll():
                    while not stop.is_set():
                        peak[0] = max(peak[0], gpu_used()); time.sleep(0.2)
                th = threading.Thread(target=poll); th.start(); t0 = time.time()
                r = subprocess.run(cmd, env={**os.environ, **env_add}, capture_output=True, text=True)
                stop.set(); th.join()
                row = {"tag": a.tag, "kind": a.kind, "pass": p, "arm": arm, "ctx": ctx, "n_gen": a.n, "reps": a.reps, "rc": r.returncode,
                       "env": env_add, "args": " ".join(cmd), "vram_idle_mib": idle, "vram_peak_mib": peak[0],
                       "wall_s": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                err = r.stderr.splitlines()
                row["route_lines"] = [l for l in err if l.startswith("temporal-") and "in one launch" not in l][:12]
                row["fused_said"] = sorted({l.split(":")[0] for l in err if l.startswith("temporal-") and "in one launch" in l})
                row["loads"] = loads(row["route_lines"])
                try:
                    j = json.loads(r.stdout)[-1]; ts = j["samples_ts"]
                    row["samples_ts"] = ts; row["tok_s"] = sum(ts[1:]) / len(ts[1:]); row["ms_per_tok"] = 1000.0 / row["tok_s"]
                except Exception as e:
                    row["error"] = repr(e); row["stderr_tail"] = r.stderr[-1500:]
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
                print(f"{a.kind} p{p} ctx {ctx} {arm}: {row.get('tok_s', 'FAILED')}", flush=True)
    summary(a.out)


if __name__ == "__main__":
    main()
