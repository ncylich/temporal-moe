#!/usr/bin/env python3
"""Decode and prefill benchmark driver for the trained 1.26B model on one GPU.

Protocol (the 11B workstation protocol, except flash attention, see --fa): batch 1, the context is filled
token by token (-b 1 -ub 1 -d <ctx>), then 100 tokens are generated and timed; four trials,
the first discarded. One llama-bench process per (arm, context); peak VRAM is the maximum of
nvidia-smi memory.used on that GPU during the process, polled every 0.2 s.

    python gpubench/run_decode.py --bin $ENGINE/build/bin --model fine_1e19-q4_0.gguf \
        --out gpubench/results/decode.jsonl --tag G1 --ctx 1024 2048 4096 --arms all
"""
import argparse, json, os, re, subprocess, threading, time

ARMS = {
    # name: (env, extra args)
    "base_trained":  ({"LLAMA_TEMPORAL_ROUTE": "trained"}, []),
    "base_topk":     ({}, []),
    "kernel_noswap": ({"LLAMA_TEMPORAL_ROUTE": "trained", "TEMPORAL_UNIFIED": "1", "TEMPORAL_SWAP_PROB": "0"}, ["-ncmoe", "14"]),
    "stream":        ({"LLAMA_TEMPORAL_ROUTE": "trained", "TEMPORAL_UNIFIED": "1"}, ["-ncmoe", "14"]),
    # free top-k at the streamed memory: stock router, R=18 slots, every missing expert fetched
    "floor":         ({"TEMPORAL_UNIFIED": "1", "TEMPORAL_UNIFIED_NOFORCE1": "1"}, ["-ncmoe", "14"]),
    # prefill control: expert-major kernel with all experts in VRAM, nothing uploaded (11B config D)
    "resident_em":   ({"LLAMA_TEMPORAL_ROUTE": "trained", "TEMPORAL_PREFILL_RESIDENT": "1"}, []),
}


def gpu_used(gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--arms", nargs="+", default=["all"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--env", nargs="*", default=[], help="extra KEY=VALUE for every arm")
    ap.add_argument("--fa", type=int, default=0, help="flash attention; the CUDA kernels have no head size 16, so 1 puts attention on the CPU")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ub", type=int, default=1, help="batch and micro-batch size used to fill the context")
    ap.add_argument("--mode", choices=["decode", "prefill"], default="decode",
                    help="prefill: time processing <ctx> prompt tokens in micro-batches of --ub, nothing generated")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    arms = list(ARMS) if a.arms == ["all"] else a.arms
    for ctx in a.ctx:
        for arm in arms:
            env_add, extra = ARMS[arm]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), **env_add)
            for kv in a.env:
                k, v = kv.split("=", 1); env[k] = v
            cmd = [os.path.join(a.bin, "llama-bench"), "-m", a.model, "-ngl", "99", "-fa", str(a.fa), "-t", str(a.threads),
                   "-ub", str(a.ub), "-b", str(a.ub), "-r", str(a.reps), "-o", "json"]
            cmd += (["-p", "0", "-n", str(a.n), "-d", str(ctx)] if a.mode == "decode" else ["-p", str(ctx), "-n", "0"]) + extra
            idle = gpu_used(a.gpu)
            peak = [idle]; stop = threading.Event()

            def poll():
                while not stop.is_set():
                    peak[0] = max(peak[0], gpu_used(a.gpu)); time.sleep(0.2)
            th = threading.Thread(target=poll); th.start()
            t0 = time.time()
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            stop.set(); th.join()
            row = {"tag": a.tag, "mode": a.mode, "ub": a.ub, "arm": arm, "ctx": ctx, "n_gen": a.n, "reps": a.reps, "rc": p.returncode,
                   "env": {**env_add, **dict(kv.split("=", 1) for kv in a.env)}, "args": " ".join(cmd[1:]),
                   "vram_idle_mib": idle, "vram_peak_mib": peak[0], "wall_s": round(time.time() - t0, 1),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": a.note}
            row["route_lines"] = [l for l in p.stderr.splitlines() if l.startswith("temporal-")]
            row["temporal_lines"] = [l for l in p.stderr.splitlines() if "[temporal" in l or "temporal UNIFIED" in l][:6]
            try:
                j = json.loads(p.stdout)[0]
                ts = j["samples_ts"]
                row["samples_ts"] = ts
                row["tok_s"] = sum(ts[1:]) / len(ts[1:])           # first trial discarded
                row["ms_per_tok"] = 1000.0 / row["tok_s"]
                if a.mode == "prefill":
                    row["prefill_ms"] = ctx * row["ms_per_tok"]
                row["build_commit"] = j.get("build_commit"); row["gpu_info"] = j.get("gpu_info")
            except Exception as e:  # keep the failure in the record
                row["error"] = repr(e); row["stderr_tail"] = p.stderr[-2000:]; row["stdout_tail"] = p.stdout[-500:]
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{a.mode} ub {a.ub} {arm:14s} ctx {ctx:6d}  " + (f"{row['tok_s']:8.1f} tok/s  {row['ms_per_tok']:.3f} ms  trials {['%.1f' % t for t in ts]}  peak {peak[0]} MiB"
                  if "tok_s" in row else f"FAILED rc={p.returncode} {row.get('error')}") + "  " + " | ".join(row["route_lines"]), flush=True)


if __name__ == "__main__":
    main()
