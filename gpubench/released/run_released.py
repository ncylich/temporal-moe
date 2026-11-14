#!/usr/bin/env python3
"""Decode timing for the released models (Qwen3.5-35B-A3B, Gemma4-26B-IT) on one GPU, every arm under the
same protocol: batch 1, llama-bench, the context filled as one prompt batch (-ub/-b >= ctx, routed freely as
in decode_state.py and cache_bias.py), then --n tokens decoded and timed; four trials, first discarded.

Context and decoded tokens are real text (TEMPORAL_BENCH_CTX / TEMPORAL_BENCH_GEN, written by
serve_client.py generate): each routing method replays its own greedy continuation, so its routing sees the
text it would produce. The streamed and no-copy arms of a method replay the same files as its all-in-VRAM arm.
Peak VRAM = max nvidia-smi memory.used during the process, polled every 0.2 s.

    python3 gpubench/released/run_released.py --bin $ENG/build/bin --model qwen35-q4km.gguf --nmoe 40 \
        --C 8 --lam 0.4 --tokens tokens/ --ctx 1024 --out gpubench/results/released_qwen_decode.jsonl --tag R1
"""
import argparse, json, os, subprocess, threading, time


def arms(nmoe, C, lam):
    sk = {"LLAMA_TEMPORAL_ROUTE": "skliar", "TEMPORAL_CB_C": str(C), "TEMPORAL_CB_LAMBDA": str(lam), "TEMPORAL_CB_J": "1"}
    ours = {"LLAMA_TEMPORAL_ROUTE": "decode"}
    st = {"TEMPORAL_UNIFIED": "1"}
    nc = ["-ncmoe", str(nmoe)]
    return {
        # name: (env, extra args, token-file tag)
        "base":            ({}, [], "base"),                                    # stock top-k, all experts in VRAM
        "ours_vram":       (ours, [], "ours"),                                  # our rule, all experts in VRAM
        "skliar_vram":     (sk, [], "skliar"),                                  # Skliar routing, all experts in VRAM
        "ours_stream":     ({**ours, **st}, nc, "ours"),                        # arm 2
        "skliar_stream":   ({**sk, **st}, nc, "skliar"),                        # arm 3
        "ours_nobytes":    ({**ours, **st, "TEMPORAL_ROUTE_NOCOPY": "1"}, nc, "ours"),     # copy kernels, empty list
        "skliar_nobytes":  ({**sk, **st, "TEMPORAL_ROUTE_NOCOPY": "1"}, nc, "skliar"),
        "ours_nocopy":     ({**ours, **st, "TEMPORAL_ROUTE_NOCOPY": "2"}, nc, "ours"),     # no copy kernels at all
        "skliar_nocopy":   ({**sk, **st, "TEMPORAL_ROUTE_NOCOPY": "2"}, nc, "skliar"),
    }


def gpu_used(gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--nmoe", type=int, required=True, help="-ncmoe value (number of layers) for streamed arms")
    ap.add_argument("--C", type=int, default=8); ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--tokens", required=True, help="dir with ctx<N>_<tag>.ctx/.gen")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--ctx", type=int, nargs="+", default=[1024])
    ap.add_argument("--arms", nargs="+", default=["all"]); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=4); ap.add_argument("--fa", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--ub", type=int, default=0, help="micro-batch for the context fill; 0 = ctx")
    ap.add_argument("--env", nargs="*", default=[], help="extra KEY=VALUE for every arm"); ap.add_argument("--note", default="")
    ap.add_argument("--prefix", default="", help="command prefix for every arm, e.g. 'taskset -c 0-31,64-95'")
    a = ap.parse_args()
    A = arms(a.nmoe, a.C, a.lam)
    names = list(A) if a.arms == ["all"] else a.arms
    for ctx in a.ctx:
        ub = a.ub or ctx
        assert ub >= ctx, "the context must be one prompt batch"
        for arm in names:
            env_add, extra, ttag = A[arm]
            env_add = {**env_add, "TEMPORAL_BENCH_CTX": os.path.join(a.tokens, f"ctx{ctx}_{ttag}.ctx"),
                       "TEMPORAL_BENCH_GEN": os.path.join(a.tokens, f"ctx{ctx}_{ttag}.gen")}
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), **env_add)
            for kv in a.env:
                k, v = kv.split("=", 1); env[k] = v
            cmd = a.prefix.split() + [os.path.join(a.bin, "llama-bench"), "-m", a.model, "-ngl", "99", "-fa", str(a.fa), "-t", str(a.threads),
                   "-ub", str(ub), "-b", str(ub), "-r", str(a.reps), "-o", "json",
                   "-p", "0", "-n", str(a.n), "-d", str(ctx)] + extra
            idle = gpu_used(a.gpu); peak = [idle]; stop = threading.Event()

            def poll():
                while not stop.is_set():
                    peak[0] = max(peak[0], gpu_used(a.gpu)); time.sleep(0.2)
            th = threading.Thread(target=poll); th.start(); t0 = time.time()
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            stop.set(); th.join()
            row = {"tag": a.tag, "arm": arm, "ctx": ctx, "ub": ub, "n_gen": a.n, "reps": a.reps, "rc": p.returncode,
                   "env": {**env_add, **dict(kv.split("=", 1) for kv in a.env)}, "args": " ".join(cmd),
                   "vram_idle_mib": idle, "vram_peak_mib": peak[0], "wall_s": round(time.time() - t0, 1),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": a.note}
            row["route_lines"] = [l for l in p.stderr.splitlines() if l.startswith("temporal-")]
            row["temporal_lines"] = [l for l in p.stderr.splitlines() if "[temporal" in l or "temporal UNIFIED" in l or "llama-bench: TEMPORAL" in l][:8]
            try:
                j = json.loads(p.stdout)[0]
                ts = j["samples_ts"]
                row["samples_ts"] = ts
                row["tok_s"] = sum(ts[1:]) / len(ts[1:])           # first trial discarded
                row["ms_per_tok"] = 1000.0 / row["tok_s"]
                row["build_commit"] = j.get("build_commit"); row["gpu_info"] = j.get("gpu_info")
            except Exception as e:
                row["error"] = repr(e); row["stderr_tail"] = p.stderr[-3000:]; row["stdout_tail"] = p.stdout[-500:]
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{arm:15s} ctx {ctx:6d}  " + (f"{row['tok_s']:8.2f} tok/s  {row['ms_per_tok']:.3f} ms  trials {['%.2f' % t for t in ts]}  peak {peak[0]} MiB"
                  if "tok_s" in row else f"FAILED rc={p.returncode} {row.get('error')}") + "  " + " | ".join(row["route_lines"]), flush=True)


if __name__ == "__main__":
    main()
