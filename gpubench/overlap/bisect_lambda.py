#!/usr/bin/env python3
"""Skliar's lambda for the load-overlap timing: bisect until its loads per decoded token-layer, counted by
the engine over the timed texts under the timing protocol itself (all in VRAM, same llama-bench arguments,
four trials), equal 1.000 within --tol. Loads fall as lambda rises.

    bisect_lambda.py --bin $BIN --model m.gguf --tokens-glob 'out/tok_v*' --C 18 --fa 0 --prefix "taskset -c 0-31,64-95"
"""
import argparse, glob, json, os, re, subprocess


def loads(a, lam):
    rows = ld = 0
    for td in sorted(glob.glob(a.tokens_glob)):
        env = dict(os.environ, LLAMA_TEMPORAL_ROUTE="skliar", TEMPORAL_CB_C=str(a.C), TEMPORAL_CB_LAMBDA=repr(lam), TEMPORAL_CB_J="1",
                   TEMPORAL_BENCH_CTX=os.path.join(td, f"ctx{a.ctx}_skliar.ctx"), TEMPORAL_BENCH_GEN=os.path.join(td, f"ctx{a.ctx}_skliar.gen"))
        for kv in a.env:
            k, v = kv.split("=", 1); env[k] = v
        cmd = a.prefix.split() + [os.path.join(a.bin, "llama-bench"), "-m", a.model, "-ngl", "99", "-fa", str(a.fa), "-t", "8",
                                  "-ub", str(a.ctx), "-b", str(a.ctx), "-r", "4", "-o", "json", "-p", "0", "-n", "100", "-d", str(a.ctx)]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        m = re.search(r"temporal-route2: .*decode_layer_tokens=(\d+) loads=(\d+)", p.stderr)
        assert m, p.stderr[-2000:]
        rows += int(m.group(1)); ld += int(m.group(2))
    return ld / rows, ld, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tokens-glob", required=True)
    ap.add_argument("--C", type=int, required=True); ap.add_argument("--ctx", type=int, default=1024); ap.add_argument("--fa", type=int, default=1)
    ap.add_argument("--lo", type=float, default=0.3); ap.add_argument("--hi", type=float, default=1.2)
    ap.add_argument("--tol", type=float, default=0.002); ap.add_argument("--prefix", default=""); ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trace = []
    lo, hi = a.lo, a.hi
    for lam in (lo, hi):
        r = loads(a, lam); trace.append({"lambda": lam, "loads_per_layer_token": r[0], "loads": r[1], "layer_tokens": r[2]}); print(trace[-1], flush=True)
    assert trace[0]["loads_per_layer_token"] > 1.0 > trace[1]["loads_per_layer_token"], "bracket does not contain 1.0"
    best = None
    for _ in range(30):
        lam = 0.5 * (lo + hi); r = loads(a, lam)
        trace.append({"lambda": lam, "loads_per_layer_token": r[0], "loads": r[1], "layer_tokens": r[2]}); print(trace[-1], flush=True)
        if best is None or abs(r[0] - 1.0) < abs(best["loads_per_layer_token"] - 1.0): best = trace[-1]
        if abs(r[0] - 1.0) <= a.tol: break
        if r[0] > 1.0: lo = lam
        else: hi = lam
    json.dump({"best": best, "trace": trace}, open(a.out, "w"), indent=1)
    print("LAMBDA", repr(best["lambda"]), best["loads_per_layer_token"])


if __name__ == "__main__":
    main()
