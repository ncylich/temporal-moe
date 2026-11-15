#!/usr/bin/env python3
"""NOT the gate used for PIXEL_SKLIAR_LEDGER.md. First attempt at "pool on flash equals all in RAM", by batch-1
perplexity. It was stopped after one arm: batch-1 llama-perplexity ran at a few tokens per second on the phone
(minutes per arm, heating it), and with -b 1 every token is a rule step, which is not the serving protocol.
The gate that was used is `run_pixel_skliar.py gate` (per-token logit hashes on the timing texts). Kept as the
record of what was run (ledger S1-2).

Output with the expert pool on flash equals output with every expert in RAM.

Batch-1 perplexity (-b 1 -ub 1: every token is a single-token batch, so every token is one rule step and goes
through the two-pass window path) on held-out real text, same binary, same production fetch and kernel flags,
same two-pass graph in both arms:
    A  LLAMA_TEMPORAL_R=192  every expert stays in RAM once loaded (evictions must be 0)
    B  LLAMA_TEMPORAL_R=18   the timed configuration: experts evicted and read back from flash (evictions > 0)
Every printed digit of the perplexity must match. B runs twice (fetches race the compute, so a missing wait
shows up as run-to-run differences as well).

    python3 pixel_skliar_gate_ppl.py [--tail N] [--chunks 2]
"""
import argparse, re, subprocess

DEV = "/data/local/tmp/tmoe"
GGUF, SIDE, BIN = "moe_fine_1e19-full-q4_0.gguf", "moe_fine_1e19-full-q4_0-repacked.bin", "llama-perplexity-skliar"
PROD = (f"LLAMA_TEMPORAL_REPACK=1 LLAMA_TEMPORAL_REPACK_FILE={SIDE} LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_MADV_FREE=1 "
        "LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6 LLAMA_TEMPORAL_SPIN_US=5000 LLAMA_TEMPORAL_TWOPASS=1")
ROUTES = {"skliar": "LLAMA_TEMPORAL_ROUTE=skliar TEMPORAL_CB_C=18 TEMPORAL_CB_LAMBDA=0.7207 TEMPORAL_CB_J=1",
          "ours": "LLAMA_TEMPORAL_ROUTE=decode"}


def arm(label, env, chunks):
    s = "#!/system/bin/sh\n" + f"cd {DEV}\n" + "echo 1000 > /proc/self/oom_score_adj\n"
    for kv in env.split():
        s += f"export {kv}\n"
    s += f"./{BIN} -m {GGUF} -f ppl_real.txt --chunks {chunks} -c 1024 -t 4 --no-mmap -ot _exps=CPU -b 1 -ub 1 2>&1\n"
    with open("/tmp/_gate_sk.sh", "w") as f:
        f.write(s)
    subprocess.run(["adb", "push", "/tmp/_gate_sk.sh", f"{DEV}/_gate_sk.sh"], capture_output=True)
    out = subprocess.run(["adb", "shell", f"sh {DEV}/_gate_sk.sh"], capture_output=True, text=True, timeout=5400).stdout
    m = re.search(r"Final estimate: PPL = ([0-9.]+ \+/- [0-9.]+)", out)
    ppl = m.group(1) if m else "NOT FOUND"
    ev = re.search(r"temporal-pool: fetches=(\d+) fetched_mib=([\d.]+) evictions=(\d+)", out)
    rt = re.search(r"temporal-route2: .*", out)
    print(f"{label:28s} PPL = {ppl}   " + (f"fetches={ev.group(1)} evictions={ev.group(3)}" if ev else "no pool line"), flush=True)
    if rt:
        print("   " + rt.group(0), flush=True)
    if not m:
        print("   tail:\n   " + "\n   ".join(out.splitlines()[-12:]), flush=True)
    return ppl, int(ev.group(3)) if ev else -1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tail", type=int, default=1); ap.add_argument("--chunks", type=int, default=2)
    a = ap.parse_args()
    ok = True
    for name, route in ROUTES.items():
        base = f"{PROD} {route} LLAMA_TEMPORAL_TAIL={a.tail}"
        pa, ea = arm(f"{name} A all in RAM R=192", base + " LLAMA_TEMPORAL_R=192", a.chunks)
        pb, eb = arm(f"{name} B on flash R=18", base + " LLAMA_TEMPORAL_R=18", a.chunks)
        pc, ec = arm(f"{name} B on flash R=18 again", base + " LLAMA_TEMPORAL_R=18", a.chunks)
        good = pa == pb == pc and pa != "NOT FOUND" and ea == 0 and eb > 0 and ec > 0
        print(f"{name}: {'PASS (bit-identical)' if good else 'FAIL'}", flush=True)
        ok = ok and good
    print("GATE " + ("PASS" if ok else "FAIL"))
