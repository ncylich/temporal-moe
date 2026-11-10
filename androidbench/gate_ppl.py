#!/usr/bin/env python3
"""Correctness gate for a changed fetch path: PPL must be bit-identical.

Pitfall #11: a missing residency barrier and a leaking eviction both LOOK like a
speedup (34.9 and 32 tok/s were both correctness bugs). Nothing gets timed until
this passes.

Self-baselining, no hard-coded oracle: the two arms are the SAME model, the SAME
production config and the SAME R -- only the fetch mechanism differs (blocking
pread worker pool vs io_uring single submitter). Every printed digit of PPL must
match, over thousands of evict/refetch cycles.

`llama-cli` is not used: with --no-mmap this fork's cli path issues ~17M tiny read
syscalls and does not finish a load in 35 minutes. llama-perplexity and llama-bench
take the lazy-expert-load path and start in seconds.

    ./gate_ppl.py                # pread vs io_uring        (Pixel, rooted)
    ./gate_ppl.py sqpoll         # pread vs io_uring+SQPOLL (Pixel, rooted)
    ./gate_ppl.py --resident 192 --noroot   # resident vs streamed, no su wrapper

Trained model (fine 1e19, trained routing, see TRAINED_MODEL_LEDGER.md):

    ./gate_ppl.py --model trained_fine --resident 192 --batch1 --noroot
    ./gate_ppl.py --model trained_fine --plain --noroot

`--model` picks the GGUF, side-file, K and binary from MODELS. `--batch1` runs perplexity
with -b 1 -ub 1, so every token is a one-token decode and goes through the two-pass window
path (a larger batch takes the prefill path, which the window op never sees). `--plain`
compares repacked against plain (LLAMA_NO_REPACK) expert kernels, both fully resident with
trained routing: kernel families accumulate in a different order, so this one is a
tolerance gate (|ln PPL difference| < 0.01 nats), not bit identity.

`--resident E` runs arm A at R=E (fully resident, no streaming at all) against arm B
at R=K. That is the STRONGEST form of the gate -- it catches a missing residency
barrier and an eviction leak in one shot -- and it needs a device on which the model
fits in RAM. `--noroot` drops the `su -c` wrapper for unrooted devices.
"""
import subprocess, sys, re

DEV   = "/data/local/tmp/tmoe"
MODEL = "qwen3moe-rand-fine-Q4pure.gguf"
SIDE  = "qwen3moe-rand-fine-Q4pure-repacked.bin"
K     = 18
NOROOT = False
BIN   = "llama-perplexity"
BATCH = ""
MODELS = {
    "rand_fine":    dict(gguf=MODEL, side=SIDE, K=18, E=192, bin="llama-perplexity", route=""),
    "trained_fine": dict(gguf="fine_1e19-q4_0.gguf", side="fine_1e19-q4_0-repacked.bin", K=18, E=192,
                         bin="llama-perplexity-trained", route="LLAMA_TEMPORAL_ROUTE=trained"),
}
ROUTE = ""

# NOTE: TWOPASS is parsed as `getenv(...) != nullptr`, so TWOPASS=0 ENABLES it.
# The single-pass policy must OMIT the variable, hence POLICY is spliced in, not set to 0.
PROD = (f"LLAMA_TEMPORAL_REPACK=1 LLAMA_TEMPORAL_REPACK_FILE={SIDE} "
        "LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_MADV_FREE=1 "
        "LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6 "
        "LLAMA_TEMPORAL_SPIN_US=5000")
POLICY = "LLAMA_TEMPORAL_TWOPASS=1"


def sh(cmd, timeout=3600):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.replace("\r", "") + r.stderr.replace("\r", "")


def arm(label, env, policy=None, prod=None):
    s = "#!/system/bin/sh\n" + f"cd {DEV}\n" + "echo 1000 > /proc/self/oom_score_adj\n"
    pol = POLICY if policy is None else policy
    for kv in ((PROD if prod is None else prod) + " " + pol + " " + ROUTE + " " + env).split():
        s += f"export {kv}\n"
    s += (f"./{BIN} -m " + MODEL + " -f ppl_input.txt --chunks 2 -c 512 "
          f"-t 4 --no-mmap -ot _exps=CPU {BATCH} 2>&1\n")
    with open("/tmp/_gate_ppl.sh", "w") as f:
        f.write(s)
    subprocess.run(["adb", "push", "/tmp/_gate_ppl.sh", f"{DEV}/_gate_ppl.sh"],
                   capture_output=True)
    cmd = (f"sh {DEV}/_gate_ppl.sh" if NOROOT else f"su -c 'sh {DEV}/_gate_ppl.sh'")
    out = sh(cmd, timeout=5400)
    m = re.search(r"Final estimate: PPL = ([0-9.]+ \+/- [0-9.]+)", out)
    rt = re.search(r"temporal-route: .*", out)
    if rt:
        print(f"   {rt.group(0)}", flush=True)
    ppl = m.group(1) if m else "NOT FOUND"
    pool = ""
    mm = re.search(r"temporal-pool: fetches=.*", out)
    if mm:
        pool = mm.group(0)
    print(f"{label:26s} PPL = {ppl}", flush=True)
    if pool:
        print(f"   {pool[:160]}", flush=True)
    if ppl == "NOT FOUND":
        print("   raw tail:\n   " + "\n   ".join(out.splitlines()[-15:]), flush=True)
    return ppl


if __name__ == "__main__":
    argv = sys.argv[1:]
    NOROOT = "--noroot" in argv
    if "--model" in argv:
        mm = MODELS[argv[argv.index("--model") + 1]]
        MODEL, SIDE, K, BIN, ROUTE = mm["gguf"], mm["side"], mm["K"], mm["bin"], mm["route"]
        PROD = PROD.replace("qwen3moe-rand-fine-Q4pure-repacked.bin", SIDE)
    if "--batch1" in argv:
        BATCH = "-b 1 -ub 1"
    if "--plain" in argv:
        # both fully resident (R=E), no two-pass; only the expert kernel family differs
        E = mm["E"] if "--model" in argv else 192
        a = arm("A repacked R=E", f"LLAMA_TEMPORAL_R={E}", policy="")
        b = arm("B plain R=E", f"LLAMA_TEMPORAL_R={E}", policy="",
                prod="LLAMA_NO_REPACK=1 LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6")
        import math
        try:
            d = abs(math.log(float(a.split()[0])) - math.log(float(b.split()[0])))
        except ValueError:
            print("GATE FAIL (no PPL)"); sys.exit(1)
        print(f"|ln PPL_repacked - ln PPL_plain| = {d:.5f} nats (gate < 0.01)")
        print("GATE PASS" if d < 0.01 else "GATE FAIL")
        sys.exit(0 if d < 0.01 else 1)
    if "--policy" in argv and argv[argv.index("--policy") + 1] == "enforce":
        POLICY = "LLAMA_TEMPORAL_ENFORCE=1"      # single-pass enforced swap
    if "--resident" in argv:
        E = int(argv[argv.index("--resident") + 1])
        a = arm(f"A resident R={E}", f"LLAMA_TEMPORAL_R={E}")
        b = arm(f"B streamed R={K}", f"LLAMA_TEMPORAL_R={K}")
        if a == b and a != "NOT FOUND":
            print("GATE PASS: PPL bit-identical (resident vs streamed)")
            sys.exit(0)
        print("GATE FAIL")
        sys.exit(1)
    sq = "sqpoll" in argv
    a = arm("A pread worker pool", f"LLAMA_TEMPORAL_R={K}")
    b = arm("B io_uring" + (" SQPOLL" if sq else ""),
            f"LLAMA_TEMPORAL_R={K} LLAMA_TEMPORAL_URING=1"
            + (" LLAMA_TEMPORAL_URING_SQPOLL=1" if sq else ""))
    if a == b and a != "NOT FOUND":
        print("GATE PASS: PPL bit-identical")
        sys.exit(0)
    print("GATE FAIL")
    sys.exit(1)
