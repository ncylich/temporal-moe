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


def arm(label, env):
    s = "#!/system/bin/sh\n" + f"cd {DEV}\n" + "echo 1000 > /proc/self/oom_score_adj\n"
    for kv in (PROD + " " + POLICY + " " + env).split():
        s += f"export {kv}\n"
    s += ("./llama-perplexity -m " + MODEL + " -f ppl_input.txt --chunks 2 -c 512 "
          "-t 4 --no-mmap -ot _exps=CPU\n")
    with open("/tmp/_gate_ppl.sh", "w") as f:
        f.write(s)
    subprocess.run(["adb", "push", "/tmp/_gate_ppl.sh", f"{DEV}/_gate_ppl.sh"],
                   capture_output=True)
    cmd = (f"sh {DEV}/_gate_ppl.sh" if NOROOT else f"su -c 'sh {DEV}/_gate_ppl.sh'")
    out = sh(cmd, timeout=5400)
    m = re.search(r"Final estimate: PPL = ([0-9.]+ \+/- [0-9.]+)", out)
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
