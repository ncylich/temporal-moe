# Session prompt: laptop decode benchmark for temporal MoE

Paste everything below the line into a fresh
agent session started from Windows PowerShell on the laptop (not from inside WSL: the
session has to write `.wslconfig`, run `wsl --shutdown`, and run native-Windows probes).

---

You are running on Mohsen's laptop (i7-11370H, 16 GB LPDDR4x, Samsung PM981a NVMe,
Windows 11) to measure decode speed of a temporal mixture-of-experts model served from the
memory of its active experts, with the other experts streamed from the NVMe. The result is a
laptop column next to the A6000 and Pixel 10a numbers in the paper. Follow
`laptopbench/PLAN.md` in the repo exactly; this prompt is the operational version of it. Where
the plan and this prompt disagree, the plan wins and you note the disagreement in the ledger.

## Inputs

- Repo: `https://github.com/ncylich/temporal-moe.git`, branch `laptop`. Clone to `C:\tmoe\temporal-moe`.
- Engine fork: `https://github.com/ncylich/llama.cpp.git`, branch `temporal-moe`. Clone to `C:\tmoe\llama.cpp` and, inside
  WSL, to `~/tmoe/llama.cpp` (two clones; WSL builds must live on the ext4 root, never on
  `/mnt/c`).
- Model: `https://huggingface.co/ncylich/temporal-moe-extras/resolve/main/serving/qwen3moe-rand-fine-Q4pure.gguf`, sha256 `d8a3bdf4a9c4a1563ad694718a44e155b7588e5ca47d3ee569bf9a8b8c3a2229`. Fine 18-of-192 random-weight Qwen3-MoE,
  uniform Q4_0, about 6 GB. Place at `C:\tmoe\models\` and copy into `~/tmoe/models/` in WSL.
- WSL distro: `Ubuntu-24.04`.

If any input is missing or the sha256 does not match, stop after Phase 0 and report. Do not
generate the model here; the generator needs 21 GB of RAM and this machine has 16.

## Read first, in this order

1. `laptopbench/PLAN.md`, all of it. Sections 1, 2, 5.2 and 8 are what you execute.
2. `androidbench/ENGINE_FLAGS.md`: every engine flag, the Pixel production config, and which
   flags were rejected on which device.
3. `androidbench/BASELINE_POLICY.md`, `androidbench/BENCHMARK_GATES.md`, and the pitfalls
   list at `androidbench/ANDROID_OPTIM_PROGRESS.md` from the line "MEASUREMENT PITFALLS"
   onward. Numbers 3, 5, 9, 11, 12, 17, 18, 19, 23 apply here verbatim.
4. `androidbench/honesty_gate.py`, `gate_ppl.py`, `bench.py`, `emit_row.py`, `analyze.py`:
   the gates and driver you are porting, not rewriting.
5. `androidbench/LEDGER.md` entries S3-32 (latency model), S3-37c (policy cost
   decomposition), S3-38 (eviction cost).

## Phases

Each phase ends with a ledger entry in `laptopbench/LEDGER.md`, numbered `L1-1`, `L1-2`, ...
in the style of `androidbench/LEDGER.md`: what was run, the exact command, the numbers, and
what was retracted. Write the entry before starting the next phase.

**Phase 0, preflight.** Record: Windows build, CPU, RAM, the NVMe model from
`Get-PhysicalDisk`, WSL kernel version, Python version on both sides, `powercfg
/getactivescheme`, AC state, free disk. Confirm power plan is Best performance, sleep and
Modern Standby are off, Defender and Windows Search exclusions cover `C:\tmoe` and the WSL
distro path. Install only: fio (Windows build and `apt` in WSL), `build-essential cmake`,
`sysbench`, and Python packages into a venv. Verify the model sha256 on both sides. If a
check needs a reboot or a firmware change, stop and ask Mohsen; do not do it.

**Phase 1, the driver.** Write `laptopbench/lapbench.py` to the CLI in PLAN.md section 8,
porting `androidbench/bench.py`'s `measure()` discipline and `emit_row.py`'s schema. It
runs under Windows Python, drives WSL stages through `wsl -d Ubuntu-24.04 -e bash -lc`, and
applies memory caps by rewriting `%UserProfile%\.wslconfig` and running `wsl --shutdown`.
The four invariants in section 8 are code, not comments: refuse to time an ungated binary
hash, refuse a row whose pool counters disagree with the requested arm, refuse `KEY=0`
overrides, stamp every row with env, cap, flags, hash, clock probe, peak RSS, counters,
tag. `--dry-run` every stage before its first real run and put the printed commands in the
ledger.

**Phase 2, physics and compute, both environments.**
`lapbench.py probe --env all --tag s1` then `lapbench.py compute --env all --tag s1`.
Probe is the fio ladder of PLAN.md section 4 on a pre-written, non-sparse 8 GB file, QD1 at
4k/32k/108k/216k/648k/1024k and QD4/8/12 at 216k, plus one DRAM bandwidth number. Fit
`latency = fixed + per_KiB × size` per environment. Compute is stock `llama-bench` on the
model fully resident, `-t 4` and `-t 8`, `-mmp 0 -p 0 -n 128 -r 8`, native and WSL2.
Apply section 5.2's rule and write the decision with its numbers as ledger entry `L1-2`. If
the rule lands on 3 (Windows port), stop here and report; the port is not this session.

**Phase 3, build and gates, in the chosen environment.**
`lapbench.py build --env <chosen>` builds `llama-bench-temporal` and `llama-perplexity` with
`-DGGML_NATIVE=ON`, then dumps the repacked side-file with `LLAMA_TEMPORAL_REPACK_DUMP`.
`lapbench.py gates --env <chosen> --tag s2` runs G1 (pool `fetched_bytes` against
`/proc/<pid>/io read_bytes`, within 10%), G2 (PPL identical in every printed digit between
R=18 and R=192), G3 (`LLAMA_NO_REPACK` and repacked: identical PPL, different tok/s). All
three pass or nothing is timed. A failed gate is a ledger entry with the raw numbers, then
you debug the fetch path; you do not lower the tolerance.

**Phase 4, the five arms.** `lapbench.py arms --env <chosen> --tag s2 --n 3`. Arms and their
required pool lines are the table in PLAN.md section 1.2: (a) ceiling `R=192` at cap 12 GB,
(b) resident control `R=192 TWOPASS=1` at 12 GB, (c) deploy `R=18 TWOPASS=1` at 4 GB, (d)
`R` in 24/36/48 at 4 GB, (e) vanilla floor `R=18` with no `TWOPASS` and no `ENFORCE` at
4 GB. Production flags from `ENGINE_FLAGS.md`, `-t 4 -p 0 -n 128 -r 8 -mmp 0 -ot "_exps=CPU"`.
Interleave a, b, c, e, a; three batches per arm from rest with one warmup; five minutes
between batches; the 20 second resident clock probe before each batch, and a batch more
than 3% under the session's first probe is marked degraded and repeated after a rest.
Then the memory demonstration: the ceiling must fail to start at cap 4 GB, deploy must run
at 4 GB and at 2.5 GB, peak RSS recorded for both. Report deploy as tok/s, ratio to the
ceiling, and fraction of the bound computed from Phase 2's fit and the formula in section 4.

**Phase 5, only if Phase 4 is complete and time remains.** Follow PLAN.md section 7 in
order: attribute the gap ((b) against (a) is policy, (c) against (b) is storage) using
`LLAMA_TEMPORAL_TRACE` and `LLAMA_TEMPORAL_FETCHPROF` before touching any knob, then one knob
per A/B, interleaved, n=3, starting with `FETCH_THREADS × SPLIT`. Gates re-run after any
fetch-path change.

**Phase 6, pack.** `lapbench.py pack --tag s2`: rows into
`results/ablations/serving_benchmarks_laptop.csv` via `emit_row.py`, raw logs and
`runs.jsonl` tarred, everything under `comms/laptop/` on branch `laptop`, then commit and
push. Commit messages describe the measurement; no assistant or vendor names anywhere in
commits or files.

## Rules

- Numbers before opinions. Nothing is tuned before the gates pass; nothing is reported
  without its pool counters, clock probe and cap in the row.
- n=3 for anything that goes in the report. One run is a ledger note, never a result.
- A surprising number is a bug in the measurement until shown otherwise. Deploy above the
  ceiling, floor above deploy, or a WSL fio latency near 10 us are bugs.
- Never read a clock from Task Manager. Never place model or build files under `/mnt/c`.
- Do not install anything system-wide beyond the listed packages. Do not reboot, change
  BIOS or firmware, or modify anything outside `C:\tmoe`, `~/tmoe`, `.wslconfig` and the
  Defender exclusions without asking Mohsen.
- Do not start the Windows-native port. If the environment rule lands there, report it.
- If a phase cannot be completed, finish everything that does not depend on it, then say
  exactly what is blocked and why, with the raw output.
- Keep going without checking in unless a rule above says to ask. A session is several
  hours; the driver's refusals are the safety net, so make them real before relying on them.

## Final report

Under 400 words, in this order: the environment decision and the two numbers that made it;
the gate results; the five-arm table (tok/s, ratio to ceiling, peak RSS, pool counters);
deploy as a fraction of the bound; the memory demonstration result; anything retracted;
what is blocked; the commit hash on `laptop`.
