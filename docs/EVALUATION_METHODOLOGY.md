# Architecture Evaluation Methodology

How we measure the performance of a model architecture on a single GPU using the **minimum number of
training runs**. This is the reusable protocol behind the Phase-0 baselines (`results/phase0/`); any
new architecture (a temporal variant, a router change, a different expert layout) is evaluated by
re-running exactly this procedure and comparing on the same axes.

The deliverable for "how good is architecture X?" is **not a single loss number** — it is a pair of
**IsoFLOP parabolas** (one per compute budget) plus a **dense floor**, which together locate X's
compute-optimal model size and quantify its advantage over a vanilla baseline. The whole point of the
protocol is to produce those curves cleanly and cheaply, then know when to **stop**.

---

## 1. The metric: bits-per-byte (BPB)

We report **BPB = CE_nats / (ln2 · bytes_per_token)**, never raw cross-entropy.

- **Why:** we use a custom 16k-vocab BPE tokenizer (smaller logits → much faster at these tiny
  scales). Raw CE is not comparable across vocabularies; BPB is tokenizer-invariant, so our numbers
  stay comparable to the 50k-vocab FLAME scaling law and to each other.
- **Divisor:** `bytes_per_token` is a property of the tokenizer+corpus. For our 16k BPE on dclm:
  **÷2.7568** (3.977 bytes/token). For the 50k pythia tokenizer it would be ÷2.9780 (4.296). Set via
  `BPB_DIVISOR`; lower BPB is better.
- Convert any law CE target the same way to get a BPB bar (e.g. FLAME's CE → ≤1.645 @1e17, ≤2.149
  @1e16).

---

## 2. The fixed harness (everything held constant across a sweep)

A sweep only varies **two** things: the **shape** (model size) and the **FLOP budget**. *Everything
else is locked* so that any BPB difference is attributable to architecture, not to a confound.

### 2a. Architecture knobs (the thing under test — fixed within one architecture's sweep)
Stock FLAME-MoE: `num_experts 64`, `top-k 6`, **1 shared expert** (intermediate `2·moe_ffn`),
`moe_layer_freq = [0]+[1]*(L−1)` (first layer dense, rest MoE), `ffn ≈ 5.34·h`,
`moe_ffn ≈ 0.6875·h`. SwiGLU, RMSNorm (ε 1e-6), RoPE, untied embed/output, no biases, dropout 0,
init-std 0.02. To evaluate a *different* architecture you change exactly these knobs and re-run §5.

### 2b. Locked hyperparameters (identical for every run, every architecture)
peak-LR **3e-3**, warmup **5% of iters**, cosine decay → 10% (min-lr 3e-4), grad-clip 1.0,
weight-decay 0.01, aux-loss 0.01, z-loss 0.001, global-batch **256**, micro-batch **32**, **bf16**,
seed **1234**. These were tuned once (LR sweep at s2, confirmed at the longer 3e16 budget, re-checked
stable across the s2/s4/s6 size range) and then frozen — we do **not** re-tune per run, because a
moving HP would make the parabola meaningless.

### 2c. Single-GPU adaptations (non-obvious; required for correctness/throughput)
- **EP=1 + `--moe-grouped-gemm`** (batch the 64 local experts; numerically equivalent to FLAME's
  EP=8 per-GPU sequential experts).
- **TransformerEngine impl** (TE 1.11 built from vendored source); `--no-gradient-accumulation-fusion`
  (apex absent, perf-only).
- **head_dim must be 16** → set `heads = hidden/16`. Fixed 16 heads gives head_dim 12/20/28 for
  s1/s3/s5, which silently drops TE fused attention to a ~3× slower path. Same params/FLOPs.
- **16k-BPE + fused cross-entropy** (the 50k logits dominate FLOPs at these tiny scales; this is the
  single biggest speedup, ~1.7×).
- **ffn must be EVEN** — an odd `ffn_hidden_size` crashes the fused-SwiGLU JIT warmup ("broadcast N
  vs N+1"). All shape ffn values are rounded to even.
- micro-batch is capped at 32 by vocab-logit memory.

All of this lives in one env-parametrized launcher, `scripts/phase0/run.sh` — you set
`SHAPE`/`TARGET_FLOPS` and it computes everything else.

---

## 3. The shape ladder (the model-size axis)

A fixed ladder of geometries (hidden/layers), each with matched ffn so the proportions hold.
`N_active` =
active **non-embedding** params (the quantity that enters the FLOP law; computed by
`scripts/phase0/shapes.py`, which excludes embeddings because they don't scale compute the same way).

| shape | hidden | layers | ffn | moe_ffn | N_active |
|---|---|---|---|---|---|
| sm1 (a.k.a. s₋₁) | 96  | 4  | 512  | 66  | 0.77M |
| s0  | 128 | 4  | 684  | 88  | 1.36M |
| s1  | 192 | 5  | 1026 | 132 | 3.81M |
| s2  | 256 | 6  | 1368 | 176 | 8.12M |
| s3  | 320 | 7  | 1710 | 220 | 14.77M |
| s4  | 384 | 8  | 2052 | 264 | 24.29M |
| s5  | 448 | 9  | 2394 | 308 | — |
| s6  | 512 | 10 | 2736 | 352 | — |

The ladder is deliberately ~2× N per step so a 3-shape window spans ~4× and reliably shows curvature.
Need a point left of sm1 or between rungs? Add a shape to the dict; keep ffn even.

---

## 4. FLOP budgeting: C = 6·N·D

Compute is fixed *per budget*, not training length. For a target `C` (e.g. 1e16, 1e17) and a shape's
`N_active`, the token budget is `D = C / (6·N)` and the iteration count is `D / (global_batch ·
seq_len)`. `run.sh` calls `shapes.py iters <shape> <flops> <gb>` to get `(N, iters)` and sizes warmup
(5%), LR-decay horizon, eval interval, and checkpointing off that. **Consequence:** smaller shapes
run *more* iterations at the same budget (e.g. s1@1e17 = 8338 iters; s4@1e17 = 1309). This matters for
§8 monitoring — run wall-clock is `iters × per-iter time`, and per-iter time depends on shape.

---

## 5. The IsoFLOP sweep — the core procedure

Goal per budget: a **clean parabola** of BPB vs `N_active`, whose **minimum is bracketed** (a point on
each side that is clearly higher). The minimum = the compute-optimal model size for that budget. Run
the **fewest shapes** that prove the bracket.

### Step 1 — predict where the minimum is
Use the scaling law (or the previous budget's result) to predict the optimal `N`. The optimum shifts
**right** (bigger) as the budget grows. For us: ~1.48M @1e16 (≈s0), ~7.74M @1e17 (≈s2).

### Step 2 — run the bracket triple
Run the predicted-optimal shape and its two ladder neighbors (e.g. s1/s2/s3 @1e17). Three points is
the minimum that can show curvature. Evaluate to **final** BPB:
- For a single-budget parabola, use `EVAL_AT_END=1` (one eval at the last iter — saves ~9 evals).
- To also read a *lower* budget's point off a higher-budget run (the loss at iter = iters/10 of a
  10× run), use the default `eval@iters/10` instead.

### Step 3 — the STOP / EXTEND decision (this is the whole game)
Look at the triple `(left, mid, right)`:

- **Bracketed & clear → STOP.** `mid` is the lowest, *and* both arms are higher by a **meaningful
  margin** (our threshold: **> ~0.01 BPB**, comfortably above run-to-run noise of ~0.009 nats / a few
  thousandths BPB). A 0.0008 gap is **not** a parabola — it's noise; do not call it.
- **Monotone (min at an edge) → EXTEND by exactly ONE run, in the downhill direction.** If
  `left < mid < right`, the true min is further left → add the next-smaller shape. If
  `left > mid > right`, add the next-larger shape. Re-evaluate. Repeat **one run at a time** until
  bracketed. (Example: dense @1e16 came back s0<s1<s2 monotone with the min at the s0 edge → we added
  sm1 on the left; sm1 1.534 > s0 1.519 < s1 1.591 → bracketed, stop.)
- **Flat / ambiguous (arms within noise) → EXTEND outward**, don't add interior points — you need a
  longer lever arm to separate signal from noise, not finer resolution near a min you can't yet see.

The discipline: **never run a shape you can't justify by the bracket rule.** Every added run must be
the single point that either confirms the bracket or extends a monotone arm. This is how the sweep
stays minimal.

### Step 4 — reproducibility check (once, at the chosen optimum)
Re-run the winning shape with a **second seed**. Accept if `|Δ loss| ≤ 0.03 nats`. (Ours: s2@1e17
seed-2 vs seed-1 = 0.009 nats.) This certifies the parabola minimum isn't a seed artifact.

### Step 5 — healthy-routing check (MoE only, once, at the optimum)
Load the trained checkpoint and tally per-expert token load (`expert_load.py`, run via
`EVAL_ONLY=1`). Accept if no expert exceeds **8× the mean** load on any MoE layer (balanced load ⇒
the aux-loss did its job and the capacity isn't being wasted). Ours: worst 2.07×.

---

## 6. The dense IsoFLOP floor (is the architecture worth its overhead?)

A low BPB is only meaningful relative to what you *could* have trained for the same compute. For each
shape we also train a **vanilla dense (no-MoE) transformer** whose `ffn_hidden` is enlarged so its
**total non-embedding params == the MoE's *active* non-embedding params** — identical FLOP budget,
tokens, and locked HPs. Run via `run.sh DENSE=1` (drops all MoE args, sets the matched even ffn).

Run the dense floor through the **same §5 procedure** (its parabola needs bracketing too — we added
dense sm1@1e16 and dense s1@1e17 left-arm points for exactly that). The architecture is justified iff
**MoE beats the dense floor at the compute-optimal shape** (and ideally everywhere). Ours: MoE wins by
~0.07 BPB at both budgets' optima, and the gap *widens* with size.

---

## 7. Acceptance-criteria template

Define pass/fail **before** running (pre-registered), then loop §5 until all hold:
1. **Best-shape ≤ bar:** the optimum's BPB clears the law-derived bar at each budget.
2. **Curve shape:** a parabola at each budget with the minimum **bracketed** (§5 Step 3), and the
   optimum shifts right with budget.
3. **Reproducible:** 2nd-seed `|Δ| ≤ 0.03 nats` at the optimum (§5 Step 4).
4. **Healthy routing:** no expert > 8× mean (§5 Step 5).
Plus the §6 floor: **beats the equal-active-param dense baseline.**

---

## 8. Execution & monitoring (the operational loop)

This is how you drive many sequential runs on one GPU **reactively** — catching a finish or a failure
within ~15 s, without busy-polling, and without ever sleeping blindly through a 3-hour run.

### 8a. One GPU ⇒ strictly serial
Runs must not overlap (GPU contention corrupts throughput and can OOM). Queue them in a **driver**:
`scripts/phase0/drive.sh <configs.txt>` reads `NAME SHAPE FLOPS LR WARMUP GB SEED [AUX]` lines, skips
any run whose final-iter checkpoint already exists (idempotent restart), runs each via `run.sh`,
parses the result, and appends to `results/phase0/log.md`. Chain different env regimes (e.g.
`EVAL_AT_END` differing between budgets) with a small wrapper script and launch it **detached**
(`nohup … &`) so it survives across monitoring cycles.

### 8b. The reactive polling pattern (the key idea)
Watch the **log file and process table**, not a wall clock. Each monitoring tool-call is a *bounded*
loop that sleeps in **short increments** and **breaks early** the instant something happens:

```bash
d=results/phase0/runs/<this_run>
for i in $(seq 1 39); do                                   # outer cap ≈ 10 min at 15s/cycle
  grep -q "after training is done" $d/train.log && break   # finished
  grep -q "nan iterations:   [1-9]" $d/train.log && break   # diverged → must fix
  [ -d results/phase0/runs/<next_run> ] && break            # driver advanced ⇒ this run is done
  pgrep -f drive.sh >/dev/null || break                     # driver died ⇒ stop waiting
  sleep 15
done
# then report the live state:
grep "consumed samples" $d/train.log | tail -1 \
  | grep -oE "iteration[ ]+[0-9]+/[ ]*<total>|lm loss: [0-9.E+]+|nan iterations:   [0-9]+"
```

Why this is both responsive and cheap:
- **15 s inner sleep** — fine enough to notice a finish/NaN within ~one log interval, coarse enough
  that you're not hammering the filesystem or the model context.
- **~10-min outer cap** — bounds a single tool-call so it *always* returns and lets you re-evaluate
  (re-read loss trend, re-check GPU, decide the next run) even when nothing dramatic happened. You
  stay in the loop instead of disappearing for hours.
- **The break conditions make it event-driven**, not timer-driven: the call returns *immediately* on
  completion, divergence, driver-advance, or driver-death — you react to the event, not to a guess.

### 8c. Sizing the wait to the run (don't over- or under-watch)
Estimate run length first, then choose how many cycles to budget:
`wall-clock ≈ iters × per-iter-time`, and `per-iter-time ≈ (6·N·gb·seq) / (throughput_TFLOPs · 1e12)`
— or just read the logged `throughput per GPU (TFLOP/s)` and `iteration X/total` once the run starts
and extrapolate. A 1300-iter run at ~30 TFLOP/s finishes in ~20 min (1–2 cycles); an 8300-iter small
shape takes ~2.5 h (~15 cycles). Spend short cycles on short runs, more cycles on long ones; never set
one giant sleep — you'd be blind to an early NaN for hours.

### 8d. The "next-dir appears ⇒ previous finished" trick
When a driver runs N configs back-to-back, the cleanest completion signal for run *k* is the
**creation of run *k+1*'s output directory** (the driver only starts it after *k*'s checkpoint lands).
Polling for that directory is more robust than parsing *k*'s final-eval line, whose format differs
between `EVAL_AT_END` (one `validation loss at iteration <total>`) and `eval@iters/10` runs.

### 8e. Per-cycle health read (what to look at every time)
On each return, check three things: **progress** (`iteration X/total` advancing), **loss** (still
decreasing, no spike), **`nan iterations: 0`**. If loss is flat-but-finite and you're past warmup, the
run is fine; if you see `nan iterations: ≥1` or an `exitcode`/`ChildFailedError` in the log, the run
failed — go to §8f.

### 8f. Failure modes & fixes (seen in practice)
- **NaN / divergence:** kill, don't let the driver log a garbage point. Usually LR-vs-budget; our
  locked HPs are stable, so a NaN more often means a config bug than a real instability.
- **Crash on launch (404 / "not a local folder" / ".idx/.bin not found"):** `run.sh` `cd`s into
  `Megatron-LM/` before the tokenizer and data paths resolve, so **`TOKENIZER_MODEL` and `DATA_DIR`
  must be absolute paths**. Relative paths work for the data *find* (it runs at repo root) but break
  the tokenizer load. (This bit us twice on the dense extensions.)
- **Orphaned GPU processes** holding memory/ports after a kill: find by `nvidia-smi` PID or
  `pgrep -f pretrain_gpt`, kill, confirm `nvidia-smi` shows ~0 MiB before relaunching. A reused run
  dir from a crashed attempt should be `rm -rf`'d so the driver restarts it clean (and stale NA rows
  pruned from `log.md`).
- **Odd ffn → fused-SwiGLU warmup crash:** round ffn to even (applies to dense matched-ffn too).

### 8g. Don't poll work the harness already tracks
If a run is launched as a tracked background task, you're notified on completion — don't also spin a
polling loop. The §8b pattern is for *external* state (a detached driver writing log files) that
nothing will notify you about.

---

## 9. Minimal-runs decision tree (the summary)

```
for each budget C:
  predict optimal shape from the law / previous budget
  run bracket triple (predicted + two neighbors), eval to final BPB
  loop:
    if mid is lowest AND both arms higher by > ~0.01 BPB:  STOP (clear parabola)
    elif monotone (min at an edge):  add ONE shape in the downhill direction; re-eval
    elif arms within noise:          extend OUTWARD (longer lever arm); re-eval
  reproduce optimum at a 2nd seed (|Δ| ≤ 0.03 nats)
  check expert load (≤ 8× mean)            # MoE only
run the dense floor through the same loop  # is the architecture worth it?
compare: architecture wins iff it beats the dense floor at the optimum
```

Everything else — the locked HPs, the FLOP budgeting, the serial driver, the reactive monitoring — is
machinery in service of producing those bracketed parabolas with the fewest runs and reacting to each
one the moment it finishes or fails.

---

## 10. Script & file map

| file | role |
|---|---|
| `scripts/phase0/run.sh` | env-parametrized single-run launcher (all of §2; `DENSE=1`, `EVAL_AT_END`, `EVAL_ONLY` modes) |
| `scripts/phase0/shapes.py` | `N_active` and iters-for-budget (§3–4) |
| `scripts/phase0/drive.sh` | serial driver over a configs file; idempotent skip; parse+log (§8a) |
| `scripts/phase0/parse_run.py` | train.log → final/at-1e16 BPB (`BPB_DIVISOR`) |
| `scripts/phase0/expert_load.py` | per-expert load, criterion 4 (§5 Step 5) |
| `scripts/phase0/plot_dense_vs_moe.py` | dense-vs-MoE parabola plot (§6) |
| `results/phase0/log.md` | append-only measured-results ledger |
| `results/phase0/{RESULTS,PASS,DENSE_BASELINES}.md` | writeups |
| `*.txt` (sweep_*, dense_*, dense_ext_*) | driver config lists |
