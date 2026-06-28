# Temporal MoE (rolling residency) — implementation handoff

**Branch:** `temporal-moe-impl` (off `phase0-flame-baselines`). **Status:** code complete + unit-tested
locally; **not yet run on a GPU**. The next agent's job: integration-smoke it on the pod, fix anything
the first real forward pass surfaces, then orchestrate the quality runs and place the result on the
dense↔MoE frontier.

## What this is (one paragraph)

A trained-from-scratch MoE variant that keeps only `K = k = 6` routed experts resident per layer and
streams one expert in per token (evicting either the LRU or the least-wanted resident — a tested knob,
see below), so resident RAM is a small fraction of all `E = 64` experts. The
goal is **not** to match full-MoE quality — it is a tunable intermediate on the RAM-footprint↔quality
frontier (between dense and MoE) for RAM-constrained deployment. Success = at the compute-optimal shape,
temporal validation **BPB (bits-per-byte, lower better)** lands **inside the already-measured band**:
@1e17/s2 dense **1.341** .. MoE **1.269**; @1e16/s0 dense **1.519** .. MoE **1.447**. Full rationale and
math: `docs/research/temporal-moe.md` (§2 "rolling residency") and `docs/EVALUATION_METHODOLOGY.md`.

## Design (why the diff is tiny and safe to review)

- **All logic is one pure function** `compute_resident_mask(logits, k, evict)` in
  `scripts/phase0/temporal_router.py` — `[seq,batch,E]` logits → boolean `[seq,batch,E]` mask, exactly `k`
  True per token. No Megatron, no GPU. This is the only novel code; it is fully unit-tested.
- **The router patch is ~6 lines** (`temporal_forward`): mask non-resident experts to `-inf`, then call
  Megatron's **unmodified** `routing()`. So z-loss, aux-loss, top-k and the alltoall dispatcher are reused
  byte-for-byte (they just see masked logits — the deliberately chosen, most-surgical option).
- **Zero edits to the Megatron-LM submodule.** The patch is installed by monkeypatch from
  `pretrain_temporal.py`, exactly mirroring the existing `scripts/phase0/expert_load.py`.
- **No new Megatron CLI flags.** Resident size `K` = the existing `--moe-router-topk` (read off the router).
  Activation is a single `TEMPORAL=1` env in `run.sh`, following the `DENSE=1`/`EVAL_ONLY=1` precedent.

### Selection policy (what the pure function computes)
**Swap-then-use** — a token pulls in one expert and uses it the SAME step (no prefetch lag), so each
token gets its top-k experts that are within +1 swap of the current set. This drops the prediction
burden (the token gets what it wants *now*) — the right semantics for a *quality* PoC.
- `t=0` cold fill: `R_0 = top-k(logits[0])` ("first token picks all experts").
- for `t ≥ 1`: `R_t = swap(R_{t-1}, logits[t])`: nominate the best **non-resident** expert; swap it in
  **iff** it beats the worst resident (≡ `R_{t-1}` ≠ global top-k); evict per the `evict` knob.
  `mask[t] = R_t` (the post-swap set the token actually uses).

> Note: the alternative **use-then-swap** (the router prefetches for `t+1`, so a token is served by the
> previous step's set) is the *systems*-faithful variant that hides SSD latency — revisit it when
> throughput is in scope. It is NOT what's implemented now; we deliberately chose swap-then-use for the
> quality measurement.

## Files (the entire change)

| file | new? | role |
|---|---|---|
| `scripts/phase0/temporal_router.py` | new | `compute_resident_mask` (pure) + `temporal_forward` + `install()` |
| `scripts/phase0/test_temporal_router.py` | new | 9 pure-function specs (TDD) — run these first |
| `scripts/phase0/pretrain_temporal.py` | new | entrypoint: `install()` then Megatron `pretrain(...)` (mirrors expert_load.py) |
| `scripts/phase0/run.sh` | edit (~6 lines) | `TEMPORAL=1` swaps entrypoint to `pretrain_temporal.py`; logs `temporal=` |
| `scripts/phase0/temporal_smoke.txt` | new | the lru/1-shared/s0@1e16 cell, run alone first as the integration smoke |
| `scripts/phase0/temporal_{lru,minlogit}_sh{1,2}.txt` | new | the 4 matrix regimes (each lists its s0@1e16 + s2@1e17 cells) |
| `scripts/phase0/temporal_matrix.sh` | new | runs the full 8-cell matrix serially (sets per-regime env; idempotent) |

## Step 0 — verify locally (no GPU; ~1s)

```bash
git switch temporal-moe-impl
python3 -m pytest scripts/phase0/test_temporal_router.py -q   # expect 9 passed
```
(Local tree's Megatron-LM submodule is uninitialized — that's fine; only the pure function is tested here.)

## Step 1 — integration smoke on the pod (THE first real test)

The pod has the initialized submodule + tokenized data + `.venv`. This first run is the integration test
(the patch firing inside a real forward pass) AND the throughput check. Common env (absolute paths — see
EVALUATION_METHODOLOGY §8f/§10a):
```bash
export TOKENIZER_MODEL=/workspace/FLAME-MoE/data/tok16k
export DATA_DIR=/workspace/FLAME-MoE/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7568
export TEMPORAL=1 TEMPORAL_EVICT=lru SHARED_MULT=2 TOPK=6
EVAL_AT_END=1 nohup bash scripts/phase0/drive.sh scripts/phase0/temporal_smoke.txt \
  > results/phase0/temporal_smoke.drive.log 2>&1 &
```
Watch with the EVALUATION_METHODOLOGY §11 reactive loop. **Pass = it launches, `[temporal] rolling-residency
router installed (evict=lru)` prints, `lm loss` descends, `nan iterations: 0`, throughput is tolerable** (see
throughput risk below). No band judgment here — this only proves the mechanism runs. (This cell is also the
matrix's `tmoe_lru_sh1_s0_1e16`, so the matrix skips it afterwards.)

## Step 2 — the full 8-cell matrix (the actual result)

We run the whole cross-product **{lru, min_logit} × {1 shared, 2 shared} × {s0@1e16, s2@1e17} = 8 runs**.
Resolving both knobs at both budgets (instead of carrying one winner) is intentional — it shows whether the
eviction and shared-expert effects are consistent across scale. One GPU ⇒ strictly serial; the wrapper sets
each regime's env and `drive.sh` skips the already-done smoke cell.

| evict \ shared | 1 shared (`SHARED_MULT=2 TOPK=6`, K=6) | 2 shared (`SHARED_MULT=3 TOPK=5`, K=5) |
|---|---|---|
| **lru** | `temporal_lru_sh1.txt` (smoke + 1e17) | `temporal_lru_sh2.txt` |
| **min_logit** | `temporal_minlogit_sh1.txt` | `temporal_minlogit_sh2.txt` |

```bash
# common env as above (TEMPORAL is set by the wrapper). Then, after the smoke passes:
nohup bash scripts/phase0/temporal_matrix.sh > results/phase0/temporal_matrix.log 2>&1 &
```
Each cell's BPB (success = inside the band, dense .. MoE: @1e16/s0 1.519..1.447, @1e17/s2 1.341..1.269):
```bash
BPB_DIVISOR=2.7568 .venv/bin/python scripts/phase0/parse_run.py results/phase0/runs/tmoe_lru_sh1_s2_1e17
# ... repeat per cell: tmoe_{lru,minlogit}_sh{1,2}_{s0_1e16,s2_1e17}
```
Then extend `scripts/phase0/plot_dense_vs_moe.py` with the temporal curve(s) — the matrix already covers both
budgets, so no extra back-fill is needed for the two-budget frontier figure.

## Risks / knobs the next agent should expect

1. **Sequential-scan throughput (most likely issue).** `compute_resident_mask` loops over the sequence
   (2048 steps/layer; ~5 small `[B,E]` tensor ops each). It's correct but launches many tiny CUDA kernels.
   If the smoke run's tok/s is unacceptable, optimize the loop (e.g. CUDA-graph/`torch.compile` the per-step
   body, or batch the nominee computation in a pre-pass) — **do not** change the semantics; the unit tests
   must stay green.
2. **Eviction policy is the #1 quality knob — `TEMPORAL_EVICT={lru,min_logit}` (default `lru`).**
   `min_logit` evicts the lowest-scoring resident (consistent with the swap trigger, quality-greedy); `lru`
   evicts the oldest-refresh resident (protects just-loaded experts from thrash, score-neutral w.r.t. the aux
   loss). Both are unit-tested and both run in the Step-2 matrix. The swap *count* is identical across policies
   (the trigger gates it) — only which expert leaves differs, so it's a clean quality comparison.
3. **Loss interaction (verified against the Megatron source — no special handling needed).** The aux/z loss is
   computed inside the **unmodified** `routing()` from the **masked** logits we pass, so it works identically
   for both eviction policies (they only change *which* experts are masked). The aux loss balances *how often
   each expert is resident-and-used*. Two consequences of masking to `-inf` worth knowing: (a) with
   `--moe-router-pre-softmax`, baseline gates sum to **<1** (mass leaks to unselected experts), but masking
   concentrates all softmax mass on the resident set so **our gates sum to 1** over the k resident — a free
   renormalization, identical for both policies; (b) this means temporal gating is *not* a pure restriction of
   baseline FLAME-MoE gating (different gate magnitudes) — fine for a from-scratch model, just don't expect
   bit-identical gating to the baseline.
4. **Coherence, not gradient, is the open empirical question.** The router gets the full gate-weight gradient
   on resident experts; the swap `argmax` is non-differentiable but rides on the trained `W_g`. The failure
   mode is the model tolerating whatever's resident instead of making usage temporally coherent (collapsing
   toward dense). If BPB sits at/under the dense floor, that's the signal. Deferred levers (only if needed):
   a churn-penalty aux loss, or a straight-through estimator on the swap.
5. **expert_load.py + temporal both patch `TopKRouter.forward`.** For a criterion-4 load check on a temporal
   model the two patches must compose (count *after* masking) — handle if/when you run that check.
6. **`K = k` tracks `--moe-router-topk`.** For s=2 (`TOPK=5`) the resident set is automatically 5. Intended.

## Pointers

- Design doc: `docs/research/temporal-moe.md` (rolling-residency section).
- Methodology (metric, harness, driver, monitoring, stop rule): `docs/EVALUATION_METHODOLOGY.md`.
- Measured bounds to beat/sit-between: `results/phase0/DENSE_BASELINES.md`, `results/phase0/RESULTS.md`.
- The research-side doc edits (the `[ROUTE]` token, masking, the gradient-correctness sentence) are staged on
  the `main` branch, intentionally uncommitted.
