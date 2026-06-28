# Temporal MoE (rolling residency) — implementation handoff

**Branch:** `temporal-moe-impl` (off `phase0-flame-baselines`). **Status:** code complete + unit-tested
locally; **not yet run on a GPU**. The next agent's job: integration-smoke it on the pod, fix anything
the first real forward pass surfaces, then orchestrate the quality runs and place the result on the
dense↔MoE frontier.

## What this is (one paragraph)

A trained-from-scratch MoE variant that keeps only `K = k = 6` routed experts resident per layer and
streams one expert in per token (LRU), so resident RAM is a small fraction of all `E = 64` experts. The
goal is **not** to match full-MoE quality — it is a tunable intermediate on the RAM-footprint↔quality
frontier (between dense and MoE) for RAM-constrained deployment. Success = at the compute-optimal shape,
temporal validation **BPB (bits-per-byte, lower better)** lands **inside the already-measured band**:
@1e17/s2 dense **1.341** .. MoE **1.269**; @1e16/s0 dense **1.519** .. MoE **1.447**. Full rationale and
math: `docs/research/temporal-moe.md` (§2 "rolling residency") and `docs/EVALUATION_METHODOLOGY.md`.

## Design (why the diff is tiny and safe to review)

- **All logic is one pure function** `compute_resident_mask(logits, k)` in `scripts/phase0/temporal_router.py`
  — `[seq,batch,E]` logits → boolean `[seq,batch,E]` mask, exactly `k` True per token. No Megatron, no GPU.
  This is the only novel code; it is fully unit-tested.
- **The router patch is ~6 lines** (`temporal_forward`): mask non-resident experts to `-inf`, then call
  Megatron's **unmodified** `routing()`. So z-loss, aux-loss, top-k and the alltoall dispatcher are reused
  byte-for-byte (they just see masked logits — the deliberately chosen, most-surgical option).
- **Zero edits to the Megatron-LM submodule.** The patch is installed by monkeypatch from
  `pretrain_temporal.py`, exactly mirroring the existing `scripts/phase0/expert_load.py`.
- **No new Megatron CLI flags.** Resident size `K` = the existing `--moe-router-topk` (read off the router).
  Activation is a single `TEMPORAL=1` env in `run.sh`, following the `DENSE=1`/`EVAL_ONLY=1` precedent.

### Selection policy (what the pure function computes)
Use-then-swap, deployment-faithful (so training matches decode-time prefetch):
- `t=0` cold fill: `R_0 = top-k(logits[0])` ("first token picks all experts").
- `mask[t] = R_t` (token `t` is served by the set available to it — a 1-token prefetch lag).
- `R_{t+1} = swap(R_t, logits[t])`: nominate the best **non-resident** expert; swap it in **iff** it beats
  the worst resident (≡ `R_t` ≠ global top-k); evict the **LRU** resident (oldest last-refresh).

## Files (the entire change)

| file | new? | role |
|---|---|---|
| `scripts/phase0/temporal_router.py` | new | `compute_resident_mask` (pure) + `temporal_forward` + `install()` |
| `scripts/phase0/test_temporal_router.py` | new | 7 pure-function specs (TDD) — run these first |
| `scripts/phase0/pretrain_temporal.py` | new | entrypoint: `install()` then Megatron `pretrain(...)` (mirrors expert_load.py) |
| `scripts/phase0/run.sh` | edit (~6 lines) | `TEMPORAL=1` swaps entrypoint to `pretrain_temporal.py`; logs `temporal=` |
| `scripts/phase0/temporal_1e16_smoke.txt` | new | driver config: s0@1e16, s=1 (smoke) |
| `scripts/phase0/temporal_1e17.txt` | new | driver config: s2@1e17, s=1 (quality) |
| `scripts/phase0/temporal_s2shared_1e17.txt` | new | driver config: s2@1e17, s=2 (run with `SHARED_MULT=3 TOPK=5`) |

## Step 0 — verify locally (no GPU; ~1s)

```bash
git switch temporal-moe-impl
python3 -m pytest scripts/phase0/test_temporal_router.py -q   # expect 7 passed
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
export TEMPORAL=1
EVAL_AT_END=1 nohup bash scripts/phase0/drive.sh scripts/phase0/temporal_1e16_smoke.txt \
  > results/phase0/temporal_smoke.drive.log 2>&1 &
```
Watch with the EVALUATION_METHODOLOGY §11 reactive loop. **Pass = it launches, `[temporal] ... installed`
prints, `lm loss` descends, `nan iterations: 0`, throughput is tolerable** (see throughput risk below). No
band judgment here — this only proves the mechanism runs.

## Step 2 — quality runs (the actual result)

```bash
# s=1 (shape s2 @1e17), default shared/topk:
export TEMPORAL=1 CE_FUSION=1   # + tokenizer/data/bpb as above
nohup bash scripts/phase0/drive.sh scripts/phase0/temporal_1e17.txt \
  > results/phase0/temporal_1e17.drive.log 2>&1 &

# s=2 (two shared experts, FLOP-matched) — note the extra env:
export TEMPORAL=1 CE_FUSION=1 SHARED_MULT=3 TOPK=5
nohup bash scripts/phase0/drive.sh scripts/phase0/temporal_s2shared_1e17.txt \
  > results/phase0/temporal_s2shared_1e17.drive.log 2>&1 &
```
Read BPB with the existing tool; success = inside (1.269, 1.341):
```bash
BPB_DIVISOR=2.7568 .venv/bin/python scripts/phase0/parse_run.py results/phase0/runs/tmoe_s2_1e17
```
Then back-fill s0@1e16 (s=1 and s=2) for the two-budget frontier figure, and extend
`scripts/phase0/plot_dense_vs_moe.py` with the temporal curve.

## Risks / knobs the next agent should expect

1. **Sequential-scan throughput (most likely issue).** `compute_resident_mask` loops over the sequence
   (2048 steps/layer; ~5 small `[B,E]` tensor ops each). It's correct but launches many tiny CUDA kernels.
   If the smoke run's tok/s is unacceptable, optimize the loop (e.g. CUDA-graph/`torch.compile` the per-step
   body, or batch the nominee computation in a pre-pass) — **do not** change the semantics; the unit tests
   must stay green.
2. **Eviction policy is the #1 quality knob.** Currently LRU-by-refresh (as specified/approved). The obvious
   alternative to A/B is "evict the least-wanted resident" (lowest current logit). It's a ~2-line change in
   `compute_resident_mask`'s `evict_i`; add a test if you switch.
3. **Coherence, not gradient, is the open empirical question.** The router gets the full gate-weight gradient
   on resident experts; the swap `argmax` is non-differentiable but rides on the trained `W_g`. The failure
   mode is the model tolerating whatever's resident instead of making usage temporally coherent (collapsing
   toward dense). If BPB sits at/under the dense floor, that's the signal. Deferred levers (only if needed):
   a churn-penalty aux loss, or a straight-through estimator on the swap.
4. **expert_load.py + temporal both patch `TopKRouter.forward`.** For a criterion-4 load check on a temporal
   model the two patches must compose (count *after* masking) — handle if/when you run that check.
5. **`K = k` tracks `--moe-router-topk`.** For s=2 (`TOPK=5`) the resident set is automatically 5. Intended.

## Pointers

- Design doc: `docs/research/temporal-moe.md` (rolling-residency section).
- Methodology (metric, harness, driver, monitoring, stop rule): `docs/EVALUATION_METHODOLOGY.md`.
- Measured bounds to beat/sit-between: `results/phase0/DENSE_BASELINES.md`, `results/phase0/RESULTS.md`.
- The research-side doc edits (the `[ROUTE]` token, masking, the gradient-correctness sentence) are staged on
  the `main` branch, intentionally uncommitted.
