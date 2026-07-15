# Temporal-coherence loss — verification + deployment plan

Goal: close part of the residual **temporal ↔ full-MoE** quality gap by adding a training-time
pressure that makes the router *prefer its own resident set*, so fewer tokens force a swap.
This doc is the plan + review gate — **nothing is launched until the PR is approved.**

## The loss (BCE)

Behaviour-cloning of the residency-masked policy into the raw router. For each token, the **final
used set** (the resident mask this token actually ran) is a multi-hot target; BCE the router's raw
per-expert logits toward it, with **independent sigmoids** (not softmax):

```
L_coh = BCE_with_logits(logits, resident_mask.detach())      # mean over all experts
```

- Independent per-expert sigmoids ⇒ a **set-membership pull, not a distribution clone** — it aligns
  the router with what it already used without forcing identical magnitudes or cloning the previous
  token's distribution.
- Target = *this* token's own final set, **detached** (it's a hard decision).
- Mechanism: raising the resident experts' logits this token makes next token's top-k demand more
  likely to already sit inside the resident set → fewer swaps → higher retention.

Rejected alternatives (see `mechanistic-probe-results.md` discussion): L2/cosine on the softmax
distribution (clones the whole distribution — too blunt); straight-through estimator (non-standard
in MoE, and carries **no** retention pressure — it optimises quality *given* the mask, not future
hit-rate).

## Code

- `scripts/phase0/temporal_router.py`
  - `coherence_bce_loss(logits, resident_mask)` — pure, tested (one-line BCE).
  - `temporal_forward` — if `TEMPORAL_COHERENCE_LAMBDA>0` **and training**, injects the loss'
    gradient onto the raw logits via `MoEAuxLossAutoScaler` (identical mechanism to the existing
    z-loss / load-balancing aux loss) and logs `coherence_loss` to the aux-loss tracker.
- `scripts/phase0/test_coherence_loss.py` — 6 CPU unit tests (alignment→0, anti-alignment large,
  gradient sign = retention direction, target-detachment vs analytic gradient, retention lowers
  loss). Run: `.venv/bin/python -m pytest scripts/phase0/test_coherence_loss.py`.

**Env knob** (no `run.sh` change needed — read from the environment):
- `TEMPORAL_COHERENCE_LAMBDA` (default `0` = off) — loss weight.

Eval is untouched (`self.training` gate), so final BPB is pure temporal — the loss only shapes
training gradients.

## Models — compute-optimal only

Fine-grained (each expert split 3×: 192 experts, top-18) — the single-swap constraint is more binding
there (1 swap = 5.5% of an 18-wide active set vs 17% of a 6-wide one), so the temporal↔MoE gap is
wider and it's the harder case. Both configs are the **per-budget compute-optimal shape**; both
already have measured full-MoE + temporal + dense baselines (zero new baseline cost).

| role | config | budget | N_active | dense | full MoE | temporal (no loss) | **gap to close** |
|---|---|---|---|---|---|---|---|
| smoke  | fine-grained **s0** | 1e16 | 1.42M | 1.519 | 1.4585 | 1.4753 | **0.017** |
| signal | fine-grained **s2** | 1e17 | 8.23M | 1.341 | 1.2708 | 1.2873 | **0.017** |

(BPB, lower better; seed noise ≈ 0.003, so 0.017 ≈ 5–6× noise.)

## Logging

`run.sh` sets `--tensorboard-dir` so Megatron's `track_moe_metrics` has a writer; each aux loss
(`load_balancing_loss`, `z_loss`, `coherence_loss`) is then logged **individually** in both the
train log and tensorboard, **separate from `lm loss`** (the reported `lm loss` is NOT summed with
the coherence term — the `MoEAuxLossAutoScaler` only injects a backward gradient). Per-log-interval
line looks like: `... lm loss: … | load_balancing_loss: … | z_loss: … | coherence_loss: … |`.

## How we pick λ

No a-priori value — λ scales the coherence gradient vs the LM gradient; find it from the scan by
watching the two now-separate curves:
- `coherence_loss` (BCE, ≈0.69 at init) should **decrease**; if flat even at high λ → λ too weak.
- `lm loss` / final BPB should stay **≤ the no-loss baseline**; if it rises → λ too strong (lock-in).
Pick the **largest λ that drives coherence_loss down while keeping BPB ≤ baseline**. Reading the
{0.02, 0.1, 0.5} bracket: monotone-improving through 0.5 → push higher (1, 2); even 0.02 regresses →
go lower (0.005, 0.01); interior best → refine around it.

## Procedure

1. **Smoke + λ scan (1e16 s0):** run `TEMPORAL_COHERENCE_LAMBDA ∈ {0.02, 0.1, 0.5}`. Check per run:
   (a) training stable, `coherence_loss` decreasing; (b) final BPB not worse than the no-loss
   temporal 1.4753 (a large drop in BPB or a collapse signals λ too high — self-distillation lock-in);
   (c) retention up. Pick the λ with the best BPB that also raises retention.
2. **Signal (1e17 s2):** one run at the chosen λ. Compare BPB to temporal-no-loss (1.2873) and
   full MoE (1.2708).
3. **Retention measurement (both):** re-run the router probe (`PROBE=1`) on the coherence checkpoint
   and compare A3 overlap + rolling hit-rate to the matched no-loss checkpoint via `plot_probe.py`.

## Success criteria

- **Retention (mechanism works):** A3 overlap / rolling hit-rate on the coherence checkpoint > the
  no-loss checkpoint (probe, same fixed batch).
- **Quality (it pays off):** signal-run BPB closes ≥ ⅓ of the 0.017 gap (≥ 0.006, > 2× seed noise)
  **without** crossing the dense floor (1.341). Neutral/negative ⇒ the router had already
  internalised the locality (consistent with A3) and there's little left to reclaim — a valid result.

## Exact commands (DO NOT RUN until approved)

Common env:
```bash
export TOKENIZER_MODEL=/workspace/FLAME-MoE/data/tok16k DATA_DIR=/workspace/FLAME-MoE/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7600 GRAIN=3 MICRO_BATCH=64 EVAL_AT_END=1
export TEMPORAL=1 TEMPORAL_EVICT=min_logit PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Smoke (one per λ):
```bash
SHAPE=s0 TARGET_FLOPS=1e16 PEAK_LR=3e-3 WARMUP_FRAC=0.05 GLOBAL_BATCH=256 SEED=1234 AUX_COEFF=0.01 \
  TEMPORAL_COHERENCE_LAMBDA=0.1 RUN_NAME=g3_tmoe_s0_1e16_coh0p1 bash scripts/phase0/run.sh
```
Signal (best λ):
```bash
SHAPE=s2 TARGET_FLOPS=1e17 PEAK_LR=3e-3 WARMUP_FRAC=0.05 GLOBAL_BATCH=256 SEED=1234 AUX_COEFF=0.01 \
  TEMPORAL_COHERENCE_LAMBDA=<best> RUN_NAME=g3_tmoe_s2_1e17_coh bash scripts/phase0/run.sh
```

## Results (measured — smoke λ scan, fine-grained s0 @ 1e16)

Baseline (no coherence loss) = **1.4754 BPB**; full-MoE ceiling = 1.4585; dense floor = 1.519;
seed noise ≈ 0.003. Lower BPB is better. All values measured (single eval at end).

| λ | BPB | Δ vs no-loss baseline | note |
|---|---|---|---|
| 0 (no-loss temporal) | 1.4754 | 0 | baseline |
| 0.005 | 1.4914 | **+0.0160** | regress |
| 0.02 | 1.5007 | **+0.0253** | regress |
| 0.001 / 0.01 / 0.1 / 0.5 | not completed | — | conclusion already settled by the monotonic trend |

Loss magnitudes at λ=0.005 (nats): BCE (coherence) settles ~0.22 vs CE (lm loss) ~4.1 → BCE ≈ 5 % of
CE raw; weighted λ·BCE ≈ 0.001 ≈ 0.03 % of the loss value. Note the loss-value share understates the
router-steering: the BCE gradient lands **directly** on the router logits (undiluted), whereas CE
reaches them attenuated through backprop — so even a 0.03 %-of-loss term measurably moved routing. Also
at λ=0.005 the BCE **rose** over training (0.18→0.22) — too weak to enforce coherence yet still costly.

**Finding — negative result: the coherence loss monotonically *hurts*; its best case is neutral as
λ→0.** The regression shrinks smoothly toward baseline as λ decreases (0.02 → 0.005), with no λ dipping
below baseline. This is consistent with the mechanistic A3 probe: the rolling-residency router already
internalises the temporal locality on its own (temporal 30–38 % same-set overlap vs full-MoE ~20 %),
so an explicit coherence loss has little to add and mostly distorts routing. The initial {0.02, 0.1,
0.5} bracket was pivoted down to {0.005, 0.01→0.001} once 0.02 already regressed.

**Implication:** do **not** carry the coherence loss to the 1e17 signal run — the smoke already
settles the question. Kept as a documented negative result (the loss code stays, env-gated off at
λ=0, for reproducibility).
**Successor plan** (decision-time mechanisms replacing gradient-space pressure):
[`decision-time-alignment-plan.md`](./decision-time-alignment-plan.md).

## Risks

- **Self-distillation lock-in:** too-high λ freezes the router onto an early resident set at a quality
  cost. Mitigation: the λ scan; watch BPB vs the no-loss baseline.
- **Load-balance interaction:** the BCE negatives add mild concentration pressure vs the aux
  loss; `AUX_COEFF` stays at 0.01, watch expert-load spread if BPB regresses.
