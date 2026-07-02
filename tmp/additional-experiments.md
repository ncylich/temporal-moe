# Tier-1 probe-replay experiments — handoff

**For an agent on the pod (or any machine with the trained checkpoints / probe logs).**
All experiments below are **offline analysis of router-probe logs — zero training**. GPU is only
needed if a probe pass has to be re-run (`PROBE=1`, a forward pass on one fixed batch) or for the
single optional eval-only run in E7. Everything else is CPU/replay work in `plot_probe.py`-land.

## Context (read first)

- Rolling-residency temporal MoE: keep `K = k` routed experts resident per layer, swap ≤1 expert
  per token (trigger: best non-resident logit > worst resident), evict `min_logit` (or `lru`).
  Router: `scripts/phase0/temporal_router.py` (pure function `compute_resident_mask`, unit-tested —
  **do not change its semantics**; these experiments only *analyze* or *replay* logged demand).
- Probe harness: `scripts/phase0/router_probe.py` (run via `run.sh PROBE=1`) logs, per MoE layer per
  token on one fixed 16×2048 batch (seed 1234), the **raw pre-mask gating logits** and (for temporal
  models) the **resident set used**. Analysis/figures: `scripts/phase0/plot_probe.py`.
  Existing results: `docs/research/mechanistic-probe-results.md` and
  `results/phase0/figures/{expert_selection_per_token_*,routing_coverage_*,expert_lifetime_*,learned_temporal_locality_*}.png`.
- Checkpoints already probed (matched temporal+MoE pairs): `tmoe_minlogit_sh1_s0_1e16` +
  `v16k_d_s0_1e16`; `tmoe_minlogit_sh1_s2_1e17` + `v16k_sweep_s2_1e17`; `tmoe_minlogit_sh1_s3_1e17`
  + `v16k_sweep_s3_1e17`; `g3_tmoe_s1_1e17` (fine-grained 18/192); `flame38m_temporal_minlogit`
  (38M @1e18, 50k vocab).
- Known baseline numbers (for sanity checks): A3 same-set overlap ≈ temporal 30–38% vs full-MoE
  ~20% vs random k/E = 9.4%; rolling hit-rate at K=k ≈ 36% (temporal s2) vs 18% (full MoE).

Deliverable: one new doc `docs/research/probe-replay-results.md` (numbers + takeaways per
experiment, self-contained) plus figures in `results/phase0/figures/` with self-explanatory
names/captions, matching the existing house style. Where an experiment compares models, run it on
at least: temporal s2@1e17 (G1), g3_tmoe_s1_1e17 (G3), and the 38M@1e18 checkpoint.

---

## E1 — Swap-rate telemetry (feasibility margin)

From each temporal probe log, compute the realized swap statistics under the shipped policy
(K=k, ≤1 swap/token, min_logit):

- mean swaps/token per layer (= fraction of tokens whose trigger fires), and the distribution
  (bursts: run-lengths of consecutive swapping tokens; per-sequence variance).
- Compare against the bandwidth budget for hiding a swap behind the *same layer's* resident-expert
  compute: `s_max = (k−1) · r / r_ram` with typical `r_ram/r ≈ 32` →
  **G1 (k=6): 5/32 ≈ 0.16; G3 (k=18): 17/32 ≈ 0.53; k=32: 31/32 ≈ 0.97** swaps/token.
  (Context: router-early placement — computing the router before attention — roughly doubles these
  budgets by adding attention+dense compute to the hiding window; see `cce/FINDINGS.md` for the
  ~17-expert-equivalents accounting. Report the measured rate against both budgets.)
- Output: table (model × layer → mean swap rate, p95 burst length, margin vs. `s_max`).

**Why:** decides whether the cap-1 policy is bandwidth-feasible as-is at each grain, and how much
slack exists for prefill bursts / latency spikes.

## E2 — Streamed-diversity attribution (+ pinning candidates)

For each temporal model, from the logged resident sets:

- **Union size per sequence**: distinct experts resident at any point over the 2048 tokens, per
  layer (mean ± spread). If ~40–60 of 64 → streamed diversity is real (the model uses far more
  experts over time than fit in RAM); if ~≤15 → the model collapsed to a small effective pool and
  a static small-E MoE would match it.
- **Per-expert residency fraction**: for each expert, fraction of tokens it is resident. Plot the
  distribution per layer. **Bimodality check**: experts with residency >0.8 are de-facto pinned
  (the s2 raster shows one, ≈ expert 12 at layer 6) — list them per layer/model. These are the
  candidates for an explicit pinned-slot architecture later.
- **Effective expert count**: exp(entropy) of the residency marginal per layer.
- Also compute per-expert *token-service* counts (tokens each expert actually served) — the
  concentration statistic behind the 1e18 "temporal beats full G3-MoE via better-trained experts"
  hypothesis. Compare temporal vs. matched full MoE.

**Why:** separates the two forces inside temporal's quality (streamed access to all E experts vs.
usage concentration), which the 1e18 result showed can both be active; finds pinning candidates.

## E3 — Mass-weighted consistency and coverage

Recompute the two headline probe metrics **weighted by gate mass** instead of set membership:

- **Mass-weighted A3**: at each token, the softmax mass (over the unconstrained top-k) that falls
  on experts already in the previous active set — vs. the current set-overlap A3 (~38%).
- **Mass-weighted hit-rate curve (B)**: fraction of each token's top-k *gate mass* already
  resident, vs. K/k — overlay on the existing set-based B curve.

**Hypothesis to test:** gate mass is concentrated in the top 1–2 experts, which are resident far
more often than the tail, so 36% set-coverage may be 60–80% mass-coverage. If confirmed, the
"only 40% self-consistent" reading overstates the problem and effort should go to anticipating
the heavy top-1/2 demand, not to raising raw set consistency.

## E4 — Trigger-margin (τ) replay

Replay each temporal model's logged logit stream through `compute_resident_mask`-equivalent
policies with a **hysteresis margin**: swap iff `nominee_logit > worst_resident_logit + τ`
(grid τ, logit space; also a probability-space variant: swap iff nominee softmax mass − worst
resident mass > τ_p). K=k, cap-1, min_logit eviction.

- Report, per τ: swap rate (E1 metric) and retained gate mass (E3 metric) → one tradeoff curve
  per model (swap rate on x, retained mass on y).
- Pick **τ\*** = max retained mass subject to the deployable swap budget (use `s ≤ 0.5` for G3
  as the working target; also mark the τ that halves the swap rate).

**Why:** quality-per-swap is concave (dropping marginal swaps should be nearly free), and τ is a
zero-training, deploy-time knob. τ\* feeds a later single training cell (train-with-τ).

## E5 — Belady replay (the policy-headroom bound)

Same replay harness, but with **offline-optimal eviction (Belady)**: on a swap, evict the resident
whose *next selection* (in the logged unconstrained top-k stream) is farthest in the future —
computable exactly from the logs. Optionally also "prescient prefetch": allow the swap for token
t's demand to fire h tokens early (h = 1, 4, 16) to bound the value of anticipation.

- Also replay a **discounted-usage oracle**: score every expert at token t by its exact discounted
  future selection mass `y_t(e) = Σ_{j≥1} γ^{j−1}·1[e ∈ top-k(t+j)]` (one reverse scan over the
  logged stream; γ grid {0.5, 0.9, 0.95}); nominate argmax non-resident y, evict argmin resident y.
  This is the exact upper bound for a *learned* discounted-lookahead nomination head (the
  anticipatory-loss direction), sitting between min_logit and Belady.
- Report set- and mass-coverage for: LRU, min_logit, min_logit+τ\*, discounted-oracle(γ),
  Belady, Belady+prescient-h — all at K=k, cap-1, same logs.

**Why:** bounds how much *any* smarter eviction/nomination policy (learned or not) can buy at this
K. If Belady ≈ min_logit, eviction-policy learning is a dead end and the residual gap must be
attacked via anticipation, structure (pinning), or robustness — this single number redirects the
whole training-side research plan. If the gap is big, a learned cache policy (Parrot-style
imitation of Belady; Liu et al., ICML 2020) is justified.

## E6 — Per-layer ranking

From E1–E3, rank MoE layers by hit-rate / swap rate / lifetime (the existing B/C machinery already
has per-layer data). Report which layers are least locally consistent (prior work — Zhu et al.
2505.16056 — finds large per-layer variance; Mixtral finds locality grows with depth).

**Why:** any future non-uniform budget (per-layer K, per-layer pinning, per-layer τ) allocates
against this ranking; zero extra measurement cost.

## E7 — EMA-logit replay (slow-feature routing preview)

Because the router is linear (`logits = W_g·h`), an EMA over hidden states equals an EMA over
logits: `W_g·EMA(h) = EMA(W_g·h)`. So **smoothed routing is exactly replayable from the logged
logits**: `logits'_t = (1−β)·logits'_{t−1} + β·logits_t`, grid β ∈ {1.0, 0.5, 0.25, 0.1}.

- Re-run the E1/E3 metrics on the smoothed stream (selection = top-k of `logits'`): swap rate,
  A3, mass-coverage. (β=1.0 must reproduce the baseline numbers exactly — use as the harness
  sanity check.)
- Caveat to state in the writeup: replay evaluates the *selection policy* on a model trained
  without it, so coverage gains are indicative, not a quality measurement.
- **Optional GPU step:** if smoothing looks strong (e.g., swap rate halves at similar mass
  coverage), run one **eval-only** pass (`EVAL_ONLY=1`) with the smoothing patched into the
  temporal router on an existing checkpoint to get a real BPB delta — no training.

**Why:** previews "shape the demand" (slow-feature routing) for free, before any architectural
commitment.

## E8 — Document-boundary attribution

The probe batch packs multiple dclm documents into each 2048-token sequence, but residency only
cold-fills at t=0 of the *sequence*. Topic shifts at packed-document boundaries are training/probe
artifacts — at deployment each request starts with a fresh cold fill.

- Locate document boundaries (EOD token id) in the probe batch; measure the fraction of swaps and
  of top-k misses occurring within w ∈ {4, 16, 64} tokens after a boundary, vs. the within-document
  rate.
- If boundary-adjacent churn is a large share, report the *within-document-only* versions of the
  E1/E3 headline numbers — the deployment-relevant ones — and flag "reset residency at EOD" as a
  candidate one-line training change (more faithful simulation, removes cross-document
  contamination from the locality statistics).

---

## Guardrails

- No training runs. No changes to `compute_resident_mask` semantics; replay variants live in
  analysis code (extend `plot_probe.py` or add `scripts/phase0/probe_replay.py`).
- Keep `test_temporal_router.py` green; add unit tests for the replay policies (Belady on a
  hand-computable 5-token example; EMA β=1.0 identity).
- All figures: self-contained filename + caption, house style of `results/phase0/figures/`.
- Write conclusions as *decisions*: E2 → is streamed diversity real? E3 → is self-consistency
  actually low? E5 → is policy learning worth it? E7 → is demand-smoothing worth a training cell?
