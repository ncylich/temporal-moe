# Local consistency, global diversity — program plan (2026-07-03, hard deadline 21:00 UTC)

**Goal (Noah, 2026-07-03 morning):** mechanisms that raise the router's LOCAL self-consistency
(A3) while PRESERVING global diversity — best-of-both-worlds candidates only. The eff-experts
residency gate stays as-is; we are not relaxing gates, we are trying to beat them.

**Formal reframe.** We want expert demand that is *bursty in time with a uniform stationary
marginal*: per-expert usage autocorrelated (runs, then dormancy), long-run marginal flat
(eff-experts ≈ plain's 184). A3 measures the autocorrelation; eff-experts the marginal. These are
independent properties — the alignment↔diversity frontier we kept hitting is an artifact of
mechanisms whose bonus was *sign-stable over time* (aux-free bias, additive momentum,
anticipatory): a time-integrated additive preference IS popularity, so it must fatten the
marginal. The fix: any boost an expert earns during a run must be paid back when the run ends —
zero time-average **per expert** (a zero-sum-per-token variant is a provable no-op; the residency
trigger only compares within a token).

**Scope guard (unchanged):** demand-history only, never cache state; wrong-directions header of
`decision-time-alignment-plan.md` still binds; free tests promote, only trained cells falsify;
one principled config per rung at the per-budget compute-optimal shape (s0@1e16 first).

**Pre-registered gates (unchanged, vs L0 1.4753 / A3 29.6% / eff 183.9):**
test BPB ≤ 1.4783 AND A3 ≥ 34% AND eff-experts ≥ 170 AND max-residency < 40% AND pinned = 0.

## Mechanisms (first three, this program)

### LG1 — log-ratio (popularity-normalized) momentum  [selection-only]
Selection bonus `beta * ln((M+eps)/(Q+eps))`, M = fast demand EMA (gamma_m=0.125, ~8 tok),
Q = slow usage EMA (gamma_q=0.015625, ~64 tok), eps = 1/E. Chronic expert: M≈Q → bonus ≈ 0.
Rarely-used expert in a fresh burst: M≫Q → large positive bonus (diversity-favoring at exactly
the moment of local demand). Just-cooled: negative. The log keeps the bonus in O(beta)
logit-scale units — the additive M−Q form was scale-inert on raw logits (mom-plain negative).
Code: `TEMPORAL_MOM_MODE=logratio` in temporal_router.py + probe_replay mirror (this commit).
- Free test (H100, task momr-replay): beta grid {0.25,0.5,1,2} on preserved plain logs;
  beta* = max setcov s.t. replayed eff ≥ 170 and max-res < 40%; tie → smaller.
- Trained cell (H100): `g3_tmoe_s0_1e16_momr`, PLAIN substrate, beta*, seed 1234, PROBE=1,
  five gates. Seed-2 replicate on A6000 in parallel (momentum family seed spread ~0.006–0.009
  BPB makes single-seed BPB claims inadmissible).

### LG2 — bursty window loss  [gradient, plain substrate]
`bursty_window_loss(logits, W)`: mean over W-token windows of H(mean-demand) — minimizing
concentrates demand within windows; the UNTOUCHED standard global aux keeps the cross-batch
marginal flat and vetoes the global-collapse shortcut. Target equilibrium: bursty rotation.
Unit-tested to be indifferent between rotation and collapse (the global aux does that job).
Distinct from the two failed gradient objectives: not self-referential (coherence: cloned own
mask) and not predictive (anticipatory: Goodharted its own target); this states the *structural*
property we want. Config (one rung): `BURSTY_LAMBDA=0.02` (the anticipatory-scale probe point),
`BURSTY_WINDOW=32` (~2 residency turnover times at k=18). No free test exists (needs gradients).
- Trained cell (H100): `g3_tmoe_s0_1e16_bursty`, PLAIN substrate, standard aux untouched,
  seed 1234, PROBE=1, five gates.

### LG3 — block-local routing  [structural; replay-only today]
Hold the resident set for T-token blocks; re-decide at boundaries from the previous block's mean
demand (causal). Within-block consistency is structural (A3=100% inside a block); every boundary
re-decides from the full pool with no accumulated preference → marginal untouched by
construction. With T ≥ k the cap-1 swap/token budget covers a full set turnover per block
(amortized; instantaneous burst reported). Temporal mirror of the spatial fine-graining story.
- Free test only today (A6000, task block-replay): T ∈ {k/2, k, 2k, 4k} on all preserved logs;
  promote (run,T) if retained gate-mass ≥ rolling-baseline − 2pt AND eff ≥ 90% of plain AND
  amortized swaps ≤ 1/token. Trained cell is a separate later decision (bigger code change).

## Schedule (deadline 21:00 UTC / 2pm PDT)
- ~17:45 both replays dispatched (no new code needed; throwaway scripts).
- ~18:15 momr-replay done → beta* → H100 trains LG1 (~50 min) then LG2 (~50 min);
  A6000 trains LG1 seed-2 (~140 min) alongside its block replay (CPU).
- ~20:30–21:00 all results + rows in alignment_cells.csv / seed_replicates.csv /
  momr_replay.csv / block_replay.csv; verdicts against the five gates.
