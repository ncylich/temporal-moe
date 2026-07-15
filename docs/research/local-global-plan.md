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

## VERDICTS (2026-07-03, program complete)

| rung | mechanism | result | gates |
|---|---|---|---|
| LG1 | log-ratio momentum (β=0.25, plain) | **diversity-safe but A3-inert** — A3 28.6% ≈ L0 29.6%; union UP (164.6 vs 160.8), eff 182.2, 0 pinned; BPB 1.4714 seed-1 (pairing below) | FAIL (A3 only) |
| LG2 | bursty window loss (λ=0.02, W=32, plain) | **catastrophic Goodhart** — eff 183.9→33.2, max-res 100%, 5 pinned/layer, BPB +0.045; loss never minimized (rising-plateau anticipatory fingerprint) | FAIL (4/5) |
| LG3 | block-local routing (replay) | **not promoted** — freezing on prev-block demand loses 16–35pt retained gate-mass at every T on all 10 logs (structural, not tunable); diversity half works (karen max-res 88→62 at T=9) | free test declines |

**LG1 seed pair (final):** seed-1 1.4714 / seed-2 1.4782, |Δ| = 0.0068 (inside family spread);
**2-seed mean 1.4748 vs L0 1.4753 = parity.** A3 reproduces at ~28.7% (baseline-level), diversity
reproduces at full (eff ~183, max-res ~20%, 0 pinned). LG1 clears the two gates Karen failed and
fails the one Karen passed — the mirror image: Karen buys alignment by spending diversity; LG1
keeps diversity and buys no alignment. The two levers are antagonistic on this substrate — the
local-vs-global tension, stated as a measured result.

**Program synthesis (now over every trained cell in both programs):** LOSS-based demand shaping
(coherence, anticipatory, bursty) always Goodharts — the gradient finds pool collapse as the
cheapest optimum regardless of how the objective is phrased (predictive, self-referential, or
structural), and the tuned global aux is too weak a counterweight at any useful λ.
SELECTION-based shaping (add/double/logratio momentum) is diversity-safe on a healthy substrate
but A3-inert there — its apparent alignment power was always the aux-free substrate's
degeneracy. The zero-time-average design goal (bursty demand, flat marginal) was achieved
mechanically by LG1 — burstiness just doesn't increase, because the router's demand turnover is
already at its equilibrium. **No mechanism moves off the alignment↔diversity frontier; the
frontier itself (plus its two clean Goodhart negatives) is the publishable finding.**

## Schedule (deadline 21:00 UTC / 2pm PDT)
- ~17:45 both replays dispatched (no new code needed; throwaway scripts).
- ~18:15 momr-replay done → beta* → H100 trains LG1 (~50 min) then LG2 (~50 min);
  A6000 trains LG1 seed-2 (~140 min) alongside its block replay (CPU).
- ~20:30–21:00 all results + rows in alignment_cells.csv / seed_replicates.csv /
  momr_replay.csv / block_replay.csv; verdicts against the five gates.

## VERDICTS — flight window + nomination-head program (2026-07-04)

All cells: plain substrate, s0@1e16, seed 1234 unless noted; five gates vs L0
(3-seed anchor: test BPB 1.4750 ± 0.0009 / A3 29.6% / eff 183.9 / max-res ~18% / 0 pinned).

### LG1 dose ladder — demand-history selection FALSIFIED at full dose (H100, 0070/0071)
| β (logratio) | test BPB | A3 % | eff | max-res % | gate |
|---|---|---|---|---|---|
| 0.25 | 1.4714 | 28.63 | 182.2 | 20.2 | A3-FAIL |
| 1.0  | 1.4686 | 29.36 | 179.0 | 21.6 | A3-FAIL |
| 2.0  | 1.4720 | 30.89 | 172.7 | 34.1 | A3-FAIL |
A3 engages monotonically with dose (co-adaptation the replay missed) but trades ~1:1 with
diversity; at β=2.0 eff/max-res sit at the screen edges with A3 still 3pt short. No β clears
A3 + diversity together. No seed-2, no s2.

### Backlog seed replicates (A6000, 0091–0093)
- **karen_soft pair:** 2-seed mean 1.4739 = BPB parity with L0 (single-seed 1.4703 "best in
  program" did not survive). A3 ~35.5% reproduces; so does concentration (eff ~128, max-res ~72%).
- **L0 anchor:** now 3 seeds (1.4754/1.4758/1.4737) → 1.4750 ± 0.0009, spread 0.0021.
- **A1 aux-free pair:** mean 1.4526 = −0.006 vs softmax 1.4585 → safe substitution CONFIRMED.
  vs sigmoid a0-ctl 1.4475 (1 seed): parity-leaning-worse, needs a0-ctl seed-2 (OPEN ITEM).

### Nomination head H1/H2 — mechanism FALSIFIED; A3 gain was learned static popularity bias
Stop-grad per-layer head (detached hidden → E logits), BCE vs discounted future demand
(anticipatory_target γ=0.5), selection bonus HEAD_BETA·zscore_E(σ(head)) after 25% warmup.
Unit-gated (stop-grad property: trunk/router grads exactly zero; 59 tests green), 200-iter smoke
4/4 (0074), then trained cells (0075/0081/0082):
| cell | test BPB | A3 % | eff | max-res % | gates |
|---|---|---|---|---|---|
| head β=1.0 | 1.4857 | **37.98** | 102.6 | 41.4 | FAIL 3/5 (first-ever A3 pass) |
| head β=1.0 + centering (H2, γ_c=1/64) | 1.4857 | 30.22 | 162.5 | 31.4 | FAIL 3/5 |
| head β=0.5 | 1.4765 | 32.42 | 134.8 | 28.2 | FAIL 2/5 (BPB ~L0) |
**The controlled triplet (L0 → head → head+centering) is decisive:** centering the bonus
(zero time-average per expert by construction) removes BOTH the diversity collapse (eff
102.6→162.5) AND the A3 gain (38.0→30.2 ≈ L0). The head never learned content-driven
anticipation — it learned a static popularity table; z-scoring turned that into a standing
per-expert boost (incumbency in disguise). β-dose moves along the same frontier monotonically.
E5's replay headroom (+20–30pt content nomination) did not survive training: the BCE target is
dominated by its stationary marginal, so popularity is the loss's cheapest optimum. A
centered-TARGET head (predict demand anomaly, popularity unlearnable — "H3") is designed but
NOT dispatched; prior modest.

### karen-centering replay (A6000, 0096, promote-only) — NO promotion
Per-expert centering of karen_soft's momentum bonus (both flavors) holds A3 (−0.6pt) but recovers
only eff 128→132 / max-res −6pt (needs 170 / <40). karen_soft's concentration is STRUCTURAL to
the aux-free residency substrate, not the bonus's time-average.

### Meta-synthesis (14-cell frontier)
Every A3 gain ever measured in this program is popularity concentration in some costume:
aux-free score scale (Karen), momentum demand-history (LG1 at dose), and now a LEARNED static
bias (head). Selection-only mechanisms are diversity-safe exactly insofar as they are A3-inert;
gradient mechanisms Goodhart. **Nothing moves off the alignment↔diversity frontier at s0.**
The head triplet is the cleanest controlled demonstration of the frontier's mechanism yet —
lead exhibit for the writeup.

### REVISION (2026-07-04 ~23:00Z) — ceiling probes OVERTURN the frontier's finality
Free probes on all 20 preserved router logs (a6000 0100/0101, oracle_a3.csv + anomaly_pred.csv):
1. **Oracle-A3 (clairvoyant nominator, same cap-1 swap budget):** +17.7–18.7pt A3 on EVERY
   temporal cell (L0: 29.6 → 48.2 discounted-γ0.5) — and on the diverse plain substrate the
   oracle KEEPS diversity (eff ~176, max-res <45). A high-A3 + high-diversity selection policy
   EXISTS. The alignment↔diversity antagonism is a property of the weak learned/hand-set
   triggers, NOT of the substrate. Next-token horizon thrashes (−6 to −11pt); the headroom needs
   SMOOTHED future demand. Swap rate ~1/token throughout: smarter swaps, not more.
2. **Anomaly predictability (causal history features, trivial fitted predictors):** AUC
   0.70–0.87 on temporal substrates (vs 0.48–0.64 full-MoE) — the discounted future demand is
   learnable from history alone; the momentum trigger was simply a weak predictor. Residency
   training itself induces temporal coherence in demand.
**Program consequence:** "nothing moves off the frontier" is now "no mechanism TRIED SO FAR
moves off the frontier — but an oracle does, with a fully learnable target." Pre-approved next
rung: H3 = head on CENTERED demand labels (above-own-baseline transients; popularity
unlearnable — implemented + unit-tested, HEAD_TARGET_CENTER=1). K2 (gate momentum, fast
activator–inhibitor state in the routed scores; smoke passed, cell training) attacks the demand
process from the other side. Candidate after H3: replace the hand-set trigger with the FITTED
history predictor (frozen offline weights → rectification-proof) — not yet approved.

### K2 + H3 verdicts (2026-07-05 ~02:40Z) — the night's two trained cells
**K2 gate momentum** (`momg_b05`: logratio bonus IN the routed scores, β=0.5): BPB 1.4754 (neutral)
/ A3 28.6 / eff 181.3 / max-res 20.1 — FAIL (A3 only). Failure signature (ii): the slow optimizer
CANCELS the fast state — dynamics engaged (top-1 run length 1.6→1.18: demand got MORE volatile,
not more cacheable) and W absorbed it at zero BPB cost. Self-contained fast dynamics cannot
entrain the demand process against co-adaptation. 15th frontier cell.

**H3 target-centered head** (`head_tc`: labels = 1[disc-γ0.75 future demand > expert's own slow
baseline], popularity unlearnable; γ re-targeted 0.5→0.75 per horizon map): BPB **1.4751** /
**A3 34.21 PASS** / eff 125.2 FAIL / max-res 49.3 FAIL / pin 0. **First A3 gate pass that is
provably genuine anticipation** — the popularity shortcut is structurally absent from the labels.
Decomposition of the uncentered head's +8.4pt: ≈ +4.6 genuine anticipation + +3.8 popularity.

**Synthesis, revised once more:** genuine transient-demand anticipation EXISTS and IS learnable
in training (+4.6pt, BPB-neutral) — but the trained head still lands ON the frontier (eff −32%),
while the same-γ oracle sits far OFF it (+20.6pt with eff ~176). So the frontier is NOT
fundamental to the policy space (the oracle refutes that) and NOT purely Goodhart (H3 refutes
that): it is where WEAK PREDICTORS land — a partial predictor concentrates residency on the few
transients it can see; a precise one spreads. The learnability gap is the whole game: head BCE
barely beat the 18% label base rate from h_t, while the offline fitted logistic on HISTORY
features hit AUC 0.72–0.74 on the same target — history is more informative than the hidden
state here. Candidate next rung (NOT yet approved): freeze the fitted history-predictor and use
it as the trigger's nomination score (offline weights → rectification-proof; approximates the
oracle ranking that provably preserves diversity).

### MECH-INTERP: why the temporal constraint wins (2026-07-05, a6000 0123-0125)
**Answer: DE-LEXICALIZATION of routing.** Full-MoE routers default to lexical specialization —
the CURRENT TOKEN predicts which expert fires at AUC 0.84 (s2) – 0.94 (s0), 99–100% of experts
token-dominated. The residency constraint breaks that shortcut (a token is served by whatever is
resident): temporal token-AUC collapses to 0.60–0.62 at both scales, and experts move to CONTEXT
(s0: context-AUC 0.76 > token 0.62, 88% context-dominated, every layer; s2: balanced). Context is
the transferable, autocorrelated feature → better held-out loss (the regularization) and
history-predictable demand (the Q2 AUC 0.85-vs-0.64 finding and the +18pt oracle headroom are the
same phenomenon). Structure (P1): the constraint reshapes USAGE not weights — flatter routing
(entropy 0.95 vs 0.85), generalist-on-token experts (PR 0.66 vs 0.25), weight orthogonality
IDENTICAL. Gradient-batching hypothesis: NULL (tokens-per-expert-per-batch identical, k/E fixed).
Logit-lens (P4): static variant inconclusive, but the DATA-WEIGHTED variant (fc2 columns
weighted by gated activation) VOTES WITH P1+P2 — full-MoE experts promote sharp/lexical vocab
clusters (eff-vocab 15439, sharpest decile 13431) vs temporal diffuse/topical (15932, 15342);
the verdict is 3-LENS (structural + input-locus + output-lens). Honest edge: the invariant is
de-lexicalization; full context-takeover is decisive at s0, balanced-only at s2. CSVs: mechinterp_{structural,locus,locus_s2,freerider,logitlens}.csv.

### R-knob program: unmask 2x2 + residency-size sweep (2026-07-06, a6000 0137/0140/0143)
**Unmask 2x2 (Q1):** the temporal advantage is SERVING CO-ADAPTATION, not a transferable
train-time regularizer. Unmasking temporal ckpts costs +0.10 (s0) / +0.12 (s2) / **+0.485 at the
1e18 winning case** (3.9037→4.389); imposing residency on full-MoE ckpts costs 2–5x more
(+0.24/+0.61) — lexical routing shatters under the constraint, contextual routing degrades
gracefully (mech-interp made causal).
**R-sweep (Q2, s0/1e16, R = residency-cache size, top-k fixed at 18):**
  R 18/36/72/128/192 → test BPB 1.4750/1.4736/1.4681/1.4580/1.4475 (monotone), eff 182–186
  everywhere (diversity never collapses), maxres 18→99.9%, first pinned expert only at R=128.
At this scale the constraint is a pure, monotone quality cost — the regularization only PAYS at
1e18 — and the curve IS the deployment quality-vs-RAM frontier (FLOPs identical at all R): R=k
costs +0.0275 BPB vs full at ~1/10 the expert RAM; R=72 recovers ~25% of the gap at 4x cache.
Permutation floors (0140): all locus AUC floors = 0.500±0.002 (circular-shift and iid) — the
de-lexicalization table is calibrated; temporal lexical residue +0.10 vs full +0.34–0.44.
