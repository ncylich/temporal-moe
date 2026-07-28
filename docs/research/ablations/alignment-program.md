# Alignment-program findings (Track A ladder + Track B anticipatory)

**Status: program COMPLETE (2026-07-03). Findings doc only — none of this is in the paper yet.**
Plan of record: `docs/research/decision-time-alignment-plan.md` (including its wrong-directions header,
which still binds). Raw rows: `results/ablations/alignment_cells.csv` and
`karen_promotion_s2_1e17.csv` (pod branches); Track-B table in `comms/h100/0027-h100-result.md`.

## TL;DR

The program asked: can we close the misalignment between what the unconstrained router wants and
what the resident set holds (A3 ≈ 30%) using only (1) the loss paradigm — the aux-free ladder up to
Karen's demand-momentum formulation — and (2) future information — the anticipatory loss?

**Answer: no mechanism improved LM quality. Every mechanism that raised A3 paid for it in expert
diversity — all trained cells lie on a single alignment↔diversity frontier.** The two durable
positives: aux-free routing is a safe (quality-neutral) substitution for FLAME's tuned aux loss, and
aux-free + momentum buys +5–7 pt A3 at quality parity. The clean negative: the anticipatory BCE
Goodharts its own target (it makes the future predictable by collapsing the pool, rather than
learning to anticipate). Plain temporal's ~30% A3 currently looks like the equilibrium price of
streamed diversity, not a removable defect.

## Metric definitions (read once)

- **BPB** — bits per byte: validation cross-entropy (nats) / BPB_DIVISOR (2.7600 at s0, 2.7568 at
  s2). Lower is better. Seed noise on plain temporal ≈ 0.0005 at s0; Karen's seed spread is ~10×
  larger (0.0039).
- **A3 overlap** — headline alignment metric: fraction of the unconstrained router's top-k choices
  already in the previous resident set (pre-swap coverage). Higher = router preferences and cache
  agree. Plain temporal reference ≈ 30%.
- **union** — mean number of distinct experts (of E=192) that ever enter the resident set during the
  probe window, per layer. Measures how much of the pool the stream actually visits.
- **eff-experts** — effective expert count: exp(entropy) of the marginal expert-selection
  distribution over the probe window ("perplexity of expert usage"). 192 = perfectly uniform
  traffic; 50 = traffic concentrated as if only ~50 equally-used experts exist. This is the
  diversity guardrail.
- **pinned / max-residency** — experts resident >80% of tokens (count per layer), and the single
  highest residency fraction. Any material pinning = degeneracy (stable capacity belongs in the
  shared expert, not in de-facto permanent residents) — disqualifying regardless of BPB.
- **Shapes** — Track A ran only at per-budget compute-optimal shapes: s0@1e16 and s2@1e17, fine
  grain (k=18 of E=192), one principled config per rung, no sweeps.
- **Epistemology** — free/eval-only replays may *promote* ideas; verdicts (kills) require
  trained-from-scratch cells.

## Track A — the aux-free ladder (s0@1e16, seed 1234 unless noted)

| cell | config | BPB | ΔBPB vs ref | A3 | union | eff-exp | verdict |
|---|---|---|---|---|---|---|---|
| L0 | plain temporal (tuned aux, softmax) | 1.4753 | — (ref) | 29.6% | 158.1 | 183.9 | alignment reference |
| A0-ctl | full MoE, sigmoid scoring + aux | 1.4475 | −0.0110 vs softmax MoE 1.4585 | 20.2% | 192.0 | 190.1 | sigmoid alone slightly *helps* baseline |
| A1 | full MoE, aux-free (DeepSeek bias) | 1.4495 | +0.0020 vs A0-ctl | 20.7% | 191.6 | 176.3 | **safe substitution**; H1 (aux-free alone raises A3) not supported |
| A2 | temporal, aux-free | 1.4765 | +0.0012 vs L0 | 34.6% | 147.4 | 126.2 | +5.0 pt A3 at parity; diversity down but no collapse |
| A3 | temporal, aux-free + momentum (Karen β=1.0, γ_M=0.125) | 1.4722 / 1.4761 (seeds 1234/2) | parity (spread > effect) | 36.1% | 153.4 | 129.3 | +6.5 pt A3 at parity; *partially recovers* diversity vs A2 |

Note on A3-Karen quality: seed 1234 looked like a genuine win (−0.0031); seed 2 flipped it
(+0.0008). Official verdict: **quality parity**, with the caveat that Karen's seed variance is ~10×
plain temporal's.

### Karen promotion cell (s2@1e17, the confirm-at-next-budget test)

| config | test BPB | A3 | union | eff-exp | pinned/layer | max-residency |
|---|---|---|---|---|---|---|
| plain temporal (min_logit) | 1.2873 | 32.9% | 160.8 | 187.8 | 0.0 | 15.2% |
| Karen (β=1.0, γ_M=0.125) | 1.2881 | 39.9% | 152.5 | 127.9 | 0.2 | **85.3%** |

**Promotion NOT confirmed.** BPB tie and +7 pt A3 replicate the s0 signal, but eff-experts fell 32%
(187.8 → 127.9) and one expert per ~5 layers became de-facto permanent (max-residency 85.3%). The
mild s0 diversity cost turned into pinning onset at scale — crossing the no-permanence principle.
Mechanism note: the DeepSeek bias b_e balances *selection* load and cannot see *residency*, so
momentum-driven permanence escapes the balance controller.

## Track B — anticipatory BCE (s0@1e16, γ = lookahead discount, λ = loss weight)

| cell | λ | γ | BPB | A3 | union | eff-exp | pinned/layer |
|---|---|---|---|---|---|---|---|
| L0 | 0 | — | 1.4753 | 29.6% | 158.1 | 183.9 | 0 |
| B1a | 0.02 | 0.5 | 1.4952 | 55.5% | 49.6 | 49.6 | 0 |
| B1b | 0.10 | 0.5 | 1.5068 | 80.5% | 24.3 | 24.3 | 7.3 (max-res 84.6%) |
| B2 | 0.02 | 0.9 | 1.5013 | 57.7% | 50.6 | 50.6 | 0 |

**Clean trained negative — the Goodhart-concentration signature.** BPB hurt is monotone in λ (every
cell ≥ 40× seed noise from parity); the working pool collapses 158 → 50 → 24 as λ grows; hard
pinning onset at λ=0.1. A3 rises to 80% *because* the pool collapsed: future demand is trivially
predictable if the router keeps the same small set resident, so the loss optimizes predictability
instead of anticipation. The anticipatory_loss term itself rises then plateaus during training — it
fights the LM objective rather than being minimized. Horizon is second-order: γ=0.9 vs 0.5 at fixed
λ leaves the pool size unchanged and is slightly *worse* on BPB (short-horizon-wins prediction
held). λ, not γ, sets the collapse.

## Synthesis — the alignment↔diversity frontier

Plotting every trained cell in (A3, eff-experts) space, all mechanisms move along one frontier:

| point | A3 | eff-experts |
|---|---|---|
| full MoE (no residency constraint) | ~20% | 176–190 |
| plain temporal | 29.6% | 183.9 |
| temporal aux-free (A2) | 34.6% | 126.2 |
| Karen s0 (A3) | 36.1% | 129.3 |
| Karen s2 (promotion) | 39.9% | 127.9 |
| anticipatory λ=0.02 | 55.5% | 49.6 |
| anticipatory λ=0.10 | 80.5% | 24.3 |

No mechanism moved *off* the frontier (more A3 at held diversity); each just picked a point on it,
and LM quality tracks diversity, not A3. Interpretation: under a residency constraint the router's
~30% self-consistency is what streamed diversity costs at equilibrium; buying more alignment means
narrowing the working pool, which is either neutral (mild, Karen) or harmful (aggressive,
anticipatory). A supporting positive remains: the residency constraint *itself* teaches temporal
locality (temporal ≥ MoE at 1e18, both granularities) without any of these mechanisms.

Frontier figure (free): both pods preserved `router_log.pt` for every cell (plain + Karen at s2 on
H100; s0 ladder on A6000; the three Track-B logs on H100) — one CPU replay can produce the
A3-vs-eff-experts scatter whenever we want it.

## Settled claims

1. **Aux-free (sigmoid + DeepSeek bias controller) is a safe substitution** for FLAME's tuned aux
   loss — neutral on baseline MoE (sigmoid scoring alone is −0.011 vs softmax) and on temporal.
2. **Aux-free alone does not raise baseline-MoE A3** (20.2 → 20.7%): H1 not supported.
3. **Temporal × aux-free (+ momentum) buys +5–7 pt A3 at quality parity** — real, seed-checked, and
   replicated at both shapes. This is the honest headline for any future alignment section.
4. **Karen full formulation: NOT promoted** — BPB tie at s2 but −32% eff-experts and pinning onset
   (max-residency 85.3%) → crosses no-permanence.
5. **Anticipatory BCE on router logits: trained negative** — Goodharts via pool collapse, monotone
   in λ; γ irrelevant.
6. **All mechanisms lie on one alignment↔diversity frontier**; quality follows diversity, not A3.

## Overnight re-tune + attribution program (night of 2026-07-03; all cells seed 1234 unless noted)

Five trained cells + one seed replicate + a free β-selection replay, run per Noah's green-lit plan
(one substitution, flagged at dispatch: "centered-M" is a per-token-uniform shift — a provable
no-op for the residency trigger, which only compares scores within a token — so the structural
slot ran Karen's full double-momentum R̃ = R + βM − αQ instead; α·Q is a slow demand-EMA penalty,
α=β cancels chronic-demand bonuses).

**Pre-registered gates** (vs L0 1.4753 / A3 29.6% / eff 183.9): test BPB ≤ 1.4783, A3 ≥ 34%,
eff-experts ≥ 170, max-residency < 40%, pinned = 0. Passing all five = off the frontier.

| cell | config | test BPB | A3 | eff-exp | max-res | gates |
|---|---|---|---|---|---|---|
| A2-s2 | s2@1e17 bare aux-free (attribution) | 1.3005 (+0.0132 vs plain) | 38.0% | 128.2 | 81.5% | — |
| A3q | s0 double momentum (β=1, γ=0.125, α=1, γ_q=1/64) | 1.4715 | 35.8% | 126.3 | 91.2% | FAIL (eff, max-res) |
| A3s | s0 gentle momentum (β=0.5, γ=0.5) | 1.4703 | 35.9% | 126.5 | 69.1% | FAIL (eff, max-res) |
| A3p | s0 momentum on PLAIN substrate (β*=2 by free replay) | 1.4717 | 28.4% | 181.8 | 16.3% | FAIL (A3) |
| A3p-seed2 | same, SEED=2 | 1.4806 | 27.9% | 183.8 | 18.4% | — |

**Findings (all trained, cross-confirmed between pods):**

1. **The substrate attribution is complete and symmetric.** Bare aux-free at s2 — no momentum —
   already collapses eff-experts to 128.2 and pins at 81.5% max-residency (≈ Karen's 127.9/85.3%),
   *and* costs real quality (+0.0132 BPB vs plain, ~25× noise). Meanwhile momentum on the plain
   substrate is A3-inert (28% both seeds): the momentum bonus lives in softmax-mass units (~1/192)
   and vanishes against raw-logit spread; its +7pt A3 existed only on the aux-free [0,1] score
   scale. So the alignment gain and the diversity collapse are BOTH downstream of the aux-free
   substrate — momentum was never an independent lever.
2. **No momentum shape reaches the diversity gate.** Across single/double/gentle momentum,
   eff-experts is structurally pinned at ~126 (gate: ≥170); softening only tunes max-residency
   (98% → 91% → 69%) — monotone but nowhere near <40%.
3. **The momentum BPB "wins" were seed noise — retracted.** A3p's −0.0036 flipped to +0.0053 on
   seed 2 (pair spread 0.0089 ≈ 18× nominal noise; 2-seed mean ties L0). The momentum family's
   real seed spread is ~0.006–0.009 test BPB, which also covers karen2t (1.4715) and karen_soft
   (1.4703) — treat every momentum BPB delta as parity pending seeds (single-mom Karen's own pair
   1.4722/1.4761 already straddled L0).
4. **Net verdict: momentum on a healthy substrate is a robust no-op** (A3-inert, diversity-safe,
   BPB-neutral — both seeds), **and the aux-free branch is disqualified at scale** by the s2 BPB
   penalty + no-permanence violation. The re-tune program closes with zero mechanisms off the
   alignment↔diversity frontier; the frontier synthesis above stands, now with 8 trained cells
   behind it (frontier CSV: `alignment_frontier.csv`, h100 branch, 7 replayed cells).

**Open question raised by the A6000** (Noah to arbitrate): is the eff-experts ≥ 170 gate too
strict for a mechanism whose premise is concentrating compute on fewer experts? (Likely moot for
aux-free given its s2 BPB penalty, but relevant to any future alignment lever.)

## Local-consistency/global-diversity program (afternoon 2026-07-03; plan: local-global-plan.md)

Reframe: the goal state is *bursty demand with a uniform stationary marginal* (autocorrelation
and marginal are independent properties, so the frontier is not a law). Three mechanisms tested;
full verdicts in `docs/research/local-global-plan.md`. Summary: LG1 log-ratio momentum achieved
the zero-time-average design (diversity-safe, union even rises) but is A3-inert on the healthy
substrate — momentum's alignment power was always the aux-free substrate's artifact. LG2 bursty
window loss is the program's worst Goodhart collapse (A3 79% via eff 33/192, max-res 100%, BPB
+0.045, loss never minimized). LG3 block-local routing declined by free replay (freezing on stale
demand costs 16–35pt retained mass at every block size). Meta-pattern across both programs:
loss-based demand shaping always Goodharts into collapse; selection-based shaping is
diversity-safe but alignment-inert on healthy substrates. Nothing moves off the frontier — the
frontier, its equilibrium interpretation (~30% A3 as the price of streamed diversity), and the
two clean Goodhart negatives are the paper-ready findings.

## Open directions (not started; Noah to arbitrate)

- **Karen re-tune** (Noah approved in principle, discussion first): softer β and/or shorter momentum
  memory, plus structural fixes (zero-sum/centered M, cap on M) aimed at the pinning failure mode
  specifically, since that — not BPB — is what killed promotion.
- **Decoupled stop-grad nomination head**: separate small head per MoE layer predicts discounted
  future demand from the hidden state; BCE gradients stop at the head (router/trunk untouched);
  predictions enter as a selection-only additive term. Goodhart-immune by construction (training
  cannot make the future predictable; it can only observe it). Motivated by the E5 replay: learned
  anticipation headroom is +20–30 pt coverage vs +6–10 pt for any eviction policy. Cost: Megatron
  surgery.
- **Frontier figure** from preserved router logs (free, CPU-only).
