# ⛔ WRONG DIRECTIONS FOR THIS WORK — READ BEFORE ANYTHING ELSE

**What this work IS (Noah's paper):** the residency *constraint itself* teaches the router
temporal locality. The only two levers under investigation for the unconstrained-vs-resident
misalignment are (1) the **loss paradigm** (standard aux → aux-free bias controller → Karen's
demand-momentum selection term) and (2) **future information** (the anticipatory loss).

**Do NOT propose, run, or re-open any of the following — each has been explicitly closed:**

| banned direction | why it is closed |
|---|---|
| rho / incumbency bias / cache-affinity bonuses / hysteresis-as-mechanism | cache-referential selection pressure; other papers' framing (CCE-style caching), not this work — and philosophically the failed coherence loss in decision space |
| pinned / permanent resident slots | permanence is a degeneracy (a routed slot acting as a shared expert); measured: no expert exceeds 39% residency |
| lag-1 serving, early expert loading, any swap-timing change | swap semantics are settled: same-token swap, fetch overlapped with resident compute |
| bandwidth-budget / feasibility framing (selecting configs against s_max, swap-rate targeting) | the memory trade-off varies per device; cap-1 swap/token is the device-independent invariant; mechanisms are judged quality-first |
| coherence BCE or any backward-looking gradient pressure on router logits | trained negative (monotone hurt at every lambda) |
| time-axis load balancing (per-token load debt) | term limits on residents = anti-locality by construction |
| re-tuning the aux-loss coefficient | already optimized by FLAME-MoE; not this paper's question |
| treating replay / eval-only negatives as verdicts | free tests can PROMOTE ideas, never falsify them — falsification requires trained-from-scratch cells |

The current plan of record is §0 below. Historical sections are retained for provenance only.

---

# Decision-time alignment & demand reduction — run plan (v3)

**Self-contained** — everything needed to execute is defined here. Deeper context:
[`temporal-moe.md`](../temporal-moe.md) (design, "rolling residency" section),
[`coherence-loss-plan.md`](./coherence-loss.md) (the measured negative result this plan
succeeds), and `docs/research/probe-replay-results.md` (**on `main`**; the E1–E8 offline
measurements that set these priorities). Scope note: we are **optimizing the technique in
isolation** — quality first, one step at a time; systems-level bandwidth accounting is
deliberately out of scope here (§6).

## 0. ALIGNMENT PROGRAM (v5, 2026-07-02 — CURRENT PLAN OF RECORD; supersedes everything below)

**Single target:** close the misalignment between the unconstrained temporal router's preferences
and the constrained resident set (A3 overlap, ~30-40%). Every cell is TRAINED FROM SCRATCH at
fine-grained s0@1e16 (free/eval-only tests may promote ideas, never falsify them) and reports the
same triple: **BPB** (vs matched baseline; seed noise 0.0005-0.0025), **A3 overlap** (headline),
**diversity guardrail** (expert union + effective experts — alignment via frozen pools is invalid).

**OUT OF SCOPE, permanently (Noah):** rho / incumbency bias / cache-affinity bonuses of any kind —
cache-referential selection pressure is other work's framing, not this paper's. The thesis here is
that the constraint itself teaches the router; the levers are the loss PARADIGM (Track A) and
FUTURE INFORMATION (Track B).

**L0 (reference):** probe the existing g3_tmoe_s0_1e16 baseline -> A3 / union / effective experts.

**Track A — the aux-free ladder (up to Karen's formulation):**
- A1: baseline MoE x aux-free (DeepSeek sign-controller bias b_e replaces the aux loss; tiny
  seq-level backstop). Q1 + hypothesis H1: the standard aux GRADIENT penalizes within-batch
  concentration and may be actively suppressing temporal streaks — does baseline locality rise?
- A2: temporal x aux-free. Q2 — the trigger reads pure LM logits; balance lives in b_e.
- A3: temporal x aux-free + Karen's demand-momentum term in the TRAINING-TIME selection score
  (score = logits + b_e + beta_M * M_{t-1}, M = causal EMA of the router's own softmax demand).
  Q3, "one level further": selection favors what the MODEL has recently wanted — demand-history,
  not cache-state (no reference to the resident set). Gradient-free, trained natively;
  ONE cell at a principled config (gamma_m = 1/8, matched to measured expert-lifetime scale;
  beta_M = 1).

**Track A efficiency rule (Noah):** for a given FLOP budget, Track A operates ONLY at that
budget's already-determined compute-optimal shape (s0@1e16; s2@1e17 on promotion). No shape
sweeps, no knob grids — one principled config per rung; a rung that shows a signal gets
PROMOTED to the next budget's optimal shape, not explored sideways.

**Track B — anticipatory loss (future information), parallel on the H100:**
- B1: temporal + standard losses + anticipatory BCE (target = discounted future demand, reverse
  scan, j>=0 term included, EOD-truncated), gamma=0.5, lambda in {0.02, 0.1}.
- B2 (horizon probe): best lambda at gamma=0.9 (~10-token horizon). Horizon theory: useful
  lookahead is bounded below by actionability (~1 swap/token means only the next few tokens are
  preparable) and above by the set-turnover time (~k tokens; beyond that the mechanism cannot use
  the information) and by predictability decay (E5 oracle: gamma=0.5 >= gamma=0.9 everywhere);
  gamma -> 1 provably degenerates to the load-balance marginal (zero temporal information).

**Capstone (only if both tracks show life):** best-of-A x best-of-B composed.

Machines: A6000 = Track A serial (after the L0 probe); H100 = A1 + Track B after the 1e18 seed
pair. Orchestrator code queue: (1) AUXFREE controller -> A1/A2; (2) anticipatory delta -> Track B;
(3) training-path momentum shaping -> A3.

Dead (settled): rho/incumbency/cache-affinity anything; budget/feasibility framing; eval-only
negatives as verdicts; aux-coefficient sweep (FLAME already optimized it); pinning; coherence BCE
(trained negative); time-axis balance (analytic); swap-semantics changes.

---

## 1. Background in five sentences

**Rolling-residency temporal MoE**: a from-scratch Mixture-of-Experts where each MoE layer keeps
only `K = k` of its `E` routed experts *resident* in fast memory (k = top-k width; the rest live
on SSD), and **at most one** expert is swapped in per token. The swap rule: swap iff the best
*non-resident* expert's router logit beats the *worst resident's*; evict the lowest-logit resident
(`min_logit`); the token is served by the post-swap resident set — the fetch overlaps the other
resident experts' compute within the same layer (`temporal/temporal_router.py`; these
semantics are settled — do not revisit). It works — ~72–82% of the full-MoE-over-dense quality
gain with 6/64 (or 18/192) experts resident — but two measured problems remain: **(P1)** the swap
rate sits at ~1.0/token (every swap is bandwidth someone must pay), and **(P2) alignment**: the
router's *unconstrained* per-token demand overlaps the resident set only ~30–42%, costing
~0.017 bits-per-byte (BPB) vs the full MoE. A training-time BCE loss pulling router logits toward
the resident set ("coherence loss") was tried and **measurably hurts** at every weight — lesson:
alignment pressure belongs at **decision time** (selection rules, triggers) or must inject
genuinely *new* (future) information; never a backward-looking gradient on the shared logits.

## 2. Notation, metrics, baselines

- `R_t ∈ ℝ^E`: router logits for token t (one per MoE layer). `p_t = softmax(R_t)`.
- `S_t`: resident set after token t's swap, `|S| = K = k`. **Demand** = top-k of `R_t` computed
  *without* any residency restriction.
- **Trigger** (shipped): swap iff `max_{e∉S} R_t[e] > min_{e∈S} R_t[e]`. **Gates** (expert mixing
  weights) always come from unbiased `R_t` restricted to the used set. Every mechanism below
  modifies *selection only*, never gates; auxiliary state carries **no gradient**.
- **Hit-rate / self-consistency** (0–1, higher better): fraction of a token's unconstrained top-k
  demand already resident on entry (before its own swap). Measured: temporal ~30–42%, full MoE
  ~20%, random `k/E` = 9.4%.
- **Mass coverage / retained mass**: same, but each demanded expert weighted by its softmax gate
  mass. Measured only +3–5 pt above set hit-rate (E3) — low consistency is real.
- **Swap rate** `s` (lower better): fraction of tokens firing a swap. Measured ≈ 1.0 everywhere.
- **Quality baselines** (BPB = CE/(ln2·bytes/token), lower better; seed noise ≈ 0.003 BPB).
  G1 = coarse 6-of-64 experts; G3 = fine-grained 18-of-192; sN = model shape at a FLOP budget:

| config | dense floor | full MoE | temporal (shipped) | gap to close |
|---|---|---|---|---|
| G3 s0@1e16 | 1.519 | 1.4585 | 1.4753 | 0.017 |
| G3 s2@1e17 | 1.341 | 1.2708 | 1.2873 | 0.017 |
| G1 s2@1e17 | 1.341 | 1.269 | 1.2821 | 0.013 |

- **Probe checkpoints** (router logs saved; replay harness `analysis/probes/probe_replay.py`):
  `tmoe_minlogit_sh1_s2_1e17` (G1 8.1M active), `g3_tmoe_s1_1e17` (G3 3.9M),
  `flame38m_temporal_minlogit` (38M @1e18).
- **Measured headroom map (E5)** at K=k / cap-1: better *eviction* is capped at **+6–10 pt**
  hit-rate (offline-optimal); choosing swaps by discounted **future** demand is worth
  **+20–30 pt** (oracle: G1 38→66%, best at short horizon γ=0.5); EMA demand-smoothing cuts the
  swap rate 43% (G1) / 31% (38M) but only 8% (G3). Invest accordingly.

## 3. Active tracks (three)

House discipline for any code change: pure reference implementation + unit tests first; the
Triton fast path bit-exact-verified or hard-fail. Env knobs follow the `TEMPORAL_*` precedent.

### T1 — Trigger shaping (one family: margin + smoothed demand)

One mechanism with three knobs, all gradient-free, all selection-only:

- **Margin τ** (`TEMPORAL_RHO`): swap only if the nominee beats the worst resident **by τ** (logit
  units). Skips low-value swaps. Measured (E4): G1 τ=2 → swap 0.40 at −5.5 pt retained mass;
  G3 τ=4 → 0.29.
- **EMA smoothing β** (`TEMPORAL_EMA_BETA`): run the trigger on `R'_t = (1−β)·R'_{t−1} + β·R_t`
  (β = weight on the current token; β=1 = shipped), so the resident set tracks *sustained* demand,
  not single-token spikes; gates still use raw `R_t`. Measured (E7): β=0.1 swap cuts — G1 −43%,
  38M −31%, **G3 only −8%** (fine-grained demand churns faster than the EMA can smooth, so on G3
  the margin must do most of the work). Replayed coverage is self-referential; only the swap-rate
  cut is clean → hence the B1 BPB pass.
- **Additive momentum (β_M, γ_m)** (`TEMPORAL_BETA_M`, `TEMPORAL_GAMMA_M`): variant that *adds* a
  history bonus instead of replacing the score: trigger score `= p_t + β_M·M_{t−1}`,
  `M_t = (1−γ_m)·M_{t−1} + γ_m·p_t` (probabilities, for scale stability). Decouples "how much
  history" (γ_m) from "how strongly it biases" (β_M). Not yet replayed — one grid, same harness.

**Conclusion this track yields:** does shaping *which* swaps happen (rather than how many) buy
quality at parity or parity at materially fewer swaps?

### T2 — Anticipatory loss (the one legitimate training-side signal)

Train the router to score experts by **near-future demand**, so the ordinary trigger swaps the
right expert in *now*. Target, computed exactly during training by one reverse scan
(`y_t = m_{t+1} + γ·y_{t+1}`, `m_t ∈ {0,1}^E` = token t's unconstrained top-k multi-hot):

```
y_t(e) = Σ_{j≥1} γ^{j−1}·1[e ∈ top-k(t+j)]      γ = 0.5 (measured best, E5; horizon ≈ 2 tokens)
```

Loss: `λ·BCE_with_logits(R_t, (1−γ)·y_t)`, injected via `MoEAuxLossAutoScaler` (~20-line delta
from the coherence-loss code already on this branch). Include the `j=0` (current-demand) term so
the loss optimum stays near the LM optimum, minimizing gate distortion. Truncate on each
sequence's last ~1/(1−γ) tokens; never let `y` cross an end-of-document boundary. Unlike the
coherence loss, the target contains **future** information the router cannot already have — new
signal, not self-imitation. Note it changes *scores only*: swap timing, swap count cap, memory
footprint, and SSD traffic are all untouched.

**Contingency (one cell, only on a specific signature):** if retention/hit-rate rises but BPB
stays flat — the gate-distortion signature — move the same target to a tiny decoupled per-layer
head `a_t = W_a·stopgrad(h_t)` (strictly-future `j≥1` target; trigger nominates/evicts by `a_t`;
logits and gates stay purely LM-trained). Report rank of the actually-entering expert at swap
events (MRR@swap) vs a predict-persistence baseline.

**Conclusion this track yields:** does future-demand signal close part of the 0.017 BPB gap —
the +20–30 pt headroom E5 measured — where backward-looking pressure failed?

### T3 — Supporting cells (independent, one-line changes, crisp answers)

- **Aux-free load balance** (`AUXFREE=1`): replace the load-balancing loss with DeepSeek-V3's
  per-expert selection bias `b_e ← b_e + u·sign(target_load − load_e)` (per micro-batch, gates
  unbiased, tiny sequence-level aux backstop). Removes the last competing gradient on `R_t`,
  making T2's effect cleanly attributable. Balance stays on the **batch axis** — never per-token
  (a per-token load debt force-evicts still-wanted residents; anti-locality).
  *Conclusion: can balance move out of gradient space at parity (within 0.003 of 1.4585 / 1.4753)?*
- **Leaky gates**: on a miss, today's masked softmax renormalizes gate mass to 1 over residents —
  substitutes get confidently over-weighted. Instead keep full-E softmax gates unrenormalized
  (resident mass < 1): absent demand ⇒ attenuated MoE update, residual carries through. One line.
  *Conclusion: does miss-renormalization hurt (BPB vs 1.4753)?*

## 4. Run plan (consolidated)

| id | tier | run | success / kill |
|---|---|---|---|
| A1 | replay (free, CPU) | **trigger-shaping frontier**: τ ∈ {0,1,2,4} × β ∈ {1.0, 0.5, 0.25, 0.1}, plus the additive-momentum grid (γ_m ∈ {1/8, 1/16, 1/32} × β_M ∈ {0.5, 1, 2}), on G1/G3/38M | deliver ≤2 configs maximizing retained mass at swap ≤ ~0.5; kill anything not above the τ-only frontier |
| B1 | eval-only GPU (minutes) | the ≤2 A1 configs patched into the trigger, `EVAL_ONLY=1` on the three probe checkpoints | ΔBPB ≥ −0.003 at the reduced swap rate; if realized swap rate deviates >20% from replay, trust B |
| C1 | training (hours, G3 s0@1e16) | **train with the B1 winner** (fixed knobs, no controller) | BPB ≤ 1.4753 − 0.006 (2× noise), or parity at that grain's replay-predicted swap cut |
| C2 | training | **anticipatory loss** (T2): λ ∈ {0.02, 0.1}, γ = 0.5 | closes ≥ ⅓ of the 0.017 gap (≥ 0.006) with no regression; monotone-hurt ⇒ kill (coherence signature); retention↑/BPB-flat ⇒ the one head-contingency cell |
| C3 | training | **aux-free parity** (T3), MoE + temporal | within 0.003 of 1.4585 / 1.4753; fail ⇒ keep aux loss, nothing else blocked |
| C4 | training | **leaky gates** (T3) | BPB ≤ 1.4753, else revert |
| C5 | training | **promote the best of C1–C4** → G3 s2@1e17; then the k=32 (E=341) geometry cell per the main roadmap | inside the dense↔MoE band, recovery > 77% |

**A1/A2 scoring rule (binding; added after the first A1 pass):** for any *shaped* trigger (EMA or
momentum), retained mass / hit-rate must be scored against the **raw** stream's demand — in
`probe_replay.py`, `replay(shaped, k, ..., eval_lg=raw)` — never against the shaped stream's own
demand, which is self-referential and inflated the first A1 pass by ~30 pt on synthetic checks.
Swap rate is a pure function of the trigger and is unaffected by the scoring stream. Unit tests:
`test_probe_replay.py` (eval_lg identity/ordering/swap-invariance, momentum causality/cold-fill).

**FRAMING CORRECTION (2026-07-02, Noah — supersedes the budget criteria above):** the memory/
bandwidth trade-off varies per device, so optimizing trigger configs to a specific `s_max` budget
is over-fitting to one hardware assumption. **Cap-1 swap/token is the design invariant** (the one
expert load per token is hidden by construction at the target geometry, and is device-independent
as an abstraction); mechanisms are evaluated **quality-first at cap-1**, never against a
feasibility budget. Consequences: (a) **tau is benched** — under cap-1 the single swap is free, so
a margin that skips it buys nothing (B1 measured it as exactly neutral); its only conceivable
future role is admission control for a *second/third* swap per token if cap-1 set-adaptation lag
ever binds at fine grain — a quality question, not a bandwidth one. (b) B1 stands as protocol
validation plus two behavior facts: eval-time trigger changes that alter WHICH experts serve
tokens hurt (smoothing, +0.08 BPB off-policy); changes that only drop marginal swaps are neutral.
(c) Remaining priority order is behaviors-first: **C3 (aux-free parity) → C2 (anticipatory loss)
→ C4 (leaky gates) → the k=32 geometry cell**; no further trigger-shaping work is scheduled.

**Ordering decision (2026-07-02):** C3 (aux-free parity) runs **before** C1/C2 when scheduling
permits — under aux-free, router logits are pure LM scores (balance moves to the separate `b_e`
term), which is the cleaner input for smoothed/momentum triggers and the only world where a
composed selection score (`R + βM − b_e`) is architecturally coherent. Fallback if parity fails:
the trigger mechanisms still run with the aux loss (they are gradient-free); note the caveat.

All training cells: locked HPs (seed 1234, lr 3e-3, gb 256, `GRAIN=3`, `EVAL_AT_END=1`) and a
closing `PROBE=1` pass → hit-rate/swap-rate delta vs the matched shipped checkpoint. Add the free
telemetry with the first cell (the scan already computes the best non-resident every token — log
hit-rate/swap-rate per interval; zero cost).

Dependencies: A1 → B1 → C1. C2, C3, C4 independent (C3 before C2 if convenient — cleaner
attribution). Winner(s) → C5.

## 5. Systems notes (facts, no runs here)

- **Victim cache**: 93–97% of swap-ins are re-loads (E1); keeping the last 16–32 evicted experts
  in RAM absorbs 46–74% (G1) / 26–45% (G3) of swap traffic at zero quality cost. Deployment
  freebie when spare RAM exists.
- **Router-early placement** (compute layer ℓ's router before attention): overlaps the fetch with
  more of the layer's compute. One training cell someday to confirm routing quality; benched.
- **EOD residency reset**: boundary tokens are <2% of the batch (E8) — low-priority nicety.

## 6. Benched (kept as ideas; not scheduled)

Adaptive swap-budget controller (τ driven to a target swap rate); bandwidth/feasibility ledgers
(deferred until the technique is optimized in isolation); early expert loading of any kind
(costs spare memory and contends for SSD bandwidth other layers need); K>k resident headroom
(final-round, opportunistic-RAM only); pressure curriculum (loose→tight over training); Gumbel
nomination noise (only if dead experts appear); per-layer knob allocation (shallow layers are
least cacheable — E6: 16–29% vs 36–48% deep); co-activation nomination prior (dominated by T2's
measured ceiling).

## 7. Do not run (measured, or twice-reasoned)

- **Eviction-policy learning** (Belady imitation): measured ceiling +6–10 pt (E5). **LRU**:
  measured 7 pt worse than shipped min_logit. *(Scope note: the `lru` policy refreshes on admission
  only, so what was measured — and trained, at s0@1e16 — is FIFO. Refresh-on-demand LRU is a
  distinct, never-trained policy that recovers 73–80% of the deficit in replay;
  [`../mechanism/lru-as-convolution.md`](../mechanism/lru-as-convolution.md) §4.1. Both derived
  kernels there are demand-referential, not cache-referential, so they sit inside the header's
  scope guard — but neither is a verdict, and neither is scheduled.)*
- **Pinned resident slots**: measured — no expert exceeds 20–39% residency (E2); permanence is a
  degeneracy (a routed slot acting as a shared expert), not a feature.
- **Coherence BCE in any form**: measured negative on this branch; no new information, and the
  gradient lands undiluted on the router logits.
- **Time-axis load balancing** (per-token load debt subtracted from scores): term limits on
  residents; anti-locality by construction.
- **Router distillation from a free-routing teacher**: fights the internalized locality (probe A3)
  that makes the method work.
