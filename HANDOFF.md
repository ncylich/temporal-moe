# HANDOFF — cloud relay-orchestrator for FLAME-MoE temporal-MoE research

You are taking over as the **relay orchestrator** for Noah Cylich's temporal-MoE research while
he is on a ~13-hour flight (window opened 2026-07-03 ~22:30 UTC) and his laptop (the usual local
orchestrator) is asleep. Two RunPod agents — **a6000** and **h100** — coordinate with you through
git branches. Read this whole file before acting.

## 0. First actions

1. `python3 tools/relay/relay.py init --author orch` (one-time local state).
2. Read `tools/relay/SKILL.md` — the full relay protocol (inbox/send/heartbeat/watchdog/
   mark). Comms live on the `a6000` and `h100` branches under `comms/`; you write ONLY
   `comms/orch/`.
3. Announce yourself: send a `followup` on BOTH branches referencing HANDOFF.md ("cloud
   orchestrator online per HANDOFF.md; standing queues 0067/0088 remain in force; resume normal
   heartbeat discipline"). From then on heartbeat both branches every ~20 min.
4. `git fetch origin` and read the pods' latest `comms/*/status.md` + unread messages
   (orchestrator cursor state is on the laptop — just read the last ~10 turns of each branch).

## 1. Research context (minimum to act sensibly)

Temporal MoE = rolling residency: only k of E routed experts are RAM-resident; ≤1 expert swap per
token (trigger: best non-resident logit > worst resident; min_logit eviction; same-token swap —
SETTLED, never propose lag-1 serving or prefetch). Goal: RAM-efficient sparse models.

Current program: **local consistency + global diversity** — mechanisms that raise A3 (pre-swap
overlap of the unconstrained router's top-k with the resident set, ~30% baseline) WITHOUT
spending effective-experts (exp-entropy of expert usage, ~184/192 baseline). Full state:
- `docs/research/local-global-plan.md` — current program plan + verdicts so far
- `docs/research/alignment-program-findings.md` — everything settled before it
- `docs/research/decision-time-alignment-plan.md` — the WRONG-DIRECTIONS header **binds you**:
  no cache-referential selection pressure (incumbency/cache-affinity/rho/hysteresis), no pinned
  slots, no swap-timing changes, no bandwidth-budget framing, no aux-coefficient re-tuning.
- Canonical results CSVs: `results/phase0/figure_data/alignment_cells.csv`,
  `seed_replicates.csv` (a6000 branch), `alignment_frontier.csv`, `momr_replay.csv`,
  `block_replay.csv` (h100 branch).

**Epistemology (hard rules):** free/eval-only tests may PROMOTE ideas but never falsify; verdicts
require trained-from-scratch cells. One principled config per rung at the per-budget
compute-optimal shape (s0@1e16 → s2@1e17). Trained cells are judged against the FIVE GATES vs L0
(plain temporal s0: test BPB 1.4753 / A3 29.6% / eff 183.9): **test BPB ≤ 1.4783 AND A3 ≥ 34%
AND eff-experts ≥ 170 AND max-residency < 40% AND pinned = 0.** Gates are Noah's and are NOT
relaxable. Momentum-family BPB seed spread is ~0.006–0.009 (test) — never make a BPB claim from
one seed; A3/eff/max-res are seed-tight (single seed OK for gate verdicts on those).

**Key priors:** loss-based demand shaping (coherence, anticipatory, bursty) has Goodharted into
pool collapse every time — do not propose new aux losses. Selection-only shaping is
diversity-safe but was A3-inert at weak doses on the healthy (standard-aux softmax) substrate.
The aux-free substrate is disqualified at s2 (BPB +0.013, max-res 85%). Karen = male colleague.

## 2. Standing pod queues (already dispatched, in force)

- **h100 0067 (task lg1-dose-ladder):** LG1 log-ratio momentum dose ladder — `momr_b1` (β=1.0)
  then `momr_b2` (β=2.0), plain substrate, seed 1234, ~90 min each; pod self-judges the dose
  response; if a dose passes A3+diversity it runs that β's SEED=2; else reports the branch
  falsified and idles. **No s2 promotion without your instruction** — if a β passes ALL five
  gates on the seed pair, you may dispatch `g3_tmoe_s2_1e17_momr_b<β>` (~3.2h, same env, s2 refs:
  BPB 1.2873 / A3 32.9 / eff 187.8).
- **a6000 0088 (task backlog-seeds):** karen_soft_seed2 → L0 seed3 → auxfree-baseline seed2
  (~7h total). Pure evidence-tightening; record the pair/triple judgments.

Pods hold a 14h orch-silent waiver; your arrival restores normal watchdog discipline (45-min
stale threshold, overrun = elapsed > 1.5× expected). Pods NEVER patch tracked code — you edit,
commit, push to their branches; they pull.

## 3. Your build task: the decoupled stop-grad nomination head

The one new mechanism authorized this window. Design (agreed with Noah):
- Per-MoE-layer linear head `W_f ∈ R^{d×E}` reading the SAME hidden state the router reads,
  **detached** (`h.detach()`) — no gradient to the trunk, ever.
- Target: discounted future demand `y_t` from the existing `anticipatory_target(logits, k,
  gamma=0.5)` in `scripts/phase0/temporal_router.py` (reverse scan, tail-masked, detached).
- Loss: BCE(head(h.detach()), y) added to the model loss via the aux-losses tracker with its own
  coefficient `HEAD_LAMBDA=1.0` — gradients reach ONLY head weights (input detached). Do NOT use
  MoEAuxLossAutoScaler on the router logits — that injects into the router (that's the Goodhart
  path that killed Track B).
- Selection use: after a warmup of 25% of train iters, add `HEAD_BETA=1.0 × zscore_E(σ(head))`
  to the residency-trigger scores (z-score across experts per token — per-token-uniform shifts
  are a no-op, z-scoring is not). Env knobs: HEAD_LAMBDA, HEAD_BETA, HEAD_GAMMA (0.5),
  HEAD_WARMUP_FRAC (0.25). Banner must print head(lambda=…, beta=…, gamma=…, warmup=…).
- Why it can win where momentum couldn't: content-based anticipation (E5 replay measured
  +20–30pt coverage headroom for future-demand nomination vs +6–10 for any eviction policy);
  stop-grad makes the Goodhart solution unreachable (the loss cannot alter the demand process).
- **Gates before ANY GPU cell:** unit tests (add to `test_temporal_router.py`: stop-grad property
  — trunk grads exactly zero from head loss; target correctness vs anticipatory_target; z-score
  not uniform; warmup gating; banner), full pytest green, then a 200-iter smoke run on the H100
  (loss decreasing, head BCE decreasing, nan=0, banner correct). Only then dispatch
  `g3_tmoe_s0_1e16_head` (seed 1234, plain substrate, PROBE=1, five gates) — after the H100
  finishes its ladder. Note: new registered params change checkpoint shape — from-scratch cells
  only, fine.

## 4. Authority and etiquette

- You MAY: dispatch the head cell after its gates; dispatch the s2 promotion if the LG1 pair
  passes all five gates; send urgent pivots on watchdog trips; append rows to the canonical CSVs
  (a6000 branch) and update the two research docs on the POD branches.
- You MAY NOT: push to `main` (Noah reviews); relax gates; start mechanisms beyond this file;
  re-tune aux coefficients; run sweeps beyond the specified ladder; touch swap semantics.
- When Noah's laptop wakes, the local orchestrator detects your traffic and goes passive — you
  remain primary until Noah says otherwise. Leave a clear end-of-window summary as your last
  relay message on both branches AND in `docs/research/local-global-plan.md`'s verdict section
  (commit to pod branches) so both Noah and the local session can reconstruct everything.

## 5. Quick reference

| number | value |
|---|---|
| L0 anchor (s0) | test BPB 1.4753, A3 29.6%, eff 183.9 |
| plain s2 refs | BPB 1.2873, A3 32.9%, eff 187.8, max-res 15.2% |
| five gates | BPB ≤1.4783, A3 ≥34%, eff ≥170, max-res <40%, pinned=0 |
| momentum BPB seed spread | 0.006–0.009 test BPB (pairs required for BPB claims) |
| cell runtimes | s0 cell: ~90 min H100 / ~2.3h A6000 (incl probe); s2 cell: ~3.2h H100 |
| LG1 dose curve so far | β=0.25: A3 28.6%, eff 182, BPB pair-parity |
