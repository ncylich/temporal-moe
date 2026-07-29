# Mechinterp Battery Re-run Plan

**Housekeeping.** Every mechanistic analysis in this repo was written against a hardcoded list of
3–8 models and, where it is per-layer at all, a hardcoded range of 3–5 layers. The numbers reported
in [`delexicalization.md`](delexicalization.md) are medians pooled over whatever subset each script
happened to cover. This plan re-runs the battery across every trained model, at every layer, with a
uniform output schema.

This is deliberately *not* hypothesis-driven work — it is making the existing measurements complete
and reproducible. The hypothesis-driven analysis that motivated the audit lives in
[`LAYER_LEXICALITY.md`](LAYER_LEXICALITY.md) and depends on Section 5 Steps 1–3 below.

## 1. Current coverage

| metric | file | per-layer? | layers covered | models |
|---|---|---|---|---|
| locus probes (A_tok, A_ctx) | `mechinterp_locus{,_1e19}.csv` | yes | 2–6 (of up to 9) | 8 |
| output logit lens (effective vocab) | `mechinterp_lens{,_1e19}.csv` | yes | **2–4 only** | 6 |
| logit lens (older) | `mechinterp_logitlens.csv` | yes | 1–3 | 2 |
| cache hit rate | `e6_per_layer_ranking.csv` | yes | all | 3 |
| swap rate / burst length | `e1_swap_rate_by_layer.csv` | yes | all | 5 |
| selectivity PR, generalist %, router entropy, weight geometry | `mechinterp_structural{,_1e19}.csv` | **no — pooled** | n/a | 11 |
| demand forecastability | `mechinterp_demand_1e19.csv` | **no — pooled over 2–6** | n/a | 3 |
| free-rider / tokens-per-expert | `mechinterp_freerider.csv` | no | n/a | — |

Where the limits are set:

- `LAYERS = [2,3,4,5,6]` at [`delex_locus.py:18`](../../../analysis/probes/delex_locus.py), with
  the note "paper convention". Layers outside the list are dropped by
  `if L not in d["layers"]: continue` — **silently**, with no warning and no record in the output.
- `HEADLINERS` (3 runs) and `ALL_TEMPORAL` (5 runs) at
  [`probe_replay.py:55-57`](../../../analysis/probes/probe_replay.py). `e6()` uses the former,
  `e1()` the latter; nothing uses the full set of captured runs.
- `delex_structural.py` and `delex_demand.py` pool experts across layers before writing, discarding
  the layer key they already have.
- `delex_lens.py` runs layers 2–4.

## 2. What is available to run against

From [`results/MANIFEST.csv`](../../../results/MANIFEST.csv):

- **69 runs with preserved checkpoints** — 1e16 through 1e19, both regimes, granularities g1/g3/g5,
  several seeds and router-recipe variants.
- **22 runs with preserved `router_log.pt`** — sufficient for every replay/cache metric (e1–e8)
  with no forward pass.
- **3 runs with a preserved `delex_capture.pt`**, all at 1e19. Every other locus/lens/structural
  number needs a fresh capture pass from its checkpoint
  ([`delex_probe.py`](../../../analysis/probes/delex_probe.py), one forward pass over the fixed
  64x2048 eval batch).

**The capture pass is the only real cost in this plan.** Everything downstream of a capture is
CPU-bound analysis measured in minutes.

## 3. The analyses to re-run

Every mechanistic analysis in the repo, with what changes about it. "Per-layer" means the output
gains a `layer` column and covers every MoE layer; "all models" means the selection set of
Section 5 Step 3 rather than the current hardcoded list.

### A. Locus / de-lexicalization family — driver [`delex_probe.py`](../../../analysis/probes/delex_probe.py) capture

| # | analysis | script -> output | change |
|---|---|---|---|
| A1 | Locus probes: token AUC vs excluded-context AUC per expert | `delex_locus.py` -> `mechinterp_locus*.csv` | layers 2–9 instead of 2–6; all models; warn instead of silently skipping |
| A2 | Null-control floors (iid permutation + circular shift) | `delex_locus.py` -> `mechinterp_floors.csv` | per layer, not pooled; gate every new model |
| A3 | Context-window sweep (w = k/2, k, 32) | `delex_locus.py` variants | run all three for **every** cell — currently the softmax-aux fine baseline has only w=32 |
| A4 | Output logit lens: data-weighted effective vocabulary per expert | `delex_lens.py` -> `mechinterp_lens*.csv` | layers 2–9 instead of 2–4; all models |
| A5 | Static (no-data) lens reference | `delex_lens.py` variant `static` | same extension; it is the no-signal control for A4 |
| A6 | Selectivity PR + generalist fraction | `delex_structural.py` -> `mechinterp_structural*.csv` | **add `layer` grouping — currently pooled across all MoE layers** |
| A7 | Router entropy (per-token routing flatness) | same | per layer |
| A8 | Weight geometry: distance-to-centroid, pairwise cosine | same | per layer; expected flat, confirm once |
| A9 | Gate-mass correlation: effective rank, strong-corr pairs | same | per layer |
| A10 | Demand forecastability (causal, history-only probe) | `delex_demand.py` -> `mechinterp_demand*.csv` | **fit per layer — currently one probe pooled over layers 2–6** |
| A11 | Free-rider: distinct experts per sequence, tokens per expert | -> `mechinterp_freerider.csv` | all models; per-layer breakout is not meaningful (architecturally fixed) — record once and note why |

### B. Replay / residency-dynamics family — [`probe_replay.py`](../../../analysis/probes/probe_replay.py), captures = `router_log.pt`

| # | analysis | output | change |
|---|---|---|---|
| B1 | Swap rate and p95 burst length by layer | `e1_swap_rate_by_layer.csv` | all 22 captured runs; **stop reporting swap rate as a depth signal** (saturated), keep burst length |
| B2 | Victim-cache hit rate vs cache size | `e1_victim_cache_hitrate.csv` | all runs |
| B3 | Streamed diversity: expert union, effective experts, pinned set | `e2_streamed_diversity.csv` | all runs |
| B4 | Mass-weighted vs set-based routing consistency | `e3_mass_vs_set_consistency.csv` | all runs |
| B5 | Swap rate vs retained mass (hysteresis tau sweep) | `e4_swap_vs_retained_mass.csv` | all runs |
| B6 | Eviction-policy headroom (min_logit / LRU / Belady bound) | `e5_eviction_policy_headroom.csv` | all runs; **per layer** — policy headroom may itself be depth-dependent |
| B7 | Per-layer ranking: hit rate, swap rate, lifetime | `e6_per_layer_ranking.csv` | **3 runs -> all 22**; this is the metric with the depth signal |
| B8 | EMA demand smoothing (beta sweep) | `e7_demand_smoothing.csv` | all runs; per layer |
| B9 | Document-boundary churn | `e8_document_boundary.csv` | all runs |
| B10 | **Counterfactual baseline replay** — impose rolling residency on an unconstrained run's router log | new | does not exist; every replay metric currently lacks a baseline arm |

### X. Cross-regime constraint swap

| # | analysis | change |
|---|---|---|
| X1 | Global swap: evaluate each trained model under the other regime (§5 of [`delexicalization.md`](delexicalization.md)) | **has no committed driver** — it was run ad hoc through `run.sh` with `EVAL_ONLY=1` and the temporal env flags. Commit one, since the per-layer version is on the layer-lexicality critical path |
| X2 | Per-layer swap sweep | new; the C3 test of [`LAYER_LEXICALITY.md`](LAYER_LEXICALITY.md) |
| X3 | Residency dose curve (uniform R) | re-run the R endpoints at 1e17 and 1e18, not only 1e16 |

### Z. Explicitly out of scope

`expert_coactivation.py` and `router_saturation.py` under
[`scripts/empirical_analysis/`](../../../scripts/empirical_analysis) are upstream FLAME-MoE
analyses that do not run in this repo. `stability_*.py`, `fakequant_eval.py` and `run_lmeval.py`
are stability, quantization and downstream-eval probes, not mechanistic interpretability.
`expert_load.py` and `activation_probe.py` produce aggregate stability statistics; pull them in
only if a specific question needs them.

## 4. Schema conventions to adopt

Applied uniformly as each script is touched, so this audit does not have to be repeated:

1. **Every per-expert CSV carries a `layer` column.** Pooling is a reporting decision, made in the
   plot script, never at write time.
2. **No silent layer skipping.** If a script cannot cover a layer present in the capture, it warns
   and records the omission in the output.
3. **Every CSV carries the run name and the budget**, not just an internal label. Several current
   files use labels (`s0_TEMPORAL`, `s2_FULL`) that require reading the source to decode.
4. **Window and variant semantics are documented in the file header.** The
   `base`/`kwin`/`kfull` variant encoding in `mechinterp_locus.csv` currently decodes only by
   cross-referencing prose in another document (`kwin` = w=k/2, `kfull` = w=k, `base` = w=32).
5. **Model lists come from a single shared registry**, not per-function constants.

## 5. Steps

Ordered so that anything usable without a GPU lands first.

### Step 1 — re-aggregate what is already on disk (no GPU, minutes)

1. Add per-layer grouping to the output-lens reporting; `mechinterp_lens{,_1e19}.csv` already carry
   `layer` and `expert` and were only ever reported pooled.
2. Replace swap rate with `p95_burst_len` in per-layer reporting. Swap rate is 0.994–1.000
   everywhere and is structural, not a finding: at R = k a swap fires iff at least one demanded
   expert is missing, so it is "fraction of tokens with >= 1 miss" and saturates. Record the reason
   in the file header so it is not re-adopted.
3. Correct §3 of [`delexicalization.md`](delexicalization.md): state the probed layer range, and
   fix the `s0_SOFTMAX_BASELINE` row label from w=18 to w=32 (only the w=32 variant was ever run
   for that cell).

### Step 2 — full coverage on existing captures and router logs (no GPU)

4. `delex_locus.py`: `LAYERS = range(2, 10)`, warn instead of skipping. Re-run on the 3 preserved
   captures.
5. `probe_replay.py`: single run registry, `e1()`–`e8()` over all 22 runs with router logs. This
   adds the 1e19 models, which no cache metric currently covers.
6. Counterfactual replay for unconstrained runs: impose rolling residency on a baseline's router
   log to produce the baseline arm that every replay metric is missing.
7. `delex_structural.py`: add `layer` as a grouping key. Selectivity PR, generalist fraction and
   router entropy become depth curves at no compute cost.
8. `delex_demand.py`: fit per layer rather than pooling layers 2–6 into one probe.

### Step 3 — capture sweep over the fleet (GPU, one forward pass per model)

9. Capture every model in the selection set below and re-run locus, lens and structural on each.
10. Selection rule: for each (budget, regime, granularity) cell take the seed-1234 run the isoFLOP
    analysis treats as the headline, plus one alternate seed where one exists. Include the dense
    control at each budget as a floor. This is where the current figures' holes get filled — most
    conspicuously **1e18, where no mechanistic measurement of any kind exists** despite it being
    the budget at which the temporal model wins.
11. Extend `delex_lens.py` past layer 4 in the same pass.

### Step 4 — regenerate and reconcile

12. Regenerate every figure from the completed CSVs.
13. Reconcile the numbers quoted in [`delexicalization.md`](delexicalization.md) against the
    re-run outputs, and record any that move.

## 6. Acceptance criteria

- Every metric in the Section 1 table has a `layer` column and covers every MoE layer of every
  model in the selection set.
- No script contains a hardcoded model list or layer range.
- The null-control gate already enforced by
  [`delex_locus_driver.py`](../../../analysis/probes/delex_locus_driver.py) — median null AUC of
  0.500 +/- 0.002 under both iid permutation and circular shift — passes for every newly captured
  model, and is extended to the lens and structural re-runs where an analogous null exists.
- Numbers quoted in prose are reconciled against the regenerated CSVs, with any movement recorded.
