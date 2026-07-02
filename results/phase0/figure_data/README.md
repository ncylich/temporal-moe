# Figure data (concise CSVs behind the probe figures)

Small, tidy CSVs holding the **exact aggregated series** plotted in every router-probe figure — the
committed, human-readable stand-in for the large raw `router_log.pt` tensors (~237 MB total, kept on
the pod under `results/phase0/runs/<run>/router_log.pt`, gitignored). Anyone can inspect the numbers
or re-plot **without** the raw logs. Regenerate all of these with:

```
.venv/bin/python scripts/phase0/probe_replay.py
```

`model` columns are self-contained (active params + coarse/fine-grained expert count). All rates are
fractions 0–1 unless a column name says `_pct`. Higher hit-rate / coverage / retained-mass = better;
lower swap-rate = better.

| CSV | figure(s) | what it holds |
|---|---|---|
| `e1_swap_rate_by_layer.csv` | `swap_rate_vs_bandwidth_budget` | per-layer mean swaps/token + p95 burst length |
| `e1_victim_cache_hitrate.csv` | `victim_cache_hitrate_vs_size` | re-load hit-rate vs victim-cache size (experts) |
| `e2_streamed_diversity.csv` | `streamed_expert_diversity_per_sequence`, `expert_residency_distribution` | union size / effective-experts / max residency per model |
| `e3_mass_vs_set_consistency.csv` | `gate_mass_vs_set_self_consistency` | set- vs gate-mass-weighted self-consistency, temporal & full MoE |
| `e4_swap_vs_retained_mass.csv` | `swap_rate_vs_retained_mass_tradeoff` | trigger-margin τ sweep: swap-rate vs retained mass |
| `e5_eviction_policy_headroom.csv` | `eviction_policy_headroom_belady_bound` | set/mass coverage per eviction policy (min_logit, LRU, τ, discounted-oracle, Belady, prefetch) |
| `e6_per_layer_ranking.csv` | `per_layer_routing_locality_ranking` | per-layer hit-rate / swap-rate / lifetime |
| `e7_demand_smoothing.csv` | `demand_smoothing_swap_vs_coverage` | EMA-β sweep: swap-rate + set/mass coverage |
| `e8_document_boundary.csv` | `document_boundary_churn` | hit-rate after-EOD vs within-document, per window |
| `learned_locality_vs_scale.csv` | `learned_temporal_locality_vs_model_size` | temporal vs full-MoE vs random same-set overlap by model size |
| `rolling_coverage_lifetime_vs_K.csv` | `routing_coverage_vs_resident_cache_size`, `expert_lifetime_vs_resident_cache_size` | rolling hit-rate + expert lifetime vs resident-cache size K/k |

Not tabulated here: the three `expert_selection_per_token_*` rasters are per-token pictures whose data
is the resident mask itself (in the raw logs), not an aggregate series.
