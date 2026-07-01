# G=3 fine-grained MoE — Phase-0 results (live, 9/12 runs)

Fine-grained (`GRAIN=3`: routed experts 64→192, top-k 6→18, shared expert unchanged) IsoFLOP sweep,
comparing MoE vs temporal (rolling-residency, K=18 resident of 192, swap 1/token, min_logit) vs the
dense floor. Config: mb=64, TE 1.11, seed 1234, `BPB_DIVISOR=2.7600`. **Lower BPB is better.**
All values **measured** from `results/phase0/runs/`. Split across H100 (@1e17 big shapes + fillers)
and A6000 (small-shape temporal) per `docs/shared-fine-grained-moe.md`.

**Status: 9/12 done** — all 6 MoE + 3 temporal (H100). Pending (A6000): temporal s0/s1@1e16, s1@1e17.

## Both MoE parabolas — complete & bracketed (min unchanged vs G=1)

| budget | left | **min** | right | vs G=1 baseline min |
|---|---|---|---|---|
| 1e16 | sm1 1.4786 | **s0 1.4585** | s1 1.5352 | G1 s0 1.447 (+0.012) |
| 1e17 | s1 1.2846 | **s2 1.2708** | s3 1.2815 | G1 s2 1.269 (+0.002) |

→ Fine-graining experts 6→18 is **quality-neutral for the MoE** (every point within ~0.001–0.012 of
G=1) and preserves the scaling geometry (same compute-optimal shape per budget).

## Temporal vs the dense↔MoE band (recovery = (dense−temporal)/(dense−MoE))

Dense floor = G=1 baseline (dense doesn't fine-grain; active params identical).

| budget | shape | N_active | dense (G1) | MoE (G3) | **temporal (G3)** | recovery | vs G1 temporal |
|---|---|---|---|---|---|---|---|
| 1e16 | sm1 | 0.81M | 1.534 | 1.4786 | **1.4976** | 66% | G1 1.4891 |
| 1e16 | s0  | 1.42M | 1.519 | 1.4585 | *A6000* | — | G1 1.4599 |
| 1e16 | s1  | 3.91M | 1.591 | 1.5352 | *A6000* | — | G1 1.5488 |
| 1e17 | s1  | 3.91M | 1.361 | 1.2846 | *A6000* | — | G1 1.3039 |
| 1e17 | **s2** | 8.23M | 1.341 | 1.2708 | **1.2873** | 77% | G1 1.2821 |
| 1e17 | s3  | 15.09M | 1.408 | 1.2815 | **1.3129** | 75% | G1 1.3073 |

**Finding so far:** every measured temporal point lands **inside the dense↔MoE band** (66–77%
recovery), costing only +0.017–0.031 BPB over the full MoE. That's slightly below the G=1 baseline's
~80–85% — a real hint that **finer experts (18/192) recover marginally less under rolling residency**
than coarse (6/64), but the headline holds: 18-of-192 resident with a single swap/token keeps most of
the MoE advantage.

## Remaining (A6000, separate machine — copy run dirs here when done)

`g3_tmoe_s0_1e16`, `g3_tmoe_s1_1e16`, `g3_tmoe_s1_1e17` → will complete both temporal parabolas.

## To finalize

`BPB_DIVISOR=2.7600 .venv/bin/python scripts/phase0/plot_g3.py` → `results/phase0/g3_isoflop.png`
(dense/MoE/temporal parabolas, G3 vs G1 overlay). Then fill the recovery column for the A6000 points
and add the acceptance checks (2nd-seed repro + expert-load ≤8× mean) if run.

## Perf note

mb=64 is the ceiling (mb≥128 OOMs; the `mb·seq·topk` dispatch dominates). `--moe-permute-fusion`
(TE≥2.1) was tested and **rejected — 2–2.5× slower** at these micro-model sizes. Details:
`docs/shared-fine-grained-moe.md`.
