# Shared run plan: G=3 fine-grained MoE + temporal (H100 + A6000)

> Historical coordination doc (numbers as published at the time, during-training-val basis for some G1 cells). Canonical final numbers: `results/ablations/FINDINGS.md` + `phase0_isoflop_points.csv`.

Two-machine execution plan for the **G=3 fine-grained** FLAME-MoE Phase-0 sweep — the same
IsoFLOP MoE-vs-temporal-vs-dense comparison as `docs/EVALUATION_METHODOLOGY.md`, but with the
routed experts subdivided 3× (DeepSeek-style segmentation). Branch: `temporal-moe-impl`.

## What "G=3 fine-grained" means

Subdivide each routed expert by `GRAIN=3` (compute-preserving): `num_experts 64→192`,
`moe-router-topk 6→18`, each expert `moe_ffn → round_even(moe_ffn/3)`. The **shared expert is NOT
fine-grained** (kept at `2×` the original moe_ffn). Active params/FLOPs are ~unchanged (only the
router term `h·num_experts` grows ~2%), so `N_active`, the token budgets, and the IsoFLOP geometry
carry over. Temporal = rolling-residency, `K = top-k = 18` resident of 192, swaps **1 expert/token**
(min_logit eviction). Goal per budget: a bracketed BPB(N) parabola; temporal should land inside the
dense↔MoE band.

## Fixed config — IDENTICAL on both machines (do not change)

- `GRAIN=3`, `MICRO_BATCH=64`, **no** `--moe-permute-fusion`, seed **1234**
- Locked HPs: peak-LR 3e-3, warmup 5%, cosine→10%, gb 256, aux 0.01, z-loss 0.001, bf16
- TransformerEngine **1.11** (see "Perf note" — do NOT use TE 2.1 / permute-fusion here)
- Tokenizer `data/tok16k` (16k BPE), corpus `data/tok16k_full` (6.06B tok), `BPB_DIVISOR=2.7600`
- Metric: **BPB = CE/(ln2·bytes_per_token)**, lower better. Single eval at end (`EVAL_AT_END=1`).

Common env:
```bash
export TOKENIZER_MODEL=/workspace/FLAME-MoE/data/tok16k
export DATA_DIR=/workspace/FLAME-MoE/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7600 GRAIN=3 MICRO_BATCH=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_AT_END=1
```

## Machine split — which system runs what

The A6000 (48 GB) can only hold the **small shapes** at mb=64 (sm1/s0/s1 ≈ 27–37 GB; s2/s3 ≈ 46+ GB
OOM), so **s2/s3 are H100-only**. The long `tmoe_s1_1e17` (8129 iters — smallest shape at 1e17 →
most tokens) is offloaded to the A6000 so it runs *alongside* the H100's fixed big-shape floor.
Balanced makespan ≈ 5.5 h (A6000 assumed 1.5× H100 per-iter).

### 🟩 H100 — big shapes + finishing + fillers
| run | shape | budget | iters | method |
|---|---|---|---|---|
| `g3_moe_s1_1e17`  | s1 | 1e17 | 8129 | MoE (finishing) |
| `g3_moe_s3_1e17`  | s3 | 1e17 | 2107 | MoE (big) |
| `g3_tmoe_s2_1e17` | s2 | 1e17 | 3861 | temporal (big) |
| `g3_tmoe_s3_1e17` | s3 | 1e17 | 2107 | temporal (big) |
| `g3_tmoe_sm1_1e16`| sm1| 1e16 | 3938 | temporal (filler) |

### 🟦 A6000 — small shapes only
| run | shape | budget | iters | method |
|---|---|---|---|---|
| `g3_tmoe_s1_1e17` | s1 | 1e17 | 8129 | temporal (long pole) |
| `g3_tmoe_s0_1e16` | s0 | 1e16 | 2232 | temporal |
| `g3_tmoe_s1_1e16` | s1 | 1e16 | 813  | temporal |

### ✅ Already done on the H100 (mb=64, TE 1.11) — do NOT re-run
`g3_moe_sm1_1e16`, `g3_moe_s0_1e16`, `g3_moe_s1_1e16`, `g3_moe_s2_1e17`

## How to run

**H100** (this machine): driven by `scripts/phase0/g3_run_all.sh` (idempotent — skips completed
runs). It has been reconfigured to run exactly the H100 list above.

**A6000** — same shared `/workspace` volume, so code + data + `.venv` are already present. Verify
the stack first, then launch its 3 runs:
```bash
cd /workspace/FLAME-MoE
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export CUDNN_PATH=$NV/cudnn LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64
.venv/bin/python -c "import transformer_engine.pytorch; print('TE ok')"   # must print TE ok (rebuild TE for sm_86 if not)

source <common env above>
export TEMPORAL=1 TEMPORAL_EVICT=min_logit
nohup bash scripts/phase0/drive.sh scripts/phase0/g3_a6000_1e16.txt > results/phase0/a6000_1e16.log 2>&1 &   # s0, s1 @1e16
nohup bash scripts/phase0/drive.sh scripts/phase0/g3_a6000_1e17.txt > results/phase0/a6000_1e17.log 2>&1 &   # s1 @1e17  (run AFTER 1e16 — one GPU, serial)
```
(Run the two serially on the A6000 — one GPU. `drive.sh` skips any run whose final checkpoint
already exists, so the shared volume auto-coordinates: neither machine will redo the other's runs.)

## Merge

Both machines write to the shared `results/phase0/runs/`, so no copy needed. Once all 12 exist:
```bash
.venv/bin/python scripts/phase0/plot_g3_curves.py   # -> results/phase0/figures/fine_grained_vs_coarse_experts_isoflop.png
```

## Current results (measured, mb=64, TE 1.11; lower BPB better)

| budget | shape | N_active | MoE (G3) | MoE (G1 baseline) | temporal (G3) |
|---|---|---|---|---|---|
| 1e16 | sm1 | 0.81M | 1.4786 | 1.478 | *pending (A6000)* |
| 1e16 | **s0** | 1.42M | **1.4585 (min)** | **1.447** | *pending (A6000)* |
| 1e16 | s1 | 3.91M | 1.5352 | 1.540 | *pending (A6000)* |
| 1e17 | s1 | 3.91M | *finishing* | 1.284 | *pending (A6000)* |
| 1e17 | **s2** | 8.23M | **1.2708** | **1.269** | *pending (H100)* |
| 1e17 | s3 | 15.09M | *pending* | 1.289 | *pending (H100)* |

Fine-graining is ~neutral vs the G=1 baseline so far (s0@1e16 +0.012, s2@1e17 +0.002 — within the
~0.003 seed-noise band; the mb=64-vs-mb=32 delta was also ≤0.003). Full parabolas + temporal band
land when the 8 remaining runs finish.

## Perf note (why mb=64 / TE 1.11 / no permute-fusion)

These micro-models (1–15M active) use ~1–3% of an H100 (overhead/HBM-bandwidth bound). Tested and
rejected: `--moe-permute-fusion` (needs TE≥2.1; measured **2–2.5× slower** at these tiny sizes),
GPU co-location (~1.2×, bandwidth contention), CUDA graphs (needs token-dropping → changes results).
torch.compile is already partially active via `megatron/core/jit.py`. `mb=64` (grad-accum 8→4,
~1.37× over mb=32) is the realistic ceiling; mb≥128 OOMs (the `mb·seq·topk` dispatch dominates).

## Caveat

Cross-GPU splitting adds ~seed-noise (~0.003 BPB) to any temporal↔MoE delta whose two runs land on
different cards — well below the ~0.02 BPB bracket margins, so it won't flip a min. The G=1 baseline
+ dense floor were themselves measured on an A6000, so a cross-GPU element is already present.
