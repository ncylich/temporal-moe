# Phase 0 — FLAME-MoE B=1 baseline establishment (single RTX A6000)

Goal: validate stock FLAME-MoE (B=1, unmodified) hits pre-registered scaling-law targets at
1e16 AND 1e17 FLOPs, per `docs/research/TEMPORAL_ABLATION_PLAN.md`. No temporal code in Phase 0.

## Environment & setup decisions (resumability)

- **Branch:** `phase0-flame-baselines`.
- **GPU:** 1× RTX A6000 48GB. `torchrun --nproc_per_node=1`, `EXPERT_MODEL_PARALLEL_SIZE=1`,
  `PIPELINE_MODEL_PARALLEL_SIZE=1`.
- **Python env:** repo `.venv` (torch 2.4.1+cu124). Original FLAME conda env `MoE` absent.
- **TransformerEngine:** not in `.venv`; Megatron imports it unconditionally
  (`language_module` → `extensions/transformer_engine`). Installed TE 2.16 (prebuilt cu12 lib +
  source-built torch bindings). **Model uses `--transformer-impl local`** (pure Megatron-core
  modules: ColumnParallelLinear / DotProductAttention / torch RMSNorm) — TE is imported but never
  executed, so numerics are stock Megatron-core. RMSNorm via `torch.nn.RMSNorm` (torch≥2.4).
- **Data:** dclm-baseline-1.0 (HF `mlfoundations/dclm-baseline-1.0-parquet`, global-shard_01
  local-shard_0). Downloaded ~85 shards → jsonl → tokenized with pythia-12b tokenizer
  (`tools/preprocess_data.py`, `--append-eod`) → `data/dclm_tokenized/dclm_text_document.{bin,idx}`.
  Single GCS dclm corpus on FLAME is identical dataset; tokenizer/vocab identical.
- **Launcher:** `scripts/phase0/run.sh` (env-parametrized SHAPE / TARGET_FLOPS / PEAK_LR /
  WARMUP_FRAC / GLOBAL_BATCH / SEED). Computes `train_iters` so C = 6·N·D, N = active
  non-embedding params (see below).

## Shapes & token budgets (N = active non-embedding params; C = 6·N·D; seq=2048)

| shape | hidden | layers | N_active | D@1e17 | iters@1e17(gb256) | D@1e16 | iters@1e16 |
|---|---|---|---|---|---|---|---|
| s1 | 192 | 5 | 3.81M | 4.37B | 8338 | 0.437B | 834 |
| s2 | 256 | 6 | 8.12M | 2.05B | 3917 | 0.205B | 392 |
| s3 | 320 | 7 | 14.77M | 1.13B | 2152 | 0.113B | 215 |
| s4 | 384 | 8 | 24.29M | 0.69B | 1309 | 0.069B | 131 |
| s5 | 448 | 9 | 37.17M | 0.45B | 855 | 0.045B | 86 |
| s6 | 512 | 10 | 53.92M | 0.31B | 590 | 0.031B | 59 |

(N matches plan: s2≈8.1M vs plan's ~7.8M optimum @1e17.) **1e16 point read from iters/10 eval of
the 1e17 run** (per plan), so eval-interval = iters/10.

## Pre-registered targets / acceptance
- L*≈4.7 @1e17, ≈6.1 @1e16. Per-shape @1e17: s1 4.83 · **s2 4.68 (min)** · s3 4.76.
- ACCEPT: (1) best-shape ≤4.9 @1e17 AND ≤6.4 @1e16; (2) parabola @1e17 (min s1–s3), monotone @1e16;
  (3) |Δloss|≤0.03 nats on 2nd seed at min shape; (4) no expert >8× mean load, aux converged.

## Locked training config (single-GPU FLAME-MoE)

Determined during bring-up (`scripts/phase0/run.sh`):
- **Transformer impl: `transformer_engine`** (FLAME's native path; TE 1.11 built from vendored
  `TransformerEngine/` source — matches Megatron-core 0.12 / torch 2.4.1). Runtime needs pip
  `nvidia-cudnn`/`nvidia-cublas` on `LD_LIBRARY_PATH` + `CUDNN_PATH`.
- **`--moe-grouped-gemm`**: batches the 64 local experts (EP=1) into one grouped GEMM
  (TEGroupedLinear). Numerically equivalent to FLAME's EP=8 sequential-per-GPU experts; required
  for throughput (sequential 64-expert loop was ~4× slower).
- **`--no-gradient-accumulation-fusion`**: that fusion needs apex (absent); perf-only.
- Datasets helper `helpers_cpp` and `python3-config` symlinked into `.venv`; `.venv/bin` on PATH so
  Megatron's `make` finds pybind11.
- **micro-batch = 32** (gb=256 → 8 microbatches). Higher mb OOMs on the **vocab cross-entropy**
  (50254-vocab fp32 logits: mb64 → 24.6 GiB). mb32 peak ≈28 GB / 48 GB. CE-loss-fusion tests were
  inconclusive (GPU contention); kept off for reliability.
- **Throughput:** mock smoke s2-shape gb256/mb32 ≈ **4.1 s/iter (~128k tok/s, ~6 TFLOP/s)**, no NaN,
  loss 10.9→8.4 over 12 iters. Low util is intrinsic (tiny matmuls + large vocab dominate FLOPs).
  → expect ~hours per 1e17 run; full Phase 0 ≈ 1–1.5 GPU-days.

## Data pipeline (parallelized per user directive)

dclm jsonl (8.02M docs, 45 GB, ~9.6B tokens) → shell `split -n l/32` into `data/dclm_parts/` →
**32 parallel `preprocess_data.py` jobs** (`scripts/phase0/tokenize_parallel.sh`, 2 workers each =
64-way) → `data/dclm_tokenized/part*_text_document.{bin,idx}`. Tokenize validated single-core vs
multi-core (identical 4808234-byte output on a 2k-doc subset). Megatron `--partitions` mode stalled
on the serial Python split of the 45 GB file → replaced with shell split + parallel jobs.

## Run log

(entries appended below as runs complete)
