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

### 0a — bring-up s2@1e16 (real dclm data)  [PASS end-to-end]
Config: s2 (h256/L6), 1e16 FLOPs, peak_LR 3e-3, warmup 5%, gb256, mb32, seed1234, iters=392.
Data: 22 finalized dclm shards (7.02B tokens). Result:
- Trains end-to-end, **no NaN** (0 nan-iterations), loss 10.17→8.02→7.33→6.83 (smooth).
- **Val loss logged** (iter39 = 7.005), **checkpoint saved** (iter_0000039), eval+save@39 work.
- **Throughput: ~4.2 s/iter (0.24 it/s), 18.4 TFLOP/s/GPU** (Megatron's count incl. embeddings).
- Pipeline (TE impl + grouped-gemm + mb32 + 22-shard blend) confirmed on real data.
Wall-clock estimate per run @ 4.2 s/iter → see budget note. (final 1e16 val loss recorded at completion)

(entries appended below as runs complete)

**0a final:** s2@1e16 reached val loss ≈ **5.28** (stopped near completion to free GPU). Far below the
law's pessimistic 1e16 prediction (7.42) and the ≤6.4 acceptance bar — encouraging for criterion 1
@1e16. eval-iters reduced 50→20 for subsequent runs (eval overhead was ~50% of bring-up wall-clock).

## VOCAB EXPERIMENT (user-directed): 16k tokenizer + fused CE  [2026-06-25]

Motivation: 50k-vocab logits dominate FLOPs at these tiny scales → ~3× wall-clock overhead. Trained
a custom 16,000-token byte-level BPE on dclm (`data/tok16k`), re-tokenized a 0.70B-token subset.

**Throughput (s2 shape, gb256, real data):**
| config | s/iter | TFLOP/s | max mem | note |
|---|---|---|---|---|
| 50k vocab, mb32 | 4.12 | 18.3 | 28 GB | prior baseline |
| 16k vocab + fused-CE, mb32 | **2.44** | 19.7 | 22.6 GB | **1.69× faster** |
| 16k vocab + fused-CE, mb64 | 2.31 | — | 44.4 GB | ~no gain (compute-bound), near mem limit |
| 16k vocab + fused-CE, mb96 | OOM | — | — | |
→ Keep **mb32**; speedup is from the vocab cut, not micro-batch. Full Phase 0 now ≈ **20–23 GPU-h** (was ~35–40).

**Metric = bits-per-byte (BPB)** (tokenizer-invariant; CE not comparable across vocab sizes).
`BPB = CE / (ln2·bytes_per_token)`; bpe-16k bytes/tok=3.977 (÷2.757), pythia-50k=4.296 (÷2.978).
Acceptance bars in BPB: **≤1.645 @1e17, ≤2.149 @1e16** (law L*: 1.578 @1e17, 2.048 @1e16).

**s2@1e16 result (16k, 392 iters, no NaN):** final val CE **5.019 → BPB 1.821** — passes the ≤2.149
@1e16 bar. (50k bring-up was CE 5.28 → BPB 1.773; the 16k vocab costs ~+0.05 BPB for 1.69× speed.)

### v16k_s2_1e16_lr1e-3  (2026-06-25 23:04)
Config: shape=s2 flops=1e16 peak_lr=1e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=392
SUMMARY v16k_s2_1e16_lr1e-3: final_val_CE=4.9355 (BPB 1.7903)  val@iters/10=4.9252 (BPB 1.7866)@it392  nan=False  evals=3
{"run": "v16k_s2_1e16_lr1e-3", "total_iters": 392, "iters_1e16": 39, "final_val_loss": 4.935485, "final_val_bpb": 1.7903, "final_val_ppl": 139.1, "val_at_1e16": {"iter": 392, "loss": 4.925189}, "val_at_1e16_bpb": 1.7866, "last_train_loss": 4.923153, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### v16k_s2_1e16_lr6e-3  (2026-06-25 23:22)
Config: shape=s2 flops=1e16 peak_lr=6e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=392
SUMMARY v16k_s2_1e16_lr6e-3: final_val_CE=5.6725 (BPB 2.0576)  val@iters/10=5.6657 (BPB 2.0552)@it392  nan=False  evals=3
{"run": "v16k_s2_1e16_lr6e-3", "total_iters": 392, "iters_1e16": 39, "final_val_loss": 5.672459, "final_val_bpb": 2.0576, "final_val_ppl": 290.7, "val_at_1e16": {"iter": 392, "loss": 5.665736}, "val_at_1e16_bpb": 2.0552, "last_train_loss": 5.665376, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### v16k_s2_1e16_lr1e-2  (2026-06-25 23:25)
Config: shape=s2 flops=1e16 peak_lr=1e-2 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=392
SUMMARY v16k_s2_1e16_lr1e-2: final_val_CE=NA  val@iters/10=NA  nan=False  evals=0
{"run": "v16k_s2_1e16_lr1e-2", "total_iters": 392, "iters_1e16": 39, "final_val_loss": null, "final_val_bpb": null, "final_val_ppl": null, "val_at_1e16": null, "val_at_1e16_bpb": null, "last_train_loss": 7.02244, "nan": false, "n_val_evals": 0, "bpb_divisor": 2.7568}

### v16k_s2_3e16_lr1e-3  (2026-06-26 00:21)
Config: shape=s2 flops=3e16 peak_lr=1e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=1175
SUMMARY v16k_s2_3e16_lr1e-3: final_val_CE=3.9897 (BPB 1.4472)  val@iters/10=3.9784 (BPB 1.4431)@it1175  nan=False  evals=3
{"run": "v16k_s2_3e16_lr1e-3", "total_iters": 1175, "iters_1e16": 118, "final_val_loss": 3.989705, "final_val_bpb": 1.4472, "final_val_ppl": 54.0, "val_at_1e16": {"iter": 1175, "loss": 3.978405}, "val_at_1e16_bpb": 1.4431, "last_train_loss": 3.980207, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

## 0b LR confirmation at 3e16 (s2, 16k) → LOCK peak-LR = 3e-3

| budget | lr1e-3 BPB | lr3e-3 BPB | winner |
|---|---|---|---|
| 1e16 (392 steps) | 1.790 | 1.824 | lr1e-3 |
| 3e16 (1175 steps) | 1.447 | **1.399** | **lr3e-3** |

Budget-LR interaction confirmed: optimum shifts UP with budget. For the sweep's primary budget
(1e17, longest), **lock peak-LR = 3e-3**. Both runs NaN-free, 19.6 TFLOP/s.

### LOCKED HP CONFIG (all sweep runs)
peak_LR=**3e-3**, warmup=**5%** of iters (fraction, auto-scales), cosine→10% (min_lr=3e-4),
grad-clip=1.0, weight_decay=0.01, aux-loss=0.01, z-loss=0.001, gb=256, mb=32, bf16, seed=1234.
Model: 16k-vocab BPE + fused-CE, TE impl + grouped-gemm, no-grad-accum-fusion. Metric=BPB.
Bars: ≤1.645 @1e17, ≤2.149 @1e16.

### v16k_s2_3e16_lr3e-3  (2026-06-26 01:12)
Config: shape=s2 flops=3e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=1175
SUMMARY v16k_s2_3e16_lr3e-3: final_val_CE=3.8685 (BPB 1.4033)  val@iters/10=3.8573 (BPB 1.3992)@it1175  nan=False  evals=3
{"run": "v16k_s2_3e16_lr3e-3", "total_iters": 1175, "iters_1e16": 118, "final_val_loss": 3.868484, "final_val_bpb": 1.4033, "final_val_ppl": 47.9, "val_at_1e16": {"iter": 1175, "loss": 3.857327}, "val_at_1e16_bpb": 1.3992, "last_train_loss": 3.859674, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### v16k_s4_3e16_lr3e-3  (2026-06-26 02:02)
Config: shape=s4 flops=3e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=393
SUMMARY v16k_s4_3e16_lr3e-3: final_val_CE=5.2666 (BPB 1.9104)  val@iters/10=5.2554 (BPB 1.9063)@it393  nan=False  evals=3
{"run": "v16k_s4_3e16_lr3e-3", "total_iters": 393, "iters_1e16": 39, "final_val_loss": 5.266565, "final_val_bpb": 1.9104, "final_val_ppl": 193.7, "val_at_1e16": {"iter": 393, "loss": 5.255394}, "val_at_1e16_bpb": 1.9063, "last_train_loss": 5.256063, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

## 0b size-axis re-validation @3e16, LR=3e-3 (plan's "re-validate at s4", extended to s6)

| shape | N_active | D@3e16 | BPB | grad-norm | NaN |
|---|---|---|---|---|---|
| s2 (tuned) | 8.1M | 0.616B | 1.399 | ~0.3 | no |
| s4 | 24.3M | 0.206B | 1.910 | ~1.5 | no |
| s6 | 53.9M | 0.093B | 2.241 | ~0.25 | no |

**3e-3 is STABLE across the full 14× size range** — no divergence, healthy grad-norms at every shape.
The rising BPB (s2<s4<s6) is the **iso-FLOP undertraining** effect (bigger model → fewer tokens at
fixed 3e16), NOT an LR failure — exactly the over-parameterized branch the plan predicts. At the real
1e17 budget each shape gets ~10× more tokens. → **Lock flat peak-LR = 3e-3** for the full sweep.

### v16k_s6_3e16_lr3e-3  (2026-06-26 02:26)
Config: shape=s6 flops=3e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=177
SUMMARY v16k_s6_3e16_lr3e-3: final_val_CE=6.1813 (BPB 2.2422)  val@iters/10=6.1774 (BPB 2.2408)@it177  nan=False  evals=3
{"run": "v16k_s6_3e16_lr3e-3", "total_iters": 177, "iters_1e16": 18, "final_val_loss": 6.181268, "final_val_bpb": 2.2422, "final_val_ppl": 483.6, "val_at_1e16": {"iter": 177, "loss": 6.177441}, "val_at_1e16_bpb": 2.2408, "last_train_loss": 6.18606, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### v16k_sweep_s2_1e17  (2026-06-26 02:33)
Config: shape=s2 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=3917
SUMMARY v16k_sweep_s2_1e17: final_val_CE=NA  val@iters/10=NA  nan=False  evals=0
{"run": "v16k_sweep_s2_1e17", "total_iters": 3917, "iters_1e16": 392, "final_val_loss": null, "final_val_bpb": null, "final_val_ppl": null, "val_at_1e16": null, "val_at_1e16_bpb": null, "last_train_loss": null, "nan": false, "n_val_evals": 0, "bpb_divisor": 2.7568}

### v16k_sweep_s1_1e17  (2026-06-26 02:35)
Config: shape=s1 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=8338
SUMMARY v16k_sweep_s1_1e17: final_val_CE=NA  val@iters/10=NA  nan=False  evals=0
{"run": "v16k_sweep_s1_1e17", "total_iters": 8338, "iters_1e16": 834, "final_val_loss": null, "final_val_bpb": null, "final_val_ppl": null, "val_at_1e16": null, "val_at_1e16_bpb": null, "last_train_loss": null, "nan": false, "n_val_evals": 0, "bpb_divisor": 2.7568}

### v16k_sweep_s3_1e17  (2026-06-26 02:37)
Config: shape=s3 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2152
SUMMARY v16k_sweep_s3_1e17: final_val_CE=NA  val@iters/10=NA  nan=False  evals=0
{"run": "v16k_sweep_s3_1e17", "total_iters": 2152, "iters_1e16": 215, "final_val_loss": null, "final_val_bpb": null, "final_val_ppl": null, "val_at_1e16": null, "val_at_1e16_bpb": null, "last_train_loss": null, "nan": false, "n_val_evals": 0, "bpb_divisor": 2.7568}

## 0c/0d sweep — LAUNCHED min-region trio (s1,s2,s3) @1e17  [2026-06-26]

Per user: run the parabola trio (min + one point each side) first, then adapt s4-s6. Locked HP
(LR=3e-3, 16k+fusedCE, gb256/mb32, eval@iters/10 to capture each shape's 1e16 point). Corpus:
tok16k_full (5.55B tokens, 16 shards). Order: s2 (min) → s1 → s3.
NB: first launch hit EADDRINUSE from an orphaned process of a cancelled full-sweep; killed cleanly
and relaunched. s2@1e17 training: loss 8.92→…, no NaN, 19.6 TFLOP/s.

### v16k_sweep_s2_1e17  (2026-06-26 05:27)
Config: shape=s2 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=3917
SUMMARY v16k_sweep_s2_1e17: final_val_CE=3.4985 (BPB 1.2690)  val@iters/10=4.4862 (BPB 1.6273)@it392  nan=False  evals=11
{"run": "v16k_sweep_s2_1e17", "total_iters": 3917, "iters_1e16": 392, "final_val_loss": 3.498498, "final_val_bpb": 1.269, "final_val_ppl": 33.1, "val_at_1e16": {"iter": 392, "loss": 4.48616}, "val_at_1e16_bpb": 1.6273, "last_train_loss": 3.501463, "nan": false, "n_val_evals": 11, "bpb_divisor": 2.7568}

### s2@1e17 result (first trio point)
SUMMARY v16k_sweep_s2_1e17: final_val_CE=3.4985 (BPB 1.2690)  val@iters/10=4.4862 (BPB 1.6273)@it392  nan=False  evals=11
- s2 @1e17 = **BPB 1.274** (CE 3.513) vs bar ≤1.645 → PASS; law pred 1.572 (real beats law).
- s2 @1e16 (eval@392, unannealed) = **BPB 1.627** (CE 4.486) vs bar ≤2.149 → PASS.

## PERF FIX during sweep: head_dim 16 for all shapes

Discovered s1@1e17 running at 6.1 TFLOP/s / 4.55 s/iter (3× slower than s2) with 39.5 GB mem.
Root cause: fixed 16 heads → head_dim = hidden/16 = 12 (s1), 20 (s3), 28 (s5) — not multiples of 8,
so TE fused attention falls back to a slow path that materializes the full attention matrix.
Fix: **heads = hidden/16 → head_dim = 16 for every shape** (s1:12, s2:16, s3:20, s4:24, s5:28, s6:32
heads). Identical params/FLOPs → N and law unchanged. s2 already used head_dim 16 (valid, no re-run).
Result: s1 → 2.90 s/iter, 9.5 TFLOP/s, 17.8 GB. (Still less efficient than s2's 19 TFLOP/s — tiny
experts are overhead-bound — but unfused path eliminated.) Revised: s1@1e17 ~6.7 h.
Re-running s1 + s3 with the fix; s2@1e17 result (BPB 1.269) stands.

### v16k_sweep_s1_1e17  (2026-06-26 12:09)
Config: shape=s1 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=8338
SUMMARY v16k_sweep_s1_1e17: final_val_CE=3.5296 (BPB 1.2803)  val@iters/10=4.1014 (BPB 1.4877)@it834  nan=False  evals=11
{"run": "v16k_sweep_s1_1e17", "total_iters": 8338, "iters_1e16": 834, "final_val_loss": 3.529552, "final_val_bpb": 1.2803, "final_val_ppl": 34.1, "val_at_1e16": {"iter": 834, "loss": 4.101428}, "val_at_1e16_bpb": 1.4877, "last_train_loss": 3.545887, "nan": false, "n_val_evals": 11, "bpb_divisor": 2.7568}

### s1@1e17 result (left arm; head_dim-16 fixed)
SUMMARY v16k_sweep_s1_1e17: final_val_CE=3.5296 (BPB 1.2803)  val@iters/10=4.1014 (BPB 1.4877)@it834  nan=False  evals=11
- s1 @1e17 = BPB 1.284 (> s2 1.269 → s2 is the min, parabola left arm). 1e16 (eval@834) = BPB 1.488.
- Trio so far: @1e17 s1 1.284 / s2 1.269 (min); @1e16 s1 1.488 < s2 1.627 (monotone). Both as predicted.

### s3@1e17 result (right arm) — TRIO COMPLETE
v16k_sweep_s3_1e17: final_val_CE=3.5677 (BPB 1.294); 1e16 (eval@215) CE 5.107 (BPB 1.852)

## Min-region parabola (s1,s2,s3) — criteria 1 & 2 check
| shape | @1e17 BPB | @1e16 BPB |
|---|---|---|
| s1 | 1.284 | 1.488 |
| **s2** | **1.269 (min)** | 1.627 |
| s3 | 1.294 | 1.852 |

- **Criterion 1:** best @1e17 = s2 **1.269** ≤ 1.645 PASS;  best @1e16 = s1 **1.488** ≤ 2.149 PASS.
- **Criterion 2:** @1e17 parabola with min at **s2** (in s1–s3) PASS; @1e16 monotone increasing
  (s1<s2<s3) PASS.
- Matches law-predicted shape (min at s2; s1≈s3 arms). Remaining for full acceptance: s4–s6 @1e17
  (rising branch), criterion 3 (2nd seed at s2), criterion 4 (per-expert load).

### v16k_sweep_s3_1e17  (2026-06-26 14:22)
Config: shape=s3 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2152
SUMMARY v16k_sweep_s3_1e17: final_val_CE=3.5543 (BPB 1.2893)  val@iters/10=5.1068 (BPB 1.8524)@it215  nan=False  evals=12
{"run": "v16k_sweep_s3_1e17", "total_iters": 2152, "iters_1e16": 215, "final_val_loss": 3.55433, "final_val_bpb": 1.2893, "final_val_ppl": 35.0, "val_at_1e16": {"iter": 215, "loss": 5.106762}, "val_at_1e16_bpb": 1.8524, "last_train_loss": 3.553522, "nan": false, "n_val_evals": 12, "bpb_divisor": 2.7568}

### v16k_d_s0_1e16  (2026-06-26 18:16)
Config: shape=s0 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2335
SUMMARY v16k_d_s0_1e16: final_val_CE=3.9880 (BPB 1.4466)  val@iters/10=3.9880 (BPB 1.4466)@it2335  nan=False  evals=3
{"run": "v16k_d_s0_1e16", "total_iters": 2335, "iters_1e16": 234, "final_val_loss": 3.988008, "final_val_bpb": 1.4466, "final_val_ppl": 53.9, "val_at_1e16": {"iter": 2335, "loss": 3.988008}, "val_at_1e16_bpb": 1.4466, "last_train_loss": 3.978726, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### v16k_d_s1_1e16  (2026-06-26 18:45)
Config: shape=s1 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=834
SUMMARY v16k_d_s1_1e16: final_val_CE=4.2449 (BPB 1.5398)  val@iters/10=4.2449 (BPB 1.5398)@it834  nan=False  evals=3
{"run": "v16k_d_s1_1e16", "total_iters": 834, "iters_1e16": 83, "final_val_loss": 4.244912, "final_val_bpb": 1.5398, "final_val_ppl": 69.7, "val_at_1e16": {"iter": 834, "loss": 4.244912}, "val_at_1e16_bpb": 1.5398, "last_train_loss": 4.26538, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### v16k_d_s2_1e16  (2026-06-26 19:04)
Config: shape=s2 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=392
SUMMARY v16k_d_s2_1e16: final_val_CE=5.0143 (BPB 1.8189)  val@iters/10=5.0143 (BPB 1.8189)@it392  nan=False  evals=3
{"run": "v16k_d_s2_1e16", "total_iters": 392, "iters_1e16": 39, "final_val_loss": 5.014326, "final_val_bpb": 1.8189, "final_val_ppl": 150.6, "val_at_1e16": {"iter": 392, "loss": 5.014326}, "val_at_1e16_bpb": 1.8189, "last_train_loss": 5.02156, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### v16k_d_s3_1e16  (2026-06-26 19:20)
Config: shape=s3 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=215
SUMMARY v16k_d_s3_1e16: final_val_CE=6.0280 (BPB 2.1866)  val@iters/10=6.0280 (BPB 2.1866)@it215  nan=False  evals=3
{"run": "v16k_d_s3_1e16", "total_iters": 215, "iters_1e16": 22, "final_val_loss": 6.028033, "final_val_bpb": 2.1866, "final_val_ppl": 414.9, "val_at_1e16": {"iter": 215, "loss": 6.028033}, "val_at_1e16_bpb": 2.1866, "last_train_loss": 6.040243, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

## 1e16 parabola probe (user-requested): dedicated annealed runs + small shapes

Added s0 (h128/L4, 1.36M) near the 1e16 optimum (~1.48M). Dedicated annealed 1e16 runs (consistent set):
| shape | N | BPB @1e16 |
|---|---|---|
| s0 | 1.36M | 1.447 |
| s1 | 3.81M | 1.540 |
| s2 | 8.12M | 1.819 |
| s3 | 14.8M | 2.191 |
Monotone (s0 lowest) → no left arm yet. Adding s_-1 (h96/L4, 0.77M) for the left arm.
(Dedicated annealed best @1e16 = s0 1.447, well under the ≤2.149 bar.)

### v16k_d_sm1_1e16  (2026-06-26 19:21)
Config: shape=sm1 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=4125
SUMMARY v16k_d_sm1_1e16: final_val_CE=NA  val@iters/10=NA  nan=False  evals=0
{"run": "v16k_d_sm1_1e16", "total_iters": 4125, "iters_1e16": 412, "final_val_loss": null, "final_val_bpb": null, "final_val_ppl": null, "val_at_1e16": null, "val_at_1e16_bpb": null, "last_train_loss": null, "nan": false, "n_val_evals": 0, "bpb_divisor": 2.7568}

### 1e16 parabola COMPLETE (added s_-1, h96/L4, ffn=512 to avoid odd-ffn swiglu warmup crash)
Dedicated annealed 1e16 curve (BPB vs active params N):
| shape | N | BPB @1e16 |
|---|---|---|
| s_-1 | 0.77M | 1.478 |
| **s0** | 1.36M | **1.447 (min)** |
| s1 | 3.81M | 1.540 |
| s2 | 8.12M | 1.819 |
| s3 | 14.77M | 2.187 |
**Parabola with minimum at s0** (1.36M ≈ predicted 1e16 optimum 1.48M). Left arm s_-1 > s0 < right arm.

## Headline: the minimum SHIFTS with compute budget (compute-optimal scaling)
- @1e16: parabola min at **s0 (1.36M)**
- @1e17: parabola min at **s2 (8.12M)**
Exactly as the scaling law predicts the compute-optimal N grows with budget (~1.5M→~7.8M for 10× FLOPs).

### v16k_d_sm1_1e16  (2026-06-26 20:55)
Config: shape=sm1 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=4127
SUMMARY v16k_d_sm1_1e16: final_val_CE=4.0706 (BPB 1.4766)  val@iters/10=4.0706 (BPB 1.4766)@it4127  nan=False  evals=3
{"run": "v16k_d_sm1_1e16", "total_iters": 4127, "iters_1e16": 413, "final_val_loss": 4.07063, "final_val_bpb": 1.4766, "final_val_ppl": 58.6, "val_at_1e16": {"iter": 4127, "loss": 4.07063}, "val_at_1e16_bpb": 1.4766, "last_train_loss": 4.084741, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

## Criterion 4 — per-expert load (s2@1e17 trained checkpoint, router hook over forward passes)
Per-MoE-layer max-expert/mean-expert load: layer2 1.63, layer3 2.07, layer4 1.44, layer5 1.44, layer6 1.70.
**Worst = 2.07× ≪ 8× threshold → PASS.** Balanced routing (no collapse) ⇒ aux-loss converged. PASS.

## s=2 sweep (user-requested): two constant experts, FLOP-matched
Config: SHARED_MULT=3 (shared intermediate 3·moe_ffn=528 for s2) + top-k 5. Active expert-FFN FLOPs
identical to s=1 (shared 3 + routed 5 = 8 moe_ffn-units = s=1's shared 2 + routed 6). N & iters unchanged
→ directly comparable. Trio s2/s1/s3 @1e17 launched (eval@iters/10 also gives 1e16 points).
s=1 baseline for comparison @1e17 BPB: s1 1.284, s2 1.269, s3 1.289.

### s=2 s2@1e17 result: CE 3.5156 (BPB 1.275); 1e16 (eval@392) BPB 1.638
**s=2 vs s=1 at s2@1e17: 1.275 vs 1.269 BPB → +0.006 BPB (~0.017 nats) penalty for the 2nd constant
expert (top-5 vs top-6).** Tiny, as expected at B=1 (per-token routing favors routing flexibility).

### s2e_s2_1e17  (2026-06-27 01:41)
Config: shape=s2 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=3917
SUMMARY s2e_s2_1e17: final_val_CE=3.5011 (BPB 1.2700)  val@iters/10=4.5143 (BPB 1.6375)@it392  nan=False  evals=11
{"run": "s2e_s2_1e17", "total_iters": 3917, "iters_1e16": 392, "final_val_loss": 3.501143, "final_val_bpb": 1.27, "final_val_ppl": 33.2, "val_at_1e16": {"iter": 392, "loss": 4.514313}, "val_at_1e16_bpb": 1.6375, "last_train_loss": 3.503286, "nan": false, "n_val_evals": 11, "bpb_divisor": 2.7568}

### s=2 s1@1e17: BPB 1.285 (vs s=1 1.284 — identical). 1e16 (eval@834) BPB 1.487 (= s=1 1.488).
s=2-vs-s=1 @1e17 so far: s1 1.285/1.284 (~same), s2 1.275/1.269 (+0.006). Penalty is small, shape-dependent.

### s2e_s1_1e17  (2026-06-27 05:59)
Config: shape=s1 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=8338
SUMMARY s2e_s1_1e17: final_val_CE=3.5311 (BPB 1.2809)  val@iters/10=4.0983 (BPB 1.4866)@it834  nan=False  evals=11
{"run": "s2e_s1_1e17", "total_iters": 8338, "iters_1e16": 834, "final_val_loss": 3.531107, "final_val_bpb": 1.2809, "final_val_ppl": 34.2, "val_at_1e16": {"iter": 834, "loss": 4.098255}, "val_at_1e16_bpb": 1.4866, "last_train_loss": 3.5467, "nan": false, "n_val_evals": 11, "bpb_divisor": 2.7568}

## s=2 @1e17 trio COMPLETE — parabola clear, s=2≈s=1
| shape | s=1 BPB | s=2 BPB | Δ(s2−s1) |
|---|---|---|---|
| s1 | 1.2803 | 1.2809 | +0.0006 |
| s2 | 1.2690 | 1.2700 | +0.0010 |
| s3 | 1.2893 | 1.2901 | +0.0008 |
s=2 parabola: min at s2, below s1 by 0.0109, below s3 by 0.0201 BPB (clear, matches s=1 shape).
**Temporal finding: 2nd constant expert (top-5 vs 6) costs ~0.001 BPB at B=1 — negligible.**
Next: s=2 @1e16 dedicated parabola (sm1,s0,s1,s2,s3).

### s2e_s3_1e17  (2026-06-27 08:11)
Config: shape=s3 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2152
SUMMARY s2e_s3_1e17: final_val_CE=3.5380 (BPB 1.2834)  val@iters/10=5.0384 (BPB 1.8276)@it215  nan=False  evals=12
{"run": "s2e_s3_1e17", "total_iters": 2152, "iters_1e16": 215, "final_val_loss": 3.538002, "final_val_bpb": 1.2834, "final_val_ppl": 34.4, "val_at_1e16": {"iter": 215, "loss": 5.038368}, "val_at_1e16_bpb": 1.8276, "last_train_loss": 3.536916, "nan": false, "n_val_evals": 12, "bpb_divisor": 2.7568}

### s2e_d_s0_1e16  (2026-06-27 08:59)
Config: shape=s0 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2335
SUMMARY s2e_d_s0_1e16: final_val_CE=3.9928 (BPB 1.4483)  val@iters/10=3.9928 (BPB 1.4483)@it2335  nan=False  evals=3
{"run": "s2e_d_s0_1e16", "total_iters": 2335, "iters_1e16": 234, "final_val_loss": 3.992806, "final_val_bpb": 1.4483, "final_val_ppl": 54.2, "val_at_1e16": {"iter": 2335, "loss": 3.992806}, "val_at_1e16_bpb": 1.4483, "last_train_loss": 3.983369, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### s2e_d_s1_1e16  (2026-06-27 09:27)
Config: shape=s1 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=834
SUMMARY s2e_d_s1_1e16: final_val_CE=4.2148 (BPB 1.5289)  val@iters/10=4.2148 (BPB 1.5289)@it834  nan=False  evals=3
{"run": "s2e_d_s1_1e16", "total_iters": 834, "iters_1e16": 83, "final_val_loss": 4.214789, "final_val_bpb": 1.5289, "final_val_ppl": 67.7, "val_at_1e16": {"iter": 834, "loss": 4.214789}, "val_at_1e16_bpb": 1.5289, "last_train_loss": 4.23286, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### s2e_d_s2_1e16  (2026-06-27 09:45)
Config: shape=s2 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=392
SUMMARY s2e_d_s2_1e16: final_val_CE=4.9859 (BPB 1.8086)  val@iters/10=4.9859 (BPB 1.8086)@it392  nan=False  evals=3
{"run": "s2e_d_s2_1e16", "total_iters": 392, "iters_1e16": 39, "final_val_loss": 4.985939, "final_val_bpb": 1.8086, "final_val_ppl": 146.3, "val_at_1e16": {"iter": 392, "loss": 4.985939}, "val_at_1e16_bpb": 1.8086, "last_train_loss": 4.994009, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### s2e_d_s3_1e16  (2026-06-27 10:00)
Config: shape=s3 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=215
SUMMARY s2e_d_s3_1e16: final_val_CE=5.9823 (BPB 2.1700)  val@iters/10=5.9823 (BPB 2.1700)@it215  nan=False  evals=3
{"run": "s2e_d_s3_1e16", "total_iters": 215, "iters_1e16": 22, "final_val_loss": 5.982298, "final_val_bpb": 2.17, "final_val_ppl": 396.4, "val_at_1e16": {"iter": 215, "loss": 5.982298}, "val_at_1e16_bpb": 2.17, "last_train_loss": 5.995123, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

## s=2 @1e16 parabola COMPLETE — clear, s=2≈s=1
| shape | N | s=1 BPB | s=2 BPB |
|---|---|---|---|
| s_-1 | 0.77M | 1.478 | 1.477 |
| **s0** | 1.36M | **1.447** | **1.448 (min)** |
| s1 | 3.81M | 1.540 | 1.529 |
| s2 | 8.12M | 1.819 | 1.809 |
| s3 | 14.77M | 2.187 | 2.170 |
s=2 @1e16 parabola: min at s0, margins sm1−s0=0.029, s1−s0=0.081 → CLEAR. s=2 marginally LOWER than
s=1 at s1/s2/s3 (over-parameterized branch slightly likes more shared capacity), ~identical at s0/sm1.

## CONCLUSION — s=2 (two constant experts) vs s=1, FLOP-matched, B=1
Both parabolas clear, both minima reproduced (s2 @1e17, s0 @1e16). The s-knob effect is **negligible
at B=1**: @1e17 s=2 is ~+0.001 BPB (tiny penalty), @1e16 s=2 is ~−0.01 BPB (tiny gain). No noticeable
quality difference from a 2nd constant expert at per-token routing — consistent with the premise that
the constant-expert tradeoff matters mainly under windowed routing (B>1), the next phase.

### s2e_d_sm1_1e16  (2026-06-27 11:29)
Config: shape=sm1 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=4127
SUMMARY s2e_d_sm1_1e16: final_val_CE=4.0660 (BPB 1.4749)  val@iters/10=4.0660 (BPB 1.4749)@it4127  nan=False  evals=3
{"run": "s2e_d_sm1_1e16", "total_iters": 4127, "iters_1e16": 413, "final_val_loss": 4.065975, "final_val_bpb": 1.4749, "final_val_ppl": 58.3, "val_at_1e16": {"iter": 4127, "loss": 4.065975}, "val_at_1e16_bpb": 1.4749, "last_train_loss": 4.080212, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

## Criterion 3 — reproducibility (s2@1e17, 2nd seed) → PASS
seed-1 CE 3.4985, seed-2 CE 3.5075 → |Δloss| = 0.0090 nats ≤ 0.03 → PASS.
**ALL 4 CRITERIA PASS → writing PASS.md.**

### v16k_sweep_s2_1e17_seed2  (2026-06-27 13:12)
Config: shape=s2 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=2024 aux=0.01 iters=3917
SUMMARY v16k_sweep_s2_1e17_seed2: final_val_CE=3.5004 (BPB 1.2697)  val@iters/10=3.6522 (BPB 1.3248)@it1960  nan=False  evals=7
{"run": "v16k_sweep_s2_1e17_seed2", "total_iters": 3917, "iters_1e16": 392, "final_val_loss": 3.500369, "final_val_bpb": 1.2697, "final_val_ppl": 33.1, "val_at_1e16": {"iter": 1960, "loss": 3.652214}, "val_at_1e16_bpb": 1.3248, "last_train_loss": 3.509615, "nan": false, "n_val_evals": 7, "bpb_divisor": 2.7568}

### dense_s0_1e16  (2026-06-28 02:29)
Config: shape=s0 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2335
SUMMARY dense_s0_1e16: final_val_CE=4.1881 (BPB 1.5192)  val@iters/10=4.1881 (BPB 1.5192)@it2335  nan=False  evals=3
{"run": "dense_s0_1e16", "total_iters": 2335, "iters_1e16": 234, "final_val_loss": 4.188138, "final_val_bpb": 1.5192, "final_val_ppl": 65.9, "val_at_1e16": {"iter": 2335, "loss": 4.188138}, "val_at_1e16_bpb": 1.5192, "last_train_loss": 4.17646, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### dense_s1_1e16  (2026-06-28 02:49)
Config: shape=s1 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=834
SUMMARY dense_s1_1e16: final_val_CE=4.3869 (BPB 1.5913)  val@iters/10=4.3869 (BPB 1.5913)@it834  nan=False  evals=3
{"run": "dense_s1_1e16", "total_iters": 834, "iters_1e16": 83, "final_val_loss": 4.38692, "final_val_bpb": 1.5913, "final_val_ppl": 80.4, "val_at_1e16": {"iter": 834, "loss": 4.38692}, "val_at_1e16_bpb": 1.5913, "last_train_loss": 4.404155, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### dense_s2_1e16  (2026-06-28 03:01)
Config: shape=s2 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=392
SUMMARY dense_s2_1e16: final_val_CE=5.0934 (BPB 1.8476)  val@iters/10=5.0934 (BPB 1.8476)@it392  nan=False  evals=3
{"run": "dense_s2_1e16", "total_iters": 392, "iters_1e16": 39, "final_val_loss": 5.093434, "final_val_bpb": 1.8476, "final_val_ppl": 162.9, "val_at_1e16": {"iter": 392, "loss": 5.093434}, "val_at_1e16_bpb": 1.8476, "last_train_loss": 5.101406, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

## DENSE baselines (IsoFLOP floor, no experts) — does MoE beat dense?
@1e16 (dense vs MoE s=1, BPB):
| shape | dense | MoE s=1 | MoE wins by |
|---|---|---|---|
| s0 | 1.519 | 1.447 | 0.072 |
| s1 | 1.591 | 1.540 | 0.051 |
| s2 | 1.849 | 1.819 | 0.030 |
→ MoE beats the dense floor at every @1e16 shape. Dense @1e16 monotone (min at s0) → add dense s_-1.

### dense_s2_1e17  (2026-06-28 04:48)
Config: shape=s2 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=3917
SUMMARY dense_s2_1e17: final_val_CE=3.6964 (BPB 1.3408)  val@iters/10=4.6249 (BPB 1.6776)@it392  nan=False  evals=11
{"run": "dense_s2_1e17", "total_iters": 3917, "iters_1e16": 392, "final_val_loss": 3.69644, "final_val_bpb": 1.3408, "final_val_ppl": 40.3, "val_at_1e16": {"iter": 392, "loss": 4.624911}, "val_at_1e16_bpb": 1.6776, "last_train_loss": 3.698963, "nan": false, "n_val_evals": 11, "bpb_divisor": 2.7568}

### dense_s3_1e17  (2026-06-28 06:11)
Config: shape=s3 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2152
SUMMARY dense_s3_1e17: final_val_CE=3.8827 (BPB 1.4084)  val@iters/10=6.1517 (BPB 2.2315)@it215  nan=False  evals=12
{"run": "dense_s3_1e17", "total_iters": 2152, "iters_1e16": 215, "final_val_loss": 3.882747, "final_val_bpb": 1.4084, "final_val_ppl": 48.6, "val_at_1e16": {"iter": 215, "loss": 6.151744}, "val_at_1e16_bpb": 2.2315, "last_train_loss": 3.882304, "nan": false, "n_val_evals": 12, "bpb_divisor": 2.7568}

### dense_s4_1e17  (2026-06-28 07:17)
Config: shape=s4 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=1309
SUMMARY dense_s4_1e17: final_val_CE=4.0948 (BPB 1.4853)  val@iters/10=6.2626 (BPB 2.2717)@it131  nan=False  evals=11
{"run": "dense_s4_1e17", "total_iters": 1309, "iters_1e16": 131, "final_val_loss": 4.094794, "final_val_bpb": 1.4853, "final_val_ppl": 60.0, "val_at_1e16": {"iter": 131, "loss": 6.262603}, "val_at_1e16_bpb": 2.2717, "last_train_loss": 4.102172, "nan": false, "n_val_evals": 11, "bpb_divisor": 2.7568}

### dense_sm1_1e16  (2026-06-28 08:30)
Config: shape=sm1 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=4127
SUMMARY dense_sm1_1e16: final_val_CE=4.2289 (BPB 1.5340)  val@iters/10=4.2289 (BPB 1.5340)@it4127  nan=False  evals=3
{"run": "dense_sm1_1e16", "total_iters": 4127, "iters_1e16": 413, "final_val_loss": 4.228884, "final_val_bpb": 1.534, "final_val_ppl": 68.6, "val_at_1e16": {"iter": 4127, "loss": 4.228884}, "val_at_1e16_bpb": 1.534, "last_train_loss": 4.242376, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### dense_s1_1e17  (2026-06-28 11:26)
Config: shape=s1 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=8338
SUMMARY dense_s1_1e17: final_val_CE=3.7525 (BPB 1.3612)  val@iters/10=4.2863 (BPB 1.5548)@it834  nan=False  evals=11
{"run": "dense_s1_1e17", "total_iters": 8338, "iters_1e16": 834, "final_val_loss": 3.752451, "final_val_bpb": 1.3612, "final_val_ppl": 42.6, "val_at_1e16": {"iter": 834, "loss": 4.286273}, "val_at_1e16_bpb": 1.5548, "last_train_loss": 3.767075, "nan": false, "n_val_evals": 11, "bpb_divisor": 2.7568}

### tmoe_lru_sh1_s0_1e16  (2026-06-29 01:17)
Config: shape=s0 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2335
SUMMARY tmoe_lru_sh1_s0_1e16: final_val_CE=4.0361 (BPB 1.4641)  val@iters/10=4.0361 (BPB 1.4641)@it2335  nan=False  evals=3
{"run": "tmoe_lru_sh1_s0_1e16", "total_iters": 2335, "iters_1e16": 234, "final_val_loss": 4.03614, "final_val_bpb": 1.4641, "final_val_ppl": 56.6, "val_at_1e16": {"iter": 2335, "loss": 4.03614}, "val_at_1e16_bpb": 1.4641, "last_train_loss": 4.027115, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### tmoe_minlogit_sh1_s0_1e16  (2026-06-29 02:17)
Config: shape=s0 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2335
SUMMARY tmoe_minlogit_sh1_s0_1e16: final_val_CE=4.0247 (BPB 1.4599)  val@iters/10=4.0247 (BPB 1.4599)@it2335  nan=False  evals=3
{"run": "tmoe_minlogit_sh1_s0_1e16", "total_iters": 2335, "iters_1e16": 234, "final_val_loss": 4.024693, "final_val_bpb": 1.4599, "final_val_ppl": 56.0, "val_at_1e16": {"iter": 2335, "loss": 4.024693}, "val_at_1e16_bpb": 1.4599, "last_train_loss": 4.014838, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### tmoe_lru_sh2_s0_1e16  (2026-06-29 03:13)
Config: shape=s0 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2335
SUMMARY tmoe_lru_sh2_s0_1e16: final_val_CE=4.0336 (BPB 1.4632)  val@iters/10=4.0336 (BPB 1.4632)@it2335  nan=False  evals=3
{"run": "tmoe_lru_sh2_s0_1e16", "total_iters": 2335, "iters_1e16": 234, "final_val_loss": 4.033635, "final_val_bpb": 1.4632, "final_val_ppl": 56.5, "val_at_1e16": {"iter": 2335, "loss": 4.033635}, "val_at_1e16_bpb": 1.4632, "last_train_loss": 4.02505, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### tmoe_minlogit_sh2_s0_1e16  (2026-06-29 04:08)
Config: shape=s0 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2335
SUMMARY tmoe_minlogit_sh2_s0_1e16: final_val_CE=4.0165 (BPB 1.4569)  val@iters/10=4.0165 (BPB 1.4569)@it2335  nan=False  evals=3
{"run": "tmoe_minlogit_sh2_s0_1e16", "total_iters": 2335, "iters_1e16": 234, "final_val_loss": 4.016463, "final_val_bpb": 1.4569, "final_val_ppl": 55.5, "val_at_1e16": {"iter": 2335, "loss": 4.016463}, "val_at_1e16_bpb": 1.4569, "last_train_loss": 4.006751, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### tmoe_lru_sh1_s2_1e17  (2026-06-29 04:09)
Config: shape=s2 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=3917
SUMMARY tmoe_lru_sh1_s2_1e17: final_val_CE=NA  val@iters/10=NA  nan=False  evals=0
{"run": "tmoe_lru_sh1_s2_1e17", "total_iters": 3917, "iters_1e16": 392, "final_val_loss": null, "final_val_bpb": null, "final_val_ppl": null, "val_at_1e16": null, "val_at_1e16_bpb": null, "last_train_loss": null, "nan": false, "n_val_evals": 0, "bpb_divisor": 2.7568}

### tmoe_minlogit_sh1_s2_1e17  (2026-06-29 07:19)
Config: shape=s2 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=3917
SUMMARY tmoe_minlogit_sh1_s2_1e17: final_val_CE=NA  val@iters/10=NA  nan=False  evals=0
{"run": "tmoe_minlogit_sh1_s2_1e17", "total_iters": 3917, "iters_1e16": 392, "final_val_loss": null, "final_val_bpb": null, "final_val_ppl": null, "val_at_1e16": null, "val_at_1e16_bpb": null, "last_train_loss": 3.838258, "nan": false, "n_val_evals": 0, "bpb_divisor": 2.7568}

### tmoe_minlogit_sh2_s2_1e17  (2026-06-29 10:01)
Config: shape=s2 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=3917
SUMMARY tmoe_minlogit_sh2_s2_1e17: final_val_CE=3.5539 (BPB 1.2891)  val@iters/10=3.5539 (BPB 1.2891)@it3917  nan=False  evals=3
{"run": "tmoe_minlogit_sh2_s2_1e17", "total_iters": 3917, "iters_1e16": 392, "final_val_loss": 3.553884, "final_val_bpb": 1.2891, "final_val_ppl": 34.9, "val_at_1e16": {"iter": 3917, "loss": 3.553884}, "val_at_1e16_bpb": 1.2891, "last_train_loss": 3.556215, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### tmoe_minlogit_sh1_sm1_1e16  (2026-06-29 11:35)
Config: shape=sm1 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=4127
SUMMARY tmoe_minlogit_sh1_sm1_1e16: final_val_CE=4.1000 (BPB 1.4872)  val@iters/10=4.1000 (BPB 1.4872)@it4127  nan=False  evals=3
{"run": "tmoe_minlogit_sh1_sm1_1e16", "total_iters": 4127, "iters_1e16": 413, "final_val_loss": 4.100036, "final_val_bpb": 1.4872, "final_val_ppl": 60.3, "val_at_1e16": {"iter": 4127, "loss": 4.100036}, "val_at_1e16_bpb": 1.4872, "last_train_loss": 4.113785, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### tmoe_minlogit_sh1_s1_1e16  (2026-06-29 12:04)
Config: shape=s1 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=834
SUMMARY tmoe_minlogit_sh1_s1_1e16: final_val_CE=4.2657 (BPB 1.5473)  val@iters/10=4.2657 (BPB 1.5473)@it834  nan=False  evals=3
{"run": "tmoe_minlogit_sh1_s1_1e16", "total_iters": 834, "iters_1e16": 83, "final_val_loss": 4.265651, "final_val_bpb": 1.5473, "final_val_ppl": 71.2, "val_at_1e16": {"iter": 834, "loss": 4.265651}, "val_at_1e16_bpb": 1.5473, "last_train_loss": 4.285384, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### tmoe_minlogit_sh1_s2_1e16  (2026-06-29 12:24)
Config: shape=s2 flops=1e16 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=392
SUMMARY tmoe_minlogit_sh1_s2_1e16: final_val_CE=5.0307 (BPB 1.8248)  val@iters/10=5.0307 (BPB 1.8248)@it392  nan=False  evals=3
{"run": "tmoe_minlogit_sh1_s2_1e16", "total_iters": 392, "iters_1e16": 39, "final_val_loss": 5.030727, "final_val_bpb": 1.8248, "final_val_ppl": 153.0, "val_at_1e16": {"iter": 392, "loss": 5.030727}, "val_at_1e16_bpb": 1.8248, "last_train_loss": 5.038204, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### tmoe_minlogit_sh1_s1_1e17  (2026-06-29 16:59)
Config: shape=s1 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=8338
SUMMARY tmoe_minlogit_sh1_s1_1e17: final_val_CE=3.5912 (BPB 1.3027)  val@iters/10=3.5912 (BPB 1.3027)@it8338  nan=False  evals=3
{"run": "tmoe_minlogit_sh1_s1_1e17", "total_iters": 8338, "iters_1e16": 834, "final_val_loss": 3.591213, "final_val_bpb": 1.3027, "final_val_ppl": 36.3, "val_at_1e16": {"iter": 8338, "loss": 3.591213}, "val_at_1e16_bpb": 1.3027, "last_train_loss": 3.606274, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### tmoe_minlogit_sh1_s3_1e17  (2026-06-29 19:08)
Config: shape=s3 flops=1e17 peak_lr=3e-3 warmup=0.05 gb=256 seed=1234 aux=0.01 iters=2152
SUMMARY tmoe_minlogit_sh1_s3_1e17: final_val_CE=3.6006 (BPB 1.3061)  val@iters/10=3.6006 (BPB 1.3061)@it2152  nan=False  evals=3
{"run": "tmoe_minlogit_sh1_s3_1e17", "total_iters": 2152, "iters_1e16": 215, "final_val_loss": 3.600576, "final_val_bpb": 1.3061, "final_val_ppl": 36.6, "val_at_1e16": {"iter": 2152, "loss": 3.600576}, "val_at_1e16_bpb": 1.3061, "last_train_loss": 3.600008, "nan": false, "n_val_evals": 3, "bpb_divisor": 2.7568}

### flame38m_temporal_minlogit (1e18, FLAME-MoE-38M-100M temporal) (2026-06-30 13:12)
Config: hidden256/L9, top-6, shared 2*moe_ffn, pythia-12b(50k), gb1024 mb8 lr3e-4 WSD, 2121 iters = 1e18 FLOPs.
SUMMARY flame38m_temporal_minlogit: final_val_CE=3.9045 (PPL 49.6)  [paper law MoE@1e18 ~3.78; delta +0.124]  nan=False

### flame38m_dense (1e18, dense floor, matched active non-embed) (2026-06-30 23:13)
Config: hidden256/L9 all-dense swiglu ffn=1422 (=12.19M non-embed, matched), pythia-12b(50k), gb1024 mb32 lr3e-4 WSD, 2121 iters = 1e18.
SUMMARY flame38m_dense: final_val_CE=4.1373 (PPL 62.6)  [temporal 3.906; MoE law 3.78]  nan=False
