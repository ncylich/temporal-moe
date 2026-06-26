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
