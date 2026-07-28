#!/bin/bash
# Master serial runner for the G=3 fine-grained sweeps: 4 cells x 3 shapes = 12 runs, one GPU.
# MoE then temporal, 1e16 then 1e17. All EVAL_AT_END (single-budget parabolas; dedicated 1e16 runs).
# drive.sh is idempotent (skips runs whose final ckpt exists), so this is safe to re-launch.
# Launch detached:  nohup bash experiments/isoflop_1e16_1e17/g3_run_all.sh > results/phase0/g3_run_all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."

# ---- common env (absolute paths required: run.sh cd's into Megatron-LM) ----
export TOKENIZER_MODEL=$ROOT/data/tok16k
export DATA_DIR=$ROOT/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7600   # measured: 16k tok bytes/token=3.9819 on dclm (baseline 2.7568)
export GRAIN=3
export EVAL_AT_END=1
export HF_TOKEN=${HF_TOKEN:-}
# Speed: mb=64 (grad-accum 8->4, ~1.6x). mb>=128 OOMs — G=3's top-k=18 triples the dispatched-expert
# activation (mb*seq*18 rows through 192 experts), so 64 is the max divisor of gb=256 that fits 80GB.
# Both MoE and temporal use the SAME mb=64 per shape, so the MoE-vs-temporal deltas stay clean.
# expandable_segments cuts allocator fragmentation (numerically neutral; the probes needed it to fit).
export MICRO_BATCH=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== G3 RUN ALL START $(date) ==="

echo "--- [1/4] MoE @1e16 ---"
bash experiments/isoflop_1e16_1e17/drive.sh experiments/isoflop_1e16_1e17/g3_moe_1e16.txt

echo "--- [2/4] MoE @1e17 ---"
bash experiments/isoflop_1e16_1e17/drive.sh experiments/isoflop_1e16_1e17/g3_moe_1e17.txt

echo "--- [3/4] Temporal @1e16 ---"
TEMPORAL=1 TEMPORAL_EVICT=min_logit bash experiments/isoflop_1e16_1e17/drive.sh experiments/isoflop_1e16_1e17/g3_tmoe_1e16.txt

echo "--- [4/4] Temporal @1e17 ---"
TEMPORAL=1 TEMPORAL_EVICT=min_logit bash experiments/isoflop_1e16_1e17/drive.sh experiments/isoflop_1e16_1e17/g3_tmoe_1e17.txt

echo "=== G3 RUN ALL DONE $(date) ==="
