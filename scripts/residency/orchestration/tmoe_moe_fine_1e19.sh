#!/usr/bin/env bash
# SCHEDULE: July's 1e19 runs used lr 3e-4, WSD (warmup 43 = 1%, exp decay over the last 427, min 3e-5),
# NOT the 1e16 cosine recipe; matched here from temporal_fine_g3_1e19's argument table.
# The missing 1e19 corner: fine-grained (GRAIN=3, 192 experts) VANILLA MoE, exactly the config
# of temporal_fine_g3_1e19 with the residency scan off. s19opt, gb 1024, mb 8, lr 3e-3,
# 4278 iters (~9B tokens), pythia-12b 50k corpus (data/dclm_tokenized). Resumable via
# run.sh --load (checkpoint every iters/10). Rope fusion off (TE 2.16 env), a tiny numerics
# difference from the 2026-07 runs, documented.  Expected ~5.3 days (~110 s/it at balanced routing x 4278) on one H100.
set -uo pipefail; cd /workspace/temporal-moe
. scripts/env.sh
export TOKENIZER_MODEL=EleutherAI/pythia-12b DATA_DIR=/root/data/dclm_tokenized   # local-disk copy (byte-identical to data/dclm_tokenized)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=5 EXTRA_ARGS="--no-rope-fusion --moe-use-legacy-grouped-gemm --save-interval 200"
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
echo "### moe_fine_g3_1e19 START $(date -u +%H:%M)"
# Auto-resume: run.sh passes --load, Megatron resumes model+optimizer+rng+data position from the
# latest COMPLETE checkpoint (the pointer file is written only after a full save). Loop until the
# final checkpoint exists, at most 6 attempts, so a crash costs <=200 iterations (~1.2h), never the run.
FINAL=results/phase0/runs/moe_fine_g3_1e19/ckpt/iter_0004278
# checkpoint pruner: each save is 34 GB; keep only the two newest iter_* dirs (never the one being written)
( while true; do
    ls -d results/phase0/runs/moe_fine_g3_1e19/ckpt/iter_* 2>/dev/null | sort | head -n -2 | xargs -r rm -rf
    sleep 300
  done ) &
PRUNER=$!
trap 'kill $PRUNER 2>/dev/null' EXIT
for attempt in 1 2 3 4 5 6; do
  [ -d "$FINAL" ] && break
  echo "### moe_fine_g3_1e19 attempt $attempt $(date -u +%H:%M)"
  env MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1 GRAIN=3 TEMPORAL=0 SHAPE=s19opt TARGET_FLOPS=1e19 PEAK_LR=3e-4 WARMUP_FRAC=0.01 LR_DECAY_STYLE=WSD GLOBAL_BATCH=1024 MICRO_BATCH=16 SEED=1234 \
    RUN_NAME=moe_fine_g3_1e19 scripts/residency/gpu_lease.sh bash experiments/run.sh
  echo "### moe_fine_g3_1e19 attempt $attempt rc=$? $(date -u +%H:%M)"
  [ -d "$FINAL" ] || sleep 120
done
"$PY" analysis/parse_run.py results/phase0/runs/moe_fine_g3_1e19 2>/dev/null | grep '^SUMMARY'
echo "### moe_fine_g3_1e19 DONE $(date -u +%H:%M)"
