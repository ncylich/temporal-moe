#!/usr/bin/env bash
# The missing 1e19 corner: fine-grained (GRAIN=3, 192 experts) VANILLA MoE, exactly the config
# of temporal_fine_g3_1e19 with the residency scan off. s19opt, gb 1024, mb 8, lr 3e-3,
# 4278 iters (~9B tokens), pythia-12b 50k corpus (data/dclm_tokenized). Resumable via
# run.sh --load (checkpoint every iters/10). Rope fusion off (TE 2.16 env), a tiny numerics
# difference from the 2026-07 runs, documented.  Expected ~55-65h on one H100.
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "### dclm50k rebuild DONE" /workspace/rerun-logs/dclm50k_rebuild.out 2>/dev/null; do sleep 300; done
. scripts/env.sh
export TOKENIZER_MODEL=EleutherAI/pythia-12b DATA_DIR=$ROOT/data/dclm_tokenized
export CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=4 EXTRA_ARGS="--no-rope-fusion"
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
echo "### moe_fine_g3_1e19 START $(date -u +%H:%M)"
GRAIN=3 TEMPORAL=0 SHAPE=s19opt TARGET_FLOPS=1e19 PEAK_LR=3e-3 WARMUP_FRAC=0.05 GLOBAL_BATCH=1024 MICRO_BATCH=8 SEED=1234 \
  RUN_NAME=moe_fine_g3_1e19 scripts/residency/gpu_lease.sh bash experiments/run.sh
echo "### moe_fine_g3_1e19 rc=$? $(date -u +%H:%M)"
"$PY" analysis/parse_run.py results/phase0/runs/moe_fine_g3_1e19 2>/dev/null | grep '^SUMMARY'
echo "### moe_fine_g3_1e19 DONE $(date -u +%H:%M)"
