#!/usr/bin/env bash
# Validation of the speed config on July's REAL schedule (lr 3e-4, WSD, 1% warmup), 40 iterations:
#  A : uninterrupted 40 iters (reference)
#  B : same run, but exit after iter 20 with a checkpoint, then resume to 40
# Checks: (1) rate on evolving routing, (2) LB loss in July's early range (~1.0-1.4),
#         (3) B's iters 25..40 losses == A's (checkpoint/resume exactness), (4) lm loss vs July run.
set -uo pipefail; cd /workspace/temporal-moe
. scripts/env.sh
export TOKENIZER_MODEL=EleutherAI/pythia-12b DATA_DIR=/root/data/dclm_tokenized
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=4 HF_TOKEN=$(cat /root/.cache/huggingface/token)
COMMON="--no-rope-fusion --moe-use-legacy-grouped-gemm --train-iters 40 --eval-iters 1 --eval-interval 1000 --log-interval 5"
launch() { # $1 name, $2 extra args
  env MOE_TORCH_GMM=${MOE_TORCH_GMM:-1} MOE_NO_LAYER_LOG=1 MOE_PERMUTE_FUSION=1 GRAIN=3 TEMPORAL=0 SHAPE=s19opt TARGET_FLOPS=1e19 PEAK_LR=3e-4 WARMUP_FRAC=0.01 LR_DECAY_STYLE=WSD \
    GLOBAL_BATCH=1024 MICRO_BATCH=16 SEED=1234 RUN_NAME=$1 EXTRA_ARGS="$COMMON $2" timeout -k 30 1500 scripts/residency/gpu_lease.sh bash experiments/run.sh
}
show() { grep -E " iteration +[0-9]+/" results/phase0/runs/$1/train.log | grep -oE "iteration +[0-9]+/|elapsed time per iteration \(ms\): [0-9.]+|lm loss: [0-9.E+-]+|load_balancing_loss: [0-9.E+-]+" | paste - - - - | sed "s/^/[$1] /"; }
echo "### validate A $(date -u +%H:%M)"; rm -rf results/phase0/runs/val1e19_A; launch val1e19_A "--save-interval 100000"; show val1e19_A
echo "### validate B1 (exit at 20 with ckpt) $(date -u +%H:%M)"; rm -rf results/phase0/runs/val1e19_B; launch val1e19_B "--save-interval 20 --exit-interval 20"; show val1e19_B
echo "### validate B2 (resume 20->40) $(date -u +%H:%M)"; launch val1e19_B "--save-interval 20"; show val1e19_B
echo "[validate july temporal_fine_g3_1e19]"; grep -E " iteration +(10|20|30|40)/" /workspace/tok16k_orig/hfruns/temporal_fine_g3_1e19/train.log | grep -oE "iteration +[0-9]+/|lm loss: [0-9.E+-]+|load_balancing_loss: [0-9.E+-]+" | paste - - -
echo "### validate DONE $(date -u +%H:%M)"
