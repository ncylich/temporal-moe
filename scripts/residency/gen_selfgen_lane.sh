#!/bin/bash
# Two-stage self-generated lane, matching the original selfmath_v2_3000.jsonl shape.
#   stage A: the model AUTHORS problems from topic seeds  -> those become pool prompts
#   stage B: build_d7_prompts.py screens and mixes them; trajectories are the model
#            SOLVING its own problems, which is the thing the original trained on.
# Only stage A lives here.
#
#     GPU=3 gen_selfgen_lane.sh gemma math 2341
set -euo pipefail
ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd $ROOT
PY=/workspace/venv_vllm312/bin/python
DATA=/workspace/olmoe-adapt/data
LOG=${LOG_DIR:-/workspace/rerun-logs}

case "${1:?usage: gen_selfgen_lane.sh gemma|qwen LANE N}" in
  gemma) MODEL=/dev/shm/gemma4-26b-it ;;
  qwen)  MODEL=/dev/shm/qwen35-35b-a3b ;;
  *) echo "unknown model: $1" >&2; exit 2 ;;
esac
LANE=${2:?lane}; N=${3:?n}
export CUDA_VISIBLE_DEVICES=${GPU:-3}

$PY analysis/residency/build_selfgen_lanes.py --lane $LANE --n $N --out $DATA
echo "### selfgen $LANE authoring pass ($1) $(date -u +%H:%M)"
$PY -u analysis/residency/gen_traj_vllm.py --model $MODEL \
    --tag selfgen_${LANE}_raw --prompts $DATA/selfgen_${LANE}_${N}.jsonl \
    --think off --max-new 512 --max-prompt-tok 1024 --gpu-mem 0.94 \
    --out /workspace/instruct-traj 2>&1 | tee $LOG/selfgen_${LANE}.log
echo "### selfgen $LANE DONE $(date -u +%H:%M)"
