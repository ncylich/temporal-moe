#!/bin/bash
# qwen: merge -> verify -> THINK-OFF grid, in one place.
#
# merge_and_remeasure.sh cannot be used for qwen's grid: its instruct_genbench_vllm call
# leaves --think at the default, which for qwen3.5 means thinking ON. The adapter trains on
# think-OFF trajectories, so a think-on grid blows the 2048 cap on 150 of 200 items and
# reads IFEval 0.27 against a true value near 0.85. This runs the merge and verification
# from that script's logic, then hands the grid to remeasure_qwen.sh, which is think-off.
set -euo pipefail
ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
cd $ROOT
TPY=/workspace/venv_fla/bin/python
LOG=${LOG_DIR:-/workspace/rerun-logs}
BASE=/dev/shm/qwen35-35b-a3b
MERGED=/dev/shm/qwen35-selfgen-merged
ADAPTER=/workspace/olmoe-adapt/data/qwen_ce_selfgen_adapter.pt
export CUDA_VISIBLE_DEVICES=${GPU:-3}

[ -s "$ADAPTER" ] || { echo "### qwen ABORT: adapter missing $ADAPTER"; exit 2; }
if [ ! -d "$MERGED" ]; then
  echo "### qwen MERGE $(date -u +%H:%M)"
  ADAPTER_PATH=$ADAPTER DST_PATH=$MERGED SRC_PATH=$BASE \
    $TPY analysis/residency/qwen_ce_patch.py 2>&1 | tee $LOG/merge_qwen_selfgen.log
fi
echo "### qwen MERGE DONE $(date -u +%H:%M)"
$TPY analysis/residency/verify_merge.py --base $BASE --merged $MERGED \
    2>&1 | tee $LOG/verify_qwen_selfgen.log
echo "### qwen VERIFY DONE $(date -u +%H:%M)"
GPU=${GPU:-3} exec $ROOT/scripts/residency/remeasure_qwen.sh
