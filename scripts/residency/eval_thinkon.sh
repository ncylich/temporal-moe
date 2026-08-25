#!/bin/bash
# B2: evaluate a think-on adapter -- merge, verify, then a think-ON grid at a budget that
# does not truncate.
#
# The question B2 exists to answer is not just accuracy. gemma_adapt_RESULTS lists think-on
# as an open item, and 01-findings calls it "an open mechanistic test of why the recipe
# works": does adaptation SHRINK the constrained thinking-lengthening? So the grid must be
# generous enough that lengths are measurable rather than clipped -- A4 showed gemma
# think-on IFEval needs 16384 to reach 0% cap-hit, and think-on trajectories ran to 8192.
set -euo pipefail
ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $ROOT

$ROOT/scripts/residency/disk_budget.sh || exit 3
# SPACE CHECK: a merge writes a full model copy (49-67GB). /dev/shm filling mid-write kills
# the merge with a bare "No space left on device" from safetensors, after minutes of work.
free_kb=$(df -k /root/models | tail -1 | awk '{print $4}')
if [ "$free_kb" -lt 75000000 ]; then
  echo "### ABORT: /root/models has only $((free_kb/1024/1024))GB free; a merge needs ~70GB." >&2
  echo "### Delete finished merged checkpoints (they are reproducible from adapter+base)." >&2
  exit 3
fi
TPY=/workspace/venv_fla/bin/python
VPY=/workspace/venv_vllm312/bin/python
LOG=${LOG_DIR:-/workspace/rerun-logs}
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}

case "${1:?usage: eval_thinkon.sh gemma|qwen}" in
  gemma) BASE=/dev/shm/gemma4-26b-it;  M=/root/models/gemma4-thinkon-merged
         A=/workspace/olmoe-adapt/data/gemma_ce_thinkon_adapter.pt
         KEY=gemma4_instruct; REC=gemma4_ce_thinkon; ARMS=free,R8,R16; RANK=32 ;;
  qwen)  BASE=/dev/shm/qwen35-35b-a3b; M=/root/models/qwen35-thinkon-merged
         A=/workspace/olmoe-adapt/data/qwen_ce_thinkon_adapter.pt
         KEY=qwen35_instruct; REC=qwen35_ce_thinkon; ARMS=free,R8,R32; RANK=16 ;;
esac
export CUDA_VISIBLE_DEVICES=${GPU:-1}
[ -s "$A" ] || { echo "### $1-thinkon ABORT: adapter missing"; exit 2; }

if [ ! -d "$M" ]; then
  echo "### $1-thinkon MERGE $(date -u +%H:%M)"
  if [ "$1" = "qwen" ]; then
    ADAPTER_PATH=$A DST_PATH=$M SRC_PATH=$BASE $TPY analysis/residency/qwen_ce_patch.py
  else
    $TPY analysis/residency/train_gemma_ce.py --model $BASE --family gemma4 --no-unsloth \
      --traj gemma4_d7think_seq8192 --max-seq 8192 --expert-lora-r $RANK --out $A --merge-out $M
    cp $BASE/processor_config.json $M/ 2>/dev/null || true
  fi
fi
echo "### $1-thinkon MERGE DONE $(date -u +%H:%M)"
$TPY analysis/residency/verify_merge.py --base $BASE --merged $M
echo "### $1-thinkon VERIFY DONE $(date -u +%H:%M)"

echo "### $1-thinkon GRID think-on @16384 $(date -u +%H:%M)"
$VPY -u analysis/residency/instruct_genbench_vllm.py --model $KEY --path $M \
    --arms $ARMS --record-as $REC --think on --tasks "ifeval=200" \
    --gen-cap 16384 --max-model-len 17920 --gpu-mem 0.94 2>&1 | tee $LOG/thinkon_grid_${1}.log
echo "### $1-thinkon ALL DONE $(date -u +%H:%M)"
