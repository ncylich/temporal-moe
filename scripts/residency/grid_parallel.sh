#!/bin/bash
# Run one model's 12-cell grid as parallel per-TASK jobs, one GPU each.
#
# NOTE: this covers GSM8K, IFEval and MMLU only. The full surface is FIVE benchmarks --
# HumanEval needs its own channel-aware producer (humaneval_gemma.py / humaneval_think.py)
# and WritingBench needs wb_arm.sh with the local critic. Run those alongside, or an arm
# ends up reported on 3 of 5 cells.
#
# Safe because the batch-fair protocol constrains ARMS, not tasks: every arm of a cell must
# share one engine boot, which each job here preserves by running free,R8,R16 together.
# Tasks were only ever sequential because the driver loops over them in one process.
#
# NOT done by tensor-parallel: vllm_glue requires VLLM_ENABLE_V1_MULTIPROCESSING=0 so its
# patches reach the in-process engine core. TP spawns worker processes the patches would
# never reach, and the residency mask would silently not apply -- arms would come out
# identical and the grid would report no constraint damage because there was no constraint.
#
#     grid_parallel.sh gemma <merged-dir> <record>   # uses GPUs from $GPUS (default 1,2,3)
set -u
ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $ROOT
PY=/workspace/venv_vllm312/bin/python
LOG=${LOG_DIR:-/workspace/rerun-logs}
MODEL=${1:?gemma|qwen}; MERGED=${2:?merged dir}; REC=${3:?record}
IFS=',' read -ra G <<< "${GPUS:-2,3}"

case "$MODEL" in
  gemma) KEY=gemma4_instruct; ARMS=free,R8,R16; THINK=(); HE=(analysis/residency/humaneval_gemma.py) ;;
  qwen)  KEY=qwen35_instruct; ARMS=free,R8,R16
         THINK=(--think off --temperature 0.7 --top-p 0.8)
         HE=() ;;
  *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac

pids=()
CUDA_VISIBLE_DEVICES=${G[0]} $PY -u analysis/residency/instruct_genbench_vllm.py \
  --model $KEY --path $MERGED --arms $ARMS --record-as $REC "${THINK[@]}" \
  --tasks "gsm8k_cot_zeroshot=200" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94 \
  > $LOG/grid_${MODEL}_gsm8k.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=${G[1]} $PY -u analysis/residency/instruct_genbench_vllm.py \
  --model $KEY --path $MERGED --arms $ARMS --record-as $REC "${THINK[@]}" \
  --tasks "ifeval=200" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94 \
  > $LOG/grid_${MODEL}_ifeval.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=${G[2]:-${G[1]}} $PY -u analysis/residency/mmlu_gptoss.py \
  --model $KEY --path $MERGED --arms $ARMS --record-as ${REC}_dual "${THINK[@]}" \
  --gpu-mem 0.94 > $LOG/grid_${MODEL}_mmlu.log 2>&1 &
pids+=($!)
echo "### grid-$MODEL launched on GPUs ${G[*]} (gsm8k | ifeval | mmlu) $(date -u +%H:%M)"
wait "${pids[@]}"
echo "### grid-$MODEL PARALLEL DONE $(date -u +%H:%M)"
