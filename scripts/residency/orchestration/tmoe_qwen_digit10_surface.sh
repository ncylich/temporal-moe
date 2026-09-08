#!/usr/bin/env bash
# Full surface for a qwen adapter beyond GSM8K, on the same arms and recipes as the rebuild's
# cells (qwen35_ce_rebuild_full ifeval, _n_dual mmlu, _code humaneval+mbpp), so the digit-weight
# GSM8K gain can be checked for a price elsewhere.   tmoe_qwen_digit10_surface.sh [digit10]
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
NAME=${1:-digit10}; M=/root/models/qwen35-${NAME}-merged; L=scripts/residency/gpu_lease.sh
PY=/workspace/venv_vllm312/bin/python
S="--model qwen35_instruct --path $M --arms free,R8,R32 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5"
[ -d $M ] || { echo "### missing $M"; exit 2; }
echo "### qwen-$NAME surface 1/3 IFEval $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py $S --record-as qwen35_ce_${NAME}_full \
  --tasks "ifeval=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
echo "### qwen-$NAME surface 2/3 MMLU $(date -u +%H:%M)"
$L $PY -u analysis/residency/mmlu_gptoss.py --model qwen35_instruct --path $M --arms free,R8,R32 \
  --think off --record-as qwen35_ce_${NAME}_n_dual --gpu-mem 0.90
echo "### qwen-$NAME surface 3/3 HumanEval + MBPP $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py $S --record-as qwen35_ce_${NAME}_code \
  --tasks "mbpp_instruct=0,humaneval_instruct=0" --gen-cap 1536 --max-model-len 4096 --gpu-mem 0.90
echo "### qwen-$NAME surface ALL DONE $(date -u +%H:%M)"
