#!/usr/bin/env bash
# Eviction deadband (ours, Skliar-inspired: swap only if the wanted expert beats the worst
# resident by more than RHO) on the FULL surface at R8 and the loose arm, with swap traffic
# measured. The free arm has no residency so RHO cannot touch it: it is not re-run.
#   tmoe_deadband_surface.sh <gemma|qwen> <rho> <merged-path> <record-prefix>
#   e.g. tmoe_deadband_surface.sh gemma 0.5 /dev/shm/gemma4-26b-it gemma4_base
#        tmoe_deadband_surface.sh qwen  0.5 /root/models/qwen35-digit3-merged qwen35_ce_digit3
set -euo pipefail
cd /workspace/temporal-moe
MODEL=$1; RHO=$2; M=$3; PFX=$4; TAG=${PFX}_rho${RHO/./p}
# third arg: a merged checkpoint dir, or adapter:<file> (engine boots from the base and applies it, gemma only)
ADAPTER=""; case "$M" in adapter:*) ADAPTER="--adapter ${M#adapter:}"; M=/dev/shm/gemma4-26b-it;; esac
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
export TEMPORAL_RHO=$RHO TEMPORAL_COUNT_SWAPS=1
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
[ -d "$M" ] || { echo "### missing $M"; exit 2; }
if [ "$MODEL" = gemma ]; then
  ARMS=R8,R16; G="--model gemma4_instruct --path $M $ADAPTER --arms $ARMS"; ML=4096
  echo "### $TAG GSM8K n=1319 $(date -u +%H:%M)"
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as ${TAG}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len $ML --gpu-mem 0.90
  echo "### $TAG IFEval $(date -u +%H:%M)"
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as ${TAG}_full --tasks "ifeval=0" --gen-cap 2048 --max-model-len $ML --gpu-mem 0.90
  echo "### $TAG MMLU $(date -u +%H:%M)"
  $L $PY -u analysis/residency/mmlu_gptoss.py $G --record-as ${TAG}_full_dual --gpu-mem 0.90
  echo "### $TAG HumanEval@8192 $(date -u +%H:%M)"
  $L $PY -u analysis/residency/humaneval_gemma.py --path $M $ADAPTER --arms $ARMS --tag ${TAG}_he8192 --max-tokens 8192 --max-model-len 9216
  echo "### $TAG MBPP@8192 $(date -u +%H:%M)"
  $L $PY -u analysis/residency/mbpp_gemma.py --path $M $ADAPTER --arms $ARMS --tag ${TAG}_m8192 --max-tokens 8192 --max-model-len 9216 --gpu-mem 0.90
  echo "### $TAG WritingBench $(date -u +%H:%M)"
  export GPU=0 TMOE_ADAPTER="${ADAPTER#--adapter }"; $L scripts/residency/wb_arm.sh $M $TAG R8,R16
else
  ARMS=R8,R32; Q="--model qwen35_instruct --path $M --arms $ARMS --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5"
  echo "### $TAG GSM8K n=1319 $(date -u +%H:%M)"
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $Q --record-as ${TAG}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
  echo "### $TAG IFEval $(date -u +%H:%M)"
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $Q --record-as ${TAG}_full --tasks "ifeval=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
  echo "### $TAG MMLU $(date -u +%H:%M)"
  $L $PY -u analysis/residency/mmlu_gptoss.py --model qwen35_instruct --path $M --arms $ARMS --think off --record-as ${TAG}_n_dual --gpu-mem 0.90
  echo "### $TAG HumanEval + MBPP $(date -u +%H:%M)"
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $Q --record-as ${TAG}_code --tasks "mbpp_instruct=0,humaneval_instruct=0" --gen-cap 1536 --max-model-len 4096 --gpu-mem 0.90
fi
echo "### $TAG SURFACE DONE $(date -u +%H:%M)"
