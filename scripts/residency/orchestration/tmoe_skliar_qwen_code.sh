#!/usr/bin/env bash
# The qwen Skliar surfaces lacked HF_ALLOW_CODE_EVAL for lm-eval's code tasks: run the two missing code stages.
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "skliar qwen DONE" /workspace/rerun-logs/skliar_qwen.out 2>/dev/null; do sleep 60; done
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
export TEMPORAL_WALKER=cache_bias TEMPORAL_CB_C=128 TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1 TMOE_PRIO=3
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; M=/root/models/qwen35-35b-a3b
G="--model qwen35_instruct --path $M --arms R8 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5"
for LAM in 0 0.4; do
  export TEMPORAL_CB_LAMBDA=$LAM; TAG=qwen35_skliar_C128_lam${LAM/./p}
  echo "### skliar qwen code lambda=$LAM $(date -u +%H:%M)"
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as ${TAG}_code --tasks "mbpp_instruct=0,humaneval_instruct=0" --gen-cap 1536 --max-model-len 4096 --gpu-mem 0.90
done
echo "### skliar qwen code DONE $(date -u +%H:%M)"
