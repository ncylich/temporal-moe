#!/usr/bin/env bash
# RHO does not transfer across models (router logit scales differ), so qwen needs its own
# sweep before a surface run. Untrained base, R8 only, GSM8K n=1319, swap traffic measured.
#   tmoe_deadband_sweep_qwen.sh [rho ...]   default 0.25 0.5 0.75 1.0
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 TEMPORAL_COUNT_SWAPS=1
L=scripts/residency/gpu_lease.sh; B=/root/models/qwen35-35b-a3b
for RHO in "${@:-0.25 0.5 0.75 1.0}"; do
  echo "### qwen deadband sweep RHO=$RHO $(date -u +%H:%M)"
  TEMPORAL_RHO=$RHO $L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
    --model qwen35_instruct --path $B --arms R8 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
    --record-as qwen35_base_rho${RHO/./p}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
done
echo "### qwen deadband sweep DONE $(date -u +%H:%M)"
