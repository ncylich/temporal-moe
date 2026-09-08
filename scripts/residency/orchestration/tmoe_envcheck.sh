#!/usr/bin/env bash
# Does the restored environment reproduce a known result? The machine, template, python
# and GPU count all changed. gemma4_ce_rebuild scored R8 81.9 / free 86.8 on GSM8K n=1319
# before the reboot; if this run lands elsewhere, every new number is suspect and the
# environment is the first thing to fix -- not the science.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path /root/models/gemma4-rebuild-merged --arms free,R8 \
  --record-as gemma4_ce_rebuild_envcheck --tasks "gsm8k_cot_zeroshot=0" \
  --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### envcheck DONE $(date -u +%H:%M)"
