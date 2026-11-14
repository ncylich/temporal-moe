#!/bin/bash
# paper/TODO.md line 189: Section 7 compares released vs adapted gemma4 on HumanEval at the
# 1536-token budget the adaptation runs share, because the adapted model was never run
# higher. Section 6 reports the same RELEASED cell at 8192, where it reads -4.3 instead of
# -5.5. So the two sections quote different numbers for what looks like one cell. Running
# the adapted checkpoint at 8192 makes them comparable.
set -euo pipefail
cd /workspace/temporal-moe
TAG=$1; M=$2
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "### he8192 $TAG $(date -u +%H:%M)"
exec /workspace/venv_vllm312/bin/python -u analysis/residency/humaneval_gemma.py \
  --path "$M" --arms free,R8,R16 --tag "${TAG}_he8192" \
  --max-tokens 8192 --max-model-len 9216
