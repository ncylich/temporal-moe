#!/bin/bash
# Does the swap-rate insensitivity hold beyond GSM8K? If quality is flat at 1/14th the
# swap traffic on math but not on code or instruction-following, the bandwidth claim has
# to be scoped to math. RHO=0 cells for these surfaces are already measured.
#   tmoe_skliar_surface.sh <what>   where what = code | instr
set -euo pipefail
cd /workspace/temporal-moe
WHAT=$1
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEMPORAL_RHO=1.0
B=/dev/shm/gemma4-26b-it
if [ "$WHAT" = "code" ]; then
  echo "### skliar-surface MBPP RHO=1.0 $(date -u +%H:%M)"
  /workspace/venv_vllm312/bin/python -u analysis/residency/mbpp_gemma.py \
    --path $B --arms R8 --tag gemma4_skliar_rho1p0 --max-model-len 4096 --gpu-mem 0.90
  echo "### skliar-surface HumanEval RHO=1.0 $(date -u +%H:%M)"
  /workspace/venv_vllm312/bin/python -u analysis/residency/humaneval_gemma.py \
    --path $B --arms R8 --tag gemma4_skliar_rho1p0
else
  echo "### skliar-surface IFEval RHO=1.0 $(date -u +%H:%M)"
  /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
    --model gemma4_instruct --path $B --arms R8 --record-as gemma4_skliar_rho1p0_full \
    --tasks "ifeval=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
  echo "### skliar-surface MMLU RHO=1.0 $(date -u +%H:%M)"
  /workspace/venv_vllm312/bin/python -u analysis/residency/mmlu_gptoss.py \
    --model gemma4_instruct --path $B --arms R8 \
    --record-as gemma4_skliar_rho1p0_full_dual --gpu-mem 0.90
fi
echo "### skliar-surface $WHAT DONE $(date -u +%H:%M)"
