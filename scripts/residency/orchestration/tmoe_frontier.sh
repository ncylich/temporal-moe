#!/bin/bash
# Fill in the R x swap-rate frontier at R32 (25% resident). R16 tolerates a 42% swap cut
# for 0.6 points where R8 loses 10.3, so resident memory and swap bandwidth substitute for
# one another. R32 says whether that keeps scaling -- i.e. whether the frontier is a smooth
# 2D surface the paper can plot, or whether R16 is already at the knee.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEMPORAL_COUNT_SWAPS=1
for RHO in 0 2.5 3.0 3.5; do
  echo "### frontier R32 RHO=$RHO $(date -u +%H:%M)"
  TEMPORAL_RHO=$RHO /workspace/venv_vllm312/bin/python -u \
    analysis/residency/instruct_genbench_vllm.py \
    --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms R32 \
    --record-as gemma4r32far_swapmeasure_rho${RHO/./p} --csv-name screening_genbench.csv \
    --tasks "gsm8k_cot_zeroshot=200" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
done
echo "### frontier R32 DONE $(date -u +%H:%M)"
