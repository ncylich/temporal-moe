#!/bin/bash
# gemma R64 = 50% resident, E/R=2 -- the low-E/R end of the frontier. The fitted power law
# (rate ~ 0.053*(E/R)^0.79) predicts ~0.092 swaps/token here. Extrapolating a fit beyond the
# range it was fitted on is where power laws usually break, so this is the honest test.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEMPORAL_COUNT_SWAPS=1
for RHO in 0 3.0 3.5 4.0; do
  echo "### frontier R64 RHO=$RHO $(date -u +%H:%M)"
  TEMPORAL_RHO=$RHO /workspace/venv_vllm312/bin/python -u \
    analysis/residency/instruct_genbench_vllm.py \
    --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms R64 \
    --record-as gemma4r64far_swapmeasure_rho${RHO/./p} --csv-name screening_genbench.csv \
    --tasks "gsm8k_cot_zeroshot=200" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
done
echo "### frontier R64 DONE $(date -u +%H:%M)"
