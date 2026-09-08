#!/bin/bash
# qwen3.5 code surface, think-off. qwen pre-closes its think block in non-thinking mode,
# so the stock instruct tasks extract correctly here -- the channel-aware producers are a
# gemma4 requirement, not a general one. Smoke first (limit 25) to confirm the score is
# credible before committing to the full run, the check that caught mbpp_instruct scoring
# gemma at 0.28 against its true 0.84.
set -euo pipefail
cd /workspace/temporal-moe
REC=$1; MPATH=$2; LIM=${3:-25}
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model qwen35_instruct --path "$MPATH" --arms free,R8,R32 --record-as "$REC" \
  --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
  --tasks "mbpp_instruct=${LIM},humaneval_instruct=0" \
  --gen-cap 1536 --max-model-len 4096 --gpu-mem 0.90
