#!/usr/bin/env bash
# The one missing cell in the fraction-matched qwen table: IFEval at R32. Base and adapted
# through the same producer, same arms, so the delta is valid.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
L=scripts/residency/gpu_lease.sh
for pair in "qwen35_base_r32:/root/models/qwen35-35b-a3b" "qwen35_ce_rebuild_r32:/root/models/qwen35-rebuild-merged"; do
  REC=${pair%%:*}; M=${pair##*:}
  echo "### $REC IFEval R32 $(date -u +%H:%M)"
  $L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
    --model qwen35_instruct --path $M --arms free,R32 \
    --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
    --record-as $REC --tasks "ifeval=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
done
echo "### qwen R32 IFEval ALL DONE $(date -u +%H:%M)"
