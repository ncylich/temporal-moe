#!/usr/bin/env bash
# The on-policy winner under the eviction deadband rho=0.5: GSM8K n=1319 R8,R16 with swap counting (10 min), not a full surface.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
export TEMPORAL_RHO=0.5 TEMPORAL_COUNT_SWAPS=1
echo "### winner deadband rho=0.5 GSM8K n=1319 R8,R16 $(date -u +%H:%M)"
scripts/residency/gpu_lease.sh /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path /dev/shm/gemma4-26b-it \
  --adapter /workspace/olmoe-adapt/data/gemma_ce_online_scratch_e16_klT2_adapter.pt --arms R8,R16 --record-as gemma4_ce_online_scratch_e16_klT2_rho0p5_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### winner deadband DONE rc=$? $(date -u +%H:%M)"
