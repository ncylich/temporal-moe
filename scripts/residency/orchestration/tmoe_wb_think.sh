#!/usr/bin/env bash
# WritingBench on the two think-on adapters (thinking-mode generation, 8192 cap), after the
# think-off finals' WB retry. Each record is self-contained: free + constrained arms, same mode.
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "wb finals RETRY ALL DONE" /workspace/rerun-logs/wb_finals.out 2>/dev/null; do sleep 300; done
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=6 TMOE_WB_THINK=on TMOE_WB_MAXNEW=8192
L=scripts/residency/gpu_lease.sh; D=/workspace/olmoe-adapt/data
echo "### wb think-on: gemma then qwen $(date -u +%H:%M)"
TMOE_ADAPTER=$D/gemma_ce_online_scratch_e16_think_adapter.pt $L scripts/residency/wb_arm.sh /dev/shm/gemma4-26b-it gemma4_ce_online_think R8,R16
echo "### wb think gemma done rc=$? $(date -u +%H:%M)"
TMOE_ADAPTER=$D/qwen_ce_online_scratch_e16_think_adapter.pt $L scripts/residency/wb_arm.sh /root/models/qwen35-35b-a3b qwen35_ce_online_think R8,R32
echo "### wb think qwen done rc=$? $(date -u +%H:%M)"
echo "### wb think ALL DONE $(date -u +%H:%M)"
