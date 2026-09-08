#!/usr/bin/env bash
# Next qwen knob if the full pool misses the bar: expert-LoRA rank 32 (gemma's rank; qwen used 16), full pool, 0.5x, then surface.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_PRIO=4 TMOE_LR=3e-5 TMOE_KL_TEMP=2 TMOE_AUX_LOSS=revkl_full TMOE_ONLINE_MEM=0.65 TMOE_ONLINE_OFFLOAD=20 TMOE_ELORA_R=32
export TMOE_PROMPTS=/workspace/olmoe-adapt/data/fullpool_prompts.jsonl TMOE_QUOTA="mathlane_v2=2341,d5_fewshot=1183,domain8k=4958,codelane=2500"
D=/workspace/olmoe-adapt/data
echo "### qwen-r32 1/2 full pool 0.5x, expert-LoRA r=32 $(date -u +%H:%M)"
TMOE_NAME_SUFFIX=_fullpool_r32 bash /workspace/tmoe_qwen_online.sh scratch 4300000 16 256 > /workspace/rerun-logs/qwen_online_fullpool_r32.out 2>&1
rc=$?; echo "### qwen-r32 1/2 done rc=$rc $(date -u +%H:%M)"; [ "$rc" = 0 ] || exit 1
echo "### qwen-r32 2/2 surface $(date -u +%H:%M)"
scripts/residency/gpu_lease.sh bash /workspace/tmoe_deadband_surface.sh qwen 0 adapter:$D/qwen_ce_online_scratch_e16_fullpool_r32_adapter.pt qwen35_ce_online_fullpool_r32 > /workspace/rerun-logs/qwen_fullpool_r32_surface.out 2>&1
echo "### qwen-r32 ALL DONE rc=$? $(date -u +%H:%M)"
