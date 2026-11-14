#!/usr/bin/env bash
# Full honest prompt pool (10,982 prompts: general 4958, math 2341, fewshot 1183, code 2500), winning recipe
# (lr 3e-5, KL T=2, 16x256). Stage 1: from scratch to 4.3M sampled tokens (0.5x coverage) -> GSM8K + full surface.
# Stage 2: resume (Adam state + prompt cursor carried) to 8.6M (1.0x) -> GSM8K + full surface.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_PRIO=4 TMOE_LR=3e-5 TMOE_KL_TEMP=2 TMOE_AUX_LOSS=revkl_full TMOE_ONLINE_MEM=0.65 TMOE_ONLINE_OFFLOAD=20
export TMOE_PROMPTS=/workspace/olmoe-adapt/data/fullpool_prompts.jsonl TMOE_QUOTA="mathlane_v2=2341,d5_fewshot=1183,domain8k=4958,codelane=2500"
D=/workspace/olmoe-adapt/data; S=scripts/residency/gpu_lease.sh
echo "### fullpool 1/4 stage 1: from scratch to 4.3M (0.5x coverage) $(date -u +%H:%M)"
TMOE_NAME_SUFFIX=_fullpool bash /workspace/tmoe_qwen_online.sh scratch 4300000 16 256 > /workspace/rerun-logs/qwen_online_fullpool.out 2>&1
rc=$?; echo "### fullpool 1/4 done rc=$rc $(date -u +%H:%M)"; [ "$rc" = 0 ] || exit 1
echo "### fullpool 2/4 surface at 0.5x $(date -u +%H:%M)"
$S bash /workspace/tmoe_deadband_surface.sh qwen 0 adapter:$D/qwen_ce_online_scratch_e16_fullpool_adapter.pt qwen35_ce_online_fullpool_half > /workspace/rerun-logs/qwen_fullpool_half_surface.out 2>&1
echo "### fullpool 2/4 done rc=$? $(date -u +%H:%M)"
echo "### fullpool 3/4 stage 2: resume to 8.6M (1.0x coverage) $(date -u +%H:%M)"
TMOE_NAME_SUFFIX=_full bash /workspace/tmoe_qwen_online.sh online_scratch_e16_fullpool 4300000 16 256 > /workspace/rerun-logs/qwen_online_fullpool2.out 2>&1
rc=$?; echo "### fullpool 3/4 done rc=$rc $(date -u +%H:%M)"; [ "$rc" = 0 ] || exit 1
echo "### fullpool 4/4 surface at 1.0x $(date -u +%H:%M)"
$S bash /workspace/tmoe_deadband_surface.sh qwen 0 adapter:$D/qwen_ce_online_online_scratch_e16_fullpool_e16_full_adapter.pt qwen35_ce_online_fullpool_full > /workspace/rerun-logs/qwen_fullpool_full_surface.out 2>&1
echo "### fullpool ALL DONE rc=$? $(date -u +%H:%M)"
