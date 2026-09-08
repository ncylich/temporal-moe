#!/usr/bin/env bash
# Every Qwen3.5 record that has a stock-task MBPP row (mbpp_instruct: 3-shot, primed fence, first
# block, 1536 budget) re-measured under the unified producer (mbpp_chat: 500 problems, 8192 budget,
# last block whole), so the Qwen MBPP column is one protocol end to end. Same recipe as the stock
# rows (think off 0.7/0.8/pp1.5; think on 0.6/0.95/pp1.5), every adapter applied on the raw dir
# (the merged text-class dirs the digit and rebuild rows used are retired for the class confound).
# Skliar cells reuse the cache_bias walker env of tmoe_skliar*.sh. Record = stock record with
# _code -> _mbpp. Skips a record whose mbpp_chat row exists. ONLY=<tag> runs one.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PATH=/workspace/venv_vllm312/bin:$PATH CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-4}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_ALLOW_CODE_EVAL=1
PY=/workspace/venv_vllm312/bin/python; D=/workspace/olmoe-adapt/data; B=/root/models/qwen35-35b-a3b
OFF="--think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5"
ON="--think on --temperature 0.6 --top-p 0.95 --presence-penalty 1.5"
run() { # tag arms extra...   (extra may include --adapter FILE)
  local tag=$1 arms=$2; shift 2
  [ -z "${ONLY:-}" ] || [ "$ONLY" = "$tag" ] || return
  grep -q "^$tag,.*,mbpp_chat," results/ablations/instruct_genbench_vllm.csv 2>/dev/null && { echo "[skip] $tag done"; return; }
  echo "### mbpp500 $tag arms=$arms $(date -u +%H:%M)"
  timeout -k 60 14400 scripts/residency/gpu_lease.sh $PY -u analysis/residency/mbpp_chat.py --model qwen35_instruct --path $B \
      --arms $arms --tag $tag "$@" > /workspace/rerun-logs/mbpp500_$tag.out 2>&1
  echo "### mbpp500 $tag rc=$? $(date -u +%H:%M)"; grep -E "^\[mbpp_chat\] .*pass@1" /workspace/rerun-logs/mbpp500_$tag.out | cut -c1-120
}
for f in qwen_ce_rebuild_adapter.pt qwen_ce_digit10_adapter.pt qwen_ce_digit3_adapter.pt qwen_ce_online_scratch_e16_klT2_mix_adapter.pt \
         qwen_ce_online_scratch_e16_klT2_mix39_adapter.pt qwen_ce_online_online_scratch_e16_klT2_mix_e16_cont_adapter.pt \
         qwen_ce_online_scratch_e16_fullpool_adapter.pt qwen_ce_online_online_scratch_e16_fullpool_e16_full_adapter.pt \
         qwen35_remoe_lr1e-4_adapter.pt qwen_ce_online_scratch_e16_think_adapter.pt; do [ -f $D/$f ] || { echo "[missing] $D/$f"; exit 1; }; done
ad() { echo "--adapter $D/$1"; }
run qwen35_ce_rebuild_mbpp                    free,R8,R32 $OFF $(ad qwen_ce_rebuild_adapter.pt)
run qwen35_ce_digit10_mbpp                    free,R8,R32 $OFF $(ad qwen_ce_digit10_adapter.pt)
run qwen35_ce_digit3_mbpp                     free,R8,R32 $OFF $(ad qwen_ce_digit3_adapter.pt)
run qwen35_ce_online_klT2_mix_rho0_mbpp       R8,R32      $OFF $(ad qwen_ce_online_scratch_e16_klT2_mix_adapter.pt)
run qwen35_ce_online_klT2_mix39_rho0_mbpp     R8,R32      $OFF $(ad qwen_ce_online_scratch_e16_klT2_mix39_adapter.pt)
run qwen35_ce_online_klT2_mix_cont_rho0_mbpp  R8,R32      $OFF $(ad qwen_ce_online_online_scratch_e16_klT2_mix_e16_cont_adapter.pt)
run qwen35_ce_online_fullpool_half_rho0_mbpp  R8,R32      $OFF $(ad qwen_ce_online_scratch_e16_fullpool_adapter.pt)
run qwen35_ce_online_fullpool_full_rho0_mbpp  free,R8,R32 $OFF $(ad qwen_ce_online_online_scratch_e16_fullpool_e16_full_adapter.pt)
run qwen35_remoe_lr1e-4_rho0_mbpp             free        $OFF $(ad qwen35_remoe_lr1e-4_adapter.pt)
CB="TEMPORAL_WALKER=cache_bias TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1"
( export $CB TEMPORAL_CB_C=128 TEMPORAL_CB_LAMBDA=0;   run qwen35_skliar_C128_lam0_mbpp   R8 $OFF )
( export $CB TEMPORAL_CB_C=128 TEMPORAL_CB_LAMBDA=0.4; run qwen35_skliar_C128_lam0p4_mbpp R8 $OFF )
( export $CB TEMPORAL_CB_C=8   TEMPORAL_CB_LAMBDA=0.4; run qwen35_skliar_C8_lam0p4_mbpp   R8 $OFF )
run qwen35_ce_online_think_mbpp               free,R8,R32 $ON  $(ad qwen_ce_online_scratch_e16_think_adapter.pt)
run qwen35_think_on_fulln_mbpp                free,R8,R32 $ON
echo "### mbpp500 qwen ALL DONE $(date -u +%H:%M)"
