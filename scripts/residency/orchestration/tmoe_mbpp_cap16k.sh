#!/usr/bin/env bash
# Fair-budget re-measurement of MBPP cells that were budget-limited at 8192 (>= 5% of items at the
# cap, DATA_CONTRACT.md rule): same producer, same recipe, budget 16384, record suffixed _cap16k.
# The 8192 twin stays in the CSV un-edited. Runs after tmoe_mbpp_full.sh (one heavy process).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PATH=/workspace/venv_vllm312/bin:$PATH CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-4}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_ALLOW_CODE_EVAL=1
PY=/workspace/venv_vllm312/bin/python
run() { # model path arms tag extra...
  local model=$1 path=$2 arms=$3 tag=$4; shift 4
  [ -d "$path" ] || { echo "[skip] $tag: $path missing"; return; }
  grep -q "^$tag,.*,mbpp_chat," results/ablations/instruct_genbench_vllm.csv 2>/dev/null && { echo "[skip] $tag done"; return; }
  echo "### mbpp16k $tag arms=$arms $(date -u +%H:%M)"
  timeout -k 60 14400 scripts/residency/gpu_lease.sh $PY -u analysis/residency/mbpp_chat.py --model $model --path $path \
      --arms $arms --tag $tag --max-tokens 16384 --max-model-len 18432 "$@" > /workspace/rerun-logs/mbpp16k_$tag.out 2>&1
  echo "### mbpp16k $tag rc=$? $(date -u +%H:%M)"; grep -E "^\[mbpp_chat\] .*pass@1" /workspace/rerun-logs/mbpp16k_$tag.out | cut -c1-120
}
sel() { [ -z "${ONLY:-}" ] || [ "$ONLY" = "$1" ]; }
sel lfm && run lfm25_instruct /workspace/instruct-models/lfm25-8b-a1b free,R4 lfm25_instruct_mbpp_cap16k --think off
echo "### mbpp16k ALL DONE $(date -u +%H:%M)"
