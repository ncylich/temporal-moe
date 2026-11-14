#!/usr/bin/env bash
# MBPP as a standard Section 6 surface: the full 500 problems at the 8192-token budget (OLMoE:
# 3328 within its 4096 window) for every released base model at free + its residency cells, through
# the one producer (analysis/residency/mbpp_chat.py, task mbpp_chat), rows into the authoritative
# CSV. gemma4's recorded mbpp_gemma rows use the identical prompt and rule and are kept; Qwen is
# re-run here because its recorded rows came from the stock lm_eval prompt.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PATH=/workspace/venv_vllm312/bin:$PATH CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-4}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_ALLOW_CODE_EVAL=1
PY=/workspace/venv_vllm312/bin/python
run() { # model path arms tag extra...
  local model=$1 path=$2 arms=$3 tag=$4; shift 4
  [ -d "$path" ] || { echo "[skip] $tag: $path missing"; return; }
  grep -q "^$tag,.*,mbpp_chat," results/ablations/instruct_genbench_vllm.csv 2>/dev/null && { echo "[skip] $tag done"; return; }
  echo "### mbpp500 $tag arms=$arms $(date -u +%H:%M)"
  timeout -k 60 14400 scripts/residency/gpu_lease.sh $PY -u analysis/residency/mbpp_chat.py --model $model --path $path \
      --arms $arms --tag $tag "$@" > /workspace/rerun-logs/mbpp500_$tag.out 2>&1
  echo "### mbpp500 $tag rc=$? $(date -u +%H:%M)"; grep -E "^\[mbpp_chat\] .*pass@1" /workspace/rerun-logs/mbpp500_$tag.out | cut -c1-120
}
[ -n "${ONLY:-}" ] && { echo "[only] $ONLY"; }
sel() { [ -z "${ONLY:-}" ] || [ "$ONLY" = "$1" ]; }
sel olmoe   && run olmoe_instruct /workspace/instruct-models/olmoe-0125-instruct free,R8 olmoe_instruct_mbpp --think default
sel lfm     && run lfm25_instruct /workspace/instruct-models/lfm25-8b-a1b free,R4 lfm25_instruct_mbpp --think off
sel qwen    && run qwen35_instruct /workspace/instruct-models/qwen35-35b-a3b-instruct free,R8,R32 qwen35_instruct_mbpp --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5
sel gptoss20 && run gptoss_20b /workspace/instruct-models/gpt-oss-20b free,R4 gptoss_20b_mbpp --think default
sel gptoss120 && run gptoss_120b /workspace/instruct-models/gpt-oss-120b free,R4,R16 gptoss_120b_mbpp --think default --gpu-mem 0.9
echo "### mbpp500 ALL DONE $(date -u +%H:%M)"
