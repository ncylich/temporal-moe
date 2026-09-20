#!/usr/bin/env bash
# MBPP sub-sample (40 problems, full 8192 budget) on every Section 6 model at free + its
# residency cells, for the coherence / parsing review before the 500-problem runs. Qwen3.5 ships
# a presence penalty, so it runs twice: TEMPORAL_FAST_PP=1 (the fast processor) and 0 (vLLM
# native), same seed, for the equivalence check. Qwen's non-thinking recipe (card, not the
# shipped generation_config) is temperature 0.7, top_p 0.8, presence_penalty 1.5, the settings
# every recorded Qwen row used. Rows go to mbpp_subsample.csv, never the
# authoritative CSV. Skips models whose weights are not on disk yet.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PATH=/workspace/venv_vllm312/bin:$PATH CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-4}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_ALLOW_CODE_EVAL=1
PY=/workspace/venv_vllm312/bin/python; N=${N:-40}
run() { # model path arms tag extra...
  local model=$1 path=$2 arms=$3 tag=$4; shift 4
  [ -d "$path" ] || { echo "[skip] $tag: $path missing"; return; }
  grep -q "^$tag," results/ablations/mbpp_subsample.csv 2>/dev/null && { echo "[skip] $tag done"; return; }
  echo "### mbpp40 $tag arms=$arms fast_pp=${TEMPORAL_FAST_PP:-1} $(date -u +%H:%M)"
  timeout -k 60 5400 scripts/residency/gpu_lease.sh $PY -u analysis/residency/mbpp_chat.py --model $model --path $path \
      --arms $arms --limit $N --tag $tag --csv-name mbpp_subsample.csv "$@" > /workspace/rerun-logs/mbpp40_$tag.out 2>&1
  echo "### mbpp40 $tag rc=$? $(date -u +%H:%M)"; grep -E "^\[mbpp_chat\] .*pass@1" /workspace/rerun-logs/mbpp40_$tag.out | cut -c1-120
}
run olmoe_instruct /workspace/instruct-models/olmoe-0125-instruct free,R8 olmoe_instruct_mbpp40 --think default
run lfm25_instruct /workspace/instruct-models/lfm25-8b-a1b free,R4 lfm25_instruct_mbpp40 --think off
TEMPORAL_FAST_PP=1 run qwen35_instruct /workspace/instruct-models/qwen35-35b-a3b-instruct free,R8,R32 qwen35_instruct_mbpp40_fastpp --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5
TEMPORAL_FAST_PP=0 run qwen35_instruct /workspace/instruct-models/qwen35-35b-a3b-instruct free,R8,R32 qwen35_instruct_mbpp40_nativepp --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5
run gptoss_20b /workspace/instruct-models/gpt-oss-20b free,R4 gptoss_20b_mbpp40 --think default
run gptoss_120b /workspace/instruct-models/gpt-oss-120b free,R4,R16 gptoss_120b_mbpp40 --think default --gpu-mem 0.9
echo "### mbpp40 ALL DONE $(date -u +%H:%M)"
