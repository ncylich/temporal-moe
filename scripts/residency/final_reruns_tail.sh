#!/bin/bash
# Tail of final_reruns with the bespoke qwen think-on humaneval inserted first.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMOE_ROOT=/workspace/temporal-moe
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_ALLOW_CODE_EVAL=1
export PATH=/opt/venv_vllm/bin:$PATH
cd /workspace/temporal-moe
PY=/opt/venv_vllm/bin/python
G=analysis/residency/instruct_genbench_vllm.py
ci() { git add results/ablations/instruct_genbench_vllm.csv results/ablations/genbench_samples         analysis/residency 2>/dev/null || true; git commit -q -m "$1"; git push -q origin layer-lexicality; }
stage() {
  for i in 1 2 3; do
    $PY - "$1" "$2" <<'PYEOF' && break || { echo "### FR: stage retry $i failed for $1"; sleep 20; [ $i -eq 3 ] && exit 1; }
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  ignore_patterns=['original/*','*.pth','*.gguf','metal/*'])
PYEOF
  done
  $PY - "$2" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(f"{d}/model.safetensors.index.json"))
missing = [f for f in sorted(set(idx["weight_map"].values()))
           if not os.path.exists(f"{d}/{f}")]
assert not missing, f"missing shards: {missing}"
print("stage verified")
PYEOF
}

echo "### FR: qwen re-stage for bespoke humaneval $(date -u +%H:%M)"
stage Qwen/Qwen3.5-35B-A3B /dev/shm/qwen35-35b-a3b-instruct
echo "### FR: qwen bespoke humaneval (think-on) $(date -u +%H:%M)"
for arm in free R8 R32; do
  $PY -u analysis/residency/humaneval_think.py --model qwen35_instruct       --path /dev/shm/qwen35-35b-a3b-instruct --arm $arm --presence-penalty 1.5       --max-tokens 4096 --gpu-mem 0.94 >> /tmp/fr_qwen_hvt.log 2>&1
done
echo "### FR: qwen bespoke humaneval OK"
ci "qwen think-on humaneval: bespoke unprimed path (gen_prefix landed inside think block)"

rm -rf /dev/shm/qwen35-35b-a3b-instruct
echo "### FR: lfm redo (ifeval + humaneval, native path)"
$PY -u $G --model lfm25_instruct --arms free,R4 --max-gen-toks 1024 --backoff-cap 4096 \
    --max-model-len 5632 --tasks "humaneval_instruct=0" > /tmp/fr_lfm.log 2>&1
$PY -u $G --model lfm25_instruct --arms free,R4 --max-gen-toks 1024 --backoff-cap 8192 \
    --max-model-len 9216 --tasks "ifeval=200" >> /tmp/fr_lfm.log 2>&1
echo "### FR: lfm OK"
ci "LFM redo: humaneval (until-in-think stops) + ifeval (1280 yaml cap), native path"

echo "### FR: gemma stage"
stage google/gemma-4-26B-A4B-it /dev/shm/gemma4-26b-it
$PY -u $G --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms free,R8,R16 \
    --think on --record-as gemma4_think_on --max-gen-toks 2048 --backoff-cap 4096 \
    --max-model-len 7168 --tasks "gsm8k_cot_zeroshot=200,mmlu_flan_cot_fewshot=4" \
    > /tmp/fr_gemma_on.log 2>&1
$PY -u $G --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms free,R8,R16 \
    --think on --record-as gemma4_think_on --max-gen-toks 2048 --backoff-cap 8192 \
    --max-model-len 9728 --tasks "ifeval=200" >> /tmp/fr_gemma_on.log 2>&1
echo "### FR: gemma think-on OK"
ci "gemma think-on: native path, budget-corrected"

rm -rf /dev/shm/gemma4-26b-it
echo "### FR: goss20 stage"
stage openai/gpt-oss-20b /dev/shm/gpt-oss-20b
for eff in low medium high; do
  REC=gptoss_20b_$eff; [ "$eff" = "medium" ] && REC=gptoss_20b
  $PY -u $G --model gptoss_20b --arms free,R4 --reasoning-effort $eff --record-as $REC \
      --max-gen-toks 2048 --backoff-cap 4096 --max-model-len 5632 --gpu-mem 0.85 \
      --tasks "ifeval=200" >> /tmp/fr_goss20.log 2>&1
done
echo "### FR: goss20 OK"
ci "gpt-oss-20b gsm8k+ifeval redo: native final-channel path, budget-corrected"

rm -rf /dev/shm/gpt-oss-20b
echo "### FR: goss120 stage"
stage openai/gpt-oss-120b /dev/shm/gpt-oss-120b
for eff in low medium high; do
  REC=gptoss_120b_$eff; [ "$eff" = "medium" ] && REC=gptoss_120b
  $PY -u $G --model gptoss_120b --arms free,R4,R16 --reasoning-effort $eff --record-as $REC \
      --max-gen-toks 2048 --backoff-cap 4096 --max-model-len 5632 --gpu-mem 0.92 \
      --tasks "ifeval=200" >> /tmp/fr_goss120.log 2>&1
done
echo "### FR: goss120 OK"
ci "gpt-oss-120b gsm8k+ifeval redo: native final-channel path, budget-corrected"

rm -rf /dev/shm/gpt-oss-120b
echo "### FR: olmoe redump"
$PY -u $G --model olmoe_instruct --arms free,R8 --max-gen-toks 640 --backoff-cap 2048 \
    --max-model-len 4096 \
    --tasks "gsm8k_cot_zeroshot=200,ifeval=200,humaneval_instruct=0,mmlu_flan_cot_fewshot=4" \
    > /tmp/fr_olmoe.log 2>&1
echo "### FR: olmoe OK"
ci "OLMoE rerun with token capture"
echo "### FR_DONE"
