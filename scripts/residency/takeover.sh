#!/bin/bash
# Takeover chain: everything remaining under the corrected protocols.
# 120b high (thinking cap 4096) -> qwen pp-probe + full pair rerun (mode recipes)
# -> gemma think-on ifeval/mmlu redo -> olmoe redump. Hard-gated throughout.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMOE_ROOT=/workspace/temporal-moe
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_ALLOW_CODE_EVAL=1
export PATH=/opt/venv_vllm/bin:$PATH
cd /workspace/temporal-moe
T4="gsm8k_cot_zeroshot=200,ifeval=200,humaneval_instruct=0,mmlu_flan_cot_fewshot=4"
PY=/opt/venv_vllm/bin/python
ci() { git add results/ablations/instruct_genbench_vllm.csv results/ablations/genbench_samples \
        analysis/residency 2>/dev/null || true; git commit -q -m "$1"; git push -q origin layer-lexicality; }

stage() {
  for i in 1 2 3; do
    $PY - "$1" "$2" <<'PYEOF' && break || { echo "### TK: stage retry $i failed for $1"; sleep 20; [ $i -eq 3 ] && exit 1; }
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
need = sorted(set(idx["weight_map"].values()))
missing = [f for f in need if not os.path.exists(f"{d}/{f}")]
assert not missing, f"missing shards: {missing}"
print(f"stage verified: {len(need)} shards")
PYEOF
}

echo "### TK: goss120 high (cap 4096) $(date -u +%H:%M)"
$PY -u analysis/residency/instruct_genbench_vllm.py --model gptoss_120b \
    --arms free,R4,R16 --reasoning-effort high --record-as gptoss_120b_high \
    --max-gen-toks 2048 --backoff-cap 4096 --max-model-len 5632 --gpu-mem 0.92 \
    --tasks "gsm8k_cot_zeroshot=200,ifeval=200" > /tmp/tk_goss120_high.log 2>&1
for arm in free R4 R16; do
  $PY -u analysis/residency/humaneval_gptoss.py --model gptoss_120b --arm $arm \
      --reasoning-effort high --tag gptoss_120b_high --max-tokens 4096 \
      >> /tmp/tk_goss120_hgo.log 2>&1
done
$PY -u analysis/residency/mmlu_gptoss.py --model gptoss_120b --arms free,R4,R16 \
    --reasoning-effort high --record-as gptoss_120b_high --backoff-cap 4096 \
    >> /tmp/tk_goss120_mmlu.log 2>&1
echo "### TK: goss120 high OK"
ci "thinking ablation: gpt-oss-120b high effort (4096 thinking cap)"

echo "### TK: 20b high humaneval redo at 4096"
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download('openai/gpt-oss-20b', local_dir='/dev/shm/gpt-oss-20b',
                  ignore_patterns=['original/*','*.pth','*.gguf','metal/*'])
PYEOF
for arm in free R4; do
  $PY -u analysis/residency/humaneval_gptoss.py --model gptoss_20b --arm $arm \
      --reasoning-effort high --tag gptoss_20b_high --max-tokens 4096 --gpu-mem 0.85 \
      >> /tmp/tk_goss20_hgo.log 2>&1
done
echo "### TK: 20b high hgo OK"
ci "thinking ablation: gpt-oss-20b high-effort humaneval redo (4096)"

rm -rf /dev/shm/gpt-oss-120b /dev/shm/gpt-oss-20b
echo "### TK: qwen stage"
stage Qwen/Qwen3.5-35B-A3B /dev/shm/qwen35-35b-a3b-instruct
echo "### TK: qwen pp probe"
$PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct \
    --path /dev/shm/qwen35-35b-a3b-instruct --arms free --record-as smoke_qwen_pp \
    --presence-penalty 1.5 --max-gen-toks 2048 --backoff-cap 4096 \
    --max-model-len 5632 --gpu-mem 0.94 --tasks "ifeval=200" > /tmp/tk_qwen_probe.log 2>&1
$PY - <<'PYEOF'
import torch, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/dev/shm/qwen35-35b-a3b-instruct")
d = torch.load("/workspace/instruct-traj/genbench_tokens/smoke_qwen_pp_free_ifeval.pt",
               weights_only=False)
noclose = sum(1 for x in d["items"] if "</think>" not in tok.decode(x["ids"]))
n = len(d["items"])
print(f"### TK: probe verdict: {noclose}/{n} unclosed think "
      f"({'PASS' if noclose <= n * 0.05 else 'STILL RAMBLING'})")
PYEOF

echo "### TK: qwen think-on rerun (pp=1.5, cap 4096)"
$PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct \
    --path /dev/shm/qwen35-35b-a3b-instruct --arms free,R8,R32 \
    --presence-penalty 1.5 --max-gen-toks 2048 --backoff-cap 4096 \
    --max-model-len 7168 --gpu-mem 0.94 --tasks "$T4" > /tmp/tk_qwen_on.log 2>&1
echo "### TK: qwen think-on OK"
ci "qwen3.5 think-on rerun: card recipe (pp=1.5), thinking cap 4096"

echo "### TK: qwen think-off arms (non-thinking recipe 0.7/0.8/pp1.5)"
$PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct \
    --path /dev/shm/qwen35-35b-a3b-instruct --arms free,R8,R32 --think off \
    --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
    --record-as qwen35_think_off --max-gen-toks 1024 --backoff-cap 2048 \
    --max-model-len 5632 --gpu-mem 0.94 --tasks "$T4" > /tmp/tk_qwen_off.log 2>&1
echo "### TK: qwen think-off OK"
ci "thinking ablation: qwen3.5 think-off (non-thinking card recipe)"

rm -rf /dev/shm/qwen35-35b-a3b-instruct
echo "### TK: gemma stage"
stage google/gemma-4-26B-A4B-it /dev/shm/gemma4-26b-it
echo "### TK: gemma think-on ifeval+mmlu redo (cap 4096, answer-only filter)"
$PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct \
    --path /dev/shm/gemma4-26b-it --arms free,R8,R16 --think on \
    --record-as gemma4_think_on --max-gen-toks 2048 --backoff-cap 4096 \
    --max-model-len 7168 --tasks "ifeval=200,mmlu_flan_cot_fewshot=4" \
    > /tmp/tk_gemma_redo.log 2>&1
echo "### TK: gemma think-on redo OK"
$PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct \
    --path /dev/shm/gemma4-26b-it --arms free,R8,R16 \
    --max-gen-toks 640 --backoff-cap 2048 --max-model-len 5632 \
    --tasks "mmlu_flan_cot_fewshot=4" > /tmp/tk_gemma_mmlu_off.log 2>&1
echo "### TK: gemma think-off mmlu (answer-filter parity) OK"
ci "gemma think-on ifeval/mmlu at thinking cap; think-off mmlu answer-filter parity"

echo "### TK: olmoe redump"
$PY -u analysis/residency/instruct_genbench_vllm.py --model olmoe_instruct \
    --arms free,R8 --max-gen-toks 640 --backoff-cap 2048 --max-model-len 4096 \
    --tasks "$T4" > /tmp/tk_olmoe.log 2>&1
echo "### TK: olmoe OK"
ci "OLMoE rerun with token capture"
echo "### TK_DONE"
