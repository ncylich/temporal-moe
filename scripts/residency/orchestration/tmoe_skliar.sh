#!/usr/bin/env bash
# Skliar et al. cache-conditional experts AT THEIR SETTING: LRU cache of half the experts (gemma 64/128, qwen 128/256),
# top-J=1 guarantee, no training. lambda sweep on GSM8K n=1319 (loads/token measured), full surface at lambda 0.5 and 1.0.
#   tmoe_skliar.sh <gemma|qwen>
set -uo pipefail; cd /workspace/temporal-moe
MODEL=$1
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
export TEMPORAL_WALKER=cache_bias TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1 TMOE_PRIO=${TMOE_PRIO:-4}
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
if [ "$MODEL" = gemma ]; then export TEMPORAL_CB_C=64; M=/dev/shm/gemma4-26b-it; G="--model gemma4_instruct --path $M --arms R8"; ML=4096; TAGP=gemma4_skliar_C64
else export TEMPORAL_CB_C=128; M=/root/models/qwen35-35b-a3b; G="--model qwen35_instruct --path $M --arms R8 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5"; ML=5632; TAGP=qwen35_skliar_C128; fi
for LAM in 0 0.05 0.1 0.2 0.4; do
  export TEMPORAL_CB_LAMBDA=$LAM; TAG=${TAGP}_lam${LAM/./p}
  echo "### skliar $MODEL C=$TEMPORAL_CB_C lambda=$LAM GSM8K n=1319 $(date -u +%H:%M)"
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as ${TAG}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len $ML --gpu-mem 0.90
done
# operating points for the surface: plain LRU (lambda 0) and the knee (largest lambda within 1 pt of plain LRU on GSM8K)
PICKS=$(/workspace/venv_vllm312/bin/python - "$MODEL" "$TAGP" <<'PY'
import sys, re, csv, glob
model, tagp = sys.argv[1], sys.argv[2]
lines = [l for l in open("results/ablations/instruct_genbench_vllm.csv") if not l.lstrip('"').startswith("#") and l.strip()]
rows = [r for r in csv.DictReader(lines) if r.get("task")]
log = open(f"/workspace/rerun-logs/skliar_{model}.out").read()
res = {}
for lam in ("0", "0.05", "0.1", "0.2", "0.4"):
    tag = f"{tagp}_lam{lam.replace('.', 'p')}_n1319"
    acc = [float(r["value"]) for r in rows if r["model"] == tag and r["arm"] == "R8" and r["task"] == "gsm8k_cot_zeroshot" and r["metric"] == "exact_match,flexible-extract"]
    m = re.findall(rf"\[swaps\] {re.escape(tag)} R8 gsm8k_cot_zeroshot: \d+ swaps over \d+ stepped rows = ([\d.]+)/token", log)
    if acc and m: res[lam] = (acc[-1], float(m[-1]))
if not res: print("0 0.1"); sys.exit()
ref = res.get("0", None) or res[min(res, key=float)]
knee = max((l for l in res if res[l][0] >= ref[0] - 0.010), key=float)   # largest lambda within 1 pt of plain LRU
print("0", knee if knee != "0" else "")
for l, (a, s) in res.items(): print(f"[skliar-pick] lambda {l}: GSM8K R8 {100*a:.1f}, loads/token {s:.3f}", file=sys.stderr)
PY
)
echo "### skliar $MODEL surface operating points: $PICKS $(date -u +%H:%M)"
for LAM in $PICKS; do
  export TEMPORAL_CB_LAMBDA=$LAM; TAG=${TAGP}_lam${LAM/./p}
  echo "### skliar $MODEL lambda=$LAM full surface $(date -u +%H:%M)"
  if [ "$MODEL" = gemma ]; then
    $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as ${TAG}_full --tasks "ifeval=0" --gen-cap 2048 --max-model-len $ML --gpu-mem 0.90
    $L $PY -u analysis/residency/mmlu_gptoss.py $G --record-as ${TAG}_full_dual --gpu-mem 0.90
    $L $PY -u analysis/residency/humaneval_gemma.py --path $M --arms R8 --tag ${TAG}_he8192 --max-tokens 8192 --max-model-len 9216
    $L $PY -u analysis/residency/mbpp_gemma.py --path $M --arms R8 --tag ${TAG}_m8192 --max-tokens 8192 --max-model-len 9216 --gpu-mem 0.90
  else
    $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as ${TAG}_full --tasks "ifeval=0" --gen-cap 2048 --max-model-len $ML --gpu-mem 0.90
    $L $PY -u analysis/residency/mmlu_gptoss.py --model qwen35_instruct --path $M --arms R8 --think off --record-as ${TAG}_n_dual --gpu-mem 0.90
    $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as ${TAG}_code --tasks "mbpp_instruct=0,humaneval_instruct=0" --gen-cap 1536 --max-model-len 4096 --gpu-mem 0.90
  fi
done
echo "### skliar $MODEL DONE $(date -u +%H:%M)"
