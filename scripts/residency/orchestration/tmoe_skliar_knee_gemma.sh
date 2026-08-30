#!/usr/bin/env bash
# After the gemma Skliar script (old pick rule surfaced lambda 0 only): full surface at the knee lambda.
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "skliar gemma DONE" /workspace/rerun-logs/skliar_gemma.out 2>/dev/null; do sleep 120; done
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
export TEMPORAL_WALKER=cache_bias TEMPORAL_CB_C=64 TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1 TMOE_PRIO=3
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; M=/dev/shm/gemma4-26b-it; G="--model gemma4_instruct --path $M --arms R8"
KNEE=$($PY - <<'PY'
import csv, re
lines = [l for l in open("results/ablations/instruct_genbench_vllm.csv") if not l.lstrip('"').startswith("#") and l.strip()]
rows = [r for r in csv.DictReader(lines) if r.get("task")]
res = {}
for lam in ("0", "0.05", "0.1", "0.2", "0.4"):
    tag = f"gemma4_skliar_C64_lam{lam.replace('.', 'p')}_n1319"
    acc = [float(r["value"]) for r in rows if r["model"] == tag and r["arm"] == "R8" and r["task"] == "gsm8k_cot_zeroshot" and r["metric"] == "exact_match,flexible-extract"]
    if acc: res[lam] = acc[-1]
ref = res["0"]; print(max((l for l in res if res[l] >= ref - 0.010), key=float))
PY
)
export TEMPORAL_CB_LAMBDA=$KNEE; TAG=gemma4_skliar_C64_lam${KNEE/./p}
echo "### skliar gemma knee lambda=$KNEE full surface $(date -u +%H:%M)"
[ "$KNEE" = "0" ] && { echo "### knee is lambda 0; nothing to add"; exit 0; }
$L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as ${TAG}_full --tasks "ifeval=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
$L $PY -u analysis/residency/mmlu_gptoss.py $G --record-as ${TAG}_full_dual --gpu-mem 0.90
$L $PY -u analysis/residency/humaneval_gemma.py --path $M --arms R8 --tag ${TAG}_he8192 --max-tokens 8192 --max-model-len 9216
$L $PY -u analysis/residency/mbpp_gemma.py --path $M --arms R8 --tag ${TAG}_m8192 --max-tokens 8192 --max-model-len 9216 --gpu-mem 0.90
echo "### skliar gemma knee DONE $(date -u +%H:%M)"
