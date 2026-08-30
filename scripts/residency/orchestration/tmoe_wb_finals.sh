#!/usr/bin/env bash
# WritingBench on the two final adapters, after all overnight chains: qwen = full pool 1.0x; gemma = the better of
# (full pool 1.0x, KL T=2 on the d7 pool) by the mean of the R8 and R16 4-cell means. Arms free,R8,R16 (qwen: free,R8,R32).
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "after-night ALL DONE" /workspace/rerun-logs/after_night.out 2>/dev/null; do sleep 300; done
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=4
L=scripts/residency/gpu_lease.sh; D=/workspace/olmoe-adapt/data
GPICK=$(/workspace/venv_vllm312/bin/python - <<'PY'
import csv, sys
lines = [l for l in open("results/ablations/instruct_genbench_vllm.csv") if not l.lstrip('"').startswith("#") and l.strip()]
rows = [r for r in csv.DictReader(lines) if r.get("task")]
met = {"gsm8k_cot_zeroshot": "exact_match,flexible-extract", "ifeval": "prompt_level_strict_acc,none", "mmlu_gptoss_relaxed": "acc,relaxed-extract", "humaneval_gemma_fixed": "pass@1,channel-aware"}
suf = {"gsm8k_cot_zeroshot": "_n1319", "ifeval": "_full", "mmlu_gptoss_relaxed": "_full_dual", "humaneval_gemma_fixed": "_he8192"}
def val(rec, arm, t):
    c = [r for r in rows if r["model"] == rec and r["arm"] == arm and r["task"] == t and r["metric"] == met[t]]
    return 100 * float(c[-1]["value"]) if c else None
def mean4(pre):
    ms = []
    for arm in ("R8", "R16"):
        ds = []
        for t in met:
            v = val(pre + suf[t], arm, t); b = val("gemma4_instruct" + suf[t], arm, t)
            if v is None or b is None: return None
            ds.append(v - b)
        ms.append(sum(ds) / 4)
    return sum(ms) / 2
c = {"gemma4_ce_online_fullpool_full_rho0": "gemma_ce_online_online_scratch_e16_fullpool_e16_full_adapter.pt", "gemma4_ce_online_scratch_e16_klT2_rho0": "gemma_ce_online_scratch_e16_klT2_adapter.pt"}
best = max(((mean4(p) or -9, p, a) for p, a in c.items()))
print(f"[wb-pick] gemma final: {best[1]} (mean of arm means {best[0]:+.2f})", file=sys.stderr); print(f"{best[1]} {best[2]}")
PY
)
set -- $GPICK; GREC=$1; GAD=$2
echo "### wb finals: gemma $GREC, qwen full pool 1.0x $(date -u +%H:%M)"
rm -f results/ablations/writingbench/responses/${GREC}_*.jsonl results/ablations/writingbench/responses/qwen35_ce_online_fullpool_full_rho0_*.jsonl
TMOE_ADAPTER=$D/$GAD $L scripts/residency/wb_arm.sh /dev/shm/gemma4-26b-it $GREC free,R8,R16
echo "### wb gemma done rc=$? $(date -u +%H:%M)"
TMOE_ADAPTER=$D/qwen_ce_online_online_scratch_e16_fullpool_e16_full_adapter.pt $L scripts/residency/wb_arm.sh /root/models/qwen35-35b-a3b qwen35_ce_online_fullpool_full_rho0 free,R8,R32
echo "### wb qwen done rc=$? $(date -u +%H:%M)"
echo "### wb finals ALL DONE $(date -u +%H:%M)"
