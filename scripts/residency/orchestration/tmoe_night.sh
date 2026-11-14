#!/usr/bin/env bash
# Overnight decision chain: wait for the qwen full-pool chain; bar = 4-cell mean >= 2.0 on either arm at 0.5x or 1.0x.
# pass -> gemma full-pool chain (0.5x, surface, 1.0x, surface); fail -> qwen r=32 cell. Then Skliar baselines (gemma, qwen).
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "fullpool ALL DONE" /workspace/rerun-logs/qwen_fullpool.out 2>/dev/null; do sleep 120; done
BEST=$(/workspace/venv_vllm312/bin/python - <<'PY'
import csv
lines = [l for l in open("results/ablations/instruct_genbench_vllm.csv") if not l.lstrip('"').startswith("#") and l.strip()]
rows = [r for r in csv.DictReader(lines) if r.get("task")]
met = {"gsm8k_cot_zeroshot": "exact_match,flexible-extract", "ifeval": "prompt_level_strict_acc,none", "mmlu_gptoss_relaxed": "acc,relaxed-extract", "humaneval_instruct": "pass@1,create_test"}
suf = {"gsm8k_cot_zeroshot": "_n1319", "ifeval": "_full", "mmlu_gptoss_relaxed": "_n_dual", "humaneval_instruct": "_code"}
base = {"R8": {"gsm8k_cot_zeroshot": "qwen35_think_off_n1319", "ifeval": "qwen35_base_full", "mmlu_gptoss_relaxed": "qwen35_base_n_dual", "humaneval_instruct": "qwen35_base_code_ref"},
        "R32": {"gsm8k_cot_zeroshot": "qwen35_think_off_n1319", "ifeval": "qwen35_base_r32", "mmlu_gptoss_relaxed": "qwen35_base_n_dual", "humaneval_instruct": "qwen35_base_code_ref"}}
def val(rec, arm, t):
    c = [r for r in rows if r["model"] == rec and r["arm"] == arm and r["task"] == t and r["metric"] == met[t]]
    return 100 * float(c[-1]["value"]) if c else None
best = -9
for pre in ("qwen35_ce_online_fullpool_half_rho0", "qwen35_ce_online_fullpool_full_rho0"):
    for arm in ("R8", "R32"):
        ds = [val(pre + suf[t], arm, t) for t in met]; bs = [val(base[arm][t], arm, t) for t in met]
        if None in ds or None in bs: continue
        m = sum(d - b for d, b in zip(ds, bs)) / 4; best = max(best, m)
print(f"{best:.2f}")
PY
)
echo "### night: best qwen 4-cell mean = $BEST (bar 2.0) $(date -u +%H:%M)"
if /workspace/venv_vllm312/bin/python -c "import sys; sys.exit(0 if float('$BEST') >= 2.0 else 1)"; then
  echo "### night: qwen passes -> gemma full-pool chain $(date -u +%H:%M)"; bash /workspace/tmoe_gemma_fullpool.sh > /workspace/rerun-logs/gemma_fullpool.out 2>&1
else
  echo "### night: qwen below the bar -> qwen expert-LoRA r=32 cell $(date -u +%H:%M)"; bash /workspace/tmoe_qwen_r32.sh > /workspace/rerun-logs/qwen_r32.out 2>&1
fi
echo "### night: Skliar baselines $(date -u +%H:%M)"
bash /workspace/tmoe_skliar.sh gemma > /workspace/rerun-logs/skliar_gemma.out 2>&1
bash /workspace/tmoe_skliar.sh qwen > /workspace/rerun-logs/skliar_qwen.out 2>&1
echo "### night ALL DONE $(date -u +%H:%M)"
