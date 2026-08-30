#!/usr/bin/env bash
# After the 39% cell's surface: pick the best mix by the mean of the R8 and R32 4-cell means, resume that
# adapter for another 3.4M sampled tokens (Adam state + prompt cursor continue), then its full surface.
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "SURFACE DONE" /workspace/rerun-logs/qwen_mix39_surface.out 2>/dev/null; do sleep 60; done
PICK=$(/workspace/venv_vllm312/bin/python - <<'PY'
import csv, sys
lines = [l for l in open("results/ablations/instruct_genbench_vllm.csv") if not l.lstrip('"').startswith("#") and l.strip()]
rows = [r for r in csv.DictReader(lines) if r.get("task")]
met = {"gsm8k_cot_zeroshot": "exact_match,flexible-extract", "ifeval": "prompt_level_strict_acc,none", "mmlu_gptoss_relaxed": "acc,relaxed-extract", "humaneval_instruct": "pass@1,create_test"}
suf = {"gsm8k_cot_zeroshot": "_n1319", "ifeval": "_full", "mmlu_gptoss_relaxed": "_n_dual", "humaneval_instruct": "_code"}
base = {"R8": {"gsm8k_cot_zeroshot": "qwen35_think_off_n1319", "ifeval": "qwen35_base_full", "mmlu_gptoss_relaxed": "qwen35_base_n_dual", "humaneval_instruct": "qwen35_base_code_ref"},
        "R32": {"gsm8k_cot_zeroshot": "qwen35_think_off_n1319", "ifeval": "qwen35_base_r32", "mmlu_gptoss_relaxed": "qwen35_base_n_dual", "humaneval_instruct": "qwen35_base_code_ref"}}
def val(rec, arm, t):
    c = [r for r in rows if r["model"] == rec and r["arm"] == arm and r["task"] == t and r["metric"] == met[t]]
    return 100 * float(c[-1]["value"]) if c else None
cands = {"online_scratch_e16_klT2_lr3e-5": "qwen35_ce_online_klT2_lr3e-5_rho0", "online_scratch_e16_klT2_mix": "qwen35_ce_online_klT2_mix_rho0", "online_scratch_e16_klT2_mix39": "qwen35_ce_online_klT2_mix39_rho0"}
best, bs = None, -9
for name, pre in cands.items():
    ms = []
    for arm in ("R8", "R32"):
        ds = []
        for t in met:
            v = val(pre + suf[t], arm, t); b = val(base[arm][t], arm, t)
            if v is None or b is None: break
            ds.append(v - b)
        if len(ds) == 4: ms.append(sum(ds) / 4)
    if len(ms) == 2:
        sc = sum(ms) / 2; print(f"[pick] {name}: R8 {ms[0]:+.2f} R32 {ms[1]:+.2f} mean {sc:+.2f}", file=sys.stderr)
        if sc > bs: best, bs = name, sc
print(best or "")
PY
)
echo "### qwen-continue pick: $PICK $(date -u +%H:%M)"; [ -n "$PICK" ] || exit 1
Q="mathlane_v2=2341,d5_fewshot=1183,domain8k=1000"; case "$PICK" in *mix39) Q="mathlane_v2=2341,d5_fewshot=1183,domain8k=2500";; *mix) Q="mathlane_v2=1200,d5_fewshot=1183,domain8k=2500";; esac
echo "### qwen-continue 1/2 resume $PICK for +3.4M sampled tokens (quota $Q) $(date -u +%H:%M)"
TMOE_PRIO=4 TMOE_LR=3e-5 TMOE_KL_TEMP=2 TMOE_AUX_LOSS=revkl_full TMOE_ONLINE_MEM=0.65 TMOE_ONLINE_OFFLOAD=20 TMOE_QUOTA="$Q" TMOE_NAME_SUFFIX=_cont bash /workspace/tmoe_qwen_online.sh $PICK 3400000 16 256 > /workspace/rerun-logs/qwen_online_cont.out 2>&1
rc=$?; echo "### qwen-continue 1/2 done rc=$rc $(date -u +%H:%M)"; [ "$rc" = 0 ] || exit 1
A=/workspace/olmoe-adapt/data/qwen_ce_online_${PICK}_e16_cont_adapter.pt
echo "### qwen-continue 2/2 full surface (no WB) $(date -u +%H:%M)"
TMOE_PRIO=4 scripts/residency/gpu_lease.sh bash /workspace/tmoe_deadband_surface.sh qwen 0 adapter:$A qwen35_ce_online_${PICK#online_scratch_e16_}_cont > /workspace/rerun-logs/qwen_cont_surface.out 2>&1
echo "### qwen-continue ALL DONE rc=$? $(date -u +%H:%M)"
