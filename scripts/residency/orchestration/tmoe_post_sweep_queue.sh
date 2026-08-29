#!/usr/bin/env bash
# After the gemma sweep runner finishes: (1) qwen on-policy from scratch at the sweep's best settings,
# (2) full surface (no WritingBench) on the gemma winner, (3) deadband surfaces (base rho=0.5, W=3 rho=0.5).
# Strictly sequential so nothing interleaves on the lease.
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "^\[sweep\] done; best = " /workspace/rerun-logs/sweep_online.out 2>/dev/null; do sleep 120; done
eval "$(/workspace/venv_vllm312/bin/python - <<'PY'
import re, ast, sys
sys.path.insert(0, "analysis/residency"); import sweep_online as S
txt = open("/workspace/rerun-logs/sweep_online.out").read()
best = re.findall(r"\[sweep\] done; best = (\S+)", txt)[-1]
bases = re.findall(r"\[sweep\] BASE is now (\{.*\})", txt)
base = ast.literal_eval(bases[-1]) if bases else S.BASE
env = base["env"]
print(f'BEST={best}; ADAPTER=/workspace/olmoe-adapt/data/{best.replace("gemma4_ce_", "gemma_ce_").replace("_n1319", "")}_adapter.pt')
print(f'TOKENS={base["tokens"]}; EVERY={base["every"]}; N={base["n"]}')
for k in ("TMOE_LR", "TMOE_KL_TEMP", "TMOE_ONLINE_TEMP", "TMOE_ANCHOR_W", "TMOE_BUDGET_ON"):
    if k in env: print(f'export {k}={env[k]}')
PY
)"
export TMOE_PRIO=4 TMOE_AUX_LOSS=revkl_full
echo "### post-sweep queue: best=$BEST adapter=$ADAPTER tokens=$TOKENS every=$EVERY n=$N lr=${TMOE_LR:-} klT=${TMOE_KL_TEMP:-1.0} temp=${TMOE_ONLINE_TEMP:-0.7} $(date -u +%H:%M)"
[ -f "$ADAPTER" ] || { echo "### post-sweep queue: winner adapter missing: $ADAPTER"; exit 2; }
echo "### post-sweep 0/4 qwen class confound: textified raw base (text class) GSM8K free,R8 n=1319 vs the raw-class base $(date -u +%H:%M)"
until grep -q "\[textify\] done" /workspace/rerun-logs/textify_qwen_base.out 2>/dev/null; do sleep 60; done
scripts/residency/gpu_lease.sh /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct --path /root/models/qwen35-base-text \
  --arms free,R8 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 --record-as qwen35_instruct_textclass_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90 > /workspace/rerun-logs/qwen_textclass_base.out 2>&1
echo "### post-sweep 0/4 done rc=$? $(date -u +%H:%M)"
echo "### post-sweep 1/4 qwen on-policy from scratch $(date -u +%H:%M)"
bash /workspace/tmoe_qwen_online.sh scratch $TOKENS $EVERY $N > /workspace/rerun-logs/qwen_online_scratch.out 2>&1
echo "### post-sweep 2/4 gemma full surface on the winner (no WB) $(date -u +%H:%M)"
bash /workspace/tmoe_deadband_surface.sh gemma 0 adapter:$ADAPTER ${BEST%_n1319} > /workspace/rerun-logs/winner_full_surface.out 2>&1
echo "### post-sweep 3/4 deadband base rho=0.5 $(date -u +%H:%M)"
bash /workspace/tmoe_deadband_surface.sh gemma 0.5 /dev/shm/gemma4-26b-it gemma4_base > /workspace/rerun-logs/deadband_base.out 2>&1
echo "### post-sweep 4/4 deadband W=3 rho=0.5 $(date -u +%H:%M)"
bash /workspace/tmoe_deadband_surface.sh gemma 0.5 adapter:/workspace/olmoe-adapt/data/gemma_ce_digit3_adapter.pt gemma4_ce_digit3 > /workspace/rerun-logs/deadband_digit3.out 2>&1
echo "### post-sweep queue ALL DONE $(date -u +%H:%M)"
