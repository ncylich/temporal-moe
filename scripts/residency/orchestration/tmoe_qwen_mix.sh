#!/usr/bin/env bash
# One-knob cell on qwen: on-policy prompt mix rebalanced from 52% math to ~25% math (mathlane 1200, d5_fewshot 1183,
# domain8k 2500), everything else the winning recipe (lr 3e-5, KL T=2, 16x256, 3.4M). GSM8K first; full surface only
# if R8 stays within 1 pt of the math-heavy cell (83.2).
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_PRIO=4
echo "### qwen-mix 1/2 on-policy from scratch, balanced prompt mix $(date -u +%H:%M)"
TMOE_LR=3e-5 TMOE_KL_TEMP=2 TMOE_AUX_LOSS=revkl_full TMOE_ONLINE_MEM=0.65 TMOE_ONLINE_OFFLOAD=20 TMOE_NAME_SUFFIX=_klT2_mix \
  TMOE_QUOTA="mathlane_v2=1200,d5_fewshot=1183,domain8k=2500" bash /workspace/tmoe_qwen_online.sh scratch 3400000 16 256 > /workspace/rerun-logs/qwen_online_klT2_mix.out 2>&1
rc=$?; echo "### qwen-mix 1/2 done rc=$rc $(date -u +%H:%M)"; [ "$rc" = 0 ] || exit 1
R8=$(/workspace/venv_vllm312/bin/python - <<'PY'
import sys; sys.path.insert(0, "analysis/residency")
from failure_filter import load_arm
d = load_arm("qwen35_ce_online_scratch_e16_klT2_mix_n1319", "R8"); print(f"{100*sum(v['correct'] for v in d.values())/len(d):.1f}")
PY
)
echo "### qwen-mix R8 GSM8K = $R8 (math-heavy cell 83.2)"
if /workspace/venv_vllm312/bin/python -c "import sys; sys.exit(0 if float('$R8') >= 82.2 else 1)"; then
  echo "### qwen-mix 2/2 full surface (no WB) $(date -u +%H:%M)"
  scripts/residency/gpu_lease.sh bash /workspace/tmoe_deadband_surface.sh qwen 0 adapter:/workspace/olmoe-adapt/data/qwen_ce_online_scratch_e16_klT2_mix_adapter.pt qwen35_ce_online_klT2_mix > /workspace/rerun-logs/qwen_mix_surface.out 2>&1
  echo "### qwen-mix 2/2 done rc=$? $(date -u +%H:%M)"
else echo "### qwen-mix: GSM8K R8 dropped more than 1 pt; surface skipped"; fi
echo "### qwen-mix ALL DONE $(date -u +%H:%M)"
