#!/usr/bin/env bash
# After the gemma winner surface: deadband surfaces, then the qwen lr ablation (the only qwen ablation):
# lr 3e-5 (qwen's original) and 6e-5 (scaled by gemma's 1e-4/5e-5), both with the winning recipe
# (KL T=2, 16x256, 3.4M sampled tokens, anchor 0, temp 0.7). Pick by paired R8 vs the raw-class base,
# then the full surface (no WritingBench) on the pick.
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "SURFACE DONE" /workspace/rerun-logs/winner_full_surface.out 2>/dev/null; do sleep 120; done
export TMOE_PRIO=4
echo "### qwen-lr 1/5 deadband base rho=0.5 $(date -u +%H:%M)"
bash /workspace/tmoe_deadband_surface.sh gemma 0.5 /dev/shm/gemma4-26b-it gemma4_base > /workspace/rerun-logs/deadband_base.out 2>&1
echo "### qwen-lr 2/5 deadband W=3 rho=0.5 $(date -u +%H:%M)"
bash /workspace/tmoe_deadband_surface.sh gemma 0.5 adapter:/workspace/olmoe-adapt/data/gemma_ce_digit3_adapter.pt gemma4_ce_digit3 > /workspace/rerun-logs/deadband_digit3.out 2>&1
for LR in 3e-5 6e-5; do
  echo "### qwen-lr 3/5 qwen on-policy from scratch, lr $LR, KL T=2 $(date -u +%H:%M)"
  TMOE_LR=$LR TMOE_KL_TEMP=2 TMOE_AUX_LOSS=revkl_full TMOE_NAME_SUFFIX=_klT2_lr$LR bash /workspace/tmoe_qwen_online.sh scratch 3400000 16 256 > /workspace/rerun-logs/qwen_online_klT2_lr$LR.out 2>&1
  echo "### qwen-lr 3/5 lr $LR done rc=$? $(date -u +%H:%M)"
done
pick() {   # best lr by paired R8 z vs the raw-class base among the completed runs given as arguments
/workspace/venv_vllm312/bin/python - "$@" <<'PY'
import sys, math; sys.path.insert(0, "analysis/residency")
from failure_filter import load_arm
def arm(t, a): return {k: v["correct"] for k, v in load_arm(t, a).items()}
def acc(d): return 100*sum(d.values())/len(d)
def z(x, y):
    f = sum(1 for k in x if not x[k] and y[k]); b = sum(1 for k in x if x[k] and not y[k]); return (f-b)/math.sqrt(f+b) if f+b else 0.0
base = {a: arm("qwen35_think_off_n1319", a) for a in ("free", "R8", "R32")}
best, bz = None, -9
for lr in sys.argv[1:]:
    rec = f"qwen35_ce_online_scratch_e16_klT2_lr{lr}_n1319"
    try: arms = {a: arm(rec, a) for a in ("free", "R8", "R32")}
    except Exception as e: print(f"[pick] {rec}: {e}", file=sys.stderr); continue
    r8 = z(base["R8"], arms["R8"])
    print(f"[pick] lr {lr}: free {acc(arms['free']):.1f} R8 {acc(arms['R8']):.1f} R32 {acc(arms['R32']):.1f}; R8 z vs base {r8:+.1f}", file=sys.stderr)
    if r8 > bz: best, bz = lr, r8
print(best or "")
PY
}
echo "### qwen-lr 4/5 pick among 3e-5, 6e-5 $(date -u +%H:%M)"
PICK=$(pick 3e-5 6e-5)
if [ "$PICK" = 6e-5 ]; then      # the scaled lr won: one more step up, then stop (user rule)
  echo "### qwen-lr 3/5 qwen on-policy from scratch, lr 1e-4, KL T=2 (6e-5 beat 3e-5) $(date -u +%H:%M)"
  TMOE_LR=1e-4 TMOE_KL_TEMP=2 TMOE_AUX_LOSS=revkl_full TMOE_NAME_SUFFIX=_klT2_lr1e-4 bash /workspace/tmoe_qwen_online.sh scratch 3400000 16 256 > /workspace/rerun-logs/qwen_online_klT2_lr1e-4.out 2>&1
  echo "### qwen-lr 3/5 lr 1e-4 done rc=$? $(date -u +%H:%M)"
  PICK=$(pick 3e-5 6e-5 1e-4)
fi
echo "### qwen-lr pick: lr $PICK"
[ -n "$PICK" ] || { echo "### qwen-lr: no completed run to pick; stopping"; exit 1; }
echo "### qwen-lr 5/5 full surface (no WB) on lr $PICK $(date -u +%H:%M)"
bash /workspace/tmoe_deadband_surface.sh qwen 0 adapter:/workspace/olmoe-adapt/data/qwen_ce_online_scratch_e16_klT2_lr${PICK}_adapter.pt qwen35_ce_online_klT2_lr$PICK > /workspace/rerun-logs/qwen_winner_surface.out 2>&1
echo "### qwen-lr ALL DONE $(date -u +%H:%M)"
