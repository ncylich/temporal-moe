#!/usr/bin/env bash
# After the runner finishes: choose the winner by the all-arm rule (the runner's rule is R8-only), append
# the override to sweep_online.out (the post-sweep chain reads the LAST 'done; best' / 'BASE is now' lines),
# then start the post-sweep chain.
until grep -q "^\[sweep\] done; best = " /workspace/rerun-logs/sweep_online.out 2>/dev/null; do sleep 10; done
cd /workspace/temporal-moe
/workspace/venv_vllm312/bin/python - <<'PY' >> /workspace/rerun-logs/sweep_online.out
import sys, math, re; sys.path.insert(0, "analysis/residency")
from failure_filter import load_arm
def arm(t, a):
    try: return {k: v["correct"] for k, v in load_arm(t, a).items()}
    except Exception: return None
def z(x, y):
    f = sum(1 for k in x if not x[k] and y[k]); b = sum(1 for k in x if x[k] and not y[k]); return (f - b) / math.sqrt(f + b) if f + b else 0.0
klT2 = "gemma4_ce_online_scratch_e16_klT2_n1319"; big = "gemma4_ce_online_scratch_e16_budget6.8M_n1319"
a, b = arm(klT2, "R8"), arm(big, "R8")
if a and b and z(a, b) >= 2.0:
    print(f"[sweep] winner rule: budget6.8M beats klT2 at R8 (z={z(a,b):+.1f}); keeping the runner's choice")
else:
    print(f"[sweep] winner rule: klT2 (R8 tie with lr1e-4/budget, best free+R16) -> override")
    print("[sweep] BASE is now {'env': {'TMOE_ANCHOR_W': '0', 'TMOE_BUDGET_ON': 'sampled', 'TMOE_LR': '1e-4', 'TMOE_KL_TEMP': '2'}, 'tokens': 3400000, 'every': 16, 'n': 256}")
    print(f"[sweep] done; best = {klT2}")
PY
nohup bash /workspace/tmoe_post_sweep_queue.sh > /workspace/rerun-logs/post_sweep_queue.out 2>&1 & echo $! > /workspace/pids/post_sweep_queue.pid
echo "### post-sweep chain started with the chosen winner $(date -u +%H:%M)"
