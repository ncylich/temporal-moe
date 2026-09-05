#!/bin/bash
# One status block. Deliberately dumb: no functions inside command substitution, no pipes
# that can SIGPIPE, every lookup guarded -- an inline version died with exit 144 and no
# error text, which is the failure heartbeats exist to CATCH rather than exhibit.
#
# Lanes are DERIVED from what is actually on the GPUs, not a hardcoded list. The previous
# version listed lanes fixed at arm time, so hours later it reported finished jobs as DOWN
# and said nothing about the four running ones.
L=/workspace/rerun-logs
echo "HEARTBEAT $(date -u '+%H:%M UTC')"
echo "CORE GOAL: reproduce/beat the published rolling-residency adaptation (gemma R8 +6.0,"
echo "  qwen +6.5) HONESTLY, then the full surface. Always GSM8K n=1319; WritingBench only on"
echo "  the final version (TMOE_WB=1). Every finished run gets the per-arm table in chat."
echo "STANDING FORMULATION (2026-08-29): on-policy reverse-KL self-distillation under the"
echo "  residency bound, from scratch, prompts only, NO CE, NO digit weight, anchor 0 (honest"
echo "  baseline, not a knob), analytic estimator --aux-loss revkl_full. The in-process"
echo "  sampler (vLLM asleep in the trainer, bit-exact GPU sync, ~24 s / 64 rows) is validated."
echo "  Reference to beat at gemma R8: CE+W=3 82.3; three recipes (CE+W=3, sampled revkl,"
echo "  analytic revkl) all stop at 82.3 +/- 0.3 (base 78.8); KL trace plateaus ~0.37."
echo "  Analytic from-scratch surface: never below base, MBPP +5.4, 4-cell mean +1.3 (CE +1.6)."
echo "SWEEP (sweep_online.py, one knob per cell, early stop if KL[100] >= KL[50]):"
echo "  baseline(anchor 0) -> lr 1e-4 -> KL T=2 -> sample temp 1.0 -> refresh 8x128 -> 6.8M."
echo "  Results append to results/ablations/online_sweep.md. Full surface on the winner (no WB)."
echo "QUEUE AFTER THE SWEEP: qwen apply_adapter check (fused raw names, prio 4) -> qwen parity"
echo "  smoke -> tmoe_qwen_online.sh scratch (no digit weight). Deadband surfaces at R8 (base"
echo "  rho=0.5 prio 5, W=3 rho=0.5 prio 6); if they hold, deadband on the final adapter."
echo "  Competitors are compared at THEIR settings on 3 axes (memory, swaps/layer/token, quality)."
echo "RULES: one heavy process at a time under gpu_lease.sh (RAM cap 233 GiB incl. /dev/shm);"
echo "  never edit a running bash script in place (new inode + mv); state expected durations;"
echo "  diagnose a failed run before blaming the algorithm; no replication seeds until final."
echo "HISTORY (details in results/ablations/REBUILD_RESULTS.md): published +6.0 came from a"
echo "  self-generated GSM8K-format lane; real-data rebuild +2.3/+3.1; digit-weight 10 moved"
echo "  qwen (+5.4) not gemma; residency breaks arithmetic (false-equation slips, uniform in"
echo "  position); eval floor ~1.2pt between identical re-runs, 10/1319 paired flips at R8."
echo "LIVE GPU WORK ($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l) GPU(s) on this machine):"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
for g in $(seq 0 $((NGPU-1))); do
  mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
  ut=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)
  pid=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "$pid" ]; then
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | sed 's|.*/python[0-9.]* *-u* *||' | cut -c1-58)
  elif [ "$g" = "0" ]; then
    cmd="idle -- give it work"
  else
    cmd="IDLE -- give it work"
  fi
  printf "  GPU %s %-11s %-6s %s\n" "$g" "$mem" "$ut" "$cmd"
done
echo "RECENT COMPLETIONS (last 6):"
grep -hoE "### .*(ALL DONE|TRAIN DONE|PARALLEL DONE|COMPLETE) [0-9:]+" $L/*.out $L/*.log 2>/dev/null \
  | tail -6 | sed 's/^/  /'
echo "STORAGE (caps: RAM 200GB, local 1TB, network 1TB):"
printf "  RAM %s  local %s  network-ours %s\n" \
  "$(du -sh /dev/shm 2>/dev/null | cut -f1)" \
  "$(du -sh /root/models 2>/dev/null | cut -f1)" \
  "$(du -s /workspace/instruct-traj /workspace/olmoe-adapt /workspace/merged-ckpts 2>/dev/null | awk '{s+=$1} END{printf "%.0fG", s/1048576}')"
