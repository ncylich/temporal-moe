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
echo "CORE GOAL: adapters FULLY fixed, trained, merged, re-measured, with GOOD results."
echo "  Verify every merge with verify_merge.py. Evaluate the FULL surface: GSM8K, IFEval,"
echo "  MMLU (grid_parallel) + HumanEval (channel-aware producer) + WritingBench (wb_arm)."
echo "  A parallel grid alone is 3 of 5 cells."
echo "STATS GATE (2026-08-26): claims are made at n=1319, NEVER at n=200 -- the 200-sample"
echo "  is biased, not just noisy (base R8 gap reads -6.0 there, -9.0 on the full split)."
echo "  RESULT (4 runs, same recipe): GSM8K R8 recovery mean +3.1, run sd 1.1 -- the"
echo "  +4.1 from the rebuild run alone was the best of four. Quote the MEAN, not a run."
echo "  CODE: HumanEval +2.2 (3/3 runs positive), MBPP -0.7 (scattered). They disagree;"
echo "  code recovery is 0..+2 at most, well under math. d7code arm tests the mix fix."
echo "  Run analysis/residency/arm_power.py and require |z|>1.96"
echo "  before calling any arm an improvement. Full split = --tasks gsm8k_cot_zeroshot=0"
echo "  (n=1319, SE ~1.2). gemma4 = flexible-extract only; strict-match is always 0.000."
echo "LIVE GPU WORK (all four GPUs available as of 2026-08-26):"
for g in 0 1 2 3; do
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
echo "BACKGROUND QUEUE:"
echo "  * unsloth: ROOT-CAUSED (deterministic algorithms -> bit-exact). Still to do:"
echo "    measure the throughput price of --no-unsloth, then decide whether to switch back."
echo "  * gemma think-on GSM8K has NO base arm at 16384 -- its +0.0/+3.5 are absolute"
echo "    numbers, not effects. Run the baseline before quoting them."
echo "  * BASELINE_METHODS_COMPARISON.md -- no published competitor implemented yet."
