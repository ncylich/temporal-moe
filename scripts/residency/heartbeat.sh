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
echo "  RESULT (5 runs, same recipe): GSM8K R8 recovery mean +3.1, sd 0.9, SE 0.4,"
echo "  z=7.5, 5/5 positive. Quote the MEAN -- rebuild's +4.1 was the best of five."
echo "  CODE (settled 2026-08-26 after 4 corrections): MBPP damage is REAL and LARGE --"
echo "  -14.6@1536, -14.0@8192 -- and the adapter recovers only ~+2 (never significant)."
echo "  Budget does NOT explain MBPP. HumanEval (n=164) is too small to settle anything:"
echo "  two D7 runs at 8192 gave +4.9 and -2.4, ~3 sigma apart. NEVER claim a code"
echo "  result from one run on one budget. d7code (26.7% code) is within run scatter."
echo "  SWAP DEADBAND (TEMPORAL_RHO, Skliar baseline #3): quality FLAT over RHO 0-1.75"
echo "  on 5 benchmarks / 2 models / base+adapted; cliff at RHO=2.0 (-7.1, z=-5.59)."
echo "  BUT the bandwidth saving is only ~11-36% -- MEASURED. The 14x/65x figures came"
echo "  from a SIMULATED swap rate whose logit scale did not match the model. Never"
echo "  quote a simulated rate: use TEMPORAL_COUNT_SWAPS=1 and swap_stats()."
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
echo "  * DONE gemma think-on: matched base at 16384, n=1319. Damage is only -2.0 with"
echo "    thinking ON vs -9.0 OFF, and the adapter adds +0.6 +/- 1.0 (null) there."
echo "    This is a think-OFF phenomenon and a think-OFF fix."
echo "  * BASELINES: #3 Skliar DONE (5 benchmarks, 2 models, base+adapted). #1 CoSMoEs"
echo "    loss implemented, needs the FLAME isoFLOP sweep. #2 ReMoE needs router-only"
echo "    training support in train_gemma_ce.py -- NOT built yet."
echo "  * FRONTIER: within GEMMA, critical swap rate ~ 0.042*(E/R)^0.85, 5 points over"
echo "    a 16x E/R range, +/-15%. R8 breaks at 0.53 swaps/tok, R64 at 0.07 (12x saving)."
echo "    BUT IT DOES NOT TRANSFER: qwen R32 has the SAME E/R=8 and breaks at ~0.78,"
echo "    3x gemma R16's 0.25. qwen is nearly FLAT (~0.8) across a 4x resident range."
echo "    Memory-for-bandwidth substitution is a gemma property, NOT a design rule."
echo "    Third scaling law tonight to die on a second model. Always test cross-model."
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
echo "  * DONE gemma think-on: matched base at 16384, n=1319. Damage is only -2.0 with"
echo "    thinking ON vs -9.0 OFF, and the adapter adds +0.6 +/- 1.0 (null) there."
echo "    This is a think-OFF phenomenon and a think-OFF fix."
echo "  * BASELINES: #3 Skliar DONE (5 benchmarks, 2 models, base+adapted). #1 CoSMoEs"
echo "    loss implemented, needs the FLAME isoFLOP sweep. #2 ReMoE needs router-only"
echo "    training support in train_gemma_ce.py -- NOT built yet."
echo "  * FRONTIER (new): resident fraction sets WHERE the swap cliff is (12.5% breaks at"
echo "    ~0.28 swaps/tok, 6.25% at ~0.53, 3.1% at ~0.75); adaptation is a flat +3 offset"
echo "    that does NOT move the cliff. Product frac x critical-rate ~0.03 -- SUGGESTIVE"
echo "    only, 3 coarse-grid points at n=200. Knee sweeps running to test it."
