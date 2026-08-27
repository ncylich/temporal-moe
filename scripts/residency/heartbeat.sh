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
echo "CHOSEN RECIPE (revised 2026-08-27): PLAIN REBUILD, not d7code. The d7code pick"
echo "  was made on math + 2 code cells before IFEval and HumanEval@8192 landed. On the"
echo "  FULL surface the code lane is net-negative: rebuild mean +1.6 vs d7code -0.2"
echo "  (4 published cells). The code lane buys MBPP (+3.6, best of any arm) and pays"
echo "  it back on IFEval (-3.0) and HumanEval (-1.8). Canonical: gemma_ce_rebuild."
echo "QWEN TRANSFER IS POOR (2026-08-27): mean -0.4 at R8 (3.1% resident), +0.2 at R32"
echo "  (12.5%), vs gemma +1.6. MMLU landed at exactly the published -2.2, so it did"
echo "  NOT rescue the mean. Only MATH is consistent across both models and bounds."
echo "RECIPE IS AT A SHARP OPTIMUM (5 knobs at n=1319, base R8 78.8; KL-arm added 08-27):"
echo "  published settings 81.9 (+3.1) > +code lane 81.1 > 5M tok 80.1 > KL0.02 79.2"
echo "  > lr5e-5 78.1 > rank64 77.3; constrained-arm KL anchor 81.0 (+2.2, worse than"
echo "  the free-arm anchor at +3.1). Every knob that adapts HARDER is worse. Do not"
echo "  retune these. THE GAP TO D12 IS SOLVED (08-27): D12 trained on SELF-GENERATED"
echo "  GSM8K-shaped word problems. selfgen adapter at n=1319: R8 +5.3 (matches D12 +6.0)"
echo "  vs rebuild +3.1; its FREE arm rises (88.2 > base 87.8) and R16 is flat (+0.1) --"
echo "  style-matching, not robustness. Rebuild (+3.1, real math) is the honest number."
echo "  OVERNIGHT 08-27 CLOSED: A (math x2) +2.7, B (constrained KL) +2.2, C (3x prompts)"
echo "  +1.0 -- all WORSE than the rebuild +3.1. SEVEN levers, seven failures. The gap to"
echo "  D12 +6.0 was its SELFGEN math lane (see above), NOT lost prompts."
echo "  RUNNING 08-27: (1) selfgen full 5-benchmark surface -- if IFEval/MMLU/code are flat"
echo "  while GSM8K is +5.3, the published gain is GSM8K-only. (2) qwen trained on the same"
echo "  authored problems -- tests whether qwen's published +6.5 has the same cause."
echo "  Do NOT run more recipe/data sweeps; every direction is downhill."
echo "DIAGNOSIS (08-27, failure_filter.py + slip_position.py): residency breaks ARITHMETIC."
echo "  About half of real R8 failures on both models contain a false equation (5+4+2=8), plan"
echo "  intact (an earlier uncommitted pass said 84/66%; the committed parser says 45-55%)."
echo "  False-eq RATE per equation: qwen 0.7->5.0->3.8% free/R8/adapted (adapter removes 29%"
echo "  of the excess), gemma 0.8->4.8->2.8% (52%). Slips are positionally uniform: not state"
echo "  drift. Scorer artifacts are a UNIFORM offset (~4pt qwen, ~1pt gemma). The adapter"
echo "  repairs ~70% of damage on BOTH models but BREAKS nearly as many (qwen 103 fixed /"
echo "  92 broken; gemma 107/75). Digit tokens = 6.3% of the CE signal, digits after"
echo "  '=' = 0.66%: the loss cannot see the failure. DOSE-RESPONSE: slip share falls 71->60%"
echo "  (qwen 3.1->12.5%) and 63->27% (gemma 6.25->12.5%). selfgen won by cutting slips 89->33"
echo "  (arithmetic-dense data upweights digits by accident). FIX RUNNING: --digit-weight 10 on"
echo "  qwen and gemma (records qwen35_ce_digit10_n1319 / gemma4_ce_digit10_n1319)."
echo "DIGIT-WEIGHT 10 WORKS ON QWEN (20:38 UTC): GSM8K n=1319 R8 82.0 (+5.4 vs base, +3.3 vs"
echo "  rebuild, McNemar z=3.3), R32 +5.3, free +1.2; broke 92->61, fixed 103->115; false-eq"
echo "  rate at R8 3.77->2.29%. Pending: gemma W=10, qwen surface (IFEval/MMLU/code, records"
echo "  qwen35_ce_digit10_{full,n_dual,code}), W=3 dose. Not a result until the surface is flat."
echo "  GEMMA W=10 (21:06): NO GAIN. R8 80.2 (+1.4 vs base, -1.7 vs rebuild z=-1.5), broke 75->101."
echo "  Lever is qwen-specific so far (gemma rebuild already removes 52% of excess; 403 digit ids"
echo "  vs qwen 22). Next cells: qwen surface (running), gemma W=3, qwen W=3, distill from digit10."
echo "  GEMMA W=10 MECHANISM: 17 R8 gens (1.3%) collapse into digit streams (0 on base/rebuild/qwen);"
echo "  false-eq rate 3.69% vs rebuild 2.75%. W=10 overshoots on 403 numeric ids. W=3 is the test."
echo "FALLBACK #2 QUEUED (tmoe_qwen_distill.sh): one round of semi-on-policy self-distillation:"
echo "  sample the constrained student (R8, eval recipe) on the D7 pool, label with the free"
echo "  base's top-50 logprobs, continue the rebuild adapter on KL-only (--kl-arm constrained)."
echo "  Record qwen35_ce_distill_rebuild_n1319. Lease order: qwen digit10 > gemma digit10 >"
echo "  distill > qwen digit3 > selfgen rest > qwen selfgen. Re-queue distill from digit10 if it wins."
echo "  Haiku analyzer written (haiku_failure_analysis.py) but NO API KEY on this pod."
echo "STATS GATE (2026-08-26): claims are made at n=1319, NEVER at n=200 -- the 200-sample"
echo "  is biased, not just noisy (base R8 gap reads -6.0 there, -9.0 on the full split)."
echo "  RESULT (5 runs, SAME-ARM framing): GSM8K R8 recovery mean +2.3, sd 0.9, SE 0.4,"
echo "  z=5.6, 5/5 positive. (+3.1 was gap-closure framing, which credits free-arm sag.)"
echo "  TWO FULL RUNS on the 5-benchmark surface: rebuild +1.6, seed3 -0.1, mean +0.8 on"
echo "  the 4 published cells vs D12 +2.2. HumanEval runs differ by 5.4pt: UNRESOLVED."
echo "  CANONICAL ARM SURFACE (gemma_ce_rebuild, same-arm R8 vs matched base):"
echo "  GSM8K n1319 +3.1 | IFEval n541 -0.7 | HumanEval@8192 +2.4 | MMLU +1.8"
echo "  WritingBench -0.007 (null) | MBPP@8192 +1.6.  MEAN of 4 published cells +1.6"
echo "  vs D12 published +2.2. The code lane (d7code) is recorded as an ablation only."
echo "  EVAL FLOOR: identical re-runs of one cell differ up to 1.2pt (vLLM batch"
echo "  composition). More test items cannot fix it. A variant must clear ~+4.5 on the"
echo "  GSM8K same-arm delta to count as beating the current +2.4/+3.1."
echo "  SWAP DEADBAND (TEMPORAL_RHO, Skliar baseline #3): quality FLAT over RHO 0-1.75"
echo "  on 5 benchmarks / 2 models / base+adapted; cliff at RHO=2.0 (-7.1, z=-5.59)."
echo "  BUT the bandwidth saving is only ~11-36% -- MEASURED. The 14x/65x figures came"
echo "  from a SIMULATED swap rate whose logit scale did not match the model. Never"
echo "  quote a simulated rate: use TEMPORAL_COUNT_SWAPS=1 and swap_stats()."
echo "  Run analysis/residency/arm_power.py and require |z|>1.96"
echo "  before calling any arm an improvement. Full split = --tasks gsm8k_cot_zeroshot=0"
echo "  (n=1319, SE ~1.2). gemma4 = flexible-extract only; strict-match is always 0.000."
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
echo "LIVE GPU WORK ($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l) GPU(s) on this machine):"
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
