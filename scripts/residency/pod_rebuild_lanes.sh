#!/bin/bash
# Group A re-run lanes for the rebuilt pod (RECOVER_DATA_PLAN Part 0).
#
# One cell = one engine boot = one GPU, so the lanes are independent processes and
# run concurrently. GPU 0 is deliberately left free; lanes take 1, 2, 3.
#
#     pod_rebuild_lanes.sh a1     # GPU 1  qwen3.5 IFEval @16384, all three arms
#     pod_rebuild_lanes.sh a4     # GPU 2  gemma4 think-on IFEval + MMLU, double budget
#     pod_rebuild_lanes.sh a2     # GPU 3  WritingBench @8192, oss120 / oss20 / LFM
#
# Every lane is skip-if-exists at the cell level, so a killed lane resumes by
# rerunning the same command.
set -euo pipefail

ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export PATH=/workspace/venv_vllm312/bin:$PATH            # ninja must be on PATH:
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}  # flashinfer JITs
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1 HF_ALLOW_CODE_EVAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENABLE_V1_MULTIPROCESSING=0                  # vllm_glue patches the in-process core
cd $ROOT

PY=/workspace/venv_vllm312/bin/python
G=analysis/residency/instruct_genbench_vllm.py
LOG=${LOG_DIR:-/workspace/rerun-logs}
mkdir -p $LOG

QWEN=/dev/shm/qwen35-35b-a3b
GEMMA=/dev/shm/gemma4-26b-it

case "${1:?usage: pod_rebuild_lanes.sh a1|a2|a4}" in

# ---------------------------------------------------------------- A1 ----------
# The one arm the truncation sweep missed. Task #80 reran R8 and R32 at 16384 and
# recorded them as qwen35_instruct_cap16k, but left the free arm at 8192, where it
# is 8.0% truncated -- so the cell has no matched free arm and falls back to 8192.
#
# All THREE arms run in one boot, not just the missing free arm: the batch-fair
# protocol this project enforces everywhere requires arms to share an engine, and
# TODO section 4 records that constrained-arm generations are not reproducible
# run-to-run. A free arm stitched onto another boot's constrained arms is exactly
# the comparison the protocol exists to prevent. Costs ~3 GPU-hours (Task #80's
# arms took 4358 s and 3729 s), which is cheap for a matched triple.
#
# Recorded under a new name so the Task #80 rows stay untouched for reference,
# following the house pattern (old row kept, new row suffixed).
# presence-penalty 1.5 is the qwen3.5-thinking model-card fallback Task #80 used
# and is required for protocol fidelity with those rows.
a1)
  export CUDA_VISIBLE_DEVICES=1
  echo "### A1: qwen3.5 IFEval @16384, arms free,R8,R32 $(date -u +%H:%M)"
  $PY -u $G --model qwen35_instruct --path $QWEN \
      --arms free,R8,R32 --tasks "ifeval=200" \
      --gen-cap 16384 --max-model-len 17920 --gpu-mem 0.94 \
      --presence-penalty 1.5 \
      --record-as qwen35_instruct_cap16k_b \
      2>&1 | tee $LOG/a1_qwen_ifeval_16k.log
  echo "### A1 DONE $(date -u +%H:%M)"
  ;;

# ---------------------------------------------------------------- A4 ----------
# The last two cells above the 2% cap-hit bar (IFEval 6.5%, MMLU 6.1%), left by
# judgment rather than evidence. Double budget = 8192, matching the cap8k sweep.
# gemma4 think-on already has cap8k HumanEval rows; these are the two that were
# never swept.
# "Double budget" is per cell, against what that cell actually ran at (verified
# from instruct_genbench_vllm.csv, not assumed): think-on IFEval sits at 8192, so
# it doubles to 16384; think-on MMLU sits at 4096, so it doubles to 8192. Record
# names follow the house cap-suffix convention and the existing
# gemma4_think_on_cap8k record, whose HumanEval dumps live under a different task
# suffix and so do not collide.
a4)
  export CUDA_VISIBLE_DEVICES=2
  echo "### A4: gemma4 think-on IFEval 8192 -> 16384 $(date -u +%H:%M)"
  $PY -u $G --model gemma4_instruct --path $GEMMA --think on \
      --arms free,R8,R16 --tasks "ifeval=200" \
      --gen-cap 16384 --max-model-len 17920 --gpu-mem 0.94 \
      --record-as gemma4_think_on_cap16k \
      2>&1 | tee $LOG/a4_gemma_ifeval_16k.log
  echo "### A4: gemma4 think-on MMLU 4096 -> 8192 $(date -u +%H:%M)"
  $PY -u $G --model gemma4_instruct --path $GEMMA --think on \
      --arms free,R8,R16 --tasks "mmlu_flan_cot_fewshot=4" \
      --gen-cap 8192 --max-model-len 11264 --gpu-mem 0.94 \
      --record-as gemma4_think_on_cap8k \
      2>&1 | tee $LOG/a4_gemma_mmlu_8k.log
  echo "### A4 DONE $(date -u +%H:%M)"
  ;;

# ---------------------------------------------------------------- A2 ----------
# WritingBench was never swept for truncation. At its 4096 budget gpt-oss-120b,
# gpt-oss-20b and LFM sit at 30-36% / 20-25% / 21-27% capped on BOTH arms, and
# Section 6 leans on WritingBench for "prose is the robust surface". The paired
# delta may survive since both arms truncate alike, but that is untested.
#
# Deliberately NOT wired up yet: the WritingBench harness needs its upstream query
# set and the local critic judge staged (both were on the deleted pod), and the LFM
# residency path needs its own smoke first -- vllm_glue's LFM factory wrap is the
# one piece the 0.27.1 port touched, and a silent no-op there produced arm-identical
# generations once before. See RERUN_ORCHESTRATION.md.
a2)
  export CUDA_VISIBLE_DEVICES=3
  WB=/workspace/writingbench
  WBSRC=$ROOT/analysis/writingbench
  cd $WB
  # Three disjoint 50-query subsets (A=0-49, B=50-99, C=100-149), matching the M3
  # matrix protocol the existing 4096 cells were measured under, so the 8192 cells are
  # comparable to them cell-for-cell and carry the same across-subset SD.
  off_for () { case "$1" in "_sB") echo 50;; "_sC") echo 100;; *) echo 0;; esac; }
  gen () {  # path record arm suffix extra...
    local path=$1 rec=$2 arm=$3 suf=$4; shift 4
    local out=responses/${rec}_${arm}${suf}.jsonl
    [ -s "$out" ] && { echo "skip $out"; return; }
    echo "### A2: gen $rec $arm ${suf:-_sA} $(date -u +%H:%M)"
    $PY $WBSRC/wb_generate.py --model-path $path --record $rec --arm $arm \
        --suffix "$suf" --offset $(off_for "$suf") --n 50 \
        --max-new 8192 --gpu-mem 0.95 "$@" > $LOG/a2_${rec}_${arm}${suf}.log 2>&1
    grep -q DONE $LOG/a2_${rec}_${arm}${suf}.log
  }
  # Engagement check: a constrained arm that reproduces the free arm verbatim means the
  # residency patch did not engage. This is not paranoia -- vllm_glue's LFM factory wrap
  # silently no-opped once before and produced exactly this signature.
  engage () {
    /workspace/venv_fla/bin/python - "$1" "$2" "$3" <<'PYEOF'
import json, sys
rec, arm, suf = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p): return {json.loads(l)["index"]: json.loads(l)["response"] for l in open(p)}
f = load(f"responses/{rec}_free{suf}.jsonl"); r = load(f"responses/{rec}_{arm}{suf}.jsonl")
same = sum(f[i] == r[i] for i in f)
print(f"[engage] {rec} {arm}{suf}: {same}/{len(f)} identical")
assert same < len(f) // 2, f"ENGAGEMENT FAIL {rec} {arm}{suf}"
PYEOF
  }
  block () {  # path record arms...
    local path=$1 rec=$2; shift 2
    for suf in "" "_sB" "_sC"; do
      gen $path $rec free "$suf" "${EXTRA[@]}"
      for arm in "$@"; do
        gen $path $rec $arm "$suf" "${EXTRA[@]}"
        engage $rec $arm "$suf"
      done
    done
  }
  EXTRA=(--think default)
  block /dev/shm/gpt-oss-120b oss120_cap8k R4 R16
  block /dev/shm/gpt-oss-20b  oss20_cap8k  R4
  EXTRA=(--think off)
  block /dev/shm/lfm25-8b-a1b lfm25_cap8k  R4
  echo "### A2: scoring with the critic $(date -u +%H:%M)"
  $PY $WBSRC/wb_score.py --responses responses/*_cap8k_*.jsonl \
      > $LOG/a2_score.log 2>&1
  grep "wb-score" $LOG/a2_score.log | tail -20
  echo "### A2 DONE $(date -u +%H:%M)"
  ;;

*)
  echo "unknown lane: $1" >&2; exit 2 ;;
esac
