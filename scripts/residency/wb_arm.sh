#!/bin/bash
# WritingBench for an adapted arm -- the FIFTH benchmark, and the one that carries Section
# 6's "prose is the robust surface" and 01-findings' "adaptation pays no fluency tax:
# D12 sits at-or-above base in every cell (+0.04 at R8)". An adapted arm measured only on
# GSM8K/IFEval/HumanEval/MMLU cannot speak to either claim.
#
# Protocol matches the published cells so the numbers are comparable: three disjoint
# 50-query subsets (A=0-49, B=50-99, C=100-149), same generation settings, scored by the
# local critic. Published gemma reference: base free 7.533, base R8 7.460, D12 R8 7.504.
#
#     GPU=3 wb_arm.sh /root/models/gemma4-realmath-merged gemma4_realmath R8,R16
set -euo pipefail
ROOT=/workspace/temporal-moe
WB=/workspace/writingbench
export TMOE_ROOT=$ROOT
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
PY=/workspace/venv_vllm312/bin/python
LOG=${LOG_DIR:-/workspace/rerun-logs}
MPATH=${1:?merged dir}; REC=${2:?record}; ARMS=${3:-R8,R16}
export CUDA_VISIBLE_DEVICES=${GPU:-3}
cd $WB
off_for () { case "$1" in "_sB") echo 50;; "_sC") echo 100;; *) echo 0;; esac; }
gen () {  # arm suffix
  local arm=$1 suf=$2 out=responses/${REC}_${arm}${suf}.jsonl
  [ -s "$out" ] && { echo "skip $out"; return; }
  echo "### wb $REC $arm ${suf:-_sA} $(date -u +%H:%M)"
  $PY $ROOT/analysis/writingbench/wb_generate.py --model-path $MPATH --record $REC \
      --arm $arm --suffix "$suf" --offset $(off_for "$suf") --n 50 --max-new 4096 \
      --gpu-mem 0.94 --think off > $LOG/wb_${REC}_${arm}${suf}.log 2>&1
  grep -q DONE $LOG/wb_${REC}_${arm}${suf}.log
}
files=()
for suf in "" "_sB" "_sC"; do
  gen free "$suf"; files+=(responses/${REC}_free${suf}.jsonl)
  for arm in ${ARMS//,/ }; do
    gen $arm "$suf"; files+=(responses/${REC}_${arm}${suf}.jsonl)
    # engagement: a constrained arm identical to free means the patch never engaged
    $PY - "$REC" "$arm" "$suf" <<'PYEOF'
import json, sys
rec, arm, suf = sys.argv[1:4]
load=lambda p: {json.loads(l)["index"]: json.loads(l)["response"] for l in open(p)}
f=load(f"responses/{rec}_free{suf}.jsonl"); r=load(f"responses/{rec}_{arm}{suf}.jsonl")
same=sum(f[i]==r[i] for i in f)
print(f"[engage] {rec} {arm}{suf}: {same}/{len(f)} identical")
assert same < len(f)//2, f"ENGAGEMENT FAIL {rec} {arm}{suf}"
PYEOF
  done
done
echo "### wb $REC SCORING $(date -u +%H:%M)"
$PY $ROOT/analysis/writingbench/wb_score.py --responses "${files[@]}" 2>&1 | tee $LOG/wb_${REC}_score.log | grep "wb-score"
echo "### wb $REC ALL DONE $(date -u +%H:%M)"
