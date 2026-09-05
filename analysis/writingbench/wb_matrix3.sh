#!/bin/bash
# WritingBench 3-replicate matrix (Noah 2026-08-19): every cell x 3 disjoint 50-query
# subsets (A=0-49 done for many cells, B=50-99, C=100-149). Skip-if-exists makes this
# resumable. Aggregation (mean +- SD across subsets) at the end.
set -euo pipefail
echo $$ > /tmp/wb_chain.pid
cd /workspace/writingbench
PY=/opt/venv_vllm/bin/python
TPY=/workspace/venv_fla/bin/python
export TMOE_ROOT=/workspace/temporal-moe PATH=/opt/venv_vllm/bin:$PATH
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1
QWEN=/workspace/instruct-models/qwen35-35b-a3b-instruct
LFM=/workspace/instruct-models/lfm25-8b-a1b
SCRATCH=/tmp/claude-0/-workspace-FLAME-MoE/d3367faa-f8a6-4828-a691-10986ec9fea6/scratchpad

off_for () { case "$1" in "_sB") echo 50;; "_sC") echo 100;; *) echo 0;; esac; }

gen () { # path rec arm suffix extra...
  local path=$1 rec=$2 arm=$3 suf=$4; shift 4
  local out=responses/${rec}_${arm}${suf}.jsonl
  [ -s $out ] && { echo "skip $out"; return; }
  echo "### M3: gen $rec $arm ${suf:-_sA} $(date -u +%H:%M)"
  $PY wb_generate.py --model-path $path --record $rec --arm $arm --suffix "$suf" \
      --offset $(off_for "$suf") --n 50 --max-new 4096 --gpu-mem 0.95 "$@" \
      > /tmp/m3_${rec}_${arm}${suf}.log 2>&1
  grep "DONE" /tmp/m3_${rec}_${arm}${suf}.log
}
engage () { # rec arm suffix
  $TPY - "$1" "$2" "$3" <<'PYEOF'
import json, sys
rec, arm, suf = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p): return {json.loads(l)["index"]: json.loads(l)["response"] for l in open(p)}
f = load(f"responses/{rec}_free{suf}.jsonl"); r = load(f"responses/{rec}_{arm}{suf}.jsonl")
same = sum(f[i] == r[i] for i in f)
print(f"[engage] {rec} {arm}{suf}: {same}/{len(f)} identical")
assert same < len(f) // 2, f"ENGAGEMENT FAIL {rec} {arm}{suf}"
PYEOF
}
score_new () { # files...
  local todo=()
  for f in "$@"; do
    local rec=$(basename $f .jsonl)
    grep -q "^${rec}," scores/summary.csv 2>/dev/null || todo+=($f)
  done
  [ ${#todo[@]} -eq 0 ] && { echo "score: nothing new"; return; }
  echo "### M3: score ${#todo[@]} records $(date -u +%H:%M)"
  $PY wb_score.py --responses "${todo[@]}" > /tmp/m3_score_$$.log 2>&1
  grep "wb-score" /tmp/m3_score_$$.log
}
model_block () { # path rec arms...
  local path=$1 rec=$2; shift 2
  local files=()
  for suf in "" "_sB" "_sC"; do
    gen $path $rec free "$suf" "${EXTRA[@]}"
    files+=(responses/${rec}_free${suf}.jsonl)
    for arm in "$@"; do
      gen $path $rec $arm "$suf" "${EXTRA[@]}"
      engage $rec $arm "$suf"
      files+=(responses/${rec}_${arm}${suf}.jsonl)
    done
  done
  score_new "${files[@]}"
}

# ---- gemma (weights hot in shm) ----
GB=/dev/shm/gemma4-26b-it
EXTRA=(--think off)
[ -s responses/gemma4_base_R16_sC.jsonl ] || model_block $GB gemma4_base R8 R16
echo "### M3: gemma D12 merge"
if [ ! -s responses/gemma4_d12_R16_sC.jsonl ] && [ ! -d /dev/shm/gemma4-d12-merged ]; then
  $TPY /workspace/temporal-moe/analysis/residency/train_gemma_ce.py --model $GB \
      --expert-lora-r 16 --out /workspace/olmoe-adapt/data/gemma_ce_d12_adapter.pt \
      --merge-out /dev/shm/gemma4-d12-merged > /tmp/m3_merge_d12.log 2>&1
  grep -q "merged model" /tmp/m3_merge_d12.log
  cp $GB/processor_config.json /dev/shm/gemma4-d12-merged/
fi
[ -s responses/gemma4_d12_R16_sC.jsonl ] || model_block /dev/shm/gemma4-d12-merged gemma4_d12 R8 R16
rm -rf /dev/shm/gemma4-d12-merged $GB
echo "### M3: gemma DONE"

# ---- gpt-oss-20b ----
echo "### M3: oss20 download $(date -u +%H:%M)"
[ -s responses/oss20_R4_sC.jsonl ] || [ -d /dev/shm/gpt-oss-20b ] || hf download openai/gpt-oss-20b --local-dir /dev/shm/gpt-oss-20b > /tmp/m3_dl_oss20.log 2>&1
EXTRA=(--think default)
[ -s responses/oss20_R4_sC.jsonl ] || model_block /dev/shm/gpt-oss-20b oss20 R4
rm -rf /dev/shm/gpt-oss-20b
echo "### M3: oss20 DONE"

# ---- gpt-oss-120b ----
echo "### M3: oss120 download $(date -u +%H:%M)"
[ -s responses/oss120_R16_sC.jsonl ] || [ -d /dev/shm/gpt-oss-120b ] || hf download openai/gpt-oss-120b --exclude "original/*" --exclude "metal/*" --local-dir /dev/shm/gpt-oss-120b > /tmp/m3_dl_oss120.log 2>&1
model_block /dev/shm/gpt-oss-120b oss120 R4 R16
rm -rf /dev/shm/gpt-oss-120b
echo "### M3: oss120 DONE"

# ---- qwen base + r2 ----
EXTRA=(--think off)
model_block $QWEN qwen35_base R8 R32
ADAPTER_PATH=/workspace/olmoe-adapt/data/qwen_ce_d12r2_adapter.pt DST_PATH=/dev/shm/qwen35-r2-merged \
  $TPY $SCRATCH/qr_patcher.py > /tmp/m3_patch_r2.log 2>&1
model_block /dev/shm/qwen35-r2-merged qwen35_r2 R8 R32
rm -rf /dev/shm/qwen35-r2-merged
echo "### M3: qwen DONE"

# ---- lfm ----
model_block $LFM lfm25 R4
echo "### M3: lfm DONE"

# ---- aggregate ----
echo "### M3: aggregate"
$TPY - <<'PYEOF'
import csv, collections, statistics as st
rows = list(csv.reader(open("scores/summary.csv")))
cells = collections.defaultdict(dict)
for r in rows:
    if len(r) < 3 or r[0].startswith("smoke"): continue
    name, mean = r[0], float(r[1])
    suf = "A"
    base = name
    for s in ("_sB", "_sC"):
        if name.endswith(s): suf, base = s[2], name[: -len(s)]
    cells[base][suf] = mean
with open("scores/cell_stats.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["cell", "mean", "sd_across_subsets", "n_subsets", "subset_means"])
    for c in sorted(cells):
        v = [cells[c][k] for k in sorted(cells[c])]
        m = st.mean(v); sd = st.stdev(v) if len(v) > 1 else float("nan")
        w.writerow([c, f"{m:.3f}", f"{sd:.3f}", len(v), " ".join(f"{x:.2f}" for x in v)])
        print(f"{c:22s} mean {m:.3f}  sd {sd:.3f}  ({len(v)} subsets)")
PYEOF
cp scores/summary.csv scores/cell_stats.csv $SCRATCH/ 2>/dev/null || true
echo "### M3_DONE $(date -u +%H:%M)"
