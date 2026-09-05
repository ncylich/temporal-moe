#!/usr/bin/env bash
# Unattended work between the attention-250M run finishing and 11:00 PST (18:00 UTC).
#
# Chosen from what the night's results left open, in order of how much each one is load-bearing for a
# conclusion already being drawn, rather than by how interesting it sounds.
#
# The night produced one result that everything else now leans on: adding 8.4M parameters of LoRA to
# q/k/v/o moved mean 10-task accuracy by +0.0123 while moving BPB by 0.001074 -- and the fourth freed
# layer had moved BPB by 0.0115 for +0.0001 accuracy. A tenth of the BPB gain, a hundred times the
# accuracy gain. If that holds it says the metric this entire line was selected on does not measure
# what the line is for. It is also only ~1.6 sigma on a conservative unpaired standard error.
#
#   A  downstream on ce_free_0_1_2 and ce_free2          ~1h    no training. Two more paired
#                                                                (BPB, accuracy) points, from
#                                                                checkpoints that already exist, on
#                                                                the two free sets never scored. The
#                                                                cheapest possible test of the
#                                                                divergence and it completes the table.
#   B  attention replicate, data seed 1, {0,1,14,15} 50M ~1h40  the headline is 1.6 sigma. Replicate
#                                                                it before anything is built on it.
#   C  attention on {0,1,15} 50M                         ~1h40  substitution: if attention delivers
#                                                                the same downstream gain on the
#                                                                cheaper free set, it replaces the
#                                                                fourth freed layer and its +43.8
#                                                                points of resident memory. This is
#                                                                the deployment question.
#   D  attention WITHOUT expert LoRA, {0,1,14,15} 50M    ~1h40  8.4M attention parameters against
#                                                                235M of expert LoRA. Asks whether
#                                                                the expensive adapter is the one
#                                                                doing the work.
#
# Order is replicate-before-build: A is free and completes a table, B decides whether B's premise
# survives, and C and D are only worth their GPU time if it does. They run regardless -- a null in B
# makes C and D the evidence that it was a null -- but the order means the decisive one lands first.
#
# DEADLINE. No cell STARTS unless its estimated duration fits before 18:00 UTC. A cell already
# running is left alone; killing a nearly-finished run to honour a clock wastes more than it saves.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/olmoe-adapt/venv/bin/python}
DATA=/workspace/olmoe-adapt/data
LOGS=/workspace/olmoe-adapt
DEADLINE=$(date -u -d "2026-08-03 18:00:00" +%s)

_left() { echo $(( (DEADLINE - $(date -u +%s)) / 60 )); }
_room() {  # $1 = minutes needed
  local l; l=$(_left)
  if [ "$l" -lt "$1" ]; then
    echo "=== $(date -u +%H:%MZ) SKIP: needs ~${1}min, ${l}min to deadline"; return 1
  fi
  return 0
}

_gitsafe() {
  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A results/ analysis/ scripts/ 2>/dev/null
  git commit -q -m "$1" 2>/dev/null
  git push -q origin ple-adaptation 2>/dev/null
  echo "[git] $1 -> $(git rev-parse --short HEAD)"
}

_wait_gpu() { while pgrep -f "analysis/residency/(train_ple|downstream)\.py" > /dev/null; do sleep 60; done; }

# Keep the volume under its quota. This is not housekeeping, it is the failure that cost two runs:
# the MooseFS volume enforces a quota that `df` does not report -- df showed 240 TB free while a
# plain write returned "Disk quota exceeded" -- and a checkpoint write hitting it killed the trainer
# mid-file twice, at byte 469762048 both times, with no traceback. Each LoRA-bearing cell writes five
# 2.95 GB checkpoints, so four queued cells are ~60 GB. Every number a finished cell produced lives
# in its result JSON; only its LAST checkpoint is needed, to score downstream.
_prune() {
  local freed
  freed=$("$PY" - <<'PYEOF2'
import os, re, glob
os.chdir("/workspace/olmoe-adapt/data")
KEEP = {"csurf_ce_free_0_1_14_15_attn_250M_at240M.pt", "csurf_ce_free_0_1_2_at50M.pt",
        "csurf_ce_free2_at50M.pt", "csurf_ce_free_0_1_14_15_at50M.pt",
        "csurf_ce_free_0_1_14_15_attn_at50M.pt"}
cells = {}
for p in glob.glob("csurf_*_at*M.pt"):
    m = re.match(r"csurf_(.+)_at(\d+)M\.pt$", p)
    if m:
        cells.setdefault(m.group(1), []).append((int(m.group(2)), p))
n = b = 0
for fs in cells.values():
    mx = max(f[0] for f in fs)
    for tok, p in fs:
        if tok != mx and p not in KEEP:
            b += os.path.getsize(p); os.remove(p); n += 1
print(f"{n} file(s), {b / 2**30:.0f} GiB")
PYEOF2
)
  echo "[prune] removed $freed of intermediate checkpoints"
}

_downstream() {  # tag, free-set, checkpoint
  local tag=$1 fs=$2 ck=$3
  [ -f "$DATA/$ck" ] || { echo "[skip] downstream $tag (no $ck)"; return 0; }
  if grep -q ",$tag," results/ablations/layer_freeing_downstream.csv 2>/dev/null; then
    echo "[skip] downstream $tag (already scored)"; return 0
  fi
  _room 35 || return 0
  echo "=== $(date -u +%H:%MZ) downstream $tag"
  "$PY" analysis/residency/downstream.py --csurf "$ck" --free-set "$fs" --tag "$tag" \
        > "$LOGS/ds_${tag}.log" 2>&1
  if [ $? -eq 0 ]; then
    grep -E "^\[ds\] (attention|identity|warn|mean)" "$LOGS/ds_${tag}.log"
    _gitsafe "downstream 10-task: $tag"
  else
    echo "[FAIL] downstream $tag; tail:" >&2; tail -20 "$LOGS/ds_${tag}.log" >&2
  fi
}

_cell() {  # tag, free-set, data-seed, expert-lora-rank, attn-lora-rank
  local tag=$1 fs=$2 ds=$3 lora=$4 attn=$5
  if [ -f "$DATA/ple_${tag}.json" ]; then echo "[skip] $tag (already has a result)"; return 0; fi
  _room 100 || return 1
  echo "=== $(date -u +%H:%MZ) $tag: free {$fs} seed $ds lora=$lora attn=$attn, 50M"
  "$PY" analysis/ple/train_ple.py --tag "$tag" --rank off --lora "$lora" --lora-attn "$attn" \
        --free-set "$fs" --data-seed "$ds" --tokens 50000000 --eval-every 10000000 --mb 16 \
        > "$LOGS/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$DATA/ple_${tag}.json" ]; then
    echo "[FAIL] $tag rc=$rc; tail:" >&2; tail -25 "$LOGS/${tag}.log" >&2; return 1
  fi
  grep -E "^\[ple\]|^\[DONE\]" "$LOGS/${tag}.log"
  "$PY" analysis/residency/consolidate.py > /dev/null || { echo "[FAIL] consolidate aborted" >&2; return 1; }
  _gitsafe "$tag: free {$fs}, expert LoRA $lora, attention LoRA $attn, seed $ds, 50M"
}

echo "=== $(date -u +%H:%MZ) autonomous queue: waiting for the GPU ($(_left)min to deadline)"
_wait_gpu
echo "=== $(date -u +%H:%MZ) GPU clear, $(_left)min to deadline"

# ---- A: score what already exists -----------------------------------------------------------------
_downstream ce_free_0_1_2 "0,1,2"  csurf_ce_free_0_1_2_at50M.pt
_downstream ce_free2      "0,1"    csurf_ce_free2_at50M.pt

# ---- B: replicate the attention result --------------------------------------------------------------
_prune
_cell ce_free_0_1_14_15_attn_ds1 "0,1,14,15" 1 32 32 \
  && _downstream ce_free_0_1_14_15_attn_ds1 "0,1,14,15" csurf_ce_free_0_1_14_15_attn_ds1_at50M.pt

# ---- C: does attention substitute for the fourth freed layer? ---------------------------------------
_prune
_cell ce_free_0_1_15_attn "0,1,15" 0 32 32 \
  && _downstream ce_free_0_1_15_attn "0,1,15" csurf_ce_free_0_1_15_attn_at50M.pt

# ---- D: does attention substitute for the expert adapter? -------------------------------------------
_prune
_cell c_free_0_1_14_15_attnonly "0,1,14,15" 0 0 32 \
  && _downstream c_free_0_1_14_15_attnonly "0,1,14,15" csurf_c_free_0_1_14_15_attnonly_at50M.pt

# ---- the table the whole queue exists to fill -------------------------------------------------------
"$PY" - <<'PYEOF'
import csv, glob, json, os
D = "/workspace/olmoe-adapt/data"
BASE, IMPOSE = 0.6727, 2.7507
ds = {}
p = "results/ablations/layer_freeing_downstream.csv"
if os.path.exists(p):
    lines = [l for l in open(p) if not l.lstrip().lstrip('"').startswith("#")]
    for r in csv.DictReader(lines):
        if r["metric"] == "acc":
            ds.setdefault(r["cell"], []).append(float(r["cell_acc"]))
rows = []
for f in glob.glob(os.path.join(D, "ple_*.json")):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if "final_bpb" not in r or str(r.get("rank")) != "off":
        continue
    fs = r.get("free_set") or ",".join(str(i) for i in range(r.get("free_layers", 0)))
    if not fs:
        continue
    n = len([x for x in fs.split(",") if x.strip()])
    acc = sum(ds[r["tag"]]) / len(ds[r["tag"]]) if r["tag"] in ds else None
    rows.append((r["final_bpb"], r["tag"], fs, (16 - n) * 8 + n * 64, r["train_tokens"],
                 r.get("lora", 0), r.get("lora_attn", 0), r.get("data_seed", 0), acc))
print(f"\n{'cell':32}{'free':11}{'slots':>6}{'tok':>6}{'xLoRA':>7}{'aLoRA':>6}{'sd':>3}"
      f"{'BPB':>10}{'rec':>8}{'acc':>8}")
for b, tag, fs, slots, tok, lo, at, sd, acc in sorted(rows):
    a = f"{acc:.4f}" if acc is not None else "  --  "
    print(f"{tag:32}{fs:11}{slots:>6}{tok // 10**6:>5}M{lo:>7}{at:>6}{sd:>3}"
          f"{b:>10.6f}{(1 - (b - BASE) / (IMPOSE - BASE)) * 100:>7.2f}%{a:>8}")
print("\nBPB lower better; rec = fraction of the constraint's BPB damage undone; acc = mean 0-shot")
print("accuracy over 10 tasks, higher better (base-free 0.6823, imposed-untrained 0.3164).")
PYEOF

_gitsafe "autonomous queue complete"
echo "=== AUTONOMOUS QUEUE COMPLETE $(date -u +%H:%MZ), $(_left)min to deadline ==="
