#!/usr/bin/env bash
# After the reference cell: measure the untrained baseline, then the best-known configuration.
#
#   1. ce_auxfix_50M            already running -- CE surface, full residency, no attention LoRA.
#                               The reference point under the unified aux.
#   2. effective-experts baseline   untrained model, both regimes, ~10 min GPU.
#   3. ce_auxfix_free_attn_50M  CE + free {0,1,14,15} + attention LoRA r32, 50M.
#
# Step 3 has a direct comparator: ce_free_0_1_14_15_attn is the identical configuration under the
# OLD aux at 0.785201, so the difference is the aux unification on the best cell in the program.
#
# Step 3 is GATED on step 1 looking sane, because chaining blindly onto a broken reference would
# produce a second cell nobody can interpret. The gate is deliberately loose -- it catches a cell
# that failed or collapsed, not one that merely came out where we did not expect.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/olmoe-adapt/venv/bin/python}
export TMOE_OLMOE_HOME=${TMOE_OLMOE_HOME:-/workspace/olmoe-adapt}
DATA=/workspace/olmoe-adapt/data
LOGS=/workspace/olmoe-adapt

_gitsafe() {
  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A results/ analysis/ scripts/ 2>/dev/null
  git commit -q -m "$1" 2>/dev/null
  git push -q origin claude/cbow-current-token-auroc-269d2e 2>/dev/null
  echo "[git] $1 -> $(git rev-parse --short HEAD)"
}

echo "=== $(date -u +%H:%MZ) waiting for the reference cell"
while pgrep -f "train_ple.py --tag ce_auxfix_50M" > /dev/null; do sleep 60; done

# ---- gate on the reference cell -------------------------------------------------------------
verdict=$("$PY" - <<'PYEOF'
import json, os, sys
p = "/workspace/olmoe-adapt/data/ple_ce_auxfix_50M.json"
if not os.path.exists(p):
    print("STOP no result JSON: the reference cell did not finish"); raise SystemExit
r = json.load(open(p))
b = r["final_bpb"]; eff = r.get("final_eff_load") or []
# Sanity, not a target. The published full-residency CE cell at 50M is 0.8269; anything in this
# window is a cell that trained. Outside it, something is wrong and a second cell will not help.
if not (0.75 < b < 0.95):
    print(f"STOP final_bpb {b:.6f} outside 0.75-0.95: the cell did not train sensibly"); raise SystemExit
if eff and min(eff) < 8:
    print(f"STOP eff_load collapsed to {min(eff):.1f} on some layer: routing degenerated"); raise SystemExit
print(f"GO final_bpb={b:.6f} eff_load_min={min(eff) if eff else float('nan'):.1f} "
      f"published_CE_50M=0.8269 delta={b-0.8269:+.6f}")
PYEOF
)
echo "  gate: $verdict"
case "$verdict" in STOP*) echo "=== chain stops here, deliberately"; exit 1 ;; esac

# ---- untrained baseline ----------------------------------------------------------------------
if [ ! -f results/ablations/effective_experts_baseline.csv ]; then
  echo "=== $(date -u +%H:%MZ) untrained effective-experts baseline"
  "$PY" analysis/ple/effective_experts_baseline.py > "$LOGS/eff_baseline.log" 2>&1 \
    && { grep -aE "^  (free|imposed)|^\[write\]" "$LOGS/eff_baseline.log"
         _gitsafe "untrained per-layer effective expert count, free and imposed regimes"; } \
    || { echo "[FAIL] baseline"; tail -12 "$LOGS/eff_baseline.log" >&2; }
fi

# ---- the best-known configuration under the unified aux ----------------------------------------
TAG=ce_auxfix_free_attn_50M
if [ -f "$DATA/ple_${TAG}.json" ]; then
  echo "[skip] $TAG (already has a result)"
else
  echo "=== $(date -u +%H:%MZ) $TAG: CE + free {0,1,14,15} + attention LoRA r32, 50M"
  # mb16 sits ~0.2 GiB inside the ceiling with the attention adapter, so expandable_segments and an
  # mb8 x accum2 fallback -- effective batch is 16 either way, so the fallback is the same cell.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  for split in "16 1" "8 2"; do
    set -- $split
    echo "  --mb $1 --accum $2"
    "$PY" analysis/ple/train_ple.py --tag "$TAG" --rank off --lora 32 --lora-attn 32 \
          --free-set "0,1,14,15" --data-seed 0 --tokens 50000000 --eval-every 10000000 \
          --mb "$1" --accum "$2" > "$LOGS/${TAG}.log" 2>&1
    [ -f "$DATA/ple_${TAG}.json" ] && break
    echo "  [warn] did not complete at mb$1"
  done
  if [ -f "$DATA/ple_${TAG}.json" ]; then
    grep -aE "^\[ple\]|^\[aux\]|^\[DONE\]" "$LOGS/${TAG}.log"
    "$PY" analysis/ple/consolidate.py > /dev/null 2>&1
    _gitsafe "$TAG: CE + free {0,1,14,15} + attention LoRA under the unified aux"
  else
    echo "[FAIL] $TAG at both micro-batches" >&2; tail -12 "$LOGS/${TAG}.log" >&2
  fi
fi

# The comparison this cell exists for.
"$PY" - <<'PYEOF'
import json, os
D = "/workspace/olmoe-adapt/data"; B, I = 0.6727, 2.7507
print(f"\n  {'cell':30}{'aux':10}{'BPB':>10}{'recovery':>10}")
for tag, aux in (("ce_free_0_1_14_15_attn", "old"), ("ce_auxfix_free_attn_50M", "unified"),
                 ("ce_free_0_1_14_15", "old"), ("ce_auxfix_50M", "unified")):
    p = os.path.join(D, f"ple_{tag}.json")
    if not os.path.exists(p):
        print(f"  {tag:30}{aux:10}{'(absent)':>10}"); continue
    b = json.load(open(p))["final_bpb"]
    print(f"  {tag:30}{aux:10}{b:>10.6f}{(1-(b-B)/(I-B))*100:>9.2f}%")
print("\n  the first two rows are the same configuration under the two aux formulas")
PYEOF
echo "=== AUXFIX CHAIN COMPLETE $(date -u +%H:%MZ) ==="
