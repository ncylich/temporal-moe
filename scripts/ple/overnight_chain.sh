#!/usr/bin/env bash
# Layer-freeing: close the controlled comparison, replicate the headline, then take the winner to 200M
# and score it downstream.
#
# `layer_freeing_RESULTS.md` §4 leaves one question open and it is the load-bearing one. Layers 2 and
# 15 have near-identical solo damage (0.14084, 0.14076) and therefore near-identical additive
# predictions, yet freed alongside {0,1} the training-free recovery is 0.573 for layer 2 and 0.409 for
# layer 15 -- a factor of 5.8 at identical memory. Only {0,1,15} was ever trained. So we do not know
# whether the damage profile predicts trained outcomes, and §4 says as much: "Without it I cannot say
# whether the damage profile is a useful design tool or a misleading one."
#
# Five stages, run in order, each skipped if its artifact already exists:
#
#   A  ce_free_0_1_2        50M   the cancelled controlled cell: same recipe, budget and memory as
#                                 the headline, differing only in whether the third freed layer is 2
#                                 or 15. This is the comparison, and it is the whole point of A.
#   B  ce_free_0_1_14_15    50M   both ends freed at +175% memory. {0,1,2} already dominates
#                                 {0,1,14,15} training-free (more recovery, less memory); this asks
#                                 whether training reverses that, as it reversed the {0,1}->{0,1,15}
#                                 ordering.
#   C  ce_free_0_1_15_ds1   50M   replicate of the published headline on a different data draw. The
#                                 headline claim -- 0.797810 beats F' = 0.810600, a full 6.92B-param
#                                 finetune -- is a margin of 0.0128 against a pre-registered noise bar
#                                 of 0.012. It clears by 6%. PLE_PLAN.md §10 says a comparison landing
#                                 at the bar gets a seed replicate; this is that replicate.
#   D  <winner> -> 200M           the best 50M free-set cell, extended. Resumed from its own 50M
#                                 checkpoint rather than retrained: LR is constant with no warmup or
#                                 schedule, and --resume-c restores the optimizer state AND the data
#                                 cursor, so continuing is arithmetically the same run and saves 45
#                                 minutes. Evals stay on the 10M cadence because the eval trigger is
#                                 `seen // eval_every > len(hist)` and hist arrives with 5 entries --
#                                 a 50M cadence here would fire nothing before 250M.
#   E  downstream                 10-task 0-shot on the 200M model, then on the same cell at 50M so
#                                 the token effect is separable from the freeing effect. Scored
#                                 against the published base-free / impose-R8 / CE-adapted columns.
#
# The winner is chosen by the rule below BEFORE any of it runs, from the artifacts on disk, and is
# printed with its margin. Ordering by BPB alone would silently prefer +175% memory over +131% for a
# difference the program calls noise, so a tie inside the 2sigma bar goes to the cheaper cell.
#
#   ARMS=A,C scripts/ple/overnight_chain.sh     # subset
#   scripts/ple/overnight_chain.sh              # everything
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/olmoe-adapt/venv/bin/python}
DATA=/workspace/olmoe-adapt/data
LOGS=/workspace/olmoe-adapt
BAR=0.012                                  # 2 sigma on this slice; smaller differences are noise

IFS=',' read -ra STAGES <<< "${ARMS:-A,B,C,D,E}"
_want() { for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

_gitsafe() {  # another queue shares this repo; wait for the index rather than racing it
  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A results/ analysis/ 2>/dev/null
  git commit -q -m "$1" 2>/dev/null
  git push -q origin ple-adaptation 2>/dev/null
  echo "[git] $1 -> $(git rev-parse --short HEAD)"
}

_consolidate() {
  if ! "$PY" analysis/ple/consolidate.py; then
    echo "[FAIL] consolidate.py aborted -- results NOT committed for this stage" >&2
    return 1
  fi
}

# ---- a 50M free-set cell ------------------------------------------------------------------------
_cell() {  # tag, free-set, data-seed
  local tag=$1 fs=$2 ds=$3
  if [ -f "$DATA/ple_${tag}.json" ]; then echo "[skip] $tag (already has a result)"; return 0; fi
  echo "=== $(date +%H:%M) $tag: free {$fs}, data-seed $ds, 50M tokens"
  "$PY" analysis/ple/train_ple.py --tag "$tag" --rank off --lora 32 --free-set "$fs" \
        --data-seed "$ds" --tokens 50000000 --eval-every 10000000 --mb 16 \
        > "$LOGS/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$DATA/ple_${tag}.json" ]; then
    echo "[FAIL] $tag rc=$rc; tail of $LOGS/${tag}.log:" >&2
    tail -20 "$LOGS/${tag}.log" >&2
    return 1
  fi
  grep -E "^\[eval\]|^\[DONE\]" "$LOGS/${tag}.log" | tail -6
  _consolidate && _gitsafe "$tag: 50M, MoE layers $fs unconstrained"
}

_want A && { _cell ce_free_0_1_2     "0,1,2"    0 || exit 1; }
_want B && { _cell ce_free_0_1_14_15 "0,1,14,15" 0 || exit 1; }
_want C && { _cell ce_free_0_1_15_ds1 "0,1,15"  1 || exit 1; }

# ---- pick the winner from the artifacts ---------------------------------------------------------
# Rules are in pick_free_set.py and were fixed before any of the cells they rank had run.
SEL=$("$PY" analysis/ple/pick_free_set.py "$DATA" "$BAR")
read -r WIN_TAG WIN_FS WIN_BPB WHY <<< "$SEL"
echo "=== winner: $WIN_TAG (free {$WIN_FS}) at BPB $WIN_BPB  [$WHY]"
[ "$WIN_TAG" = "NONE" ] && { echo "[FAIL] no eligible cell to extend" >&2; exit 1; }

# ---- D: extend the winner to 200M ---------------------------------------------------------------
EXT="${WIN_TAG}_200M"
if _want D; then
  if [ -f "$DATA/ple_${EXT}.json" ]; then
    echo "[skip] $EXT (already has a result)"
  elif [ ! -f "$DATA/csurf_${WIN_TAG}_at50M.pt" ]; then
    echo "[FAIL] no 50M checkpoint for $WIN_TAG to resume from" >&2; exit 1
  else
    echo "=== $(date +%H:%M) $EXT: resume $WIN_TAG at 50M, train to 200M"
    "$PY" analysis/ple/train_ple.py --tag "$EXT" --rank off --lora 32 --free-set "$WIN_FS" \
          --data-seed 0 --tokens 200000000 --eval-every 10000000 --mb 16 \
          --resume-c "$DATA/csurf_${WIN_TAG}_at50M.pt" > "$LOGS/${EXT}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$DATA/ple_${EXT}.json" ]; then
      echo "[FAIL] $EXT rc=$rc; tail:" >&2; tail -20 "$LOGS/${EXT}.log" >&2; exit 1
    fi
    grep -E "^\[resume\]|^\[eval\]|^\[DONE\]" "$LOGS/${EXT}.log" | tail -8
    _consolidate && _gitsafe "$EXT: winner extended 50M -> 200M"
  fi
fi

# ---- E: downstream ------------------------------------------------------------------------------
# 200M first: it is the headline, and if the night runs out the control is the cheaper thing to lose.
if _want E; then
  for spec in "$EXT:csurf_${EXT}_at200M.pt" "$WIN_TAG:csurf_${WIN_TAG}_at50M.pt"; do
    tg=${spec%%:*}; ck=${spec##*:}
    if [ ! -f "$DATA/$ck" ]; then echo "[skip] downstream $tg (no $ck)"; continue; fi
    if grep -q ",$tg," results/ablations/layer_freeing_downstream.csv 2>/dev/null; then
      echo "[skip] downstream $tg (already scored)"; continue
    fi
    echo "=== $(date +%H:%M) downstream $tg"
    "$PY" analysis/ple/downstream.py --csurf "$ck" --free-set "$WIN_FS" --tag "$tg" \
          > "$LOGS/ds_${tg}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[FAIL] downstream $tg rc=$rc; tail:" >&2; tail -20 "$LOGS/ds_${tg}.log" >&2; continue
    fi
    grep -E "^\[ds\] (identity|mean|wrote)" "$LOGS/ds_${tg}.log"
    _gitsafe "downstream 10-task: $tg"
  done
fi

echo "=== CHAIN COMPLETE $(date +%H:%M) ==="
