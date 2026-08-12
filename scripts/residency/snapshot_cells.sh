#!/usr/bin/env bash
# Copy the authoritative per-cell result JSONs into the repository.
#
# consolidate.py's own header calls the per-cell JSON the trainer writes at exit "authoritative" for
# the trained_cell group -- every BPB, recovery, eval curve, swap rate and usage entropy in both
# committed CSVs is derived from these files. They live in $OLMOE/data, which is gitignored, on a
# network volume that today truncated two checkpoints and one log mid-write when it hit a quota that
# `df` does not report. If that directory is lost, the CSVs survive in git but nothing can rebuild
# or verify them, and the eval curves -- which are in the JSONs and only summarised in the CSVs --
# are gone outright.
#
# 24 files, 56 KB. There is no reason for the record of a program's results to be less durable than
# the code that produced them.
#
# The JSONs remain the live artifact; this is a snapshot, refreshed by running the script.
set -uo pipefail
cd "$(dirname "$0")/../.."
DATA=${TMOE_OLMOE_DATA:-/workspace/olmoe-adapt/data}
OUT=results/ablations/cells

mkdir -p "$OUT"
n=0
for f in "$DATA"/ple_*.json; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in ple_calib_meta*) continue ;; esac   # already tracked in results/ablations
  cp -p "$f" "$OUT/" && n=$((n + 1))
done
echo "[snapshot] $n per-cell JSON(s) -> $OUT ($(du -sh "$OUT" | cut -f1))"

# A snapshot nobody checks is a snapshot that silently goes stale, so verify it can stand in.
python3 - "$OUT" "$DATA" <<'PYEOF'
import glob, json, os, sys
out, data = sys.argv[1], sys.argv[2]
live = {os.path.basename(p) for p in glob.glob(os.path.join(data, "ple_*.json"))
        if not os.path.basename(p).startswith("ple_calib_meta")}
snap = {os.path.basename(p) for p in glob.glob(os.path.join(out, "*.json"))}
missing, extra = sorted(live - snap), sorted(snap - live)
bad = []
for p in glob.glob(os.path.join(out, "*.json")):
    try:
        json.load(open(p))
    except Exception as e:
        bad.append(f"{os.path.basename(p)}: {e}")
print(f"[snapshot] live={len(live)} snapshot={len(snap)}"
      + (f" MISSING={missing}" if missing else "")
      + (f" STALE-EXTRA={extra}" if extra else "")
      + (f" UNPARSEABLE={bad}" if bad else ""))
sys.exit(1 if (missing or bad) else 0)
PYEOF
