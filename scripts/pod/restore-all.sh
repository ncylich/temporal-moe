#!/usr/bin/env bash
# ONE COMMAND to bring the pod back after a stop. No flags, no env vars needed.
#
#     /workspace/temporal-moe/scripts/pod/restore-all.sh
#
# Add --check to see what it WOULD do without changing anything.
#
# The container disk (/) is wiped on stop; /workspace survives. This restores /root from
# the snapshot, re-stages the two base models, and tells you what is ready. It is safe to
# re-run: every step is skipped if already satisfied.
set -uo pipefail

SNAP="${SNAPSHOT_DIR:-/workspace/pod-snapshot}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
ok(){ printf '  \033[32mok\033[0m   %s\n' "$*"; }
todo(){ printf '  \033[33mtodo\033[0m %s\n' "$*"; }
bad(){ printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
run(){ if [ $CHECK -eq 1 ]; then todo "$1"; else shift; "$@"; fi }

echo "== 1. /root from snapshot =="
if [ ! -d "$SNAP/root" ]; then
  bad "no snapshot at $SNAP/root -- nothing to restore from"
elif [ -f /root/.gitconfig ] && [ -d /root/.ssh ]; then
  ok "/root already has credentials; skipping (pass --force to snapshot.sh to overwrite)"
else
  if [ $CHECK -eq 1 ]; then todo "restore /root from $SNAP/root (credentials, .claude, repos)"
  else "$REPO/scripts/pod/snapshot.sh" restore --force; fi
fi

echo "== 2. base models =="
stage(){ # repo-id  destination  label
  if [ -f "$2/config.json" ]; then ok "$3 already staged at $2"; return; fi
  if [ $CHECK -eq 1 ]; then todo "download $1 -> $2"; return; fi
  echo "  downloading $1 -> $2 (this is the slow part)"
  mkdir -p "$2"
  HF_TOKEN="${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}" \
  "$REPO/../venv_fla/bin/python" - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  ignore_patterns=['original/*','*.pth','*.gguf','metal/*'])
PY
  [ -f "$2/config.json" ] && ok "$3 staged" || bad "$3 download failed"
}
stage google/gemma-4-26B-A4B-it /dev/shm/gemma4-26b-it "gemma4 base"
stage Qwen/Qwen3.5-35B-A3B      /root/models/qwen35-35b-a3b "qwen3.5 base"

echo "== 3. things that should have survived on /workspace =="
for p in "$REPO/.git:this repo" \
         /workspace/venv_fla:"venv (training)" \
         /workspace/venv_vllm312:"venv (vllm)" \
         /workspace/instruct-traj:"trajectories" \
         /workspace/olmoe-adapt/data:"adapters"; do
  d="${p%%:*}"; l="${p##*:}"
  [ -e "$d" ] && ok "$l  ($(du -sh "$d" 2>/dev/null | cut -f1))" || bad "$l MISSING at $d"
done
n=$(ls /workspace/olmoe-adapt/data/*_adapter.pt 2>/dev/null | wc -l)
[ "$n" -gt 0 ] && ok "$n adapters present (merged models rebuild from these on demand)"

echo "== 4. restart the background workers =="
if [ $CHECK -eq 1 ]; then
  todo "start backup daemon and the 4 GPU queue runners"
else
  BP=/workspace/tmoe_queue/pids/backup.pid
  if [ -f "$BP" ] && kill -0 "$(cat $BP)" 2>/dev/null; then ok "backup daemon already running"
  else nohup "$REPO/scripts/pod/backup.sh" > /root/backup.log 2>&1 &
       sleep 1; ok "backup daemon started (log: /root/backup.log)"; fi
  for g in 0 1 2 3; do
    f=/workspace/tmoe_queue/pids/runner$g.pid
    if [ -f "$f" ] && kill -0 "$(cat $f)" 2>/dev/null; then ok "queue runner $g already up"
    else nohup "$REPO/scripts/residency/orchestration/tmoe_runner.sh" $g >/dev/null 2>&1 &
         ok "queue runner $g started"; fi
  done
fi

echo
echo "Done. Merged checkpoints are NOT restored by design -- rebuild any you need with"
echo "  scripts/residency/orchestration/tmoe_variant.sh, or the chain in that dir's README."
