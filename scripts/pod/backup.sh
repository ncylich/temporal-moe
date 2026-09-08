#!/usr/bin/env bash
# Passive, low-priority backup of everything irreplaceable on the container disk to the
# network volume. On this pod the container disk is wiped when the pod stops.
#
#   ./scripts/pod/backup.sh once    one-shot (run this before stopping the pod)
#   ./scripts/pod/backup.sh         daemon, every $BACKUP_INTERVAL seconds (default 120)
#
# Deliberately NOT a full-disk copy. Excluded on purpose:
#   - the Hugging Face cache   large and re-downloadable; snapshot.sh keeps the tokens
#   - .venv                    rebuildable in minutes
#   - OS directories           supplied by the image
#
# Writes into the same tree as snapshot.sh, which is the complete on-demand version.
#
# Verified 2026-08-26 against a scratch destination: .venv / __pycache__ / *.pyc excluded
# (7.2G source copied as 1.1G), and a canary file placed in the destination survived a
# re-sync, confirming the no---delete guarantee below.
set -euo pipefail

DST="${BACKUP_DST:-/workspace/pod-snapshot/root}"
INTERVAL="${BACKUP_INTERVAL:-120}"

# Override BACKUP_SRCS (colon-separated) to back up a different set. The default is the
# Claude Code transcripts plus this repository, the repo path discovered from this script's
# own location rather than hardcoded, so a clone in a different directory still works.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IFS=':' read -ra SRCS <<< "${BACKUP_SRCS:-/root/.claude:$REPO_ROOT}"

mkdir -p "$DST"

# Liveness is checked via this pidfile, never with `pgrep -f`: that pattern matches the
# shell running it, so a check false-positives and a kill takes out the caller.
PIDFILE="${BACKUP_PIDFILE:-/workspace/tmoe_queue/pids/backup.pid}"
mkdir -p "$(dirname "$PIDFILE")"

# ionice idle class + nice 19: yields to the experiments, never competes for I/O.
# No --delete: this is a backup, not a mirror. Removing a file from the container
# disk must never remove it from the durable copy.
backup_once() {
  local src
  for src in "${SRCS[@]}"; do
    [[ -e "$src" ]] || continue
    ionice -c3 nice -n19 rsync -a --partial \
      --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
      "$src" "$DST/"
  done
}

if [[ "${1:-loop}" == "once" ]]; then
  backup_once
  echo "[$(date -u +%FT%TZ)] backup complete -> $DST ($(du -sh "$DST" 2>/dev/null | cut -f1))"
  exit 0
fi

echo $$ > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
echo "[$(date -u +%FT%TZ)] backup daemon started (pid $$) -> $DST every ${INTERVAL}s"
while true; do
  backup_once
  echo "[$(date -u +%FT%TZ)] synced ($(du -sh "$DST" 2>/dev/null | cut -f1))"
  sleep "$INTERVAL"
done
