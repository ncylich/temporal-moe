#!/usr/bin/env bash
# Full save/restore of the container disk to the network volume.
#
#   ./scripts/pod/snapshot.sh save      mirror everything durable to /workspace
#   ./scripts/pod/snapshot.sh restore   rebuild /root after a pod stop
#
# On this pod the container disk (/) is WIPED when the pod stops; /workspace is a network
# volume and survives. backup.sh mirrors the small, fast-changing things every 2 min; this
# is the complete version, to run before stopping the pod and after restarting it.
#
# Everything on the container disk is copied, INCLUDING credentials (.ssh, .gitconfig,
# .git-credentials) so a restored pod needs nothing from a laptop. The exceptions are model
# weights and merged checkpoints, which are large and reproducible. HF *credentials* live
# inside the excluded cache and are saved separately, then restored with mode 600.
#
# Verified 2026-08-26 by rehearsing both directions against a synthetic tree with
# SNAPSHOT_DIR / SNAPSHOT_TARGET overridden: exclusions honoured, credentials present,
# permissions 700/.ssh 600/key 644/pub 600/git-credentials 600/hf-token, contents
# byte-identical, and the overwrite guard refuses a populated target without --force.
set -euo pipefail

SNAP="${SNAPSHOT_DIR:-/workspace/pod-snapshot}"
TARGET="${SNAPSHOT_TARGET:-/root}"          # overridable so restore can be rehearsed
HUB="$TARGET/.cache/huggingface/hub"

# The only exclusion: the entire Hugging Face cache. Excluding just the model weights
# would leave partial state behind - orphaned .locks entries, CACHEDIR.TAG, xet/ - and
# restoring that produces a cache that looks populated but has no weights. All of it is
# re-downloadable. HF *credentials* live in the same tree and ARE kept, separately.
EXCLUDES=(
  --exclude '/.cache/huggingface/'   # weights: large, re-downloadable. Tokens kept below.
  --exclude '/models/'               # merged checkpoints, owned by a separate agent
)
HF_CREDS=(token stored_tokens)

# Must be present in every snapshot. Verified after the copy, not assumed.
REQUIRED=(
  .ssh/id_ed25519 .ssh/id_ed25519.pub .ssh/known_hosts
  .gitconfig .git-credentials
)

case "${1:-}" in
save)
  mkdir -p "$SNAP/root" "$SNAP/hf-credentials"
  echo "[1/3] container disk -> $SNAP/root (Hugging Face cache excluded entirely)"
  ionice -c3 nice -n19 rsync -a --partial "${EXCLUDES[@]}" "$TARGET/" "$SNAP/root/"
  echo "[2/3] HF credentials (not the cache) -> $SNAP/hf-credentials"
  for f in "${HF_CREDS[@]}"; do
    [ -f "$TARGET/.cache/huggingface/$f" ] && cp "$TARGET/.cache/huggingface/$f" "$SNAP/hf-credentials/$f"
  done
  # Purge any HF cache left by an earlier run that excluded less; rsync never deletes.
  if [ -d "$SNAP/root/.cache/huggingface" ]; then
    echo "  purging stale HF cache from a previous snapshot"
    rm -rf "$SNAP/root/.cache/huggingface"
  fi
  echo "[3/3] verify credentials and manifest"
  missing=0
  for f in "${REQUIRED[@]}"; do
    [ -e "$TARGET/$f" ] || continue          # not on this host: nothing to save
    [ -e "$SNAP/root/$f" ] || { echo "  MISSING from snapshot: $f" >&2; missing=1; }
  done
  for f in "${HF_CREDS[@]}"; do
    [ -e "$TARGET/.cache/huggingface/$f" ] || continue
    [ -e "$SNAP/hf-credentials/$f" ] || { echo "  MISSING HF credential: $f" >&2; missing=1; }
  done
  [ "$missing" -eq 0 ] && echo "  credentials, keys, and HF tokens all present"

  {
    echo "saved:     $(date -u +%FT%TZ)"
    echo "host:      $(hostname)"
    echo "source:    $TARGET"
    echo "snapshot:  $(du -sh "$SNAP/root" | cut -f1)"
    echo "models:    NOT backed up (large and reproducible; re-stage or re-download)"
    echo "excluded:  HF weights (tokens kept), /models"
    echo "included:  credentials, SSH keys, HF tokens, non-HF caches, datasets, venv, transcripts"
  } > "$SNAP/MANIFEST.txt"
  cat "$SNAP/MANIFEST.txt"
  ;;

restore)
  [ -d "$SNAP/root" ] || { echo "no snapshot at $SNAP/root" >&2; exit 1; }
  if [ -n "$(ls -A "$TARGET" 2>/dev/null)" ] && [ "${2:-}" != "--force" ]; then
    echo "$TARGET is not empty; pass --force to overwrite" >&2; exit 1
  fi
  echo "[1/3] container disk <- $SNAP/root"
  rsync -a --partial "$SNAP/root/" "$TARGET/"
  echo "[2/3] permissions"
  chmod 700 "$TARGET/.ssh" 2>/dev/null || true
  chmod 600 "$TARGET/.ssh/id_ed25519" "$TARGET/.git-credentials" 2>/dev/null || true
  chmod 644 "$TARGET/.ssh/id_ed25519.pub" 2>/dev/null || true
  echo "[3/3] HF credentials"
  mkdir -p "$TARGET/.cache/huggingface"
  for f in "${HF_CREDS[@]}"; do
    [ -f "$SNAP/hf-credentials/$f" ] && install -m 600 "$SNAP/hf-credentials/$f" "$TARGET/.cache/huggingface/$f"
  done
  echo "  models/datasets are NOT restored - the cache starts empty by design."
  echo "  let HF re-download, or re-stage from your own model mirror."
  echo "restored. venv and repo are back; re-run scripts/pod/backup.sh in tmux."
  ;;

*) echo "usage: snapshot.sh save|restore [--force]" >&2; exit 1 ;;
esac
