#!/bin/bash
# Custom merged checkpoints are OUR artifacts, not re-downloadable from HF, so they get a
# network copy that survives a pod stop. Base weights are skipped: HF has them.
# They ARE reproducible from adapter + base in minutes, so this is convenience insurance,
# not the primary record -- the adapters on /workspace and Hugging Face are that.
for d in /root/models/*-merged; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); dst=/workspace/merged-ckpts/$n
  if [ -d "$dst" ] && [ "$(du -sb "$d" | cut -f1)" = "$(du -sb "$dst" 2>/dev/null | cut -f1)" ]; then
    echo "$(date -u +%H:%M) skip $n (already backed up)"; continue
  fi
  echo "$(date -u +%H:%M) backing up $n -> network"
  cp -r "$d" "$dst".tmp && mv "$dst".tmp "$dst" && echo "$(date -u +%H:%M) done $n"
done
echo "$(date -u +%H:%M) BACKUP COMPLETE"
