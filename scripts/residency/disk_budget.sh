#!/bin/bash
# Enforce this project's storage budget: no more than 1TB on RAM (/dev/shm) or on local
# disk (/root/models). The box is shared -- other jobs need the space, and a 469GB tmpfs
# is RAM taken away from everyone. Run before any merge; call with --report to just print.
#
# Ordering matters when trimming: merged checkpoints are DERIVED (rebuildable from adapter
# + base in minutes and the adapters are all mirrored to HF), so they go first. Base
# weights are re-downloadable but cost bandwidth. Adapters and trajectories are NEVER
# deleted here -- they are the artifacts.
CAP_GB=${CAP_GB:-1000}
usage_gb () { du -sb "$1" 2>/dev/null | awk '{printf "%d", $1/1000000000}'; }
shm=$(usage_gb /dev/shm); loc=$(usage_gb /root/models)
echo "[budget] /dev/shm ${shm:-0}GB  /root/models ${loc:-0}GB  (cap ${CAP_GB}GB each)"
[ "${1:-}" = "--report" ] && exit 0
over=0
[ "${shm:-0}" -gt "$CAP_GB" ] && { echo "[budget] OVER on /dev/shm"; over=1; }
[ "${loc:-0}" -gt "$CAP_GB" ] && { echo "[budget] OVER on /root/models"; over=1; }
if [ "$over" = 1 ]; then
  echo "[budget] merged checkpoints are derived artifacts; delete the ones whose grids are"
  echo "[budget] already recorded in results/ablations/instruct_genbench_vllm.csv:"
  du -sh /dev/shm/*merged /root/models/*merged 2>/dev/null | sort -rh | head
  exit 3
fi
