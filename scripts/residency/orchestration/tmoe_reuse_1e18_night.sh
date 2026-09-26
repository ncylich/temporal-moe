#!/usr/bin/env bash
# Overnight 1e18 package (grain 1, paper's config, speed recipe): 1/6 + coherence loss, same-environment
# free MoE, half policy, 1/6 replicate. Log: /workspace/rerun-logs/curriculum_reuse_night.out
set -uo pipefail; cd "$(dirname "$0")/../../.."
for A in WK5C0p01 C0 WK3 WK5b; do ARM=$A GRAIN=1 bash scripts/residency/orchestration/tmoe_curriculum_1e18.sh; done
echo "### night ALL DONE $(date -u +%H:%M)"
