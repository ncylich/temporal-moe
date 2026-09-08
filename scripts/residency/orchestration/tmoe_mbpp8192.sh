#!/bin/bash
# MBPP at 8192. Every MBPP cell so far used the 1536 default. HumanEval at 8192 showed the
# adapter eliminating code residency damage entirely (+0.0 gap vs base -4.9) where at 1536
# it showed nothing -- so the "code damage is structural" conclusion may be an artifact of
# the generation cap. This is the test.
set -euo pipefail
export MAXTOK=8192 MML=9216
exec /workspace/tmoe_mbpp.sh "$@"
