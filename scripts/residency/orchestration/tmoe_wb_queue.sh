#!/bin/bash
# WritingBench for arm C, queued behind the HumanEval job already on GPU 3.
while pgrep -f "[h]umaneval_gemma.py --path /root/models/gemma4-realmath-merged" >/dev/null; do
  sleep 60
done
echo "$(date -u +%H:%M) HumanEval done -> starting WritingBench for arm C"
GPU=3 exec /workspace/temporal-moe/scripts/residency/wb_arm.sh \
    /root/models/gemma4-realmath-merged gemma4_realmath R8,R16
