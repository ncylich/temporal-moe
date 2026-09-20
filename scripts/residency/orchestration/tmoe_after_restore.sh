#!/usr/bin/env bash
# container restarted 20:31: stage the gemma base back into /dev/shm (restore-all is idempotent), then relaunch in order
cd /workspace/temporal-moe
echo "### restore start $(date -u +%H:%M)"
bash scripts/pod/restore-all.sh
[ -f /dev/shm/gemma4-26b-it/config.json ] || { echo "### restore FAILED: no base model"; exit 2; }
echo "### restore done $(date -u +%H:%M); base $(du -sh /dev/shm/gemma4-26b-it | cut -f1)"
TMOE_PRIO=0 nohup bash /workspace/tmoe_online_smoke_eager.sh >> /workspace/rerun-logs/online_smoke_eager.out 2>&1 & echo $! > /workspace/pids/online_smoke_eager.pid
sleep 2
TMOE_PRIO=3 nohup bash /workspace/tmoe_gemma_onpol.sh digit3 mix >> /workspace/rerun-logs/gemma_onpol.out 2>&1 & echo $! > /workspace/pids/gemma_onpol.pid
echo "### relaunched: eager parity smoke (prio 0), mix merge/eval (prio 3) $(date -u +%H:%M)"
