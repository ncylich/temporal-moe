#!/bin/bash
# One status block for the overnight run. Deliberately dumb: no functions inside command
# substitution, no pipes that can SIGPIPE, every lookup guarded. An inline version of this
# died with exit 144 and no error text, which is exactly the failure mode heartbeats are
# supposed to CATCH rather than exhibit.
#
# Process-independent by design: it reports every lane whether or not anything is running,
# so a dead job reads as DOWN instead of as silence.
L=/workspace/rerun-logs
PS=$(ps -eo cmd 2>/dev/null)

status_of () {          # $1 = process pattern, $2 = logfile
  local st="DOWN" stage step
  case "$PS" in *"$1"*) st="up" ;; esac
  if [ -f "$2" ]; then
    stage=$(grep -E "^### |^\[chain\] " "$2" 2>/dev/null | tail -1)
    stage=${stage#\#\#\# }
    stage=${stage#\[chain\] }
    step=$(grep -oE "step [0-9]+ seen [0-9.]+M loss [0-9.]+" "$2" 2>/dev/null | tail -1)
  fi
  [ -n "$stage" ] || stage="no marker yet"
  [ -z "$step" ] || stage="$stage | $step"
  echo "$st|$stage"
}

echo "HEARTBEAT $(date -u '+%H:%M UTC')"
echo "CORE GOAL: adapters FULLY fixed, trained, merged, re-measured, with GOOD results --"
echo "  not merely 'ran'. Verify every merge with verify_merge.py (diff merged vs base):"
echo "  a LoRA silently attached to the wrong module tree already produced one"
echo "  'successful' merge that carried no attention LoRA at all."
for entry in \
  "gemma-train:train_adapters.sh gemma:$L/adapt_gemma.out" \
  "qwen-train:train_adapters.sh qwen:$L/adapt_qwen.out" \
  "gemma-merge:night_chain.sh gemma-merge:$L/chain_gemma-merge.out" \
  "qwen-merge:night_chain.sh qwen-merge:$L/chain_qwen-merge.out" \
  "qwen-think:regen_trajectories.sh qwen-think:$L/traj_qwen_think.out" \
  "gpu0-chain:night_chain.sh gpu0-think:$L/chain_gpu0-think.out"
do
  lbl=${entry%%:*}; rest=${entry#*:}; pat=${rest%%:*}; log=${rest#*:}
  res=$(status_of "$pat" "$log")
  printf '  [%-11s|%-4s] %s\n' "$lbl" "${res%%|*}" "${res#*|}"
done
echo "GPU:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'
echo "NEXT AFTER ADAPTERS: (1) OLMoE item  (2) BASELINE_METHODS_COMPARISON.md -- read,"
echo "  execute, run it. No published competitor (skliar/cosmoe/promoe/pregated/"
echo "  oracle-moe/eliseev/blockffn) is implemented anywhere yet. Update the doc as fit."
echo "NOTE: GPU 0 usable from 08:30 UTC (01:30 PDT)."
