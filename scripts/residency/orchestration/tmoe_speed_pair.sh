#!/usr/bin/env bash
bash /workspace/tmoe_speed.sh qwen ${1:-fla_kv065} 2>&1 | grep -E "^\[speed\]|^### "
bash /workspace/tmoe_speed.sh gemma ${1:-fla_kv065} 2>&1 | grep -E "^\[speed\]|^### "
