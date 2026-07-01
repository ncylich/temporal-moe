#!/bin/bash
set -uo pipefail
cd /workspace/FLAME-MoE
ROOT=$(pwd); NV=/usr/local/lib/python3.11/dist-packages/nvidia
echo "=== waiting for downloads to finish $(date) ==="
while pgrep -f download_parts.py >/dev/null; do sleep 10; done
echo "=== downloads done; parts: $(ls data/dclm_parts/part*.jsonl | wc -l) $(date) ==="
export PATH=$ROOT/.venv/bin:$PATH CUDNN_PATH=$NV/cudnn \
  LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PARTS_GLOB="$ROOT/data/dclm_parts/part*.jsonl" OUT_DIR="$ROOT/data/tok16k_full" \
  TOKENIZER_MODEL="$ROOT/data/tok16k" EOD=0 TOKENIZERS_PARALLELISM=false
echo "=== tokenizing (idempotent) $(date) ==="
python scripts/phase0/fast_tokenize.py 12
echo "=== TOTAL CORPUS $(date) ==="
python - <<'PY'
import glob, os
bins = sorted(glob.glob("data/tok16k_full/part*_text_document.bin"))
toks = sum(os.path.getsize(b)//2 for b in bins)
print(f"parts={len(bins)} total_tokens={toks/1e9:.3f}B train_tokens@90%={0.9*toks/1e9:.3f}B")
PY
echo "=== G3 DATA READY $(date) ==="
