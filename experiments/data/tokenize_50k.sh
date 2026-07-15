#!/bin/bash
set -uo pipefail
cd /workspace/FLAME-MoE
ROOT=$(pwd); NV=/usr/local/lib/python3.11/dist-packages/nvidia
export PATH=$ROOT/.venv/bin:$PATH CUDNN_PATH=$NV/cudnn \
  LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PARTS_GLOB="$ROOT/data/dclm_parts/part*.jsonl" OUT_DIR="$ROOT/data/dclm_tokenized" \
  TOKENIZER_MODEL="EleutherAI/pythia-12b" EOD=0 TOKENIZERS_PARALLELISM=false HF_TOKEN=${HF_TOKEN:-}
echo "=== 50k tokenize start $(date) ==="
python experiments/data/fast_tokenize.py 12
python - <<'PY'
import glob, os
b=sorted(glob.glob("data/dclm_tokenized/part*_text_document.bin")); t=sum(os.path.getsize(x)//2 for x in b)
print(f"50k parts={len(b)} tokens={t/1e9:.3f}B train@90%={0.9*t/1e9:.3f}B  (need 4.45B/run)")
PY
echo "=== 50K TOKENIZE DONE $(date) ==="
