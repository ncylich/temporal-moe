#!/usr/bin/env bash
# Rebuild the phase0 16k corpus end-to-end with the repo's own pipeline (CPU/network only):
# exact parquet shards -> data/dclm_parts/partNN.jsonl -> train tok16k (parts 00+01, same recipe)
# -> fast_tokenize all parts into data/tok16k_full. Idempotent at every stage.
set -uo pipefail; cd /workspace/temporal-moe
. scripts/env.sh
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 TMOE_ROOT=/workspace/temporal-moe
N_PARTS=${1:-24}
echo "### tok16k rebuild: download parts 0..$((N_PARTS-1)) $(date -u +%H:%M)"
.venv/bin/python experiments/data/download_parts.py 0 $((N_PARTS-1)) 4
echo "### tok16k rebuild: train tokenizer $(date -u +%H:%M)"
[ -f data/tok16k/tokenizer.json ] || .venv/bin/python experiments/data/train_tok16k.py
echo "### tok16k rebuild: tokenize $(date -u +%H:%M)"
PARTS_GLOB="$TMOE_ROOT/data/dclm_parts/part*.jsonl" OUT_DIR="$TMOE_ROOT/data/tok16k_full" \
  TOKENIZER_MODEL="$TMOE_ROOT/data/tok16k" EOD=0 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python experiments/data/fast_tokenize.py 12
.venv/bin/python - <<'PY'
import glob, os
bins = sorted(glob.glob("data/tok16k_full/part*_text_document.bin"))
toks = sum(os.path.getsize(b)//2 for b in bins)
print(f"parts={len(bins)} total_tokens={toks/1e9:.3f}B train@90%={0.9*toks/1e9:.3f}B")
PY
echo "### tok16k rebuild DONE $(date -u +%H:%M)"
