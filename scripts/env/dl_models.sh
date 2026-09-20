set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
P=/workspace/venv_vllm312/bin/python
$P - <<'PY'
from huggingface_hub import snapshot_download
for mid, dest in [("google/gemma-4-26B-A4B-it", "/dev/shm/gemma4-26b-it"),
                  ("Qwen/Qwen3.5-35B-A3B",      "/dev/shm/qwen35-35b-a3b")]:
    print("=== downloading", mid, "->", dest, flush=True)
    p = snapshot_download(mid, local_dir=dest, max_workers=16)
    print("=== done", p, flush=True)
PY
echo "DL_DONE"
