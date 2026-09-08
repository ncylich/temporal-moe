#!/usr/bin/env bash
# Speed harness: the real on-policy training path for 32 steps (2 refreshes of 256 rows), per-4-step timing.
#   tmoe_speed.sh <gemma|qwen> [tag]     -> /workspace/rerun-logs/speed_<model>_<tag>.out + a summary line
set -uo pipefail; cd /workspace/temporal-moe
MODEL=$1; TAG=${2:-base}; OUT=/workspace/rerun-logs/speed_${MODEL}_${TAG}.out
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; D=/workspace/olmoe-adapt/data
if [ "$MODEL" = gemma ]; then
  COMMON="--model /dev/shm/gemma4-26b-it --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16"; MEM="--online-gpu-mem ${TMOE_ONLINE_MEM:-0.45}"
else
  COMMON="--model /root/models/qwen35-35b-a3b --family qwen35 --no-unsloth --traj qwen35_d7_seq4096 --max-seq 4096 --expert-lora-r 16 --opt adamw --micro-batch 16"; MEM="--online-gpu-mem ${TMOE_ONLINE_MEM:-0.65} --online-offload ${TMOE_ONLINE_OFFLOAD:-20} --online-presence-penalty ${TMOE_ONLINE_PP:-1.5}"
fi
rm -f /tmp/speed_${MODEL}_adapter.pt
echo "### speed $MODEL $TAG: 32 steps, refresh every 16 x 256, $(date -u +%H:%M)" | tee $OUT
$L $PY -u analysis/residency/train_gemma_ce.py $COMMON --out /tmp/speed_${MODEL}_adapter.pt --accum 16 --lr 1e-4 --tokens 400000 --kl-only \
  --aux-loss revkl_full --aux-kl-weight 1.0 --aux-kl-temp 2 --online-every 16 --online-n 256 --online-max-new 1024 --online-temp 0.7 \
  --online-quota "mathlane_v2=2341,d5_fewshot=1183,domain8k=1000" --budget-on sampled --log-every 4 $MEM ${TMOE_SPEED_EXTRA:-} >> $OUT 2>&1
echo "### speed $MODEL $TAG DONE rc=$? $(date -u +%H:%M)" | tee -a $OUT
$PY - "$OUT" "$MODEL" "$TAG" <<'PY'
import re, sys, statistics as st
t = open(sys.argv[1]).read()
steps = [float(x) for x in re.findall(r"window ([\d.]+) s/step", t)]
ref = [float(x) for x in re.findall(r"fresh on-policy rows in (\d+)s total", t)]
smp = [float(x) for x in re.findall(r"sampled 256 rows, \d+ tokens in \d+s \((\d+) tok/s\)", t)]
boot = re.findall(r"engine up and asleep in (\d+)s", t)
# windows containing a refresh (steps 4 after 0 and 16) are inflated; report the median of the rest
clean = sorted(steps)[: max(1, len(steps) - 2)]
print(f"[speed] {sys.argv[2]} {sys.argv[3]}: median {st.median(clean) if clean else float('nan'):.1f} s/step (train), refresh {ref} s, sampling {smp} tok/s, engine boot {boot} s, fallback={'yes' if 'Falling back to torch' in t else 'no'}")
PY
