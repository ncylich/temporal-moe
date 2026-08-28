#!/usr/bin/env bash
# End-to-end smoke of the in-process on-policy sampler. Runs after tmoe_online_smoke.sh (parity).
#  1. short REAL training from the W=3 adapter: +120k tokens (~11 steps), refresh every 4 steps
#     x 64 rows (4 refreshes incl. step 0), reverse-KL + anchor, no CE
#  2. merge + verify + GSM8K n=1319 (record gemma4_ce_onlinesmoke_n1319; should sit near W=3's +3.6)
#  3. timing table from the log and hard checks (online_e2e_check.py)
set -uo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; D=/workspace/olmoe-adapt/data; B=/dev/shm/gemma4-26b-it
until grep -q "### smoke DONE rc=0" /workspace/rerun-logs/online_smoke.out 2>/dev/null; do sleep 30; done
grep -q "identical to the merged checkpoint" /workspace/rerun-logs/online_smoke.out || { echo "### e2e ABORT: parity smoke did not report"; exit 2; }
KL=/workspace/instruct-traj/gemma4_d7_seq4096_klref.pt
COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16"
A=$D/gemma_ce_onlinesmoke_adapter.pt; M=/root/models/gemma4-onlinesmoke-merged
SEEN=$($PY -c "import torch,sys; print(int(torch.load(sys.argv[1], weights_only=False, map_location='cpu')['seen']))" $D/gemma_ce_digit3_adapter.pt)
echo "### e2e 1/3 short online training from digit3: +120k tokens, refresh every 4 steps x 64 rows $(date -u +%H:%M) (expect ~9 min incl. load)"
rm -f $A $A.done; cp $D/gemma_ce_digit3_adapter.pt $A
$L $PY -u analysis/residency/train_gemma_ce.py $COMMON --out $A --resume --accum 16 --lr 3e-5 --tokens $((SEEN+120000)) \
  --kl-only --kl-anchor $KL --kl-weight 0.05 --aux-loss revkl --aux-kl-weight 1.0 --online-every 4 --online-n 64 --save-every 1000000
echo "### e2e 2/3 merge + verify + GSM8K n=1319 $(date -u +%H:%M) (expect ~12 min)"
rm -rf $M; $L $PY analysis/residency/train_gemma_ce.py $COMMON --out $A --merge-out $M && cp $B/processor_config.json $M/ 2>/dev/null
$PY analysis/residency/verify_merge.py --base $B --merged $M
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_onlinesmoke_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### e2e 3/3 checks $(date -u +%H:%M)"
$PY analysis/residency/online_e2e_check.py /workspace/rerun-logs/online_smoke.out /workspace/rerun-logs/online_e2e.out
echo "### e2e DONE rc=$? $(date -u +%H:%M)"
