#!/usr/bin/env bash
# Thinking ON, same recipe and sampled-token budget as the think-off full-pool run (equal compute; thinking rows are
# ~4x longer so fewer prompts are covered). Sampler cap 8192 / context 10240 (qwen thinking runs 3-6k tok on non-GSM8K prompts; 4096 would truncate mid-think — see think_ablation_summary.csv cap-hits). max_num_seqs stays 256. Evals with --think on at 16384-token caps:
# GSM8K free/R8/R16(R32) first; the full surface only if GSM8K R8 beats the think-off base R8 damage by >1 pt.
#   tmoe_think_on.sh <gemma|qwen>
set -uo pipefail; cd /workspace/temporal-moe
MODEL=$1
export TMOE_PRIO=4 TMOE_KL_TEMP=2 TMOE_AUX_LOSS=revkl_full TMOE_ANCHOR_W=0 TMOE_BUDGET_ON=sampled TMOE_THINK=on TMOE_ONLINE_MML=10240
export TMOE_PROMPTS=/workspace/olmoe-adapt/data/fullpool_prompts.jsonl TMOE_QUOTA="mathlane_v2=2341,d5_fewshot=1183,domain8k=4958,codelane=2500"
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; D=/workspace/olmoe-adapt/data
if [ "$MODEL" = gemma ]; then export TMOE_LR=1e-4 TMOE_ONLINE_MEM=0.55 TMOE_ONLINE_OFFLOAD=8; CH=/workspace/tmoe_gemma_online.sh; A=$D/gemma_ce_online_scratch_e16_think_adapter.pt
  B=/dev/shm/gemma4-26b-it; G="--model gemma4_instruct --path $B --think on"; ARMS=free,R8,R16; PFX=gemma4_ce_online_think
else export TMOE_LR=3e-5 TMOE_ONLINE_MEM=0.65 TMOE_ONLINE_OFFLOAD=20; CH=/workspace/tmoe_qwen_online.sh; A=$D/qwen_ce_online_scratch_e16_think_adapter.pt
  B=/root/models/qwen35-35b-a3b; G="--model qwen35_instruct --path $B --think on --temperature 0.6 --top-p 0.95 --presence-penalty 1.5"; ARMS=free,R8,R32; PFX=qwen35_ce_online_think; fi
echo "### think-on $MODEL 1/2 train (full pool, 8.6M sampled tokens, thinking on, cap 8192) $(date -u +%H:%M)"
TMOE_NAME_SUFFIX=_think TMOE_MAXNEW=8192 bash $CH scratch 8600000 16 256 > /workspace/rerun-logs/${MODEL}_online_think.out 2>&1
rc=$?; echo "### think-on $MODEL 1/2 done rc=$rc $(date -u +%H:%M)"; [ "$rc" = 0 ] || exit 1
echo "### think-on $MODEL 2/2 GSM8K n=1319 think on, caps 16384 $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py $G --adapter $A --arms $ARMS --record-as ${PFX}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 16384 --max-model-len 18432 --gpu-mem 0.90
echo "### think-on $MODEL DONE $(date -u +%H:%M)"
