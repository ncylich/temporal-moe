#!/usr/bin/env bash
# Fills: (1) full surface at the matched Skliar C=8 points (gemma lam0.5, qwen lam0.4);
# (2) qwen per-token transfer histogram at its matched point; (3) gemma C8 lam0.1/0.2.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 TMOE_PRIO=4
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
CB="TEMPORAL_WALKER=cache_bias TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1 TEMPORAL_CB_C=8"
G="--model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms R8"
Q="--model qwen35_instruct --path /root/models/qwen35-35b-a3b --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 --arms R8"
# (3) gemma upper-curve points
for LAM in 0.1 0.2; do
  echo "### fill gemma C8 lam$LAM GSM8K $(date -u +%H:%M)"
  ( export $CB TEMPORAL_CB_LAMBDA=$LAM; $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as gemma4_skliar_C8_lam${LAM/./p}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90 )
done
# (2) qwen histogram at matched lam0.4
echo "### fill qwen C8 lam0.4 histogram $(date -u +%H:%M)"
( export $CB TEMPORAL_CB_LAMBDA=0.4 TEMPORAL_SWAP_HIST=/workspace/rerun-logs/skliar_c8_qwen_lam04_hist.json; $L $PY -u analysis/residency/instruct_genbench_vllm.py $Q --record-as qwen35_skliar_C8_lam0p4_hist --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90 )
# (1) full surfaces at the matched points
echo "### fill gemma C8 lam0.5 surface (IFEval,MMLU,HE,MBPP) $(date -u +%H:%M)"
( export $CB TEMPORAL_CB_LAMBDA=0.5
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $G --record-as gemma4_skliar_C8_lam0p5_full --tasks "ifeval=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
  $L $PY -u analysis/residency/mmlu_gptoss.py --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms R8 --record-as gemma4_skliar_C8_lam0p5_full_dual --gpu-mem 0.90
  $L $PY -u analysis/residency/humaneval_gemma.py --path /dev/shm/gemma4-26b-it --arms R8 --tag gemma4_skliar_C8_lam0p5_he8192 --max-tokens 8192 --max-model-len 9216
  $L $PY -u analysis/residency/mbpp_gemma.py --path /dev/shm/gemma4-26b-it --arms R8 --tag gemma4_skliar_C8_lam0p5_m8192 --max-tokens 8192 --max-model-len 9216 --gpu-mem 0.90 )
echo "### fill qwen C8 lam0.4 surface (IFEval,MMLU,code) $(date -u +%H:%M)"
( export $CB TEMPORAL_CB_LAMBDA=0.4
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $Q --record-as qwen35_skliar_C8_lam0p4_full --tasks "ifeval=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
  $L $PY -u analysis/residency/mmlu_gptoss.py --model qwen35_instruct --path /root/models/qwen35-35b-a3b --think off --arms R8 --record-as qwen35_skliar_C8_lam0p4_n_dual --gpu-mem 0.90
  $L $PY -u analysis/residency/instruct_genbench_vllm.py $Q --record-as qwen35_skliar_C8_lam0p4_code --tasks "mbpp_instruct=0,humaneval_instruct=0" --gen-cap 1536 --max-model-len 4096 --gpu-mem 0.90 )
echo "### matched fill ALL DONE $(date -u +%H:%M)"
