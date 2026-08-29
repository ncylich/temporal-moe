#!/usr/bin/env bash
# Finish qwen before the gemma sweep resumes: (1) wait for the class-faithfulness smoke, (2) class-confound
# measurement (textified base, text class, GSM8K free/R8 n=1319), (3) short real on-policy run with refreshes
# + GSM8K (timing + first reading), then (4) relaunch the gemma sweep and the post-sweep chain.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
until grep -q "qwen smoke DONE" /workspace/rerun-logs/online_smoke_qwen.out 2>/dev/null; do sleep 30; done
echo "### qwen-finish 1/3 class confound: textified raw base (text class) GSM8K free,R8 n=1319 $(date -u +%H:%M)"
TMOE_PRIO=3 scripts/residency/gpu_lease.sh /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct --path /root/models/qwen35-base-text \
  --arms free,R8 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 --record-as qwen35_instruct_textclass_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90 > /workspace/rerun-logs/qwen_textclass_base.out 2>&1
echo "### qwen-finish 1/3 done rc=$? $(date -u +%H:%M)"
echo "### qwen-finish 2/3 short on-policy run (0.45M sampled tokens, ~2 refreshes) + GSM8K free/R8/R32 n=1319 $(date -u +%H:%M)"
TMOE_PRIO=3 TMOE_AUX_LOSS=revkl_full TMOE_NAME_SUFFIX=_e2e bash /workspace/tmoe_qwen_online.sh scratch 450000 16 256 > /workspace/rerun-logs/qwen_online_e2e.out 2>&1
echo "### qwen-finish 2/3 done rc=$? $(date -u +%H:%M)"
grep -E "\[online\] (offloaded|restored|wake|sampled|refresh)|\[gce\] step" /workspace/rerun-logs/qwen_online_e2e.out | tail -12
echo "### qwen-finish 3/3 relaunch the gemma sweep + post-sweep chain $(date -u +%H:%M)"
nohup /workspace/venv_vllm312/bin/python -u analysis/residency/sweep_online.py --aux-loss revkl_full --prio 4 --best gemma4_ce_online_scratch_e16_lr1e-4_n1319 --cells lr2e-4,klT2,temp1.0,refresh8x128,budget6.8M > /workspace/rerun-logs/sweep_online.out 2>&1 & echo $! > /workspace/pids/sweep_online.pid
sleep 5; nohup bash /workspace/tmoe_post_sweep_queue.sh > /workspace/rerun-logs/post_sweep_queue.out 2>&1 & echo $! > /workspace/pids/post_sweep_queue.pid
echo "### qwen-finish ALL DONE $(date -u +%H:%M)"
