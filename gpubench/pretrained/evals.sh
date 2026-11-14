#!/usr/bin/env bash
# Accuracy for the 1e19 fine models on the reference forward (run on the eval pod, after both
# safetensors are in /workspace). Stages skip when their output exists, so a rerun resumes.
#   bash gpubench/pretrained/evals.sh
set -uo pipefail
cd /workspace/tm
W=/workspace; O=/workspace/out; mkdir -p $O
MOE=$W/moe_fine_1e19.safetensors; TMOE=$W/fine_1e19.safetensors
RM="python3 scripts/serving_convert/ref_model.py --device cuda --seq 2048 --batch 16"
SWEEP="--tokens $W/test_tokens_2048.npz --start 1536 --n 64"        # held out from the 512-row CE eval
TASKS=arc_challenge,arc_easy,hellaswag,openbookqa,piqa,winogrande,boolq,copa,lambada_openai,sciq
done_() { [ -s "$1" ] && grep -q "^loss " "$1"; }

echo "### gate $(date -u +%H:%M)"
[ -s $O/gate_moe.log ] || python3 gpubench/pretrained/gate_skliar_ref.py --weights $MOE --tokens $W/test_tokens.npz --n 4 --seq 256 --lam 0.5 --device cuda > $O/gate_moe.log 2>&1
tail -1 $O/gate_moe.log

echo "### sweep $(date -u +%H:%M)"
done_ $O/sweep_ours.log || $RM --weights $MOE $SWEEP --routing temporal > $O/sweep_ours.log 2>&1
for L in 0.05 0.1 0.2 0.3 0.5 0.75 1.0 1.5 2.0 3.0; do
  done_ $O/sweep_skliar_$L.log || $RM --weights $MOE $SWEEP --routing skliar --cb-lambda $L > $O/sweep_skliar_$L.log 2>&1
done
python3 gpubench/pretrained/pick_lambda.py $O/sweep_ours.log $O/sweep_skliar_*.log | tee $O/lambda.txt
LAM=$(grep -oE "^LAMBDA [0-9.]+" $O/lambda.txt | awk '{print $2}')
[ -n "$LAM" ] || { echo "no lambda"; exit 1; }
# the interpolated lambda gave 0.958 loads against ours 1.000: bisect between the grid point above and it
TGT=$(grep -oE "loads per token-layer [0-9.]+" $O/sweep_ours.log | awk '{print $4}')
[ -s $O/bisect.txt ] || python3 gpubench/pretrained/bisect_lambda.py --target $TGT --lo 0.75 --hi $LAM --log-dir $O -- \
    --device cuda --seq 2048 --batch 16 --weights $MOE $SWEEP | tee $O/bisect.txt
LAM=$(grep -oE "^LAMBDA [0-9.]+" $O/bisect.txt | awk '{print $2}')
echo "lambda* $LAM"

echo "### test CE, 512 rows $(date -u +%H:%M)"
CE="--tokens $W/test_tokens.npz --n 512"
done_ $O/ce_moe_topk.log    || $RM --weights $MOE  $CE --routing topk     > $O/ce_moe_topk.log 2>&1
done_ $O/ce_moe_ours.log    || $RM --weights $MOE  $CE --routing temporal > $O/ce_moe_ours.log 2>&1
done_ $O/ce_moe_skliar.log  || $RM --weights $MOE  $CE --routing skliar --cb-lambda $LAM > $O/ce_moe_skliar.log 2>&1
done_ $O/ce_tmoe_trained.log || $RM --weights $TMOE $CE --routing temporal > $O/ce_tmoe_trained.log 2>&1
grep -hE "^routing|loads per|^loss" $O/ce_*.log

echo "### lm-eval, 10 tasks $(date -u +%H:%M)"
lme() { # name weights routing [lambda]
  [ -s $O/lme_$1/done ] && return
  mkdir -p $O/lme_$1
  python3 gpubench/pretrained/lmeval_ref.py --model refmoe \
    --model_args weights=$2,routing=$3,cb_lambda=${4:-0.5},batch=64,loads_out=$O/lme_$1/loads.jsonl \
    --tasks $TASKS --num_fewshot 0 --seed 42 --output_path $O/lme_$1 > $O/lme_$1.log 2>&1 && touch $O/lme_$1/done
}
lme moe_skliar $MOE skliar $LAM & lme moe_topk $MOE topk & lme moe_ours $MOE temporal & lme tmoe_trained $TMOE temporal & wait
for n in moe_topk moe_ours moe_skliar tmoe_trained; do echo "== $n"; grep -E "^\|" $O/lme_$n.log | head -40; done
echo "### ALL DONE $(date -u +%H:%M)"
