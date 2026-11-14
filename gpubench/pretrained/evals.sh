#!/usr/bin/env bash
# Accuracy for the 1e19 fine models on the reference forward (eval pod; both safetensors in /workspace).
# Protocol (ledger P1-6):
#   plain MoE (no temporal training): its own gates (softmax over all experts, not renormalised); the prompt
#     is routed by stock top-k and the rule (ours / Skliar) applies to the tokens after it;
#   trained model: its rule on every token, gates over the resident window.
# Test CE: the first 1024 tokens of a row are the prompt, the loss is on tokens 1024..2047, same for every arm.
# lm-eval: the request's context is the prompt, the continuation is scored.
# Stages skip when their output exists, so a rerun resumes.
set -uo pipefail
cd /workspace/tm
W=/workspace; O=${OUT_DIR:-/workspace/out2}; mkdir -p $O
MOE=$W/moe_fine_1e19.safetensors; TMOE=$W/fine_1e19.safetensors
RM="python3 scripts/serving_convert/ref_model.py --device cuda --seq 2048 --batch 16 --score-from 1024"
PL="--weights $MOE --gates full"; RULE="--prefill 1024"
SWEEP="--tokens $W/test_tokens_2048.npz --start 1536 --n 64"        # held out from the 512-row CE eval
TASKS=arc_challenge,arc_easy,hellaswag,openbookqa,piqa,winogrande,boolq,copa,lambada_openai,sciq
done_() { [ -s "$1" ] && grep -q "^loss " "$1"; }

echo "### gate $(date -u +%H:%M)"
[ -s $O/gate_moe.log ] || python3 gpubench/pretrained/gate_skliar_ref.py --weights $MOE --tokens $W/test_tokens.npz --n 8 --seq 512 --lam 0.5 --prefill 256 --device cuda > $O/gate_moe.log 2>&1
tail -3 $O/gate_moe.log

echo "### sweep $(date -u +%H:%M)"
done_ $O/sweep_ours.log || $RM $PL $RULE $SWEEP --routing temporal > $O/sweep_ours.log 2>&1
for L in 0.05 0.1 0.2 0.3 0.5 0.75 1.0 1.5 2.0 3.0; do
  done_ $O/sweep_skliar_$L.log || $RM $PL $RULE $SWEEP --routing skliar --cb-lambda $L > $O/sweep_skliar_$L.log 2>&1
done
python3 gpubench/pretrained/pick_lambda.py $O/sweep_ours.log $O/sweep_skliar_*.log | tee $O/lambda.txt
read BLO BHI < <(grep -oE "^BRACKET [0-9.]+ [0-9.]+" $O/lambda.txt | awk '{print $2, $3}')
TGT=$(grep -oE "loads per token-layer [0-9.]+" $O/sweep_ours.log | awk '{print $4}')
[ -s $O/bisect.txt ] || python3 gpubench/pretrained/bisect_lambda.py --target $TGT --lo $BLO --hi $BHI --log-dir $O -- \
    --device cuda --seq 2048 --batch 16 --score-from 1024 $PL $RULE $SWEEP | tee $O/bisect.txt
LAM=$(grep -oE "^LAMBDA [0-9.]+" $O/bisect.txt | awk '{print $2}')
[ -n "$LAM" ] || { echo "no lambda"; exit 1; }
echo "lambda* $LAM"

echo "### test CE, 512 rows, tokens 1024..2047 $(date -u +%H:%M)"
CE="--tokens $W/test_tokens.npz --n 512"
done_ $O/ce_moe_topk.log     || $RM $PL $CE --routing topk > $O/ce_moe_topk.log 2>&1
done_ $O/ce_moe_ours.log     || $RM $PL $RULE $CE --routing temporal > $O/ce_moe_ours.log 2>&1
done_ $O/ce_moe_skliar.log   || $RM $PL $RULE $CE --routing skliar --cb-lambda $LAM > $O/ce_moe_skliar.log 2>&1
done_ $O/ce_tmoe_trained.log || $RM --weights $TMOE $CE --routing temporal > $O/ce_tmoe_trained.log 2>&1
grep -hE "^routing|loads per|^loss" $O/ce_*.log

echo "### lm-eval, 10 tasks $(date -u +%H:%M)"
lme() { # name weights routing lambda gates prefill
  [ -e $O/lme_$1/done ] && return
  mkdir -p $O/lme_$1
  python3 gpubench/pretrained/lmeval_ref.py --model refmoe \
    --model_args weights=$2,routing=$3,cb_lambda=$4,gates=$5,prefill=$6,batch=32,loads_out=$O/lme_$1/loads.jsonl \
    --tasks $TASKS --num_fewshot 0 --seed 42 --output_path $O/lme_$1 > $O/lme_$1.log 2>&1 && touch $O/lme_$1/done
}
lme moe_skliar $MOE skliar $LAM full free & lme moe_ours $MOE temporal 0.5 full free & wait
lme moe_topk $MOE topk 0.5 full rule & lme tmoe_trained $TMOE temporal 0.5 norm rule & wait
echo "### ALL DONE $(date -u +%H:%M)"
