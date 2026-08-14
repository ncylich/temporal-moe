# Recipe ablations: annealing then distillation (this order — annealing needs live judgment)

All arms: Qwen3-30B, lr=1e-4, 15M tokens unless stated; compare vs CE baseline 0.676359, noise 3e-03.
R and swap-budget are both damage axes and complements: R is fine when loose but coarse near k
(k+2→k+1→k are big jumps); s gives granularity exactly at the tight end (R=k, s=4→2→1).

## Track 1 — annealing (FIRST: schedule design + kernel decisions need user input)

1. Measure both damage curves, training-free (~1.5 h): BPB at R ∈ {128,64,32,16,12,10,8};
   extend the EAGER scan to s swaps/token, BPB at s ∈ {8,4,2,1} at R=8. No fast kernel needed.
2. Derive schedules from the curves (damage-paced: allocate time where the curve is steep);
   user reviews curves + schedule choice vs the E/4 → 60% cool → 40% hold proposal.
3. Train arms: (i) user schedule R-only; (ii) damage-paced R-only; (iii) DROPPED by decision.
Agreed specs (08-07): both arms hold R=8 for EXACTLY the final 40% (366/915 steps), R >= 9
strictly in-window. Arm (ii): d(R) = A·R^-γ fitted to the 6-point qwen3 damage curve (γ=1.282,
log-log LSQ); R(u) inverts the fit so damage rises linearly from d(128) (warm-up sweep, ~1
step per R above 32) to d(8), clamped [9,128] — dwell monotone into the hold (…55, 111, 366).
Arm (i): linear 32 → 9 over the window, same clamp and hold.

## Track 2 — distillation (SECOND: mechanical, runs overnight unattended)

0. Teacher cache, once (~45 min): free-model pass over 15M tokens, top-256 logits bf16 + int32
   idx (~23 GB), teacher probs renormalized over the stored set; bias identical across arms.
1. T-screen at 5M: T ∈ {1,2,4}, lr 1e-4, pure KL scaled by T² (keeps gradients T-invariant so
   one LR bracket serves all T). Select argmin held-out BPB@5M; ties <1.5e-03 → T=1.
2. LR bracket at 15M with winning T: {3e-5, 1e-4, 3e-4}.
3. Head-to-heads: best distill vs CE baseline; plus distill+anneal combined if Track 1 won.
Deferred until these verdicts: λ<1 KL/CE mixtures, any 100M/1B commitment.

## Queue (08-07): verdicts in, distillation won everywhere — 100M campaign

Verdicts: anneal NEGATIVE (100%-hold beats both schedules); distill T=1 optimum on all three
models; T=0.5 collapses. 15M bests: qwen3 0.671301 (lr 1e-4), olmoe 0.788727 (lr 3e-5),
q35 distill15M lr 3e-5 in flight (10M eval 0.664297 already beats CE winner 0.665780).
Chain (autonomous, sequential, fires on q35 15M save):
1. Downstream gate on the three 15M bests (pre-100M confirmation; ten-task 0-shot suite).
2. Profiler re-measure of MoE permute cost on the unsloth path (stale 37.5% figure); then
   cached-teacher smoke (gate: top-K mass coverage > 0.995) and q35 mb4 smoke.
3. 100M runs, evals every 10M, winning recipe (distill T=1) per model vs its dense floor:
   qwen3 lr 1e-4 mb4 + 10M rolling teacher cache (floor Qwen3-4B 0.678077, null 0.616034);
   q35 lr 3e-5 mb4-if-smoke-passes + cache (floor Qwen3.5-4B 0.689223, null 0.623235);
   olmoe lr 3e-5 (floor OLMo-1B shared-tok 0.672723 — only model still failing its floor).
4. Wrap-up: tables (recipe grid, 100M curves, dense verdicts), update sweep_RESULTS.md (rewrite pending) +
   TRAINING_OPTIM_PLAN.md (profiler findings), commit producers, reproduce.sh, push.
Parked for 1B: fp8 compute, permutation kernels tier 2-3, causal-conv1d (CUDA-13 toolchain).

OUTCOME (08-08): campaign complete — see sweep_RESULTS.md (rewrite pending) "100M distillation campaign".
All three clear dense downstream bars; both Qwens clear dense BPB floors; token axis
saturates ~20-40M (q35 flat 15M->100M). 1B levers now: rank (r64+), 100M-tuned LR, data.
