# Recipe ablations: annealing then distillation (this order — annealing needs live judgment)

All arms: Qwen3-30B, lr=1e-4, 15M tokens unless stated; compare vs CE baseline 0.676359, noise 3e-03.
R and swap-budget are both damage axes and complements: R is fine when loose but coarse near k
(k+2→k+1→k are big jumps); s gives granularity exactly at the tight end (R=k, s=4→2→1).

## Track 1 — annealing (FIRST: schedule design + kernel decisions need user input)

1. Measure both damage curves, training-free (~1.5 h): BPB at R ∈ {128,64,32,16,12,10,8};
   extend the EAGER scan to s swaps/token, BPB at s ∈ {8,4,2,1} at R=8. No fast kernel needed.
2. Derive schedules from the curves (damage-paced: allocate time where the curve is steep);
   user reviews curves + schedule choice vs the E/4 → 60% cool → 40% hold proposal.
3. Train arms: (i) user schedule R-only; (ii) damage-paced R-only; (iii) R-then-s combined —
   gated on the s-curve being smooth near s=1, since only (iii) needs the multi-swap triton kernel.

## Track 2 — distillation (SECOND: mechanical, runs overnight unattended)

0. Teacher cache, once (~45 min): free-model pass over 15M tokens, top-256 logits bf16 + int32
   idx (~23 GB), teacher probs renormalized over the stored set; bias identical across arms.
1. T-screen at 5M: T ∈ {1,2,4}, lr 1e-4, pure KL scaled by T² (keeps gradients T-invariant so
   one LR bracket serves all T). Select argmin held-out BPB@5M; ties <1.5e-03 → T=1.
2. LR bracket at 15M with winning T: {3e-5, 1e-4, 3e-4}.
3. Head-to-heads: best distill vs CE baseline; plus distill+anneal combined if Track 1 won.
Deferred until these verdicts: λ<1 KL/CE mixtures, any 100M/1B commitment.
