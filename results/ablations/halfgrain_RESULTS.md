# Half-grain experts: full program record (closed 2026-08-22)

Cut every expert into two half-size experts, duplicate its router entry, double
top-k, double each half's down_proj: the composed function is exactly the
original model (fp64 transform error ~1e-13; full-model bf16 logits 95-100%
top-1 agreement). Residency then operates at half-expert grain. Program
question: does finer grain make rolling residency cheaper or better? Answer:
no at equal bandwidth (provably a no-op), partially at half bandwidth after
adaptation, with two structural walls that survived every training-side lever.
Program closed; artifacts below.

Models: gemma4-26B-A4B-IT (128 experts, d_i 704) and Qwen3.5-35B-A3B
(256 experts, d_i 512, shared expert untouched). Serving rule throughout:
min-logit eviction, prefill free, constraint on generated tokens, thinking
off; "half-bandwidth" = 1 half-expert swap per token, "byte-matched" = 2.

## 1. Zero-training results

- **Byte-matched half-grain is exactly whole-grain.** Duplicated router rows
  make selection and eviction pair-locked; the byte-matched scan replays the
  whole-grain trajectory relabeled. Functional-displacement (FD) cells match
  to 3-4 decimals (qwen R64s2 0.3777 vs whole R32 0.3778; gemma R32s2 0.2086
  vs whole R16 0.2012, tie-flip noise from its bf16 router softmax).
- **Half bandwidth is strictly worse and memory buys it back badly.** Parity
  sweep (qwen, swaps=1): R16 0.715, R64 0.556, R96 0.512, R128 0.479, R192
  0.415, R256 0.346 FD rel_out. Crossing whole-R8 (0.534) costs ~5x resident
  memory; crossing whole-R32 (0.378) ~3.5x.
- **FD underestimates generation damage.** FD rated half-grain R96s1 slightly
  better than whole R8; downstream GSM8K was 4x worse (0.580 vs 0.780).
  Teacher-forced imposed-mask metrics miss autoregressive compounding under
  sustained swap starvation. Instrument caveat now attached to FD.
- Downstream walls of the unadapted half-bandwidth models: gemma HumanEval
  0.482 (free 0.994), qwen GSM8K 0.580 (free 0.850); other cells mild.

## 2. Adaptation ladder (all on the D12/r2 recipe skeleton)

Per-variant, constrained arm at 18.75% resident, swaps=1; free arms stayed at
base level everywhere unless noted.

- **v1 (router at lr/5, as D12/r2)**: gemma GSM8K 0.825->0.855; qwen GSM8K
  0.580->0.680, IFEval/HumanEval improved. Gemma code unmoved.
- **E1 (router at full lr)**: batch-parity drift 9.1%->1.13% (pair
  desymmetrization completes; ladder 9.1 -> 5.8 -> 1.1 across router-lr
  settings), gemma GSM8K 0.860 (healed), code recovered v1's harm only.
- **E2 (termination-weighted CE): falsified.** Weighting golden-trajectory
  endings can't teach stopping on degraded trajectories (exposure bias);
  runtime unmoved, scores below E1.
- **E4 (spectral co-activation partition): no downstream effect either model.**
  Explained by the offline screen (section 4).
- Trainer/infra gained: --R, --smoke-tol (pair-degenerate inits widen the
  batch-parity gates; documented in-code), --kl-only/--kl-arm, qwen router
  trainability, prefix-tolerant + stack-guarded merges, streaming
  split+patch merge for the 72G class, antisymmetric router noise injector.

## 3. Semi-on-policy distillation (the phase that worked)

Generate WITH the adapted model UNDER the constraint (harvesting real degraded
states, ~21% of gemma / ~29% of qwen general prompts ramble to the cap), score
those states with BASE free logprobs (KL refs computed on the naive split's
free arm, which is provably base), train KL-only on the constrained arm.
Iterate with fresh generations per round; stop when round-start KL flattens.

- Loop depth scales with initial divergence: gemma (start KL 0.31) took 2
  productive rounds; qwen (0.115) took 1; extra rounds show over-distillation
  (IFEval erosion, free-arm sag).
- **Final models** (vs own base free):
  - gemma dz2, R48: GSM8K 0.870 (>= free), IFEval 0.820, HumanEval 0.524 /
    0.567@3k. dz3 variant: HumanEval@3k 0.634 (first above unadapted 0.622)
    at some IFEval cost - the code-optimal pick.
  - qwen dzq, R96: GSM8K 0.725 (from 0.580), IFEval 0.815, HumanEval 0.872.
- Falsified follow-ups: gemma rambler-only round (targeting the cap-hit
  trajectories: scores at/below plateau - the deliberation loop is a robust
  attractor, not undertrained); qwen math-lane round (right state
  distribution, GSM8K flat at 0.710).

## 4. Why partitions can't help, and what rotations could buy

- Exact-preservation group for gated-FFN splits = channel permutations x
  per-channel rescalings (elementwise gate pins the basis; sign flips and
  rotations change the function).
- Offline screen (single-half reconstruction error, 300 WildChat prompts,
  n=3765 expert-layer cells): naive/spectral/interleave/snake/random all
  0.860-0.868 - **expert intermediates are dense; no channel choice makes a
  self-sufficient half**. This is the mechanism behind E4's null.
- **Rotation oracle** (best rank-d/2 linear bottleneck, full-rank-conditioned
  covariance, n=1141 experts): **0.039 - a 22x gap** over any permutation.
  LaRoSA-style rotated re-basing (eigenbasis folded into weights, halves =
  top/bottom rotated coordinates) is the one untried lever with proven
  headroom; it is LOSSY (free arm no longer equals base). Infrastructure is
  built and launch-ready (--save-rotations / --rotate / ROTATION env), left
  unlaunched at program close.

## 5. The two walls (triangulated three ways each)

- **gemma constrained code: non-convergent deliberation.** Budget-saturating
  (0.524@1536 / 0.634@3k / 0.628@6k), ~35% of problems ruminate in comments
  indefinitely; invariant to router lr, termination weighting, partition,
  and rambler-targeted distillation.
- **qwen constrained math: incoherent, not truncated.** Only 2% of GSM8K
  generations hit the cap (same as free arm); +33% residency buys +1pt;
  math-state-targeted distillation flat. Median generation 270 tokens of a
  2048 budget - it finishes and is wrong.
- Both are swap-starvation effects on the workloads with the fastest expert
  turnover; they are serving-side, not training-side.

## 6. Hygiene

- Every self-generation harvest 8-gram-screened against GSM8K test before
  training: mathlane prompts 0/2793 overlaps; generation hits are 1-3
  generic-boilerplate grams ("and there are 7 days in a week"), zero
  question/answer content. Lineage rule maintained (mathlane_v2 =
  benchmark-free by construction, D12-vetted).

## Producers and data

- split_experts.py (--partition/--rotate), inject_router_noise (scratch),
  functional_displacement{,_oss120_vllm}.py (--swaps/--name-suffix),
  partition_screen.py (--save-rotations), selfgen_traj.py,
  train_gemma_ce.py (--R/--smoke-tol/--kl-only/--kl-arm), streaming
  split+patch merge (scratch qwen_half_split_patch.py; PARTITION/ROTATION env).
- Data: instruct_genbench_vllm.csv rows gemma4_halfgrain* / qwen35_halfgrain*
  (base, ce/ce2/ce3/sp, dz/dz2/dz3/dzh, dzq/dzq2/dzm + R-curve arms);
  functional_displacement.csv halfgrain rows + parity sweep;
  partition_screen.csv; genbench_samples/ per-cell trajectories;
  figures/halfgrain_adapt.png. Adapters kept: gemma dz2/dz3/ce2, qwen dzq/ce2
  (superseded variants deleted per disk policy; all regenerable from the
  recipes above). Selfgen trajectory files in /workspace/instruct-traj.
