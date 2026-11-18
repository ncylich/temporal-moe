# Starting the load first and computing under it: what it does to decode speed for our rule and for Skliar

Question (2026-11-17/18): the streamed engine starts an incoming expert's copy at the router, but the routed
experts then wait for it. If the shared expert and the experts already in VRAM are computed while the copy
is in flight, and only the incoming expert waits, does our single load per layer become free, and does that
separate our rule from Skliar's cache-conditional routing at matched loads? This ledger tests that on the
RTX A6000 with one binary and a switch. The load can be hidden, and that speeds up both rules by about the
same amount (O1-0).

Earlier numbers quoted here come from `PRETRAINED_LEDGER.md` (P1-x), `RELEASED_MODELS_LEDGER.md` (R1-x) and
`TRAINED_MODEL_LEDGER.md` (G1-x). Entries are in the order things happened. All times UTC.

## O1-0 -- Summary

- **1.26 B models, real text, matched loads** (Skliar at 0.997 loads per decoded token-layer, ours 1.000;
  15 runs per arm, sd over passes at most 2 tok/s):

  | arm | switch off | order only, fused (`LOAD_FIRST=3` + `UNIFIED_FUSE=1`) | two passes (`LOAD_FIRST=1`) | all in VRAM |
  |---|---|---|---|---|
  | ours, plain MoE | 608.8 tok/s (1.643 ms) | **662.2** (1.510 ms) | 608.9 (1.642 ms) | 692.5 (1.444 ms) |
  | Skliar, plain MoE | 589.8 (1.696 ms) | **636.2** (1.572 ms) | 598.4 (1.671 ms) | 677.3 (1.476 ms) |
  | ours, trained model | 587.5 (1.702 ms) | **636.6** (1.571 ms) | 587.7 (1.702 ms) | 670.2 (1.492 ms) |
  | ours against Skliar | +3.2 % | +4.1 % | +1.8 % | +2.2 % |
  | streamed share of all in VRAM, ours / Skliar / trained | 0.879 / 0.871 / 0.877 | 0.956 / 0.939 / 0.950 | 0.879 / 0.884 / 0.877 | |

- **The load can be hidden completely.** Exposed load time (streamed minus the same kernels with no bytes
  moved) for ours falls from 0.065 ms per token to 0.00 with either setting; for Skliar from 0.076 to 0.014
  (order only) and 0.007 (two passes).
- **The two-pass split does what was asked and does not pay on this model.** It hides every byte, but its
  second pass costs four more kernel launches per layer, 0.13 ms per token, twice what the bytes cost.
  Ordering the shared expert under the copy and keeping gate/up fusion gets the same hiding with fewer
  launches than the old path, and is 8 to 9 % faster than the switch off for every arm.
- **The gap to Skliar moves by about one point either way**, +3.2 % to +4.1 % with the order change and to
  +1.8 % with the split. Of the 0.05 ms per token between the arms, 0.03 is Skliar's routing kernel (the gap
  with all experts in VRAM) and at most 0.02 is load time.
- **First load against later loads** (pinned loads, O1-6): with the switch off the first load of a layer
  costs 0.070 ms per token and later ones 0.10 to 0.11. With two passes the first two together cost about
  0.03 and the third 0.07. So the first load did become cheaper than later ones, and on this model so did
  the second.
- **11 B random-weight model, forced loads** (O1-7): with the split the first load costs 0.53 ms per token
  against 0.78 off, but the second pass costs 0.42 ms per token before any load, so one load per layer runs
  at 169.5 tok/s against 174.4 off. Exactly one load against Skliar-shaped loads at matched mean: +2.0 %
  off, +2.2 % with the split, a change inside what three runs resolve.
- **Answer to the question.** Starting the load first and computing under it makes our one load free where
  there is a shared expert long enough to cover it (the 1.26 B models), and that is worth 8 to 9 % for every
  arm, Skliar included. It does not separate the two rules by more than a point, and computing the resident
  experts under the copy costs more in launches than it hides on both models measured.
- **Qwen3.5-35B was not run** (O1-8): its pod could not be resumed.

## O1-1 -- Order of operations per MoE layer on the streamed path, as patched through 0012

Read from the engine tree `26f21c79` (patches 0001 to 0012; rebuilt locally from the patch files, same tree
hash as P1-12). Line numbers are in that tree.

Both paths, single-token step, `TEMPORAL_ROUTE_SLOTS=2`:

1. Router logits (`build_moe_ffn`, the `ffn_gate_inp` matmul), then the gate softmax over all experts
   (`llama-graph.cpp:1851`). The softmax precedes the routing op because the graph is expanded from
   `weights = get_rows(probs, selected)`, source 0 first (`llama-graph.cpp:1985`).
2. The routing op: `k_route` for the trained rule and for our rule on the plain MoE
   (`temporal.cu:442`, launched at `:1073`), `k_skliar` for Skliar (`:829`, launched at `:1010`), `k_route2`
   for Skliar with `TEMPORAL_CB_K2=1` or a prompt batch (`:635`, `:1016`). The kernel picks the experts,
   updates the slot table, writes the swap list and the slot id of each selected position.
3. Right after it, on the compute stream, an event; the copy stream waits for that event and runs three
   copy kernels in a row, gate, up, down, each followed by its own event (`temporal.cu:1023-1027` for
   `k_skliar`/`k_route2`, `:1083-1087` for `k_route`). The copies of all of a layer's loads are in the same
   three kernels.
4. On the compute stream meanwhile: gather of the gate weights, and for the trained model the
   renormalisation (sum, clamp, divide). Then whatever `tm_after_router` holds (`llama-graph.cpp:1986`).
5. The routed experts: gate `mul_mat_id`, up `mul_mat_id`, SwiGLU, down `mul_mat_id`. Each `mul_mat_id`
   enters `ggml_cuda_temporal_unified_remap` (`ggml-cuda.cu:2053`), which with slots 2 does nothing but
   `cudaStreamWaitEvent` on its own tensor's copy event (`temporal.cu:292-293`), and then runs one
   `mul_mat_vec_q` over all K positions (`ggml-cuda.cu:2072`). Gate/up fusion is off on the streamed path
   (`ggml-cuda.cu:1597`).
6. Weights, sum over experts, and after `build_moe_ffn` returns, whatever the architecture adds.

With `TEMPORAL_ROUTE_SLOTS=1` (P1-9's setting for Skliar and the trained model) step 3 moves to the gate
`mul_mat_id` (`temporal.cu:313-346`), so nothing overlaps the gate copy.

**The quoted reading is confirmed, with two corrections.**

- Confirmed: the gate GEMV waits for the gate copy of all of the layer's loads, the up GEMV for the up
  copy (which is queued behind the gate copy), the down GEMV for the down copy. There is no per-expert
  pipelining. What makes that fixable cheaply: `mul_mat_vec_q` for a single token already launches one
  block set per expert position (`blockIdx.y` is the position, `mmvq.cu:507`, slot read at `:514`), so
  positions can be skipped individually.
- Correction 1, the trained-model path: **for the 1.26 B architecture (`glm4moe`) the shared expert was
  never under the copy.** `TEMPORAL_SHEXP_FIRST=1` there expands the shared expert after `build_moe_ffn`
  has returned (`glm4-moe.cpp:266`), but `build_moe_ffn` has by then expanded the routed experts itself
  (`llama-graph.cpp:2178`), so the order does not change. Second check: P1-9's own rows are equal with and
  without the switch (ours 616.2 / 616.3, Skliar 593.6 / 591.2, trained 598.4 / 598.2 tok/s at slots 2), and
  G1-7 saw "nothing more" from it. So on `k_route` and `k_skliar` with this model only the gather (and the
  renormalisation) ran under the copy. "Shared expert first" in P1-9's setting column was a no-op.
- Correction 2, the released-model path: for `qwen35moe` and `gemma4` the shared expert or dense MLP is
  under the copy with `TEMPORAL_SHEXP_FIRST=1` (`qwen35moe.cpp:519`, through `tm_after_router`), as R1-6
  says. The structure of `k_route2` and `k_route`/`k_skliar` is otherwise the same; the copy and wait code
  is shared.

## O1-2 -- Engine patch 0013, `TEMPORAL_LOAD_FIRST` (off by default)

`scripts/serving_convert/engine_patch_cuda/0013-*.patch`, engine tree `bcf3fe54` on top of `26f21c79`. Only
the streamed path with the routing op on the device and single-token steps changes; `TEMPORAL_LOAD_FIRST`
forces `TEMPORAL_ROUTE_SLOTS=2`. Both rules go through the same code; nothing is specific to either.

- **Order** (any value): the routing op is expanded right after the router logits, before the softmax, so
  the copies start one kernel earlier. For `glm4moe` the shared expert is built first and handed to
  `build_moe_ffn` through `tm_after_router`, which puts it between the routing op and the routed experts.
- **Two passes** (`=1` or `=2`): the routed experts are built twice over the same selection. Pass A
  computes the experts already in their slots and waits for nothing. Pass B computes the experts being
  loaded: its fused gate/up kernel waits for the gate and up copies, its down projection for the down copy,
  and the down projection writes into pass A's output tensor (an in-place `mul_mat_id`). The routing
  kernels write one slot-id list per pass with the other pass's positions set to -1, and `mul_mat_vec_q`
  returns at once for a negative id. Every expert is computed once, by the same kernel, at its own
  position, so values and summation order are those of the single pass. Layers with no load run pass B as
  empty launches; loads beyond the first share pass B.
- `=1` keeps gate/up fusion (legal now because pass A waits for nothing; single-token ops only), `=2` runs
  unfused, `=3` changes the order only. `TEMPORAL_UNIFIED_FUSE=1` (patch 0003) combines with `=3`.
- For the measurements: `TEMPORAL_SWAP_N` and `TEMPORAL_SWAP_HIST` now also work on the routing-op path
  (`k_force_route`: n cold experts into slots 0..n-1, the positions computed from those slots are the
  incoming ones; timing only), and `TEMPORAL_ROUTE_NOCOPY=1` now also applies to `k_route`.

Kernel launches per layer for the routed experts (quantise input, GEMV, ...): switch off 7; fused single
pass 4; two passes fused 8; two passes unfused 14. Checked on the captured CUDA graph of the plain MoE
(`TEMPORAL_GRAPHDBG=1`): 601 nodes off, 520 fused, 692 for `=2` (601 + 7 x 13), 572 for `=1` (520 + 4 x 13).

## O1-3 -- Pod, build, what went wrong on the way

- **Pods.** `i65msin9gh64dw` (holds the Qwen3.5 GGUF) and `bcnqix1ewwwjr3` could not resume ("not enough
  free GPUs on the host"); `i65...` was retried 18 times between 04:51 and 06:46 and never came up.
  `8xcjjlxgekub1c` (P1-9's pod, RTX A6000, $0.69/h, container disk only) resumed at 04:51 with an empty
  disk. GPU0 sits on NUMA node 0 this time (CPUs 0-31,64-95; P1-9 recorded node 1).
- **No route to GitHub from the pod**, and SSH connects about one time in three. The engine base went up as
  a `git archive` of `43bec5f3`; `build_engine.sh` takes it through `ENGINE_BASE_TGZ` and checks the base
  tree (`56fc59a1`) and the tree after the CPU patches (`b69151b6`) as before. The pod's tree hash after all
  patches equals the local one.
- **Models.** `moe_fine_1e19-full-q4_0.gguf` (sha256 `bee4834d9d3dbb31...`), `fine_1e19-q4_0.gguf`
  (`c34d3445064a9b14...`, the published file), `test_tokens_2048.npz` and `ppl.txt` copied from the laptop.
- **Two engine trees.** The 1.26 B timing rows (O1-5, the first pinned sweep) were taken on tree `e170b658`.
  The final tree `bcf3fe54` differs only in `k_force_route` (it gained the histogram draw, used by O1-7).
  The gates were run on both trees; the second pinned sweep and O1-7 ran on the final one.
- **Went wrong, and how it was caught.**
  1. The first gate script printed "FAILED" for every streamed row. Its `grep` read standard output and
     `llama-perplexity` wrote the estimate to standard error on those calls; the values were in the log and
     were equal. Fixed and rerun.
  2. The routing-equality gate failed twice on a missing Python module: the pod copy lacked the `temporal`
     package, then had only two of its files. It passed on the third run with the whole package.
  3. `TEMPORAL_UNIFIED_FUSE=1` without the order change makes things worse, not better (exposed load time
     0.13 ms against 0.065): the fused kernel needs both the gate and the up copy before it starts, while
     the unfused chain hid the up and down copies behind the gate GEMV. G1-7 had seen "none or worse". It
     only pays once something else covers the copy.
  4. My first read of the task assumed the shared expert was already under the copy on the 1.26 B model.
     It was not (O1-1), so "switch off" in this ledger is a weaker baseline than R1-6's Qwen rows were.

## O1-4 -- Gates with the switch on (final tree `bcf3fe54`; the same results on `e170b658`)

`gpubench/overlap/gates.sh`, outputs in `gpubench/results/overlap/gates_final.log`.

- **Streamed equals all in VRAM**, batch-1 perplexity on 4 x 1,024 held-out tokens, two repeats per setting:

  | rule (model) | all in VRAM | streamed, slots 2 | `LOAD_FIRST=3` | `=2` | `=1` |
  |---|---|---|---|---|---|
  | ours (plain MoE) | 42.6913 +/- 2.98935 | same | same | same | same |
  | Skliar, lambda 0.739453125 (plain MoE) | 40.0690 +/- 2.84434 | same | same | same | same |
  | trained model | 22.7367 +/- 1.50944 | same | same | same | same |

  Every digit equal in all 24 streamed runs. Stock top-k 22.6868. Ours and the trained model reproduce
  P1-8's values (42.6913, 22.7367); Skliar's differs from P1-8 because lambda does.
- **Routing equality** (`route_check.py`, graphs off, 512-token prompt plus 100 greedy tokens, `=1`): decoded
  token-layers equal to `decode_state.py` / `cache_bias.py` on 1,287 of 1,287 for ours and for Skliar,
  streamed. Prompt token-layers 6,508 of 6,708 (97.0 %), the exact ties of P1-3.
- Not gated: the fused single pass without the order change, and the forced-load and no-bytes rows, which
  compute wrong values by construction.

## O1-5 -- Decode speed on the 1.26 B models, switch off against on

Protocol of P1-9: batch 1, Q4_0, `-fa 0 -t 8`, pinned to the GPU's NUMA node, 1,024-token prompt as one batch,
the test row's own next 100 tokens decoded, rows 601-605, the same token files for every arm, four trials
with the first dropped. `gpubench/overlap/speed.sh`: three passes over all arms, arm order reversed in the
middle pass; 15 runs per arm (3 passes x 5 texts). `TEMPORAL_SHEXP_FIRST=1` on every arm as in P1-9.

**Skliar's lambda.** 0.739453125, chosen by `bisect_lambda.py`: bisection on the five timed texts themselves,
under the timing protocol (all in VRAM, four trials), until the engine's own count is within 0.2 % of 1.000
(reached at 1.0014, 26,036 loads on 26,000 decoded token-layers). The streamed arm counts 0.9972 on the same
texts, since its prompt goes through a different prefill kernel and warms the cache slightly differently.
Both are inside 0.5 %. P1-9 ran lambda 0.72070 at 1.054 to 1.059. Skliar's histogram at this lambda:
0 loads 21.0 %, 1 load 64.4 %, 2 loads 10.2 %, 3 or more 4.4 %. lambda is tuned on the timed texts, so
it says nothing about accuracy; P1-10's accuracy rows are at 0.72070.

Settings: `off` = `TEMPORAL_ROUTE_SLOTS=2`; `fuse` = off plus `TEMPORAL_UNIFIED_FUSE=1`; `order` =
`TEMPORAL_LOAD_FIRST=3`; `order+fuse` = that plus `TEMPORAL_UNIFIED_FUSE=1`; `split unfused` =
`TEMPORAL_LOAD_FIRST=2`; `split` = `TEMPORAL_LOAD_FIRST=1`. tok/s, mean of 15 runs, ms per token in brackets,
then the sd of the three pass means.

| setting | ours, plain MoE | Skliar, plain MoE | ours, trained model |
|---|---|---|---|
| all in VRAM | 692.5 (1.4440) 0.5 | 677.3 (1.4764) 0.5 | 670.2 (1.4922) 0.3 |
| P1-9's setting (slots 2 / 1 / 1) | 608.1 (1.6445) 1.1 | 590.6 (1.6931) 0.7 | 587.5 (1.7023) 0.6 |
| off | 608.8 (1.6427) 0.3 | 589.8 (1.6956) 0.2 | 587.5 (1.7021) 0.6 |
| fuse | 609.0 (1.6421) 1.0 | 595.8 (1.6785) 0.7 | 587.0 (1.7036) 0.5 |
| order | 632.0 (1.5823) 0.4 | 614.9 (1.6263) 1.8 | 612.0 (1.6339) 1.3 |
| **order+fuse** | **662.2 (1.5102)** 0.6 | **636.2 (1.5719)** 2.0 | **636.6 (1.5708)** 1.5 |
| split unfused | 573.3 (1.7443) 0.7 | 563.6 (1.7743) 1.6 | 556.3 (1.7977) 0.1 |
| split | 608.9 (1.6424) 0.6 | 598.4 (1.6712) 0.3 | 587.7 (1.7015) 0.6 |

Stock top-k with all experts in VRAM: 647.6 tok/s (1.5441 ms). Peak VRAM 918 MiB streamed, 1,438 in VRAM, in
every setting. The same host gave 616.3 / 592.8 / 598.5 for P1-9's settings on 2026-11-14; today's rows are
1 to 2 % lower in every arm, the all-in-VRAM rows included, so only same-day rows are compared.

**No-bytes controls** (`TEMPORAL_ROUTE_NOCOPY=1`: same kernels, copy kernels launched with an empty list,
output wrong) and **exposed load time** = streamed minus no-bytes, ms per token. The `off` and `order+fuse`
pairs were rerun together in a second three-pass run (`speed_nb.jsonl`); where a pair exists in one run that
pair is used.

| setting | ours: no bytes / exposed | Skliar: no bytes / exposed | trained: no bytes / exposed |
|---|---|---|---|
| off | 1.5773 / 0.067 (first run 0.063) | 1.6186 / 0.075 (0.078) | 1.6372 / 0.065 (0.067) |
| fuse | 1.5136 / 0.129 | 1.5555 / 0.123 | 1.5713 / 0.132 |
| order | 1.5726 / 0.010 | 1.6094 / 0.017 | 1.6300 / 0.004 |
| order+fuse | 1.5144 / -0.004 | 1.5518 / 0.014 | 1.5702 / -0.003 |
| split | 1.6445 / -0.002 | 1.6638 / 0.007 | 1.7010 / 0.001 |

One pass mean moves by 0.003 to 0.005 ms, so anything under about 0.007 ms is zero here.

- **Hidden load time.** Ours: 0 of 0.065 ms per token with the switch off, all of it with `order+fuse` or
  `split`. Skliar: 0 of 0.076 off, 0.062 with `order+fuse`, 0.069 with `split`. Trained: all of 0.066.
- **Ours against Skliar, tok/s**, with the three pass ratios: off +3.2 % (3.2, 3.2, 3.3); order+fuse +4.1 %
  (4.5, 4.0, 3.7; second run +3.7 %); split +1.8 % (1.7, 1.7, 1.8); all in VRAM +2.2 %. In ms per token
  Skliar is behind by 0.053 off, 0.062 with order+fuse, 0.029 with split, and 0.032 with everything in
  VRAM, which is its routing kernel (max and min of the logits, bias, forced top-1) and not load time.
- **Why the split loses to order+fuse here.** Both hide our load completely. The split adds a second
  pass, four launches per layer and 52 per token, and costs 1.6424 - 1.5102 = 0.13 ms; the bytes it could
  still hide for Skliar are 0.014 ms. The no-bytes rows say the same: 1.6445 against 1.5144.
- **Why the gap to Skliar grew with order+fuse and shrank with the split.** With order+fuse the window is
  the shared expert. One load fits, so ours is fully hidden; Skliar's layers with two or more loads
  (14.6 % of token-layers) outlast it and its layers with none (21 %) leave it unused: 0.014 ms stays
  exposed. With the split the window also contains the resident pass, two loads fit (O1-6), and
  Skliar's exposed time halves, while both arms pay the same 0.13 ms for the second pass.

## O1-6 -- Pinned loads: what the first load costs against later ones (1.26 B, plain MoE, our kernel)

`gpubench/overlap/pinned.sh`: `TEMPORAL_SWAP_N=n` copies n cold experts on every layer-token whatever the
router selects (timing only), row 601, three passes with alternating order, same protocol. ms per token and
the cost of each added load:

| loads per layer-token | off (slots 2) | order+fuse | split |
|---|---|---|---|
| 0 | 1.5850 | 1.5241 | 1.6114 |
| 1 | 1.6545 (+0.070) | 1.5199 (-0.004) | 1.6432 (+0.032) |
| 2 | 1.7559 (+0.101) | 1.6067 (+0.087) | 1.6427 (+0.000) |
| 3 | 1.8690 (+0.113) | 1.7147 (+0.108) | 1.7103 (+0.068) |

Final tree, `pinned_fine2.jsonl`; sd of the three pass means 1.3 to 6 tok/s (0.003 to 0.015 ms). A first
sweep on tree `e170b658` without the order+fuse column (`pinned_fine.jsonl`) gave the same increments: off
+0.070, +0.103, +0.109; split +0.029, -0.006, +0.067. Its levels are about 1 % lower.

- **Off:** the first load costs about two thirds of a later one (0.070 against 0.10 to 0.11), as on the
  11 B model in P1-12 (about 75 %).
- **Order+fuse:** the first load is free and the second and third cost what a later load costs with the
  switch off. This is the shape the hypothesis predicts: one load fits under the shared expert, nothing
  beyond it does.
- **Split:** the first two loads together cost 0.03 ms and the third 0.07. The window (shared expert plus
  the resident pass) holds two loads of this size, so a second load is hidden as well as the first. The
  zero-load row is 0.026 ms above off and 0.087 above order+fuse: the price of the second pass.
- On this model a load is 259,200 bytes and the whole effect is 0.07 ms of a 1.6 ms token.

## O1-7 -- 11 B random-weight model, forced loads

Asked for only if time was left; run because the Qwen3.5 pod never came up (O1-8) and this is the other
model where bytes dominate. `gpubench/overlap/big11b.sh`, final tree, 06:26 to 07:17.

**Model.** `qwen3moe-rand-fine-Q4_K_M.gguf`, regenerated on the pod with `llamacpp-bench/gen_random_qwen3moe.py`,
7,032,387,840 bytes, sha256 `af987fed...c172d004`, the file of P1-12. 45 layers, 192 experts, 18 active, 840 KiB
per load, no shared expert, so the only thing the switch can put under the copy is the resident pass.

**Protocol** of P1-12: `llama-bench -ngl 99 -fa 1 -ub 1 -b 1 -d 1024 -n 128 -r 8`, streamed arms `-ncmoe 48
TEMPORAL_UNIFIED=1`, pinned to the GPU's NUMA node, three passes with the arm order reversed in the middle
pass. Random weights route nothing meaningful, so loads are forced on the routing-op path (`LLAMA_TEMPORAL_ROUTE=decode`
plus `TEMPORAL_SWAP_N` or `TEMPORAL_SWAP_HIST`). "Skliar-shaped" draws each layer-token's count from the
histogram Skliar produced in O1-5 (`16413,50220,7956,2391,732,186,96,6`); realised mean 0.9955 loads
(62,904 / 193,263 / 30,579 / 8,959 / 2,799 / 680 / 355 / 26 of 299,565 layer-tokens), against exactly 1.

ms per token (tok/s), mean of three runs; sd over runs at most 0.3 tok/s for every streamed row:

| loads per layer-token | off (slots 2) | fuse, single pass | split (`LOAD_FIRST=1`) |
|---|---|---|---|
| 0 pinned | 4.956 (201.8) | 4.996 (200.2) | 5.373 (186.1) |
| 1 pinned | 5.735 (174.4), +0.779 | 6.243 (160.2), +1.247 | 5.899 (169.5), +0.526 |
| 2 pinned | 7.081 (141.2), +1.346 | 7.577 (132.0), +1.334 | 7.017 (142.5), +1.118 |
| 3 pinned | 8.397 (119.1), +1.316 | 8.866 (112.8), +1.289 | 8.179 (122.3), +1.163 |
| exactly one (histogram path) | 5.762 (173.6) | 6.284 (159.1) | 5.927 (168.7) |
| Skliar-shaped, mean 0.9955 | 5.876 (170.2) | 6.283 (159.2) | 6.059 (165.1) |
| exactly one against Skliar-shaped | +2.0 % | 0.0 % | +2.2 % |

All in VRAM: 203.7 tok/s stock (209.3 / 200.8 / 200.9), 213.2 with the routing op (216.3 / 211.5 / 211.8).
Share of the routing-op row: exactly one 0.814 off, 0.791 split; Skliar-shaped 0.798 off, 0.774 split.

- **The first load became cheaper relative to later ones, but a token with one load per layer got slower.** Off, the
  first load costs 0.78 ms per token, 59 % of a later one (1.33). With the split it costs 0.53 ms, 46 % of
  a later one (1.14), and later loads are 0.15 to 0.2 ms cheaper too. But the second pass costs 0.42 ms
  per token before any load (5.373 against 4.956: four launches on each of 45 layers). At one load per
  layer the split is 2.8 % slower than off (169.5 against 174.4 tok/s); it breaks even at two loads and
  is 2.7 % faster at three.
- **The resident pass hides 0.25 ms of a 1.33 ms load**, about 6 us of a 30 us
  copy per layer. The copy runs at roughly 28 GB/s and the 17 resident GEMVs finish long before it. The
  fused single pass, which waits for the gate and up copies before doing anything, hides nothing (1.25 of
  1.33 ms): the 0.5 ms the old path hides comes from the up and down copies running behind the unfused
  gate and up GEMVs, which is per-tensor pipelining the engine already had.
- **Ours against Skliar-shaped loads at matched mean: +2.0 % off, +2.2 % with the split.** 0.114 ms against
  0.131 ms per token, so the split did not open the gap. The fused single pass removes it (0.0 %): when no
  load is hidden, cost follows the mean and the two means are equal within 0.5 %.
- So the 2 % that separates the two shapes at equal mean comes from the convexity already in the off
  column (first load 0.78, later 1.33), and making the first load cheaper still (0.53) moved it by 0.2
  points.

Limits, as in P1-12: counts are drawn independently per layer-token, the experts copied are cold and
arbitrary, the histogram comes from the 1.26 B model, routing-kernel cost differences between the rules are
not in these rows (both arms run `k_route`), context 1,024 only. The pinned and histogram rows of P1-12 used
the residency-kernel path, these use the routing-op path, so levels are not comparable across the ledgers.

## O1-8 -- Not done or not verified

- **Qwen3.5-35B-A3B was not run.** Its Q4_K_M file and token files are on pod `i65msin9gh64dw`, whose
  host refused to resume every time between 04:51 and 06:46 (18 attempts). The pod that did resume has a
  60 GB container disk (the conversion needs about 160 GB) and an unreliable route to Hugging Face, and
  creating a pod was outside what was approved. This is the model where the shared expert is already under
  the copy with the switch off (R1-6) and bytes are 5 ms of a 12 ms token, so it is the missing test of
  the split with a real router. The 11 B rows suggest what to expect (a resident pass too short to hide a
  2 MB load, a per-layer price for the second pass), but that is an inference, not a measurement.
- The split's second pass re-quantises the layer input and launches four kernels even when a layer has no
  load. Reusing pass A's quantised input would save one launch per layer; a single kernel for pass B would
  save more. Neither was tried, so "the split does not pay" holds for this implementation.
- The split was not tried with the copies ordered incoming-expert-first within a tensor, or with pass B
  per expert; loads beyond the first share one pass B and wait for all of the layer's copies.
- Gemma4 not run. Context 1,024 only. Five texts for the 1.26 B rows.
- lambda 0.739453125 was tuned on the five timed texts for load matching only. No accuracy number in any
  ledger was measured at this lambda.
- The fused single pass without the order change (`fuse`) and all forced-load and no-bytes rows were not
  put through the perplexity gate; the forced-load and no-bytes rows cannot pass it by construction.
- Routing equality was checked for ours and Skliar on the plain MoE, not for the trained model's mode
  (its gate is the perplexity identity), as in P1-8.
- Why the first load costs 0.03 ms and the second nothing under the split on the 1.26 B model (O1-6) was
  not chased; both are near the noise of that sweep.
- Whether the paper's Figure 1 condition is met on this card was not derived here. The measurement says
  that at 28 GB/s host-to-device and with 17 or 18 resident experts of this size the resident GEMVs
  cover about a fifth of one load on the 11 B model.

## O1-9 -- Pod and cost

Pod `8xcjjlxgekub1c` ran 04:51 to 07:19 UTC on 2026-11-18, 2.47 h at $0.69/h, about $1.70, and is stopped
(not terminated; it has no volume, so nothing on it survives). The other three pods from the earlier
ledgers stayed stopped. A fifth pod, `k425im1s3drxei`, was running at 07:19; it belongs to another session
and was not touched. Raw output: `gpubench/results/overlap/`; scripts: `gpubench/overlap/`.
