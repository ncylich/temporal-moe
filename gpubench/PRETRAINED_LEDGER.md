# Our 1e19 models end to end: plain MoE under stock / our rule / Skliar, against the trained model

Goal: the comparison of the released-model ledger (RELEASED_MODELS_LEDGER.md) on models we trained
ourselves, so routing rule, quality and decode speed are measured on one architecture with one engine.

- `moe_fine_g3_1e19` (plain MoE, no temporal training; iteration 4278; 1.26 B parameters, 14 blocks,
  layer 0 dense, 13 MoE layers of 192 experts, 18 active, shared expert) under three rules:
  stock top-k (all experts available), our rule without training (the min_logit window, R = 18), and
  Skliar's cache-conditional routing (`cache_bias.py`, C = 18, J = 1, lambda chosen below).
- `temporal_fine_g3_1e19` (same shape, temporally trained) under its own rule, R = 18.

Every routed arm has 18 expert slots per layer. Entries are in the order things happened.

## P1-0 -- Protocol and choices (2026-11-14)

- **Which plain model.** `moe_fine_g3_1e19` is the full-MoE twin of the trained model (same shape,
  data, tokens and schedule), so the only difference between the rows is the router's training.
- **Skliar's lambda.** No published value exists for this model. Chosen to match our rule's traffic:
  bisected on 64 held-out test rows (rows 1536-1599 of the 2048-row test dump, disjoint from the 512
  rows scored) until Skliar's loads per token-layer equal ours (1.000) within 0.5 %.
- **Accuracy harness.** Everything is scored by `scripts/serving_convert/ref_model.py`, the
  plain-PyTorch forward already checked against Megatron's logged loss, run in FP32 on GPU, batched.
  A `skliar` routing mode was added (it calls `cache_bias.select` and `_cap` unchanged), and
  `gpubench/pretrained/lmeval_ref.py` puts lm-eval on top of it (model `refmoe`). Using one forward
  for every arm keeps Megatron-versus-reference differences (bf16, kernels) out of the comparison.
- **Routing in the evals.** Both rules run on every token of a scored sequence, prompt included
  (teacher-forced), which is how the trained model is scored natively. Skliar's delta is the running
  mean of max - min logit over the sequence so far (cache_bias keeps one running mean per process;
  for a single sequence the two are the same, which the gate checks).
- **Speed.** Batch 1, RTX A6000, Q4_0 GGUF, the engine of the released runs (tree 20c50f0c), real
  text (held-out test rows as context, each method's own greedy continuation), four trials with the
  first discarded, every arm `TEMPORAL_SHEXP_FIRST=1` and pinned to the GPU's NUMA node.
- **Hardware split.** No A6000 was in stock when the evals started, so accuracy ran on an L40S pod
  (`hvcggjwvmohc7a`). Accuracy does not depend on the card; speed is A6000 only.
- **Credentials.** Copying the GitHub key and HF token to the new pod was blocked, so the checkpoint
  weights were extracted on the laptop (`dcp_extract.py`, 2.53 GB of the 17.8 GB checkpoint) and
  the code shipped as tarballs.

## P1-1 -- Gates and conversion

- Skliar in the reference equals `cache_bias.apply` fed the same router logits one token at a time:
  0 mismatches on every layer-token, equal load counts (laptop on the trained weights at lambda 0.5,
  2 x 96 tokens; pod on the plain weights, CUDA, 4 x 256 and 8 x 512 tokens).
- Our rule on CUDA uses the Triton scan (`compute_resident_mask_accel`); it equals the CPU reference
  scan on all 13 layers of the gate's logits. (The first pass ran the CPU scan at 4.6 s per
  2048-token row with the GPU idle; it was stopped and rerun on the GPU scan, about 35x faster.)
- GGUF: `moe_fine_1e19-q4_0.gguf`, 744,281,312 bytes, sha256 `1963a01d...` (same size as the trained
  model's file). F16 engine against the reference under stock top-k, 2 x 512 tokens: mean loss
  3.34190 against 3.34273 (-0.0008), argmax agreement 0.994 to 0.996.

## P1-2 -- Lambda and test CE (512 test rows, 1,048,576 tokens)

Sweep on the 64 held-out rows (loads = experts fetched per token-layer, after token 0):

| lambda | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| loads | 13.55 | 12.37 | 9.82 | 7.28 | 3.23 | 1.030 | 0.744 | 0.723 | 0.723 | 0.723 |
| CE | 3.753 | 3.758 | 3.788 | 3.850 | 4.171 | 4.877 | 5.085 | 5.114 | 5.115 | 5.115 |

Ours on the same rows: loads 1.000, CE 4.950. Loads floor at 0.72 because J = 1 always admits the
top-1 expert. Bisection: lambda 0.75954 gives 1.0028 loads (lambda* = 0.75954).

Test CE (nats per token; lower is better; SE over the 512 rows about 0.015):

| model | routing | loads per token-layer | test CE | bpb |
|---|---|---|---|---|
| plain MoE | stock top-k, all 192 available | -- | 3.72878 | 1.25211 |
| plain MoE | ours, no training, R = 18 | 1.000 | 4.94461 | 1.66038 |
| plain MoE | Skliar, C = 18, lambda 0.75954 | 1.012 | 4.88407 | 1.64005 |
| trained | its own rule, R = 18 | 1.000 | 3.19208 | 1.07189 |

(The trained row equals RESULTS.md's reference value on the same rows, 3.19202, to 6e-5.)

## P1-3 -- Engine gates on the A6000 (pod `i65msin9gh64dw`, resumed 20:46 UTC)

Engine tree 20c50f0c (the released runs' build), Q4_0 files above, GPU0 on NUMA node 1
(CPUs 32-63,96-127; a different host from the released runs, so the pinning differs).

- **Routing equality** (`route_check.py`, one 512-token prompt plus 100 greedy tokens per arm, graphs
  off): decode tokens equal on every layer-token for ours in VRAM, ours streamed and Skliar streamed
  (1,287 of 1,287 each); Skliar in VRAM 1,188 of 1,287. Prompt tokens 97.3 to 97.5 % equal in every
  arm. Diagnosed on a fresh dump: all 179 differing prompt token-layers have the 18th and 19th router
  logits exactly equal (gap 0), so engine and `torch.topk` pick different, equally valid members of a
  tie. Skliar warms its cache from the prompt, which is where its VRAM decode differences start. The
  released models had no such mismatches; why this model's logits tie exactly was not checked.
- **Streamed equals all in VRAM** (batch-1 perplexity, 4 x 1024 tokens of held-out test text): ours
  181.5047 both, Skliar 166.1783 both, trained 22.6354 both.

## P1-4 -- Decode speed (batch 1, 1k context, five held-out test rows 601-605)

**Flash attention.** The first pass used the released harness's `-fa 1` and every arm, stock
included, ran at about 68 tok/s. This model has 50 heads of size 16 and the CUDA flash-attention
kernels have no head size 16, so attention ran on the CPU (TRAINED_MODEL_LEDGER G1-2 found the same).
Those rows are kept as `speed_fa1_invalid.jsonl` and not used. Every row below is `-fa 0 -t 8`.
Contexts of 2k and 4k were refused by the server: the model was trained at 2,048 positions.

Mean of five rows (each the mean of trials 2 to 4); range over rows in brackets:

| model | arm | tok/s | ms per token | loads per decode token-layer | peak VRAM MiB |
|---|---|---|---|---|---|
| plain | stock top-k, all in VRAM | 633.9 [632.5, 636.2] | 1.578 | -- | 1438 |
| plain | ours, streamed | 504.1 [501.7, 506.4] | 1.984 | 1.000 | 918 |
| plain | Skliar, streamed | 495.7 [490.7, 508.1] | 2.017 | 0.806 | 918 |
| trained | own rule, all in VRAM | 675.9 [675.3, 677.1] | 1.479 | 1.000 (rule) | 1438 |
| trained | own rule, streamed | 587.9 [586.3, 590.0] | 1.701 | 1.000 (rule) | 918 |

Row 601 only: ours in VRAM 656.1, Skliar in VRAM 632.1, ours with no bytes moved 554.7, Skliar with
no bytes moved 540.4 (loads 0.665 on that run's text).

**The trained rows run a different kernel path.** `LLAMA_TEMPORAL_ROUTE=trained` is the original
device routing of patches 0001-0005; `decode` and `skliar` are patch 0006's kernel (needed for Skliar,
which has no other implementation). Check: the plain model on the trained path, same text files as
ours, gives 676.8 in VRAM and 588.2 streamed, equal to the trained model. So the 17 % gap between the
trained-model row and ours streamed is the kernel path, not the model, and speed does not depend on
which model is loaded. Ours against Skliar is a like-for-like comparison (same 0006 kernel); the
trained row against either is not, and the fair statement is that the trained model decodes at the
same speed as our rule on whichever path both use.

**Skliar's loads in serving are lower than in the evals** (0.81 against 1.01). Serving routes the
prompt freely and applies the rule from the first decoded token (decode_state.py / cache_bias.py
protocol); the evals apply it to every token. lambda* was matched on the eval protocol. Skliar is
still the slower streamed arm, by about 0.03 ms per token, because its routing step costs about
0.05 ms more (no-bytes rows).

## P1-5 -- Correction: the plain MoE was scored with the wrong gates (found 21:25 UTC)

Every plain-MoE row in P1-2 (sweep, lambda*, test CE) and the Skliar lm-eval scores are **invalid**.
`ref_model.py` and `to_gguf.py` both renormalised the gates over the selected experts. That is right
for the temporal model (its router masks the logits before the softmax, so the softmax runs over the
resident experts), but the plain MoE was trained with Megatron's pre-softmax router: softmax over all
192 experts, top-18 probabilities kept as they are. Caught because the plain MoE scored 3.729 against
the trained model's 3.192, where Megatron logged 3.158 against 3.172.

Check on 8 test rows x 512 tokens (laptop): plain MoE, renormalised 3.65239; softmax over all, not
renormalised 3.15758; trained model 3.18880. The plain MoE is back above the trained model, as in
Megatron. Fix: `ref_model.py --gates full` / `lmeval_ref.py gates=full` for every plain-MoE arm
(stock, ours, Skliar: the model's own gate weights on whichever experts the rule selects), and
`to_gguf.py --no-weights-norm` for its GGUF (`moe_fine_1e19-full-q4_0.gguf`; F16 against the
reference with full gates, 2 x 512 tokens: 2.88649 against 2.88621). The trained model's rows are
unaffected. The Megatron "imposed R18" value (4.93) was computed with the temporal router installed,
i.e. renormalised over the window, so it is not the same quantity as the corrected "ours" row.

The speed rows in P1-4 timed the renormalising GGUF on each method's own greedy continuation. The
gate weights do not enter the routing decision, so the loads and timings are expected to be close,
but the text each arm replayed came from the wrong gating. They will be retimed on the corrected file,
on the same real text for every arm (the test row's own next 100 tokens), so all arms replay identical
tokens. Two speed questions are open for that run: (1) the 0006 kernel path streams 0.46 ms per token
above its VRAM row against 0.22 ms on the trained path; 0006 forces `TEMPORAL_ROUTE_SLOTS=2` (copies
issued at the router on the copy stream), the trained path defaults to copies at the expert op, so the
first test is the trained path with `TEMPORAL_ROUTE_SLOTS=2`; (2) Skliar's serving loads (0.81)
against its eval loads (1.01), separated by the real-text replay.

## P1-6 -- Corrected protocol (21:35 UTC); everything above for the plain MoE is superseded

Two things were wrong before, both now fixed for every plain-MoE arm (stock, temporal base, Skliar):
the gates (P1-5), and the prompt. The first passes applied our rule and Skliar to every token of a
scored sequence. That is the trained model's protocol, not the one for a model without temporal
training.

- **Plain MoE** (stock / temporal base = our rule without training / Skliar): its own gates; the prompt
  is routed by stock top-k while the rule observes (decode_state.py and cache_bias.py, unchanged); the
  rule applies to the tokens after the prompt. Loads are counted on those tokens only.
- **Trained model**: its rule on every token, prompt included; gates over the resident window.
- **Test CE**: the first 1,024 tokens of a test row are the prompt, the loss is on tokens 1,024 to
  2,047, the same tokens for every arm. 512 rows, 524,288 scored tokens.
- **lm-eval**: a request's context is the prompt, its continuation is scored. A one-token
  continuation is predicted from the prompt alone, so on such requests the plain arms equal stock.
- **Skliar's lambda**: re-tuned under this protocol, bisected to our rule's loads on decoded tokens.
- **Gate** (pod, CUDA, 8 x 512 tokens, 256-token prompt): Skliar in the reference equals
  `cache_bias.apply` (prompt as one prefill batch, then one step per token) on every layer-token with
  equal load counts, apart from one exact logit tie at the cut; our rule equals `decode_state.prefill`
  plus `step` on every step; GPU scan equals CPU scan on all 13 layers.
- **Speed**: every arm replays the same real text (the test row's own next 100 tokens). Engine patch
  0008 lets `TEMPORAL_ROUTE_SLOTS` apply to the decode and skliar modes (0006 forced the value 2, which
  is the suspected cause of the 84 tok/s gap to the trained mode); every streamed method is timed at
  slots 0/1/2 with the shared expert first or not, all settings are reported, and methods are compared
  at their fastest.
