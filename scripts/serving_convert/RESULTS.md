# Conversion results: temporal_fine_g3_1e19, iter 4278

Run on 2026-11-10 on an Apple M4 Pro (CPU only), engine `ncylich/llama.cpp` branch
`temporal-moe` @ 61f6d1b4, host build. Commands are in [README.md](README.md). Losses are
nats per token; bits per byte (bpb) = loss / 2.9780, lower is better.

## Summary

| step | check | result | status |
|---|---|---|---|
| 1 | key list with shapes; layouts recorded | 126 tensors, 1.2652 B parameters, [fine_1e19.keys.txt](fine_1e19.keys.txt) | pass |
| 2 | reference test bpb within about 0.002 of the logged 1.06553 | 1.06977 on the first 936 of 20480 test samples, +0.00424, standard error about 0.0039 | not resolved at this sample size |
| 3 | engine logits match the reference on the same ids | F32 GGUF: zero difference to 4 decimals on 512 tokens; on 4 x 2048 tokens loss differs by 0.00005 nats, max-abs logit difference 0.63 from top-k near-ties | pass, with the caveat below |
| 3 | tokenizers give identical ids | 131,133 ids from 571,615 bytes of test text, identical | pass |
| 4 | Q4_0 size and tensor types | 744,281,312 bytes; all 39 routed expert tensors Q4_0; router F16 | pass |

## Step 1: weights

`dcp_extract.py` read the checkpoint's `.metadata` without Megatron and fetched 2.54 GB in 40
range requests. The other 15.2 GB of the checkpoint is optimizer state, including an fp32
master copy of the weights. The model tensors themselves are bf16, which is what the training
run evaluated, so the extraction keeps bf16.

- Embedding and output layer: 50304 rows each, untied. The pythia tokenizer defines 50277 ids,
  so 27 rows are never produced by the tokenizer.
- Fused qkv, `[2400, 800]`: rows are grouped per head as `[q_h (16) | k_h (16) | v_h (16)]`
  for 50 heads. Megatron views the output as `[.., 50, 48]` and splits the last axis.
- Dense fc1, `[8544, 800]`: rows `[gate (4272) | up (4272)]`, output `fc2(silu(gate) * up)`.
  Routed experts `[192, 368, 800]` and the shared expert `[2200, 800]` use the same order.
- The pre-MLP norm is stored as `mlp.linear_fc1.layer_norm_weight` in layer 0 and as
  `pre_mlp_layernorm.weight` in layers 1 to 13.

## Step 2: reference forward

The run's log line is `validation loss at iteration 4278 on test set | lm loss value:
3.173160E+00`, which is 1.06553 bpb. That evaluation covered 20 eval iterations of 128
micro-batches of 8, so 20480 samples of 2048 tokens (41.9 M tokens), not the 160 samples the
`--eval-iters 20` flag suggests.

`test_tokens.py` rebuilds that split with Megatron's dataset code. The per-part test sample
counts it logs match the 22 counts in `train.log` exactly, and the first 512 rows of a
2048-sample dump equal the 512-sample dump.

| routing | samples | loss | bpb | vs logged bpb |
|---|---|---|---|---|
| logged by Megatron (bf16, GPU) | 20480 | 3.17316 | 1.06553 | |
| reference, temporal, `evict=min_logit` | first 512 | 3.19202 | 1.07187 | +0.00634 (SE 0.00533) |
| same | 512 to 935 | 3.17826 | 1.06725 | +0.00172 |
| same, pooled | first 936 | 3.18579 | 1.06977 | +0.00424 (SE about 0.0039) |

The second row comes from a run over samples 512 to 2047 that was terminated at 424 of 1536
while other jobs were loading the machine. Its per-sample losses were not saved, only the
running mean in the log, so the pooled standard error uses the per-sample spread of the first
512. `ref_model.py` now rewrites `--save-losses` after every batch.

The pooled difference is 1.1 standard errors, so the reference is consistent with the log but
the check as written (0.002 bpb) is not met and cannot be at this sample size. Per-sample mean loss has
a standard deviation of 0.359 nats, so resolving 0.002 bpb needs about 3,600 samples at one
standard error, and an exact comparison needs all 20480. The reference runs at 2.4 s per sample
on this CPU, so the full split is about 13.6 hours here or a few minutes on a GPU pod.

Two paired comparisons on the first 32 samples show how much a wrong routing rule would move
the number. Each shift is under two pooled standard errors (0.012 nats), so these runs could
not have excluded either mistake. The table shows sensitivity, not correctness.

| change from the reference | loss shift (nats) | SE |
|---|---|---|
| `evict=lru` instead of `min_logit` | +0.0205 | 0.0015 |
| unconstrained top-18 instead of the resident window | +0.0240 | 0.0018 |

What went wrong along the way: the background notes for this task described the eviction rule
as the `compute_resident_mask` default, which is `lru`. The run's own banner in `train.log`
says `evict=min_logit`, and that is what the reference uses.

## Step 3: GGUF against the reference

Both sides use unconstrained top-18 routing with gates renormalised over the selected experts.
The reference loss on these four samples is 3.16636.

| GGUF | engine attention | engine loss | loss difference | max-abs logit difference | median per-position max |
|---|---|---|---|---|---|
| F32 | plain | 3.16641 | +0.00005 | 0.634 | 1.6e-4 |
| F16 | plain | 3.16645 | +0.00009 | 0.905 | 1.5e-2 |
| F16 | flash (engine default) | 3.16674 | +0.00039 | 0.969 | 2.4e-2 |
| Q4_0 | flash (engine default) | 3.18747 | +0.02112 | 5.22 | 1.28 |

On the first 512 tokens of sample 0 the F32 file agrees with the reference to four decimals at
every position (max-abs difference below 5e-5). On full 2048-token sequences a few positions
differ by up to 0.63. Those come from top-k near-ties: when the 18th and 19th router logits are
almost equal, float rounding picks a different expert and the change propagates to later
positions through attention. The control for this is the reference against itself in float64,
which shows the same thing with no engine involved: max-abs differences of 0.41 and 0.23 on
samples 0 and 1, with 174 and 987 positions above 1e-3, and none on samples 2 and 3. So the
max-abs number is not a useful pass criterion for this model under top-k routing, and the loss
difference of 0.00005 nats is the number to rely on.

What went wrong along the way: the first comparison showed a systematic difference of about
1e-3 at position 0 that grew with position, in F16 and F32 alike. Ablating components in a
two-layer truncation showed it vanished when the attention output was zeroed. The cause is that
the engine's flash-attention path casts K and V to F16 whatever the KV cache type. With flash
attention off the F32 file matches. Flash attention is the engine default, so device runs carry
that rounding; it costs 0.0003 nats here.

F16 holds the bf16 weights exactly except for magnitudes below 6e-8, which F16 cannot
represent. The F16 rows differ from F32 because ggml rounds activations to F16 inside matmuls
against F16 weights.

Tokenizer: text decoded from 64 test samples, 571,615 bytes, tokenizes to the same 131,133 ids
in the engine and in the Hugging Face tokenizer, including `<|endoftext|>`. Re-tokenized text
is 3 ids shorter than the stored ids because 2048-token samples cut documents mid-character.

## Step 4: Q4_0

`llama-quantize --pure ... Q4_0` produced 744,281,312 bytes (4.52 bits per weight) from the
2,631,311,648-byte F16 file.

| tensors | type | size |
|---|---|---|
| routed experts, `ffn_{gate,up,down}_exps` x 13 layers | Q4_0, all 39 | 647 MB |
| token embedding, output layer | Q4_0 | 45 MB |
| shared experts, attention, dense layer 0 | Q4_0 | 46 MB |
| router, `ffn_gate_inp` x 13 | F16 | 4 MB |
| norms, `exp_probs_b` | F32 | under 0.1 MB |

Under top-k routing with the engine's default attention, Q4_0 costs +0.0211 nats (+0.0071 bpb)
against the reference on 8192 tokens. The `--pure` flag also quantizes the embedding and output
layers to Q4_0; how much of the loss comes from those was not measured.

## Not done

- The full 20480-sample test score for step 2.
- The coarse checkpoint (`g1_tmoe_coarse_1e19`). The author chose the fine model, so it is not
  needed. The scripts take its widths from the shapes but none of them has been run on it.
- Anything on the phone or laptop, and the engine's resident-window routing. Every engine number
  above uses the stock top-k router, which the model was not trained with.

## Next steps

1. Engine routing. `ggml_temporal_window_fill` has to implement the training rule: cold fill
   with the top-18 of the first token, then at most one swap per token when the best
   non-resident logit beats the worst resident logit, evicting the lowest-logit resident
   (`min_logit`). The paired numbers above say that `lru` or no window each cost about 0.02
   nats, so the device quality number depends on getting this right. `ref_model.py` returns the
   per-layer resident masks (`return_masks=True`) to check the engine's window against.
2. Score the engine with that routing on these test samples and compare with the reference's
   3.19202 on the first 512 (per-token losses were kept for those).
3. On a Metal build with the model on GPU, the fork fails to compile
   `kernel_mul_mm_id_map0_ne20_18` for 18 used experts. It does not affect the phone or the
   Windows laptop but will affect any Mac run.
4. Benchmark drivers and device runs.
