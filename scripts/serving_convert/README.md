# Serving conversion: Megatron checkpoint to GGUF

Turns a trained Temporal MoE checkpoint (Megatron torch-DCP on Hugging Face) into a GGUF file
that the device engine (`ncylich/llama.cpp`, architecture `glm4moe`) loads without any engine
change, and checks each stage against the one before it. Measured numbers for the 1e19 fine
model are in [RESULTS.md](RESULTS.md).

Everything runs on a laptop CPU. No Megatron import is needed for the weights or the reference
forward; only `test_tokens.py` imports Megatron's dataset code (pure Python) so the test split
is rebuilt by the same code that built it for training.

| file | what it does |
|---|---|
| `dcp_extract.py` | fetch only the model tensors from the DCP checkpoint (HTTP range requests), write safetensors + key list |
| `test_tokens.py` | rebuild the Megatron test split and dump its first N samples |
| `ref_model.py` | plain-PyTorch forward with the training-time routing rule; scores the test samples |
| `to_gguf.py` | write the GGUF (`glm4moe`, zero-padded widths, pythia BPE vocab) |
| `gguf_logits.cpp` | run a GGUF through the engine's `libllama` on given ids and dump logits; also tokenizes |
| `compare_gguf.py` | engine logits vs reference logits on the same ids |
| `check_tokenizer.py` | engine tokenizer vs Hugging Face tokenizer on test text |
| `gguf_types.py` | tensor types and sizes of a GGUF |

## Commands

Requirements: `torch`, `safetensors`, `huggingface_hub`, `tokenizers`, `requests`, `numpy`; a
checkout of the engine with a host build (`build-host/bin/libllama.dylib`, `llama-quantize`);
the `Megatron-LM` submodule checked out (source only, nothing compiled).

```bash
WORK=/path/to/scratch            # about 12 GB
LLAMA=~/Documents/llama.cpp-android
cd /path/to/temporal-moe

# 1. weights: 2.5 GB of the 17.8 GB checkpoint
python scripts/serving_convert/dcp_extract.py --run temporal_fine_g3_1e19 --iter 4278 --out $WORK/fine_1e19

# 2. test samples (the directory must be named dclm_tokenized), then the reference score
python scripts/serving_convert/test_tokens.py --megatron Megatron-LM \
    --data $WORK/dclm_tokenized --n 512 --out $WORK/test_tokens.npz
python scripts/serving_convert/ref_model.py --weights $WORK/fine_1e19.safetensors \
    --tokens $WORK/test_tokens.npz --n 512 --routing temporal

# 3. GGUF, engine check tool, logit and tokenizer comparisons
python scripts/serving_convert/to_gguf.py --llama-cpp $LLAMA --weights $WORK/fine_1e19.safetensors \
    --topk 18 --out $WORK/fine_1e19-f16.gguf
clang++ -std=c++17 -O2 scripts/serving_convert/gguf_logits.cpp -I $LLAMA/include -I $LLAMA/ggml/include \
    -L $LLAMA/build-host/bin -lllama -Wl,-rpath,$LLAMA/build-host/bin -o $WORK/gguf_logits
python scripts/serving_convert/compare_gguf.py --weights $WORK/fine_1e19.safetensors \
    --gguf $WORK/fine_1e19-f16.gguf --bin $WORK/gguf_logits --tokens $WORK/test_tokens.npz --samples 4
python scripts/serving_convert/check_tokenizer.py --gguf $WORK/fine_1e19-f16.gguf \
    --bin $WORK/gguf_logits --tokens $WORK/test_tokens.npz --samples 64

# 4. quantize and inspect
$LLAMA/build-host/bin/llama-quantize --pure $WORK/fine_1e19-f16.gguf $WORK/fine_1e19-q4_0.gguf Q4_0
python scripts/serving_convert/gguf_types.py --llama-cpp $LLAMA $WORK/fine_1e19-q4_0.gguf
```

For the coarse model use `--run g1_tmoe_coarse_1e19 --iter 4318` in step 1 and `--topk 6` in
steps 2 and 3. Widths are read from the tensor shapes, so nothing else changes (550 pads to 576,
the shared expert 1100 to 1152 = 2 x 576). The coarse path has not been run.

`to_gguf.py --ftype f32` writes an F32 file. It is only useful for `compare_gguf.py`, where it
removes the engine's half-precision rounding from the comparison.

## What the converter assumes about the engine

Each of these was read in the fork (`temporal-moe` @ 61f6d1b4) and then confirmed numerically by
the F32 comparison in RESULTS.md.

- `glm4moe` runs `attn_norm -> attention -> post_attention_norm -> FFN`, so Megatron's pre-MLP
  norm goes into `post_attention_norm`.
- Layers below `leading_dense_block_count` use `ffn_gate/up/down`; the rest use the routed
  experts plus an ungated shared expert added to the routed output.
- The shared expert's width must be `expert_feed_forward_length * expert_shared_count`. With
  routed width padded to 192 the shared expert pads from 1100 to 1152 and the count is 6.
- `exp_probs_b` is a required tensor. It is written as zeros.
- RoPE for `glm4moe` is the NEOX pairing, which is what Megatron's non-interleaved RoPE does.
- Gating is `softmax` over all experts, top-k, then (`expert_weights_norm = true`) division by
  the sum over the selected set. Under a resident window this equals the training-time softmax
  over the resident logits.
- The training run logged `evict=min_logit`, not the `lru` default of `compute_resident_mask`.
  `ref_model.py` defaults to `min_logit`.

## Two things that look like bugs and are not

- With flash attention on (the engine default), K and V are cast to F16 whatever the cache type.
  That moves logits by about 1e-3 at the first position and more later. `gguf_logits` turns
  flash attention off so the comparison is exact; pass `--flash-attn` to `compare_gguf.py` to
  measure the default path.
- On a Metal build, `gguf_logits` must pass an empty device list. With Metal in the graph the
  fork fails to compile `kernel_mul_mm_id_map0_ne20_18` (18 experts used) and the run dies.
