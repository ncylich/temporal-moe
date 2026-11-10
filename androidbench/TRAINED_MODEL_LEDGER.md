# Trained fine 1e19 model on the Pixel 10a: ledger

Model: `temporal_fine_g3_1e19/ckpt/iter_0004278` (HF `ncylich/temporal-moe-ckpts`), 14 layers
(layer 0 dense, 13 MoE), 192 routed experts per MoE layer, 18 active, 1.26 B parameters.
Training routing: rolling resident window, cold fill top-18, at most one swap per layer per
token, eviction `min_logit`. One entry per step, written before the next step starts.

## Step 1: converter and GGUF files (2026-11-10)

- `git cherry-pick 1e7e4cab` onto this branch: `scripts/serving_convert/` (README, RESULTS, scripts).
- Files were not rebuilt. They were copied from the converting session's scratch directory to a
  durable location outside the repo: `~/Documents/temporal-moe-serving/fine_1e19/`
  (`fine_1e19-f16.gguf`, `fine_1e19-q4_0.gguf`, `fine_1e19.safetensors`, `test_tokens.npz`,
  `test_tokens_2048.npz`, `ref_temporal_512.npy`, reference logs).
- Command: `shasum -a 256 *.gguf; ls -l`

| file | bytes | sha256 |
|---|---|---|
| fine_1e19-q4_0.gguf | 744,281,312 | c34d3445064a9b14cadacee1de9ec74d568995f078de6da0d8aef767b412c8ff |
| fine_1e19-f16.gguf | 2,631,311,648 | 47d771bb61157b517eed9c7f35e231cd6c18c2a4cdf17ac9431ac64a3e97738f |

Check: Q4_0 size equals the expected 744,281,312 bytes. Pass.

Reference (from the converting session's log, `ref_temporal_512.log`): routing temporal,
evict min_logit, k 18, 512 samples, 1,048,576 tokens, loss 3.19202 nats/token (SE 0.01587),
1.07187 bits per byte.

Device note found during step 1 (2026-11-10 23:24 device time): `su` is not available. The
Magisk app is installed and the bootloader is unlocked, but Magisk is not active this boot
(uptime 1 h 28 min, `/debug_ramdisk` empty). Shell-level access works: `/data/local/tmp/tmoe`
is writable, binaries run, VmHWM is readable. Not available: `drop_caches`, block-queue tuning,
frequency pinning.

## Step 2: trained routing in the engine (2026-11-10)

Engine checkout: `~/Documents/llama.cpp-trained-routing` (a `git worktree` of
`~/Documents/llama.cpp-android`), local branch `temporal-moe-trained-routing`, based on
`temporal-moe-win` @ 43bec5f3 (fetched from the `mine` remote). Nothing pushed to the fork.

What changed:

- New custom op `ggml_temporal_route` (ggml-cpu.c, compiled on every platform). Input: router
  logits [E, n_tokens] F32. Per layer and per token: cold fill = top-K of the logits; later
  tokens admit the best non-resident expert iff its logit is strictly above the worst
  resident's, evicting the lowest-logit resident. The newest admission sits in slot K-1.
  Works for any batch size, so a prefill advances the window token by token exactly like
  decode.
- `build_moe_ffn` uses it whenever trained routing is on. Gate weights are unchanged engine
  code: softmax over all experts, gathered to the window, renormalised, which equals the
  softmax over the resident logits.
- `ggml_temporal_window_fill` (two-pass, n_tokens == 1) no longer picks experts in trained
  mode. It takes the router's ids, evicts the expert the router just dropped, makes every
  window expert resident (a no-op for ones already resident or in flight), and keeps the
  existing trim.
- `llama_context::decode` resets every layer's window at position 0, and refuses a batch
  that mixes sequences while trained routing is on (the state is per layer, not per sequence).
- Flags: `LLAMA_TEMPORAL_ROUTE=trained` turns it on; it is also the default whenever
  `LLAMA_TEMPORAL_TWOPASS` is set. `LLAMA_TEMPORAL_ROUTE=random` restores the earlier random
  one-swap policy. With neither flag the engine runs stock top-k, as before.
  `LLAMA_TEMPORAL_ROUTE_DUMP=path` writes every token's window (int32 layer + K ids).
  At exit the engine prints `temporal-route: ... swaps_per_layer_token=...`.
- Host-build fixes, needed because `temporal-moe-win` does not build on macOS: guard two trace
  lines in `mul_mat_id`, add macOS stubs for `ggml_temporal_window_fill` and
  `ggml_tm_wait_src_expert`, and map `mincore`/`posix_fadvise` in `llama-model-loader.cpp`.
  None of these touch Linux or Windows code paths.

Check tool: `scripts/serving_convert/gguf_logits.cpp` gained an `nll` mode (one model load,
rows of seq+1 ids, KV cleared per row, writes per-token NLL). `check_engine_routing.py`
compares the engine's window dump with `ref_model.py` masks and per-token losses with
`ref_temporal_512.npy`.

Commands (host, Apple M4 Pro, CPU only, flash attention off, F32 KV):

```bash
LLAMA_TEMPORAL_ROUTE=trained LLAMA_TEMPORAL_ROUTE_DUMP=route_f32_4.bin \
  gguf_logits nll fine_1e19-f32.gguf rows4.i32 4 2048 nll_f32_4.f32
python scripts/serving_convert/check_engine_routing.py --weights fine_1e19.safetensors \
  --tokens test_tokens.npz --rows 4 --dump route_f32_4.bin --nll nll_f32_4.f32 \
  --ref-losses ref_temporal_512.npy
```

| GGUF | rows x tokens | resident sets equal to reference | engine loss | reference loss | diff |
|---|---|---|---|---|---|
| F32 | 4 x 2048 | 106,496 of 106,496 layer-tokens (13 layers) | 3.14413 | 3.14409 | +0.00004 |
| F16 | 4 x 2048 | 12,287 of 106,496 | 3.14407 | 3.14409 | -0.00002 |

- F32: the mask check passes exactly, token by token, on every layer.
- F16: the sets diverge from F16 rounding at near-ties, and the window is path-dependent, so
  a divergence persists. Mean overlap on row 0 is 15.73 of 18 experts (17.39 in layer 1, about
  15.4 in deeper layers). The loss is unaffected (-0.00002 nats, max per-token |d| 1.24,
  median 0.048). The resident-set equality check is therefore only meaningful on the F32
  file; F16 is checked by loss.
- Swap rate: both the reference and the engine swap on every non-cold token
  (swaps_per_layer_token = 1.0000 in all four rows). The trained router never settles on a
  window equal to its current top-18, so each layer admits exactly one expert per token.
