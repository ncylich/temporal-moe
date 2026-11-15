# Engine patch: trained temporal routing

`0001-*.patch` and `0002-*.patch` add the training-time routing rule to the device engine (`ncylich/llama.cpp`).
It applies to branch `temporal-moe-win` at commit `43bec5f3` (which builds on `temporal-moe`
@ `61f6d1b4`), so one change serves the phone (Android build) and the Windows laptop.

```bash
cd llama.cpp
git fetch <remote> temporal-moe-win
git checkout -b temporal-moe-trained-routing 43bec5f3
git am /path/to/temporal-moe/scripts/serving_convert/engine_patch/000*.patch
```

Checked: applied with `git am` to a fresh clone at `43bec5f3`, the result has the same tree
(`b69151b6`) as the branch it was exported from (engine commit `759d43c5`).

`0002` fixes a bug `0001` had: glm4moe drops rows that need no logits before the last layer's
FFN, so the last layer's window skipped those tokens. Under trained routing every layer now
routes every token. Builds made before `0002` are wrong for prefills and llama-perplexity.

## What it does

- `LLAMA_TEMPORAL_ROUTE=trained`: experts come from the router logits under the rule of
  `temporal/temporal_router.py` with `evict="min_logit"`. Cold fill is the top-K of the first
  token; each later token admits the best non-resident expert only if its logit beats the worst
  resident, evicting the lowest-logit resident. That is zero or one swap per layer per token.
  Gate weights are the softmax over the resident logits. Works for prefill and decode alike.
- It is the default whenever `LLAMA_TEMPORAL_TWOPASS` is set. `LLAMA_TEMPORAL_ROUTE=random`
  restores the earlier random one-swap policy. With neither flag the engine is stock top-k.
- The window resets at sequence position 0. A batch must hold a single sequence.
- `LLAMA_TEMPORAL_ROUTE_DUMP=<file>` writes every token's window (int32 layer, then K int32 ids).
- At exit: `temporal-route: ... swaps_per_layer_token=<x>`.
- Also makes the branch build on macOS (CPU only) for host checks.

Checks are in `androidbench/TRAINED_MODEL_LEDGER.md` (step 2) and
`scripts/serving_convert/check_engine_routing.py`.

## Decode-time rules on the CPU path (patch 0003)

For a model without temporal training (the plain 1e19 MoE): our rule applied at decode time, and Skliar
et al.'s cache-conditional routing. Before these patches both existed only as CUDA kernels
(`../engine_patch_cuda/0006` to `0011`) and aborted on the CPU backend. `0003` applies after `0001` and `0002`:

```bash
git checkout -b pixel-skliar 43bec5f3
git am /path/to/temporal-moe/scripts/serving_convert/engine_patch/000*.patch
```

| switch | what it does |
|---|---|
| `LLAMA_TEMPORAL_ROUTE=decode` | our rule as `analysis/residency/decode_state.py`: a prompt batch (more than one token) is routed by the stock top-k while the min_logit window scan observes it; each single-token batch is one rule step (0 or 1 expert admitted) and routes inside the K residents |
| `LLAMA_TEMPORAL_ROUTE=skliar` + `TEMPORAL_CB_C`, `TEMPORAL_CB_LAMBDA` (`TEMPORAL_CB_J`, 0 or 1, default 1) | Skliar's rule as `analysis/residency/cache_bias.py`: LRU cache of C experts per layer, lambda times the running mean of max-min logit added to cached experts, top-1 guarantee; the prompt is routed by the stock top-k and warms the cache. Any number of experts can enter on one token |
| `LLAMA_TEMPORAL_ROUTE=replay` + `TEMPORAL_REPLAY_HIST="c0,c1,c2,..."` | for models whose router is not meaningful (random weights): each layer-token draws its load count from the histogram (weights for 0, 1, 2, ... loads) and swaps that many random experts; `0,1` is exactly one load per layer per token. Same downstream path as `skliar` |
| `TEMPORAL_CB_FULLSEL=1` | Skliar by a plain top-K of the biased logits instead of the exchange loop used when C = K (same selections; cross-check only) |
| `LLAMA_TEMPORAL_TWOPASS=1` with either mode | streamed: K experts per layer in RAM, the pool on flash. The window op evicts every expert that left the window and queues all missing ones at once, so the fetch workers read them in parallel. Experts that entered on this token sit last in the window; the first sub-pass does not wait for them up front, and the repacked kernel computes that layer's residents first. With `skliar`, C must equal K |
| `LLAMA_TEMPORAL_TAIL=<n>` (default 1, any mode) | the two-pass FFN ends with n single-expert sub-passes instead of one |
| `LLAMA_TEMPORAL_R >= n_expert` with either mode | nothing is evicted: the all-in-RAM twin of the streamed path, same graph and kernels |
| `TEMPORAL_ROUTE_HIST=<json>` | loads-per-layer-token histogram at exit, format of `cache_bias.stats()` |
| `LLAMA_TEMPORAL_ROUTE_DUMP2=<file>` | logits and ids per layer-token plus a marker at each sequence reset, for `gpubench/released/route_check.py` |
| `TEMPORAL_BENCH_CTX`, `TEMPORAL_BENCH_GEN` | `llama-bench` fills the context and feeds the decode steps from int32 token files (real text) instead of random tokens |
| `TEMPORAL_BENCH_REPREFILL=1` | `llama-bench` fills the context again on every repetition. Without it the saved KV state is restored, and the router state (window, cache) is not part of it, so later repetitions start from where the previous one ended |
| `TEMPORAL_BENCH_TOKLOG=<csv>` | per decoded token: time, loads summed over the layers, largest per-layer load count, time blocked on fetches |
| `TEMPORAL_BENCH_LOGIT_HASH=1` | adds a 64-bit hash of each decoded token's logits to that file (output-identity gate) |

Exit line: `temporal-route2: mode=... decode_layer_tokens=... loads=... loads_per_layer_token=... hist=[...]`.
Loads are counted on single-token batches only. Skliar's running mean of max-min is per layer and per
process, as in the reference; it is not reset between sequences.

`LLAMA_TEMPORAL_ROUTE=trained` is unchanged by this patch (same rule, same window order, same waits).

Checked: `git am` of `0001` to `0003` on a fresh clone at `43bec5f3` gives tree `536002e3` (local engine commit `4f9948d8`, not pushed to the fork). The 1.26 B rows of the ledger were
timed on tree `b1f58983` (commit `d9f0f2d1`), which is this patch without the `replay` mode; the 11 B rows on `536002e3`.

Checks and numbers: `androidbench/PIXEL_SKLIAR_LEDGER.md`.
