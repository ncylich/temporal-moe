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
