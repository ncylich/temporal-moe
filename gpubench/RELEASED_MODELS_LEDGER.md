# Released checkpoints on the RTX A6000: our rule against Skliar et al. in the streaming engine (session R1)

Goal: decode tokens/s at batch 1 for the unadapted released checkpoints, Qwen3.5-35B-A3B first and then
Gemma4-26B-IT, under (1) stock top-k with every expert in VRAM, (2) our one-swap rule with K = 8 expert
slots per layer in VRAM and the pool in CPU RAM, (3) Skliar et al.'s cache-conditional routing with a cache
of C = 8 per layer in the same VRAM, every miss fetched, and (4) the no-bytes-moved controls of (2) and (3).
The reviewer (BZ2a) asked for tokens/s at matched accuracy; the accuracies are already measured (GSM8K,
n = 1319): Qwen3.5 ours 76.6, Skliar (C 8, lambda 0.4) 74.5; Gemma4 ours 78.8, Skliar (C 8, lambda 0.5) 80.7.

Entries are in the order things happened.

## R1-0 -- Settings the accuracy runs used, and the design (2026-11-14, before any pod)

**Checkpoints and chat settings** (from `analysis/residency/instruct_selfce.py`, `rescore_answer_only.py`,
`scripts/residency/audit_fixes.sh`, and the record names in `results/ablations/instruct_genbench_vllm.csv`):

| model | Hugging Face id | experts, k | accuracy record (ours / Skliar) | thinking |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | `Qwen/Qwen3.5-35B-A3B` | 256, 8 | `qwen35_think_off_n1319` R8 = 0.7665 / `qwen35_skliar_C8_lam0p4_n1319` = 0.7453 | off (`enable_thinking=False`) |
| Gemma4-26B-IT | `google/gemma-4-26B-A4B-it` | 128, 8 | `gemma4_instruct_n1319` R8 = 0.7877 / `gemma4_skliar_C8_lam0p5_n1319` = 0.8067 | template default (off) |

Task `gsm8k_cot_zeroshot`, flexible-extract, sampled (never greedy), 2048-token budget.

**Selection signal.** Both references act on the raw router logits that vLLM hands to its top-k
(`vllm_residency.apply(layer, router_logits)`). In llama.cpp that is `logits` in `build_moe_ffn`, the
input of the stock softmax; the routing op of patch 0001 already takes exactly that tensor.

**Skliar's rule, as implemented in `cache_bias.py`** (the reference to match, not the paper): per layer an
LRU cache of C experts; delta = running mean over every token seen (prompt included, never reset across
requests) of max(z) minus min(z); decode token: z' = z + lambda * delta on cached experts, top-k of z', the
top-1 of z forced in (J = 1) by dropping the selected expert with the lowest z'; loads = selected experts not
cached; the cache keeps the C most recently used (ties: higher expert index). The prompt is routed freely
and warms the cache; its loads are not counted. With C = k = 8 the cache after a decode step is exactly the
step's selection, so the 8 VRAM slots and the cache are the same set and every load is one copy.

**Our rule, `decode_state.py`.** Prompt routed freely while the training scan runs observe-only over it
(cold fill with the first prompt token's top-8, then one rule step per token); every generated token is one
rule step (admit the best non-resident iff its logit is strictly above the worst resident's, evict the
lowest-logit resident) and routes inside the 8 residents.

**Engine.** The fork base (`43bec5f3`, upstream of July 2026) already has `qwen35moe` and `gemma4`
(`src/models/qwen35moe.cpp`, `gemma4.cpp`). The slot registration of the systems patch is
architecture-neutral (needs separate `ffn_{gate,up,down}_exps`; the converter fuses gate/up only when asked
with `--fuse-gate-up-exps`, so the default conversion is fine). Qwen3.5 has a shared expert (sigmoid-gated,
stays in VRAM) and softmax gating with renormalisation over the selected experts, which `build_moe_ffn`
does after the routing op, so the gate weights equal the references' softmax over the selected logits.

**New engine patch** `scripts/serving_convert/engine_patch_cuda/0006-*.patch` (engine tree `7da7823d` on
top of `a235a715`):

- `LLAMA_TEMPORAL_ROUTE=decode` and `=skliar`, one CUDA kernel (`k_route2`) for both, per-layer state on
  the device, one captured decode graph. Prompt batches (more than one token) are routed with the stock
  top-8 in both modes.
- On the streamed path the same kernel keeps the 8 VRAM slots with one routine shared by both modes (keep
  slots whose expert is selected, load each missing expert into the lowest free slot) and the copies start
  at the router on the copy stream (the `TEMPORAL_ROUTE_SLOTS=2` design, the best variant of ledger G1-7,
  forced on for both modes). So arms 2 and 3 use identical copy machinery and differ only in what they
  select. What overlaps the copies, in both arms: whatever the graph runs between the router and the
  routed-expert GEMMs (gate-weight softmax, gather, normalisation, and the shared expert if
  `TEMPORAL_SHEXP_FIRST=1` puts it first). The gate GEMM waits for the gate copy of all of that layer's
  loads, the up GEMM for the up copy, the down GEMM for the down copy; there is no per-expert pipelining
  in either arm, since one `mul_mat_id` computes all 8 experts.
- `TEMPORAL_ROUTE_NOCOPY=1`: copy kernels launched with an empty swap list (no bytes moved);
  `=2`: no copy kernels, no events. Both give wrong output and are timing controls only (ledger G1-7
  showed that the first still pays for launching and joining the copy kernels).
- Exit line `temporal-route2:` with loads per decode layer-token and the histogram; `TEMPORAL_ROUTE_HIST`
  writes it as JSON in the format of the saved Python histograms; `LLAMA_TEMPORAL_ROUTE_DUMP2` writes
  every layer-token's logits and ids for the routing gate.
- `llama-bench` replays real tokens (`TEMPORAL_BENCH_CTX`, `TEMPORAL_BENCH_GEN`). **Why this is needed:**
  stock `llama-bench` fills the context with random tokens and feeds a random token at every decode step.
  For the trained 1.26B model that did not matter much (its swap rate was 1.0 either way), but Skliar's
  loads depend on how stable routing is from one token to the next, and random tokens would inflate them.
- `qwen35moe`: `TEMPORAL_SHEXP_FIRST=1` builds the shared expert before the routed experts (same ops).

Checked on the Mac: the CPU-side files and `llama-bench.cpp` compile (CPU-only build; the CUDA file
cannot be compiled here). The link of that CPU-only build fails on a pre-existing reference from the
systems patch's `qwen3moe.cpp` to CUDA symbols; not touched.

**Protocol for every arm** (`gpubench/released/run_released.py`): `llama-bench -ngl 99 -fa 1 -t 8 -p 0
-n 100 -d <ctx> -ub <ctx> -b <ctx> -r 4`, context filled as one prompt batch (so it is routed freely, as
in both references), the 100 decoded tokens replayed from a real continuation, four trials, first
discarded. Each routing method replays its own greedy continuation of the same chat context (earlier GSM8K
questions with their reference solutions as finished turns, then an open question, thinking off), built
by `gpubench/released/serve_client.py`. Known artifact, same for all arms: `llama-bench` restores the KV
state of the context for trials 2 to 4 but the routing state carries over from the end of the previous
trial; the reload of all 8 slots after the prompt batch therefore happens in trial 1 only, which is
discarded.

**Gates planned.**
1. Routing equality, engine against the Python references, on the engine's own router logits
   (`gpubench/released/route_check.py` imports `decode_state.py` and `cache_bias.py` unmodified and replays
   the dumped logits). Both sides see the same logits, so quantization cannot flip a near-tie between them;
   this checks the rule implementation, not the forward pass.
2. Skliar's loads per layer-token on GSM8K generations from the engine against the saved histogram.
3. Batch-1 perplexity streamed against all in VRAM under each routing mode (must be identical).

**Blocked at the start.** The pod of the trained-model run (`bcnqix1ewwwjr3`, EU-SE-1) cannot resume:
`podResume` answered "There are not enough free GPUs on the host machine". Creating a new A6000 pod (and,
before that, enlarging the stopped pod's disk) was refused by this session's permission check, so the
orchestrating session was asked for approval at 08:15 UTC.
