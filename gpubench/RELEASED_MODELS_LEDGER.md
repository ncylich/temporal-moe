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

## R1-1 -- Pod, build, conversion (2026-11-14 08:37-09:05 UTC)

**Pod.** The user approved a new pod at about 08:35 UTC. A request with at least 120 GB RAM and 300 GB
container disk was refused for supply in both clouds; the next one succeeded: **pod `i65msin9gh64dw`, one
RTX A6000, US, secure cloud, $0.69/h, created 08:37 UTC**, 200 GB container disk, 60 GB volume.

| item | value |
|---|---|
| GPU | NVIDIA RTX A6000, 48,541 MiB, driver 595.91.07, PCIe gen 4 x16 |
| host | AMD EPYC 7543 (128 threads visible), cgroup memory limit 62 GB (503 GB on the host) |
| image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, torch 2.4.1+cu124 (not reinstalled) |
| added | apt: tmux cmake; pip: huggingface_hub hf_transfer safetensors tokenizers sentencepiece `pyarrow<16` (numpy stays 1.26) transformers 5.23 (tokenizer side only, it disables itself under torch 2.4) |

No Hugging Face token exists on the Mac (`~/.cache/huggingface` has no `token`); both checkpoints are
public, so the download ran unauthenticated.

**Build.** `gpubench/build_engine.sh` (patches 0001 to 0006), engine tree **`96b1c0e3`** after the fixes
below, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release`.

**Model.** `Qwen/Qwen3.5-35B-A3B` snapshot (67 GB) converted with the engine tree's
`convert_hf_to_gguf.py --outtype bf16 --no-mtp` (69.4 GB, 733 tensors), then `llama-quantize ... Q4_K_M`:
**`qwen35-q4km.gguf`, 21,166,758,080 bytes**. The routed experts are separate tensors
(`ffn_{gate,up,down}_exps`, `{2048, 512, 256}`), router `ffn_gate_inp` F32, shared expert
`ffn_*_shexp` with its sigmoid gate. Q4_K_M puts gate and up in Q4_K (576 KiB per expert) and down in Q6_K
(840 KiB per expert), so one load is about 2.0 MB. 40 layers, all MoE; 30 Gated DeltaNet layers and 10
full-attention layers (head size 256, which the CUDA flash-attention kernels support, so `-fa 1` is used,
unlike the 1.26B model's `-fa 0`).

**Went wrong, and how it was caught.**
1. The first pod edit and pod creation were refused by the session's permission check (R1-0); the old pod
   could not resume.
2. First CUDA compile of 0006 failed: two functions used before their declaration (the Mac check could
   not compile CUDA). Fixed by forward declarations.
3. `llama-server` did not build: its web-UI asset step fails offline. `-DLLAMA_BUILD_UI=OFF
   -DLLAMA_USE_PREBUILT_UI=OFF` and a clean of `build/tools/ui` fixed it; `build_engine.sh` now passes both.
4. The first conversion carried the multi-token-prediction draft layer (`blk.40`, with its own 256
   experts), which plain decoding never runs but whose experts would sit in VRAM in the baseline. Caught
   in the conversion log; reconverted with `--no-mtp`.

## R1-2 -- Making Skliar's routing step as cheap as ours (09:03-09:12 UTC)

Smoke test with stock `llama-bench` (random tokens, `-d 512 -n 32/64`, all experts in VRAM): stock top-k
151 to 159 tok/s, our routing 145 to 150, Skliar's 132 to 138. So the first Skliar kernel cost about
0.7 ms per token more than ours, about 18 us per layer: an implementation cost, not a property of the
method, and it would have counted against Skliar. Found by elimination (no GPU counters on this pod,
`ERR_NVGPUCTRPERM`): with the top-k over the biased logits skipped (debug switch, not shipped), Skliar's
arm ran at our speed (146/147 against 145/150). Changes, all in 0006 and used by both modes:

- contended shared-memory atomics replaced by warp-shuffle block reductions (no measurable change);
- the cache cap after a decode step uses the exact shortcut for C = K (the K selected experts carry the
  newest time, every other cached expert is older, so the cap keeps exactly the selection);
- top-k: first a vectorized rank count (138 to 145), then the scheme of the stock fused top-k MoE kernel:
  one warp, 8 values per lane in registers, 8 rounds of warp argmax. Result **base 151.6 / 159.3, ours
  148.7 / 147.8, Skliar 147.0 / 145.9 tok/s** (two passes). The two routing steps now cost the same to
  within noise; both cost about 0.3 to 0.7 ms per token more than stock, which is the price of replacing
  the stock fused softmax/top-k kernel by a custom op plus the unfused gate-weight ops (as for the 1.26B
  model). The timed comparison of arms 2 and 3 is therefore not tilted by routing-kernel cost.

These smoke numbers use random tokens and are not results.

## R1-3 -- Gates (09:18 UTC onward, `gpubench/released/gates.sh`, outputs in `gpubench/results/released/qwen35/gates/`)

### Gate 1: routing equality against the Python references (pass, all four configurations)

Three sampled GSM8K generations per configuration (zero-shot CoT prompt, thinking off, up to 400 tokens),
`llama-server` with `GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_TEMPORAL_ROUTE_DUMP2=...`; `route_check.py` feeds
every dumped logit vector to `decode_state.py` (prefill, then step) or `cache_bias.py` (apply with the
vLLM span protocol), imported unmodified. **Weights: Q4_K_M**; since both sides read the same logits,
quantization cannot flip a near-tie between them, so this checks the rule, not the forward pass.

| configuration | prompt layer-tokens equal (stock top-8) | decode layer-tokens equal | loads equal |
|---|---|---|---|
| ours, all in VRAM | 8,920 / 8,920 | 26,320 / 26,320 | (0 or 1 by construction) |
| ours, streamed | 8,920 / 8,920 | 26,320 / 26,320 | |
| Skliar, all in VRAM | 8,920 / 8,920 | 28,480 / 28,480 | 28,480 / 28,480 |
| Skliar, streamed | 8,920 / 8,920 | 26,680 / 26,680 | 26,680 / 26,680 |

(5 sequences each: the server's two 2-token warm-up batches, which the engine and the checker both treat
as sequences, and the 3 prompts.)

**Went wrong first, and why it was the checker.** The first pass gave ours 21,363 of 26,000 while Skliar
matched exactly. `llama-server` was splitting every prompt into two batches (84 + 4 tokens: a context
checkpoint before the last tokens, made for recurrent models), and also runs two 2-token warm-up batches.
The engine carries the observe-only scan across batches of one sequence, which is what the reference's
`observe_chunk` does for chunked prefill; the checker had restarted the scan at every batch with
`prefill()`. Skliar was unaffected because with C = 8 a re-warm from the last chunk gives the same cache
(the last prompt token's top-8). Fixed on both sides: the dump now writes a marker record at every
sequence reset (position 0), the checker starts a sequence only there and continues the scan for later
batches, and the server runs with `-ctxcp 0` so a prompt is one batch, as in vLLM.

Streamed and all-in-VRAM generations of the same prompts are not textually identical (sampling is seeded
identically, the routing of each is exactly the reference's for its own logits). The streamed decode
path uses the unfused gate/up kernels and the prompt goes through the expert-major prefill kernel, so the
logits differ at rounding level and a sampled token can flip. Gate 3 checks the computation at batch 1.

### Gate 2: Skliar's loads per layer-token on GSM8K, engine against the saved histogram (pass)

First 200 GSM8K test items, zero-shot CoT prompt as one user turn, thinking off, sampled (temperature 0.7,
top-p 0.8, top-k 20, seed 1234 + item), up to 2048 tokens, one request at a time, prompt cache off; all
experts in VRAM (gate 1 shows the streamed path selects the same experts). Engine histogram from
`TEMPORAL_ROUTE_HIST`; saved one is `results/ablations/skliar_c8_qwen_lam04_hist.json` (vLLM, all 1319
items). `gpubench/released/compare_hist.py`.

| loads in a layer-token | engine (2,569,880 layer-tokens) | saved (17,540,400) |
|---|---|---|
| 0 | 0.2648 | 0.2587 |
| 1 | 0.5573 | 0.5625 |
| 2 | 0.1167 | 0.1173 |
| 3 | 0.0413 | 0.0411 |
| 4 | 0.0144 | 0.0144 |
| 5 | 0.0042 | 0.0043 |
| 6 | 0.0010 | 0.0013 |
| 7 | 0.0002 | 0.0004 |
| 8 | 0.0000 | 0.0001 |
| **mean** | **1.0011** | **1.0105** |

Total variation distance 0.0064. The remaining difference is within what a different sample (200 of 1319
items), different sampled text and Q4_K_M against BF16 weights would give; not chased further.

Our rule on the same 200 items: 2,620,160 decode layer-tokens, 2,620,022 with one load and 138 with none,
mean 0.9999 (the rule's cap is 1).

Sanity check, not a gate and not a replacement for the n = 1319 accuracies: GSM8K flexible-extract on these
200 engine generations is stock 178/200 (0.890), ours 165/200 (0.825), Skliar 156/200 (0.780). The vLLM
numbers on all 1319 items are 0.859 / 0.766 / 0.745; same order, and n = 200 has a standard error of
about 0.03. Mean generated length 293 / 329 / 322 tokens, 2 / 4 / 3 of 200 hit the 2048 limit.

### Gate 3: batch-1 perplexity, streamed against all in VRAM (pass, identical)

`llama-perplexity -f ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa on -t 8` (`-ncmoe 40` and
`TEMPORAL_UNIFIED=1` for streamed), `ppl.txt` = the GSM8K test questions and solutions concatenated (an
identity oracle). At batch 1 every token after a chunk's first is a decode step under the rule.

| routing | all in VRAM | streamed | running estimates |
|---|---|---|---|
| ours | 2.1260 +/- 0.07134 | 2.1260 +/- 0.07134 | [1]2.3679 [2]2.2148 [3]2.2192 [4]2.1260 in both |
| Skliar | 1.9938 +/- 0.06096 | 1.9938 +/- 0.06096 | [1]2.1992 [2]2.1349 [3]2.0931 [4]1.9938 in both |
| stock top-k | 1.7345 +/- 0.04634 | | |

Loads per layer-token in these runs: ours 0.9998, Skliar 1.0001 (163,680 layer-tokens each).

## R1-4 -- Timing at 1k, and why every arm is pinned to the GPU's NUMA node (09:51-10:13 UTC)

**Token files** (`tokens.sh`): chat contexts of 957 (1k), 1990 (2k) and 3989 (4k) prompt tokens made of
earlier GSM8K questions with their reference solutions as finished turns plus an open question; each
method's greedy continuation (all experts in VRAM, its own routing) fills the context to exactly 1024 /
2048 / 4096 tokens and gives the 100 replayed decode tokens. Where an answer ended early the chat went
on with another question (1k: one extra turn for every method).

**First pass (void, kept as raw rows in `qwen_decode_unpinned.jsonl`).** All nine arms once, unpinned:
base 152.4, ours in VRAM 140.7, Skliar in VRAM 139.2, ours streamed 71.6, Skliar streamed 76.1 tok/s.
A copy-grid check right after gave ours streamed 78.7 and Skliar 70.3, the opposite order. Six
alternations of the two streamed arms (`qwen_alt_unpinned.jsonl`) gave ours 72.2 to 78.8 and Skliar 70.7 to
76.3, with ours varying inside one process (trials 78.1, 70.1, 68.4). The host has two NUMA nodes and the
GPU sits on node 0 (`nvidia-smi topo -m`: CPU affinity 0-31,64-95); the 20 GB pinned expert pool and the
decode threads land on whichever node the process runs on, and the scheduler moves threads. `numactl` is
not permitted in the container; `taskset` is. With `taskset -c 0-31,64-95` (four alternations,
`qwen_alt_node0.jsonl`): ours streamed 78.54 / 78.62 / 78.67 / 78.80, Skliar streamed 76.09 / 76.12 /
76.21 / 76.36. **Deviation, applied to every arm including the baseline: all timing runs are pinned to
the GPU's NUMA node with `taskset -c 0-31,64-95`.**

Also measured: host-to-device pinned `cudaMemcpy` on this machine reaches 17.6 GB/s for 2 MB and 22 GB/s
for 80 MB. Copy grid: 128 blocks 79.3 / 69.2, 256 (default) 78.7 / 70.3, 1024 73.3 / 70.0 tok/s (ours /
Skliar, unpinned, single runs); the default 256 is kept for both arms.

## R1-5 -- Qwen3.5 decode timing, final protocol (engine tree `96b1c0e3` = patches 0001 to 0006, 10:13-10:33 UTC)

`gpubench/released/run_released.py --prefix "taskset -c 0-31,64-95"`, all nine arms per pass, arm order
reversed in the middle pass; three passes at 1k, two at 2k and 4k. Rows: `gpubench/results/released/qwen35/
qwen_decode_final.jsonl` (63 rows, no failures). "Pooled" is the mean of every kept trial (trials 2 to 4
of each pass). Loads per layer-token come from the engine's exit line (all four trials and the warm-up
token of the last pass; the routing is deterministic, so every pass gives the same counts). The
no-bytes-moved controls compute with stale slot contents, so their routing, and their loads, drift from
the real arm's; they move no bytes either way, so this does not affect what they measure.

### 1024 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 158.26 | 160.46 / 156.83 / 157.50 | 6.319 | 21336 | 1.000 |  |  |
| ours_vram | 147.69 | 149.62 / 146.81 / 146.63 | 6.771 | 21338 | 0.933 | 0.9996 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_vram | 146.11 | 147.36 / 145.96 / 145.00 | 6.844 | 21338 | 0.923 | 1.0207 | 0.323 / 0.466 / 0.126 / 0.085 |
| ours_stream | 78.46 | 78.89 / 78.19 / 78.29 | 12.746 | 3636 | 0.496 | 0.9997 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_stream | 76.29 | 76.36 / 76.22 / 76.28 | 13.109 | 3636 | 0.482 | 1.0211 | 0.323 / 0.465 / 0.126 / 0.085 |
| ours_nobytes | 134.76 | 135.09 / 134.35 / 134.83 | 7.421 | 3636 | 0.851 | 0.9998 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nobytes | 133.44 | 133.78 / 132.99 / 133.55 | 7.494 | 3636 | 0.843 | 0.8317 | 0.393 / 0.468 / 0.084 / 0.055 |
| ours_nocopy | 135.64 | 135.85 / 135.60 / 135.45 | 7.373 | 3636 | 0.857 | 0.9998 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nocopy | 134.06 | 133.77 / 134.41 / 133.98 | 7.460 | 3636 | 0.847 | 0.8317 | 0.393 / 0.468 / 0.084 / 0.055 |

### 2048 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 156.43 | 156.48 / 156.38 | 6.393 | 22374 | 1.000 |  |  |
| ours_vram | 146.15 | 145.75 / 146.55 | 6.842 | 22376 | 0.934 | 1.0000 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_vram | 144.75 | 144.36 / 145.14 | 6.909 | 22376 | 0.925 | 0.9939 | 0.240 / 0.604 / 0.101 / 0.055 |
| ours_stream | 78.27 | 78.04 / 78.49 | 12.777 | 4738 | 0.500 | 1.0000 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_stream | 77.59 | 77.72 / 77.45 | 12.889 | 4738 | 0.496 | 0.9980 | 0.236 / 0.605 / 0.105 / 0.054 |
| ours_nobytes | 134.07 | 133.95 / 134.19 | 7.459 | 4738 | 0.857 | 0.9997 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nobytes | 132.57 | 132.95 / 132.19 | 7.543 | 4738 | 0.847 | 0.8821 | 0.355 / 0.483 / 0.107 / 0.055 |
| ours_nocopy | 135.27 | 134.96 / 135.58 | 7.393 | 4738 | 0.865 | 0.9999 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nocopy | 133.39 | 132.75 / 134.03 | 7.497 | 4738 | 0.853 | 0.8821 | 0.355 / 0.483 / 0.107 / 0.055 |

### 4096 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 155.36 | 155.41 / 155.30 | 6.437 | 24432 | 1.000 |  |  |
| ours_vram | 145.25 | 145.29 / 145.21 | 6.885 | 24434 | 0.935 | 1.0000 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_vram | 143.51 | 143.27 / 143.76 | 6.968 | 24434 | 0.924 | 0.9908 | 0.257 / 0.579 / 0.109 / 0.055 |
| ours_stream | 77.75 | 77.56 / 77.94 | 12.861 | 6924 | 0.500 | 1.0000 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_stream | 76.64 | 76.51 / 76.76 | 13.049 | 6924 | 0.493 | 0.9914 | 0.251 / 0.586 / 0.109 / 0.054 |
| ours_nobytes | 132.25 | 132.71 / 131.79 | 7.561 | 6924 | 0.851 | 0.9998 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nobytes | 131.02 | 131.15 / 130.88 | 7.633 | 6924 | 0.843 | 0.9204 | 0.304 / 0.541 / 0.105 / 0.050 |
| ours_nocopy | 133.39 | 133.23 / 133.55 | 7.497 | 6924 | 0.859 | 0.9999 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nocopy | 131.55 | 131.42 / 131.69 | 7.601 | 6924 | 0.847 | 0.9266 | 0.299 / 0.546 / 0.103 / 0.052 |

**Where the time goes** (ms per token, pooled; differences of rows above):

| step | ours 1k | Skliar 1k | ours 2k | Skliar 2k | ours 4k | Skliar 4k |
|---|---|---|---|---|---|---|
| stock top-k, all in VRAM | 6.319 | 6.319 | 6.393 | 6.393 | 6.437 | 6.437 |
| + routing op (all in VRAM) | +0.452 | +0.526 | +0.450 | +0.516 | +0.448 | +0.531 |
| + slot machinery, no copy kernels | +0.602 | +0.615 | +0.551 | +0.588 | +0.612 | +0.633 |
| + copy kernels launched, nothing moved | +0.048 | +0.034 | +0.066 | +0.046 | +0.065 | +0.031 |
| + bytes moved | +5.325 | +5.615 | +5.318 | +5.346 | +5.300 | +5.416 |
| = streamed | 12.746 | 13.109 | 12.777 | 12.889 | 12.861 | 13.049 |

The bytes dominate: 40 layers times about 1.0 load of 2.0 MB is about 80 MB per token, and the measured
cost per load is the same for both arms (1k: ours 5.325 ms / 39.99 loads = 133 us, Skliar 5.615 / 40.84 =
137 us; 2k: 133 and 134 us), about 15 GB/s, against 17.6 GB/s for a plain 2 MB pinned `cudaMemcpy` on this
host. Skliar's arm is slower by its slightly higher mean load at 1k (1.021 against 1.000 on its own replayed
text) and by about 0.07 ms of routing step (top-k over the biased logits, J check, cache update).

## R1-6 -- The shared expert under the copies (patch 0007, engine tree `20c50f0c`, 10:37-11:06 UTC)

In 0006 the graph order per layer is router, gate weights, routed experts, then the shared expert, so the
copies started at the router overlap only the few small gate-weight kernels. (0006's own
`TEMPORAL_SHEXP_FIRST` for `qwen35moe` put the shared expert before the router and could not help; found by
reading `build_moe_ffn`, which expands the routed-expert GEMMs itself, before it was ever timed.) Patch 0007
adds `llm_graph_context::tm_after_router`, a node `build_moe_ffn` expands right after the router and the
gate weights, and uses it for Qwen3.5's shared expert and Gemma4's dense MLP under
`TEMPORAL_SHEXP_FIRST=1`. Same ops and same add order: batch-1 perplexity is unchanged (ours streamed
2.1260 +/- 0.07134 with and without; stock 1.7345 +/- 0.04634).

Same protocol as R1-5 with `--env TEMPORAL_SHEXP_FIRST=1` for **every** arm, three passes at 1k, two at 2k
and 4k (the 1k controls have three passes too). Rows: `qwen_decode_shexp.jsonl` (63 rows, no failures).

### 1024 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 157.30 | 155.40 / 158.20 / 158.31 | 6.357 | 21328 | 1.000 |  |  |
| ours_vram | 146.89 | 145.24 / 147.83 / 147.60 | 6.808 | 21330 | 0.934 | 0.9996 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_vram | 144.84 | 143.30 / 145.61 / 145.60 | 6.904 | 21330 | 0.921 | 1.0207 | 0.323 / 0.466 / 0.126 / 0.085 |
| ours_stream | 82.73 | 82.45 / 82.92 / 82.81 | 12.088 | 3628 | 0.526 | 0.9997 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_stream | 78.79 | 78.79 / 78.77 / 78.81 | 12.692 | 3628 | 0.501 | 1.0211 | 0.323 / 0.465 / 0.126 / 0.085 |
| ours_nobytes | 135.33 | 135.99 / 134.92 / 135.09 | 7.389 | 3628 | 0.860 | 0.9998 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nobytes | 133.49 | 134.11 / 133.64 / 132.73 | 7.491 | 3628 | 0.849 | 0.8317 | 0.393 / 0.468 / 0.084 / 0.055 |
| ours_nocopy | 136.11 | 135.77 / 136.29 / 136.28 | 7.347 | 3628 | 0.865 | 0.9998 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nocopy | 134.38 | 134.63 / 133.78 / 134.72 | 7.442 | 3628 | 0.854 | 0.8317 | 0.393 / 0.468 / 0.084 / 0.055 |

### 2048 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 156.82 | 157.32 / 156.32 | 6.377 | 22358 | 1.000 |  |  |
| ours_vram | 146.29 | 146.70 / 145.88 | 6.836 | 22360 | 0.933 | 1.0000 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_vram | 144.84 | 144.52 / 145.15 | 6.904 | 22360 | 0.924 | 0.9939 | 0.240 / 0.604 / 0.101 / 0.055 |
| ours_stream | 82.47 | 82.54 / 82.41 | 12.125 | 4722 | 0.526 | 1.0000 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_stream | 80.74 | 80.97 / 80.50 | 12.386 | 4722 | 0.515 | 0.9940 | 0.236 / 0.606 / 0.106 / 0.052 |
| ours_nobytes | 134.17 | 133.87 / 134.48 | 7.453 | 4722 | 0.856 | 0.9999 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nobytes | 133.07 | 133.07 / 133.07 | 7.515 | 4722 | 0.849 | 0.8821 | 0.355 / 0.483 / 0.107 / 0.055 |
| ours_nocopy | 134.91 | 134.62 / 135.19 | 7.413 | 4722 | 0.860 | 0.9999 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nocopy | 133.72 | 133.45 / 133.98 | 7.479 | 4722 | 0.853 | 0.8874 | 0.354 / 0.476 / 0.116 / 0.053 |

### 4096 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 155.40 | 155.99 / 154.81 | 6.435 | 24396 | 1.000 |  |  |
| ours_vram | 145.25 | 145.51 / 145.00 | 6.885 | 24398 | 0.935 | 1.0000 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_vram | 143.45 | 143.46 / 143.45 | 6.971 | 24398 | 0.923 | 0.9908 | 0.257 / 0.579 / 0.109 / 0.055 |
| ours_stream | 81.90 | 81.73 / 82.07 | 12.210 | 6888 | 0.527 | 1.0000 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_stream | 79.44 | 79.56 / 79.32 | 12.588 | 6888 | 0.511 | 0.9944 | 0.252 / 0.583 / 0.110 / 0.055 |
| ours_nobytes | 133.02 | 132.43 / 133.61 | 7.518 | 6888 | 0.856 | 0.9998 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nobytes | 130.85 | 130.27 / 131.43 | 7.642 | 6888 | 0.842 | 0.9266 | 0.299 / 0.546 / 0.103 / 0.052 |
| ours_nocopy | 133.46 | 133.31 / 133.61 | 7.493 | 6888 | 0.859 | 0.9999 | 0.000 / 1.000 / 0.000 / 0.000 |
| skliar_nocopy | 132.32 | 132.57 / 132.06 | 7.558 | 6888 | 0.851 | 0.9204 | 0.304 / 0.541 / 0.105 / 0.050 |

Exposed transfer (streamed minus no bytes moved): ours 4.70 / 4.67 / 4.69 ms at 1k / 2k / 4k, against
5.33 / 5.32 / 5.30 without 0007, so about 0.63 ms per token now runs under the shared expert. Skliar 5.20
/ 4.87 / 4.95 against 5.62 / 5.35 / 5.42, about 0.45 ms hidden. Ours gains more because each of its layers
has exactly one load to hide, while about a quarter of Skliar's layer-tokens have no load (nothing to hide)
and its layers with two or more loads outlast the shared expert. Stock is unchanged (157.3 against 158.3
tok/s at 1k, within pass-to-pass spread).

## Results, Qwen3.5-35B-A3B (Q4_K_M, RTX A6000, batch 1, real-text replay)

Headline configuration: patches 0001 to 0007, `TEMPORAL_SHEXP_FIRST=1` for every arm, `-fa 1 -t 8`,
pinned to the GPU's NUMA node. tok/s pooled over passes (three at 1k, two at 2k and 4k); ratio to stock
top-k with every expert in VRAM. GSM8K accuracies are the n = 1319 vLLM numbers of the paper's Section 7
(not re-measured here).

| arm | GSM8K | 1k tok/s (ms) | vs stock | 2k tok/s | vs stock | 4k tok/s | vs stock | peak VRAM at 1k | loads per layer-token (1k replay / GSM8K) |
|---|---|---|---|---|---|---|---|---|---|
| 1. stock top-k, all 256 experts in VRAM | 85.9 | 157.30 (6.357) | 1.000 | 156.82 | 1.000 | 155.40 | 1.000 | 21,328 MiB | n/a |
| 2. ours, 8 slots per layer streamed | 76.6 | **82.73** (12.088) | 0.526 | **82.47** | 0.526 | **81.90** | 0.527 | 3,628 MiB | 1.000 / 1.000 |
| 3. Skliar C 8, lambda 0.4, 8 slots streamed | 74.5 | 78.79 (12.692) | 0.501 | 80.74 | 0.515 | 79.44 | 0.511 | 3,628 MiB | 1.021 / 1.001 |
| 4a. arm 2's kernels, no bytes moved | | 135.33 | 0.860 | 134.17 | 0.856 | 133.02 | 0.856 | 3,628 MiB | |
| 4b. arm 3's kernels, no bytes moved | | 133.49 | 0.849 | 133.07 | 0.849 | 130.85 | 0.842 | 3,628 MiB | |
| ours, all experts in VRAM | | 146.89 | 0.934 | 146.29 | 0.933 | 145.25 | 0.935 | 21,330 MiB | |
| Skliar, all experts in VRAM | | 144.84 | 0.921 | 144.84 | 0.924 | 143.45 | 0.923 | 21,330 MiB | |

Ours against Skliar, streamed: 1.050x at 1k, 1.021x at 2k, 1.031x at 4k (tok/s). Without 0007 (R1-5): 78.46
against 76.29 at 1k (1.028x), 78.27 against 77.59 at 2k (1.009x), 77.75 against 76.64 at 4k (1.015x).

Loads per decode layer-token, distribution (fraction of layer-tokens with 0 / 1 / 2 / 3 or more loads):
ours 0 / 1.000 / 0 / 0 everywhere (138 of 2.62 M GSM8K layer-tokens had none). Skliar on GSM8K (gate 2,
engine, 200 items) 0.265 / 0.557 / 0.117 / 0.061, saved vLLM histogram 0.259 / 0.563 / 0.117 / 0.062; on the
1k / 2k / 4k replays 0.323 / 0.465 / 0.126 / 0.085, 0.236 / 0.606 / 0.106 / 0.052, 0.252 / 0.583 / 0.110 /
0.055.

Peak VRAM is `nvidia-smi memory.used` during the process, CUDA context and KV cache included: 21,328 against
3,628 MiB at 1k (5.9x less), 22,358 against 4,722 at 2k, 24,396 against 6,888 at 4k. The streamed arms also
hold the expert pool twice in host RAM (the `-ncmoe` CPU buffer and a pinned copy, about 20 GB each).

### Statements a reply or the paper could quote

- (measured) On one RTX A6000 with Qwen3.5-35B-A3B at Q4_K_M, batch-1 decode at 1k context runs at 157.3
  tok/s with all 256 experts per layer in VRAM (21.3 GB peak). With only 8 expert slots per layer in VRAM
  and the pool in CPU RAM (3.6 GB peak), our one-swap rule decodes at 82.7 tok/s and Skliar et al.'s
  cache-conditional routing (cache 8, lambda 0.4) at 78.8 tok/s.
- (measured, with the paper's accuracies) At the same 8 experts per layer in fast memory, our rule is both
  more accurate on GSM8K (76.6 against 74.5) and faster in the streaming engine (82.7 against 78.8 tok/s at
  1k, 82.5 against 80.7 at 2k, 81.9 against 79.4 at 4k): it is ahead on both axes, so there is no trade-off
  to match.
- (measured) The speed difference is small, 2 to 5% depending on context, and the two methods load almost
  the same number of experts (1.00 against about 1.00 per layer and token on GSM8K). Our advantage comes
  from the load being exactly one in every layer, which overlaps better with the shared expert and costs
  no top-k over biased logits; Skliar's loads are spread from 0 to 8 per layer.
- (measured) Both streamed configurations run at about half the all-in-VRAM rate on this card (0.53 and
  0.50); the cost is the PCIe transfer, about 2 MB per load, 40 loads per token, at about 15 GB/s effective
  (4.7 to 5.2 ms of a 12.1 to 12.7 ms token), not the routing or slot kernels (about 1.0 to 1.1 ms
  together).
- (measured) Routing equality: on GSM8K generations, the engine's selected expert sets equal the Python
  reference implementations' on every layer-token, for both methods and both memory placements (26,320 to
  28,480 decode layer-tokens per configuration, weights Q4_K_M, references fed the engine's own logits).
- (measured) Skliar's per-layer load distribution in the engine matches the vLLM histogram the accuracy
  numbers were measured with (mean 1.001 against 1.011, total variation distance 0.006).
- (derived) The speed ratio to the all-in-VRAM baseline depends on the host link; with the 22 GB/s that
  large pinned copies reach on this host the transfer term would shrink by about a third.

### Not done or not verified (Qwen)

- The routing-equality gate used the engine's own logits; an end-to-end comparison of the engine's
  forward pass with the vLLM/BF16 forward (where near-ties can flip) was not run.
- Accuracy in the engine was only sanity-checked on 200 items (stock 0.890, ours 0.825, Skliar 0.780,
  Q4_K_M); the comparison uses the paper's vLLM accuracies.
- Peak VRAM includes the prompt-batch compute buffer (micro-batch = context length) for every arm.
- The first decode token after the prompt reloads all 8 slots in both arms; with `llama-bench` restoring
  the KV state between trials this happens in the discarded trial only.

## R1-7 -- Pod and cost so far

Pod `i65msin9gh64dw` running since 08:37 UTC, $0.69/h: about $1.8 at 11:10 UTC. The old pod
`bcnqix1ewwwjr3` is stopped and was not resumed (its host had no free GPU). The new pod is left running for
Gemma4 rather than stopped, because the morning showed a stopped pod may not get its GPU back; it will be
stopped at the end.

## R1-8 -- Gemma4-26B-IT: download, conversion, quantization (11:06-11:28 UTC)

`google/gemma-4-26B-A4B-it` (public, not gated; 49 GB). 30 layers, all with a routed MoE block (128
experts, top-8, expert width 704) next to a dense GELU MLP (width 2112), hidden size 2816, sliding-window
attention with head size 256 and global attention with head size 512 (the CUDA flash-attention kernels
have head size 512, so `-fa 1` as for Qwen; on and off measured the same speed in a smoke test).

**Went wrong, and how it was handled.**
1. The converter failed with `KeyError: 'global_head_dim'`. transformers 5.23 (installed for the tokenizer)
   rebuilds Gemma4's config without that key, while the engine's July converter reads it. The converter
   already falls back to the raw `config.json` when `AutoConfig` fails; `gpubench/released/
   convert_rawconfig.py` makes only that call fail (a first version also broke the tokenizer's
   `AutoConfig` call and was narrowed).
2. The checkpoint stores gate and up projections fused, and the converter writes
   `blk.N.ffn_gate_up_exps` [2816, 1408, 128]; the streamed slot path needs separate gate, up and down
   tensors. `gpubench/released/split_gate_up.py` rewrites the BF16 GGUF with each fused tensor split into
   `ffn_gate_exps` (first 704 rows, as `build_moe_ffn`'s merged path reads them) and `ffn_up_exps`; rows are
   independent, so quantizing the split file gives the same quantized rows. **All arms use the split file.**
   So that this cannot weaken the baseline, stock top-k is also timed on the fused file (R1-10).
3. The first fused-file quantization ran out of container disk (200 GB) and failed; rerun after deleting the
   safetensors.

Files: `gemma4-q4km.gguf` (split, 16,796,019,744 bytes) and `gemma4-q4km-fused.gguf` (16,796,017,664
bytes). Q4_K_M puts gate and up in Q4_K (1.06 MiB per expert each) and, because 704 is not a multiple of
256, the down projection falls back from Q6_K to Q8_0 (2.01 MiB per expert): **one load is about 4.3 MB**,
twice Qwen's, with 30 layers instead of 40. Per-expert down scales (`ffn_down_exps.scale`, [128]) stay in
VRAM and are gathered by expert id, so the slot remap does not touch them.

## R1-9 -- Gemma4 gates (11:28-11:57 UTC, `gates.sh ... 128 8 0.5 30 200`, outputs in `gpubench/results/released/gemma4/gates/`)

**Gate 1, routing equality (pass, all four configurations).** Same procedure as R1-3, Q4_K_M weights, the
references fed the engine's own logits.

| configuration | prompt layer-tokens equal | decode layer-tokens equal | loads equal |
|---|---|---|---|
| ours, all in VRAM | 6,870 / 6,870 | 17,130 / 17,130 | |
| ours, streamed | 6,870 / 6,870 | 18,150 / 18,150 | |
| Skliar, all in VRAM | 6,870 / 6,870 | 19,260 / 19,260 | 19,260 / 19,260 |
| Skliar, streamed | 6,870 / 6,870 | 15,000 / 15,000 | 15,000 / 15,000 |

**Gate 2, Skliar's loads on GSM8K against `results/ablations/skliar_c8_lam05_hist.json` (pass).** 200 items,
same sampling as for Qwen.

| loads in a layer-token | engine (1,610,100 layer-tokens) | saved (9,204,210) |
|---|---|---|
| 0 | 0.3265 | 0.3275 |
| 1 | 0.4756 | 0.4736 |
| 2 | 0.1442 | 0.1443 |
| 3 | 0.0426 | 0.0437 |
| 4 | 0.0094 | 0.0094 |
| 5 | 0.0015 | 0.0014 |
| 6 or more | 0.0001 | 0.0001 |
| **mean** | **0.9379** | **0.9389** |

Total variation distance 0.0021. The task brief's "about 20% of token-layers needing two or more" is
19.8% here and 19.9% in the saved histogram. Ours on the same items: 1,553,340 layer-tokens, mean 0.9986
(2,181 with no load).

Sanity check only: GSM8K flexible-extract on the 200 engine generations, stock 182/200 (0.910), ours
163/200 (0.815), Skliar 161/200 (0.805). The vLLM numbers on 1319 items are 0.878 / 0.788 / 0.807: on these
200 items the order of ours and Skliar is reversed, inside the n = 200 standard error of about 0.03.

**Gate 3, batch-1 perplexity, streamed against all in VRAM (pass, identical).**

| routing | all in VRAM | streamed | running estimates |
|---|---|---|---|
| ours | 73.2257 +/- 9.74357 | 73.2257 +/- 9.74357 | [1]100.8822 [2]62.2031 [3]92.5127 [4]73.2257 in both |
| Skliar | 77.8332 +/- 10.62048 | 77.8332 +/- 10.62048 | [1]55.6226 [2]31.6268 [3]66.9017 [4]77.8332 in both |
| stock top-k | 20.7269 +/- 2.39479 | | |

(The text is GSM8K questions and solutions without the chat template, and at batch 1 every token after a
chunk's first is under the constraint, which an instruction-tuned model handles badly; this is an identity
check, not a quality measurement.)

## R1-10 -- Gemma4 decode timing (engine tree `20c50f0c`, 11:58-12:41 UTC)

`gpubench/released/final_gemma.sh` (pod-side, kept in `gpubench/results/released/gemma4/logs/`): token files
as for Qwen (1k / 2k / 4k contexts of 961 / 1999 / 4007 prompt tokens), then the same nine arms and pass
structure as R1-5, once without and once with `TEMPORAL_SHEXP_FIRST=1` (which for Gemma4 places the dense
MLP between the router and the routed experts), then stock top-k on the checkpoint's fused gate/up file.
Rows: `gemma_decode_final.jsonl`, `gemma_decode_shexp.jsonl` (63 rows each, no failures),
`gemma_decode_fusedbase.jsonl`.

**The split file does not weaken the baseline.** Stock top-k on the fused file: 133.83 / 131.66 / 130.58
tok/s at 1k / 2k / 4k (three passes each), 133.26 / 132.17 / 131.08 with `TEMPORAL_SHEXP_FIRST=1`;
on the split file 132.86 / 131.52 / 130.51 and 133.40 / 132.50 / 130.78. The differences are below 0.8%
and go both ways.

### Without `TEMPORAL_SHEXP_FIRST`

### 1024 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 132.86 | 133.24 / 132.59 / 132.75 | 7.527 | 17704 | 1.000 |  |  |
| ours_vram | 127.56 | 127.99 / 127.51 / 127.18 | 7.839 | 17704 | 0.960 | 0.9987 | 0.001 / 0.999 / 0.000 / 0.000 |
| skliar_vram | 126.72 | 127.07 / 126.72 / 126.37 | 7.892 | 17704 | 0.954 | 0.7924 | 0.401 / 0.451 / 0.113 / 0.036 |
| ours_stream | 60.41 | 60.49 / 60.48 / 60.25 | 16.554 | 4750 | 0.455 | 0.9980 | 0.002 / 0.998 / 0.000 / 0.000 |
| skliar_stream | 66.23 | 66.19 / 66.29 / 66.21 | 15.099 | 5018 | 0.498 | 0.8017 | 0.395 / 0.452 / 0.117 / 0.036 |
| ours_nobytes | 118.94 | 118.72 / 118.93 / 119.17 | 8.407 | 4750 | 0.895 | 0.9945 | 0.005 / 0.995 / 0.000 / 0.000 |
| skliar_nobytes | 117.70 | 117.89 / 117.27 / 117.95 | 8.496 | 4750 | 0.886 | 0.5674 | 0.539 / 0.380 / 0.064 / 0.017 |
| ours_nocopy | 119.31 | 119.79 / 119.27 / 118.85 | 8.382 | 4750 | 0.898 | 0.9945 | 0.005 / 0.995 / 0.000 / 0.000 |
| skliar_nocopy | 118.30 | 118.18 / 118.02 / 118.71 | 8.453 | 4750 | 0.890 | 0.5674 | 0.539 / 0.380 / 0.064 / 0.017 |

### 2048 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 131.52 | 131.99 / 131.06 | 7.603 | 19030 | 1.000 |  |  |
| ours_vram | 126.53 | 126.63 / 126.43 | 7.903 | 19030 | 0.962 | 0.9992 | 0.001 / 0.999 / 0.000 / 0.000 |
| skliar_vram | 125.32 | 125.67 / 124.97 | 7.980 | 19030 | 0.953 | 1.0188 | 0.316 / 0.440 / 0.175 / 0.069 |
| ours_stream | 60.12 | 60.07 / 60.18 | 16.632 | 6160 | 0.457 | 0.9988 | 0.001 / 0.999 / 0.000 / 0.000 |
| skliar_stream | 57.71 | 57.79 / 57.64 | 17.327 | 6160 | 0.439 | 1.0450 | 0.308 / 0.444 / 0.171 / 0.077 |
| ours_nobytes | 116.91 | 117.50 / 116.32 | 8.553 | 6160 | 0.889 | 0.9977 | 0.002 / 0.998 / 0.000 / 0.000 |
| skliar_nobytes | 116.50 | 116.41 / 116.59 | 8.584 | 6160 | 0.886 | 0.5592 | 0.526 / 0.402 / 0.061 / 0.010 |
| ours_nocopy | 118.30 | 118.37 / 118.22 | 8.453 | 6160 | 0.899 | 0.9968 | 0.003 / 0.997 / 0.000 / 0.000 |
| skliar_nocopy | 116.94 | 116.84 / 117.04 | 8.551 | 6160 | 0.889 | 0.5705 | 0.523 / 0.400 / 0.064 / 0.013 |

### 4096 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 130.51 | 130.26 / 130.77 | 7.662 | 21672 | 1.000 |  |  |
| ours_vram | 124.90 | 124.25 / 125.54 | 8.007 | 21672 | 0.957 | 0.9980 | 0.002 / 0.998 / 0.000 / 0.000 |
| skliar_vram | 123.77 | 122.80 / 124.74 | 8.080 | 21672 | 0.948 | 0.7623 | 0.414 / 0.450 / 0.102 / 0.034 |
| ours_stream | 59.80 | 59.62 / 59.98 | 16.723 | 8976 | 0.458 | 0.9982 | 0.002 / 0.998 / 0.000 / 0.000 |
| skliar_stream | 66.56 | 66.50 / 66.61 | 15.024 | 8976 | 0.510 | 0.7755 | 0.408 / 0.450 / 0.107 / 0.035 |
| ours_nobytes | 116.36 | 116.47 / 116.25 | 8.594 | 8976 | 0.892 | 0.9953 | 0.005 / 0.995 / 0.000 / 0.000 |
| skliar_nobytes | 115.80 | 115.38 / 116.22 | 8.636 | 8976 | 0.887 | 0.5719 | 0.524 / 0.401 / 0.059 / 0.017 |
| ours_nocopy | 117.29 | 117.52 / 117.05 | 8.526 | 8976 | 0.899 | 0.9954 | 0.005 / 0.995 / 0.000 / 0.000 |
| skliar_nocopy | 116.84 | 116.71 / 116.97 | 8.559 | 8976 | 0.895 | 0.5640 | 0.529 / 0.397 / 0.059 / 0.015 |

### With `TEMPORAL_SHEXP_FIRST=1` (dense MLP under the copies)

### 1024 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 133.40 | 133.33 / 133.54 / 133.35 | 7.496 | 17692 | 1.000 |  |  |
| ours_vram | 127.68 | 127.43 / 127.70 / 127.91 | 7.832 | 17692 | 0.957 | 0.9987 | 0.001 / 0.999 / 0.000 / 0.000 |
| skliar_vram | 126.46 | 126.05 / 126.53 / 126.79 | 7.908 | 17692 | 0.948 | 0.7924 | 0.401 / 0.451 / 0.113 / 0.036 |
| ours_stream | 62.90 | 62.73 / 63.16 / 62.80 | 15.899 | 4738 | 0.471 | 0.9980 | 0.002 / 0.998 / 0.000 / 0.000 |
| skliar_stream | 68.04 | 68.03 / 68.02 / 68.07 | 14.697 | 4738 | 0.510 | 0.8017 | 0.395 / 0.452 / 0.117 / 0.036 |
| ours_nobytes | 118.64 | 118.67 / 119.04 / 118.22 | 8.429 | 4738 | 0.889 | 0.9945 | 0.005 / 0.995 / 0.000 / 0.000 |
| skliar_nobytes | 117.67 | 117.64 / 117.53 / 117.84 | 8.498 | 4738 | 0.882 | 0.5674 | 0.539 / 0.380 / 0.064 / 0.017 |
| ours_nocopy | 119.67 | 119.36 / 119.65 / 120.00 | 8.356 | 4738 | 0.897 | 0.9945 | 0.005 / 0.995 / 0.000 / 0.000 |
| skliar_nocopy | 118.79 | 118.85 / 118.63 / 118.88 | 8.418 | 4738 | 0.890 | 0.5674 | 0.539 / 0.380 / 0.064 / 0.017 |

### 2048 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 132.50 | 132.55 / 132.45 | 7.547 | 19008 | 1.000 |  |  |
| ours_vram | 126.69 | 126.60 / 126.78 | 7.893 | 19008 | 0.956 | 0.9992 | 0.001 / 0.999 / 0.000 / 0.000 |
| skliar_vram | 125.89 | 125.85 / 125.94 | 7.943 | 19008 | 0.950 | 1.0188 | 0.316 / 0.440 / 0.175 / 0.069 |
| ours_stream | 62.31 | 62.59 / 62.02 | 16.050 | 6138 | 0.470 | 0.9991 | 0.001 / 0.999 / 0.000 / 0.000 |
| skliar_stream | 59.00 | 59.26 / 58.74 | 16.948 | 6138 | 0.445 | 1.0450 | 0.308 / 0.444 / 0.171 / 0.077 |
| ours_nobytes | 117.41 | 117.85 / 116.97 | 8.517 | 6138 | 0.886 | 0.9972 | 0.003 / 0.997 / 0.000 / 0.000 |
| skliar_nobytes | 116.58 | 116.52 / 116.64 | 8.578 | 6138 | 0.880 | 0.5534 | 0.535 / 0.392 / 0.061 / 0.012 |
| ours_nocopy | 118.40 | 118.38 / 118.42 | 8.446 | 6138 | 0.894 | 0.9968 | 0.003 / 0.997 / 0.000 / 0.000 |
| skliar_nocopy | 117.16 | 117.43 / 116.89 | 8.535 | 6138 | 0.884 | 0.5829 | 0.511 / 0.411 / 0.066 / 0.013 |

### 4096 context

| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |
|---|---|---|---|---|---|---|---|
| base | 130.78 | 131.03 / 130.53 | 7.646 | 21628 | 1.000 |  |  |
| ours_vram | 125.33 | 125.51 / 125.15 | 7.979 | 21628 | 0.958 | 0.9980 | 0.002 / 0.998 / 0.000 / 0.000 |
| skliar_vram | 124.15 | 124.18 / 124.12 | 8.055 | 21628 | 0.949 | 0.7623 | 0.414 / 0.450 / 0.102 / 0.034 |
| ours_stream | 62.32 | 62.28 / 62.36 | 16.046 | 8932 | 0.477 | 0.9977 | 0.002 / 0.998 / 0.000 / 0.000 |
| skliar_stream | 68.16 | 67.90 / 68.41 | 14.672 | 8932 | 0.521 | 0.7742 | 0.404 / 0.457 / 0.106 / 0.033 |
| ours_nobytes | 116.54 | 116.41 / 116.66 | 8.581 | 8932 | 0.891 | 0.9958 | 0.004 / 0.996 / 0.000 / 0.000 |
| skliar_nobytes | 115.58 | 115.54 / 115.61 | 8.652 | 8932 | 0.884 | 0.5633 | 0.531 / 0.395 / 0.058 / 0.016 |
| ours_nocopy | 117.73 | 117.56 / 117.89 | 8.494 | 8932 | 0.900 | 0.9956 | 0.004 / 0.996 / 0.000 / 0.000 |
| skliar_nocopy | 116.78 | 116.84 / 116.71 | 8.563 | 8932 | 0.893 | 0.5674 | 0.532 / 0.390 / 0.061 / 0.017 |

**Where the time goes** (ms per token, pooled):

| step | ours 1k | Skliar 1k | ours 2k | Skliar 2k | ours 4k | Skliar 4k |
|---|---|---|---|---|---|---|
| routing op (all in VRAM) over stock | +0.313 | +0.365 | +0.300 | +0.376 | +0.345 | +0.417 |
| slot machinery, no copy kernels | +0.542 | +0.561 | +0.550 | +0.572 | +0.519 | +0.479 |
| copy kernels launched, nothing moved | +0.026 | +0.043 | +0.100 | +0.033 | +0.068 | +0.077 |
| bytes moved | +8.147 | +6.603 | +8.079 | +8.744 | +8.129 | +6.389 |
| cost per load (bytes moved / loads) | 272 us | 275 us | 270 us | 279 us | 271 us | 275 us |
| same, with the dense MLP under the copies | 250 us | 258 us | 251 us | 267 us | 249 us | 259 us |

(Without `TEMPORAL_SHEXP_FIRST` in the upper rows.) One load is 4.3 MB here, so about 16 GB/s; the per-load
cost is the same for both arms, so **the streamed time is set by the number of loads**. Ours is at 1.00 on
every text. Skliar's depends on the text it replays: 0.80, 1.05 and 0.78 loads per layer-token on the 1k,
2k and 4k continuations, against 0.938 on 200 GSM8K generations (gate 2). Skliar's arm is therefore faster
than ours at 1k and 4k (68.0 against 62.9 and 68.2 against 62.3 tok/s with the dense MLP under the copies)
and slower at 2k (59.0 against 62.3). One 100-token replay per context is too small a sample of a
text-dependent load rate, which is why R1-11 times the streamed arms on several conversations.
