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
