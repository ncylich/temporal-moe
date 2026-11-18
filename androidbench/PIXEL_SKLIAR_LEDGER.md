# Skliar's cache-conditional routing on the Pixel 10a: ledger

Question: `gpubench/PRETRAINED_LEDGER.md` (P1-9) found no burst penalty for Skliar on an RTX A6000, where
loading an expert costs microseconds. The phone reads experts from flash, so this ledger times Skliar there,
on the plain 1.26 B MoE, and compares it with the phone's existing temporal rows.

Entries are in the order things happened. Statements are tagged **measured** or **derived**.

Headline (S1-5): Skliar decodes at **32.20 tok/s** (31.05 ms per token) with 18 experts per layer in RAM and
the pool on flash, at 1.029 loads per decoded token-layer. The trained model's streamed row is 32.91 tok/s in
`TRAINED_MODEL_LEDGER.md` and 32.60 tok/s re-run on this build. Skliar is 1.2 % behind the re-run row and
inside the spread of its own texts (30.03 to 33.23).

On the 11 B random-weight model with Skliar's load pattern imposed (S1-6): 13.21 tok/s against 14.30 with
exactly one load per layer per token, 7.6 % slower.

## S1-0 -- Protocol (2026-11-14)

Fixed by the task and not changed:

- **Model.** `moe_fine_g3_1e19`, the plain MoE (14 blocks, layer 0 dense, 13 MoE layers, 192 experts, 18
  active), with its own gates: `moe_fine_1e19-full-q4_0.gguf` (converted with `--no-weights-norm`),
  744,281,312 bytes, sha256 `bee4834d9d3d...3ee910`, same hash on the Mac and on the phone.
- **Skliar.** Cache C = 18 = k, J = 1, lambda = 0.7207 (P1-10). Reference `analysis/residency/cache_bias.py`,
  not edited.
- **Serving protocol for a model without temporal training.** The prompt is routed by the stock top-k while
  the cache warms; the rule decides from the first decoded token; loads are counted on decoded tokens.

Scope, changed by the user during the work: only the Skliar arm is timed. Our rule on the plain MoE and a new
stock all-resident baseline were dropped. Skliar is compared with the rows already in
`TRAINED_MODEL_LEDGER.md` (step 6, context 1,024), which are:

| prior row (trained model, 2026-11-10, engine 759d43c5) | tok/s (sd over 3 reps) | VmHWM MB |
|---|---|---|
| (a) deploy: R = 18, pool on flash, trained rule | 32.91 (0.34) | 762 |
| (b) resident_trained: all experts in RAM, trained rule | 48.56 (0.05) | 887 |
| (c) resident_topk: all experts in RAM, stock top-k | 48.11 (0.07) | 887 |

Phone protocol, the one of that ledger's step 6 (`run_ctx_designpoint.py --model trained_fine --noroot`):

- Pixel 10a `5C111JEA320125`, Android 17 (`CP2A.260705.006`), unrooted, governor not pinned, on AC.
- `llama-bench -t 4 -p 0 -n 128 -d 1024 -r 3 -mmp 0 -ot "_exps=CPU"`, Q4_0. Three repetitions, none
  discarded; the context is filled once and its KV state restored for repetitions 2 and 3.
- `LLAMA_TEMPORAL_REPACK=1` with the repacked side-file (built on the phone as in that ledger's step 5),
  `LLAMA_TEMPORAL_ODIRECT=1`, `LLAMA_TEMPORAL_MADV_FREE=1`, `LLAMA_TEMPORAL_R=18`, `LLAMA_TEMPORAL_SPLIT=2`,
  `LLAMA_TEMPORAL_FETCH_THREADS=6`, `LLAMA_TEMPORAL_SPIN_US=5000`, `LLAMA_TEMPORAL_FETCHPROF=1`,
  `LLAMA_TEMPORAL_TWOPASS=1`.
- Void if policy 7's `scaling_max_freq` drops below 3.105 GHz during the run, swap above 100 MB, no decode
  result, no evictions, or no router line. 60 s rest after every run.

Deviations from that protocol:

1. **Real text instead of llama-bench's random tokens.** Skliar's loads depend on the text and lambda was
   matched on real text, so random tokens would not measure it. Rows 601 to 605 of the held-out test dump:
   1,024 tokens as the prompt, the row's own next 128 tokens decoded (`gpubench/pretrained/tokens_base.py`
   without `--url`, `--n-gen 128`; files in `results/pixel_skliar/tokens/`). The trained rule loads one
   expert per layer per token on any input, so its prior rows do not depend on this.
2. **Context 1,024 only.** The prior ledger also has 0, 2,048 and 4,096. The text files give one context.
3. **Cool gate.** The prior run waited for `soc_therm` <= 33 C and every `scaling_max_freq` at rating.
   `soc_therm` read a constant 27.854 C through this whole session, hot or cold, so that test was always
   true. Added: Thermal Status 0 and `VIRTUAL-SKIN` <= 34.0 C. Applied to the re-run trained row too.
4. **Engine.** The prior engine (759d43c5 = base 43bec5f3 plus patches 0001 and 0002) plus patch 0003
   (S1-1): tree `b1f58983`, local commit d9f0f2d1, not pushed to the fork. The patch touches code the trained
   rule runs through (S1-1), so one trained row was re-timed on the new build (S1-5).
5. **Prefill.** The plain MoE's prompt is routed by the stock top-k (the fixed serving protocol); the
   trained model's prompt is routed by its rule. This changes the prefill's memory, not the decode loop.

## S1-1 -- Engine: Skliar on the CPU path (patch 0003)

`scripts/serving_convert/engine_patch/0003-*.patch`, README section there. Before it, Skliar existed only as
CUDA kernels.

- **Rule.** `LLAMA_TEMPORAL_ROUTE=skliar` in `ggml_temporal_route`: per layer an LRU cache of C experts, a
  running mean of max - min logit (per layer and per process, as in the reference), z' = z + lambda x mean
  on cached experts, top-K of z', the top-1 of z forced in, loads = selected and not cached. A batch of more
  than one token is a prompt batch: stock top-k, the cache is warmed. A single-token batch is one rule step.
  With C = K the cache is the previous selection, and the top-K is reached by exchanging the best outsider
  with the weakest member until it stops winning (`TEMPORAL_CB_FULLSEL=1` runs the plain selection; both
  give identical selections on the five texts, host check). `LLAMA_TEMPORAL_ROUTE=decode` (our rule at
  decode time) is in the same patch; it was gated but, after the scope change, not timed.
- **Several loads per token.** The window op evicts every expert that left the window and queues all missing
  ones at once, so the six workers read them in parallel with the same direct reads. Experts that entered on
  this token sit last in the window. The first FFN sub-pass does not wait for them up front; the repacked
  kernel computes that layer's residents first and waits for each entered expert when it reaches it.
- **What changed on the trained rule's path.** Three things, none changing what it computes: the two-pass
  graph is built by a loop (`LLAMA_TEMPORAL_TAIL`, default 1 = the old graph); the wait op derives its slot
  range from that setting (same range at 1); the repacked `mul_mat_id` asks for a compute order once per op
  (always "none" under the trained rule). S1-5 re-times one trained row.
- `LLAMA_TEMPORAL_TAIL=<n>` (n single-expert sub-passes) is in the patch and passed the gates at 2 and 4 on
  text 1. It was not timed under the cool gate; the only readings are throttled exploration runs.
- llama-bench: `TEMPORAL_BENCH_CTX/GEN` (real-text replay, ported from the CUDA patch 0006),
  `TEMPORAL_BENCH_TOKLOG` (time, loads and fetch wait per decoded token), `TEMPORAL_BENCH_LOGIT_HASH`,
  `TEMPORAL_BENCH_REPREFILL` (not used in the timed rows).
- Export check: `git am` of 0001 to 0003 on a fresh clone at 43bec5f3 gives tree `b1f58983`, the tree the
  phone binary was built from (binary sha256 `dfba2337...f7ace4`).

## S1-2 -- Gates, first pass (engine 1a69eddf, superseded by S1-4)

- **Host, routing.** Mac, CPU build, Q4_0, the five texts: every decoded token-layer equals
  `cache_bias.py` (`gpubench/released/route_check.py`), loads equal, with the exchange loop and with the plain
  selection; identical histograms from both.
- **Flash against RAM by perplexity: abandoned.** `pixel_skliar_gate_ppl.py` (batch-1 llama-perplexity).
  The all-in-RAM arm gave 40.8885; the streamed arm was stopped after about ten minutes. Batch-1 perplexity
  runs at a few tokens per second on the phone and heated it, and with `-b 1` every token is a rule step,
  which is not the serving protocol. Replaced by per-token logit hashes on the timing texts.
- **Phone, texts 1 to 5** (`run_pixel_skliar.py gate`): routing and flash-equals-RAM passed for both rules,
  except three Skliar token-layers on text 5. Diagnosed on a fresh dump: experts 94 and 126 of layer 7 have
  the same logit (-7.267578125) and are both cached, the engine evicts one and `torch.topk` the other, and
  the two caches then differ for two more tokens. Loads are equal throughout.

## S1-3 -- Fault found before timing: 3.8 s stalls under thermal throttling

The gate runs heated the phone (Thermal Status 1, skin 39 C, big-core cap 1.4 to 2.5 GHz instead of 3.1).
In that state Skliar's streamed arm ran at 0.3 to 2 tok/s where our rule and the trained deploy arm ran at
24 to 31: runs of consecutive tokens took 3.77 s each (within 10 ms of each other), whatever their loads.

- Reproduced by alternating arms on the hot phone: trained deploy 30.9 and 29.9 tok/s with no token above
  45 ms; Skliar 0.51 and 0.95 tok/s with 52 and 28 tokens above 200 ms.
- Counters added to a diagnostic build showed the stalled time was not in any fetch wait (all threads), nor
  in the window op or the wait ops: 279 s of stalls in one run against 18 s of waits.
- A longer spin before sleeping (`LLAMA_TEMPORAL_SPIN_US=20000`, no thread slept) did not help (96 stalled
  tokens). `LLAMA_TEMPORAL_TAIL=8` removed them (0 stalled tokens, twice). With 8 single-expert sub-passes
  only thread 0 ever waits for an expert; with 1, all four compute threads reach experts still in flight and
  each took the pool mutex.
- Fix (in patch 0003): under the decode-time rules, compute threads other than 0 poll the expert's state and
  never take the mutex or sleep. Five hot runs after it: 16.8 to 28.0 tok/s, 2 stalled tokens in total.
- **Not closed.** Later hot runs on the fixed build still stalled in 3 of about 18 streamed runs, and an arm with
  every expert in RAM (nothing to wait for) showed the same pattern for the first three or four tokens after
  the prefill (5.1 s each) in 2 of 5 runs. So a second cause exists that is not the fetch path. It was seen
  only with the big-core cap below rating. Every timed row below has the cap at rating for the whole run and
  no token above 100 ms. The mechanism (my reading: a compute thread placed on a throttled core after
  blocking) was not established.

This is why the cool gate was tightened (S1-0, deviation 3): a run started at skin 36.8 C with all caps at
rating was throttled mid-run and void.

## S1-4 -- Gates on the final engine (d9f0f2d1), in the timing configuration

`run_pixel_skliar.py gate --texts 1 2 3 4 5 --reps 3 --env TEMPORAL_BENCH_REPREFILL=0`, i.e. the timed
command with the routing dump and logit hashes on. Raw results `results/pixel_skliar/gates/gate_g2.jsonl`.

**(a) Routing equals `cache_bias.py`** (the engine's own router logits replayed through the unmodified
reference). Decoded token-layers per text: 3 repetitions x 128 tokens x 13 layers = 4,992.

| text | per-step check: equal / exact tie / real difference | whole-sequence check (`route_check.py`): equal of 5,005 | loads equal | prompt token-layers equal to `torch.topk` | exact ties at the cut in the prompt |
|---|---|---|---|---|---|
| 1 | 4,991 / 1 / 0 | 4,992 | 5,002 | 13,302 of 13,312 | 19 |
| 2 | 4,992 / 0 / 0 | 5,005 | 5,005 | 13,302 of 13,312 | 17 |
| 3 | 4,992 / 0 / 0 | 5,005 | 5,005 | 13,299 of 13,312 | 28 |
| 4 | 4,991 / 1 / 0 | 4,898 | 5,002 | 13,302 of 13,312 | 18 |
| 5 | 4,991 / 1 / 0 | 5,002 | 5,005 | 13,303 of 13,312 | 22 |

- The whole-sequence check lets the reference keep its own cache, so one exact tie makes later steps differ
  until the caches meet again (107 token-layers on text 4). `pixel_skliar_stepcheck.py` gives the reference's
  `select` the engine's cache at every step: 24,957 of 24,960 steps equal, 3 exact ties (the experts the two
  sides disagree on have equal biased logits), 0 real differences, loads equal on every step. The per-step
  check ran on separate dumps of the same command (the gate deletes its 38 MB dumps).
- Prompt token-layers that differ from `torch.topk` are all exact ties at the 18th logit; the engine's
  selection is a valid top-18 on every prompt token-layer (0 invalid).

**(b) Pool on flash equals every expert in RAM.** Same command with `LLAMA_TEMPORAL_R=192` (0 fetches, 0
evictions) against R = 18 (21,315 to 24,114 evictions per run), and R = 18 a second time. A 64-bit hash of
the logit vector of every decoded token (385 per run: one warm-up and 3 x 128) is identical across the
three runs on all five texts. Load counts are identical too.

**(c) Loads histogram.** Pooled over the five texts, 24,960 decoded token-layers:

| | mean | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|---|
| Pixel, this run | 1.029 | 20.5 % | 63.2 % | 11.3 % | 3.4 % | 1.5 % |
| A6000 (P1-9) | 1.059 | 19.4 % | 63.2 % | 11.8 % | 3.9 % | 1.7 % |

Per text on the phone: 1.014, 0.992, 1.002, 1.159, 0.976. The A6000 rows used 100 decoded tokens and four
repetitions; on the same 100 tokens and four repetitions the host CPU build gives 1.041, 1.022, 1.063,
1.150, 1.002 (mean 1.056) against the A6000's 1.051, 1.014, 1.073, 1.159, 0.999 (mean 1.059)
(`results/pixel_skliar/host/`). The difference to 1.029 is the 28 extra tokens per repetition.

## S1-5 -- Timing (18:53 to 19:03, `results/pixel_skliar/run_timing.sh`, rows tagged `t2`)

Six runs, none void: big-core cap at 3.105 GHz throughout each, no swap, skin 32.0 to 33.4 C at start and
at most 34.6 C at the end, slowest token after the first 93 ms.

| arm | model | tok/s (sd over 3 reps) | ms per token | loads per decoded token-layer | peak RSS MB |
|---|---|---|---|---|---|
| trained deploy, prior ledger row (a) | trained | 32.91 (0.34) | 30.39 | 1.000 | 762 |
| trained deploy, re-run on this build, same command | trained | 32.60 (0.17) | 30.68 | 1.000 | 762 |
| **Skliar, R = C = 18, pool on flash, mean of 5 texts** | plain MoE | **32.20** [30.03, 33.23] | **31.05** | **1.029** | 889 |
| resident, trained rule, prior ledger row (b) | trained | 48.56 (0.05) | 20.59 | 1.000 | 887 |
| resident, stock top-k, prior ledger row (c) | trained | 48.11 (0.07) | 20.79 | -- | 887 |

Skliar per text (tok/s, the three repetitions, loads): text 1 32.76 (32.0, 33.6, 32.7) 1.014; text 2 32.98
(32.1, 33.7, 33.1) 0.992; text 3 30.03 (29.2, 30.5, 30.5) 1.002; text 4 32.01 (30.9, 32.7, 32.4) 1.159;
text 5 33.23 (32.6, 33.6, 33.5) 0.976. Mean sd over repetitions 0.78.

Loads distribution of the timed rows: 0: 20.5 %, 1: 63.2 %, 2: 11.3 %, 3: 3.4 %, 4+: 1.5 % (largest 8).

Time per decoded token by how bursty the token was (engine token log, tokens after the first of each
repetition, 1,905 tokens):

| largest number of experts one layer loaded on the token | tokens | share | ms per token | loads on the token, 13 layers | ms the main thread was blocked on fetches |
|---|---|---|---|---|---|
| 1 | 552 | 29.0 % | 28.45 | 9.3 | 4.49 |
| 2 | 708 | 37.2 % | 30.24 | 12.3 | 5.47 |
| 3 | 373 | 19.6 % | 31.79 | 15.6 | 6.49 |
| 4 | 171 | 9.0 % | 33.55 | 19.8 | 7.91 |
| 5 or more | 101 | 5.3 % | 36.21 | 23.2 | 9.46 |

Least squares over those tokens: ms = 24.29 + 0.477 x (loads on the token), r^2 0.32. With the largest
per-layer count as a second term: 24.50 + 0.361 x loads + 0.589 x largest. The trained deploy re-run: 30.37
ms per token after the first, every token at 13 loads. First token after the prompt: 90 ms for Skliar, 69 ms
for the trained deploy arm (the window op evicts what the prefill left).

Statements:

- **Measured.** Skliar streams at 32.20 tok/s on the phone, 1.2 % below the trained deploy arm on the same
  build (32.60) and 2.2 % below the prior ledger's row (32.91). Text-to-text spread (30.0 to 33.2) is larger
  than that gap. A later run of text 3 for the per-step routing check gave 32.7 (dump on, no cool
  gate, not a timed row), so the 30.03 is not a property of that text.
- **Measured.** The re-run trained row is 0.9 % below the prior row with the same evictions (19,845) and the
  same peak RSS (762 MB). The patch does not change the trained rule's speed beyond run-to-run noise (sd
  0.17 and 0.34).
- **Measured.** Bursts cost time in proportion to what they load. A token whose worst layer loaded 5 or
  more experts takes 36.2 ms against 28.5 ms for one whose layers loaded at most one each, and it loads 23.2
  experts against 9.3.
- **Derived.** About 0.48 ms per expert load, and the trained rule's own tokens sit on the same line
  (24.29 + 0.477 x 13 = 30.5 ms against 30.37 measured). Skliar's mean matches because 20.5 % of its
  token-layers load nothing, which pays for the 16.2 % that load two or more. At equal loads per token the
  two rules cost about the same on this phone; the burst shows up as per-token jitter (p95 38.5 ms against
  35.6 ms, slowest 93 ms against 51 ms), not as a lower mean.
- **Derived.** Against all experts in RAM (prior rows b and c, about 48.3 tok/s), Skliar keeps 0.67 of the
  speed, the trained deploy arm 0.68.
- **Measured.** Peak RSS is 889 MB for Skliar against 762 MB for the trained deploy arm. The peak is set
  during the prefill: the plain MoE's prompt is routed by the stock top-k, which touches more distinct
  experts than the trained rule's window, and the batched path does not evict until decoding starts. Decode
  -phase RSS was not sampled for Skliar.

## S1-6 -- The 11 B random-weight model with Skliar-shaped loads (20:23 to 20:30)

Asked afterwards out of curiosity. The phone's 11 B model (`qwen3moe-rand-fine-Q4pure.gguf`, 45 MoE layers,
192 experts, 18 active, 5.9 GB, the model of `CTX_DESIGNPOINT_2026-08-24.md`) has random weights. Its router
decides nothing meaningful, so Skliar's rule cannot be run on it, and its earlier temporal rows used an imposed
policy (one random swap per layer per token). This entry does the same for Skliar: `LLAMA_TEMPORAL_ROUTE=replay`
(added to patch 0003, engine tree `536002e3`) draws each layer-token's load count from the histogram Skliar
produced on the 1.26 B model in S1-5 and swaps that many random experts. The comparison arm draws from "0,1",
exactly one load, on the same code path and the same binary.

Same protocol as S1-0 (context 1,024, 128 tokens, 3 repetitions, cool gate), llama-bench's random tokens,
arms alternated, four runs, none void (big-core cap at rating, no swap, no token above 141 ms).

| arm | tok/s, the two runs (sd over 3 reps) | mean | ms per token | loads per decoded token-layer |
|---|---|---|---|---|
| one load per layer per token | 14.18 (0.66), 14.42 (0.44) | 14.30 | 69.9 | 1.000 |
| Skliar-shaped loads | 13.03 (0.22), 13.40 (0.40) | 13.21 | 75.7 | 1.028 |

Per token after the first of each repetition: 68.81 ms against 73.14 ms; p95 83.1 against 91.6 ms; first token
after the prompt 143 against 269 ms. Fit over the Skliar-shaped tokens: ms = 43.75 + 0.636 x loads (r^2 0.13).

- **Measured.** On the 11 B geometry Skliar-shaped loads are 7.6 % slower than one load per token, for 2.8 %
  more loads.
- **Derived.** At the one-load arm's 45 loads per token the fit gives 72.4 ms against 68.8 ms measured for
  the one-load arm: at equal loads the bursty pattern is about 5 % slower per token, and the extra loads
  and the slower first token after the prompt account for the rest of the 7.6 %. On the
  1.26 B model the same comparison showed no gap (S1-5). An expert here is about 0.66 MB over its three
  tensors against 0.26 MB there, and each layer's compute hides less of a multi-expert read.
- **Limits.** Load counts are drawn independently per layer and token, so the way real Skliar bursts line up
  across the layers of one token is not reproduced. The experts swapped are random. The histogram comes from
  the 1.26 B model; a trained 11 B model's Skliar histogram is unknown. Two runs per arm.
- Not comparable with the 2026-08-24 row for this model (17.80 tok/s at context 1,024): that run was rooted
  with the governor pinned, on an older build and the earlier random-swap policy. Peak RSS (3.7 to 4.4 GB,
  rising run to run) is set by the prefill of the 5.9 GB model and was not controlled.

## S1-7 -- Where the shared expert runs, and moving it into the load's overlap window (2026-11-17)

Asked by the review session: the paper's bandwidth condition counts the shared expert in the computation that
hides a load, and the CUDA path has `TEMPORAL_SHEXP_FIRST` for that. The CPU patches had no equivalent.

**Where it sat.** After the wait, hiding nothing. In `src/models/glm4-moe.cpp` (the trained model and the
plain MoE are both `glm4moe`) the layer calls `build_moe_ffn` (line 237 before this change), which builds the
route op, the window op (`ffn_moe_window`, queues the reads), the resident sub-pass, the wait and the
new-expert sub-pass (`src/llama-graph.cpp`, `one_pass` and the `tm_tail` loop) and returns their sum. Only
then is the shared expert built (`build_ffn` on `ffn_*_shexp`, line 251) and added. The CPU backend runs
nodes in graph order. Node order of layer 1 on a one-token graph, printed from the built graph on the host:

| switch | route | window | shared expert (gate, up, down) | routed sum `ffn_moe_out` | `ffn_out` |
|---|---|---|---|---|---|
| off | 61 | 62 | 124, 125, 127 | 123 | 128 |
| on | 61 | 62 | 69, 70, 72 | 127 | 128 |

**Change (patch 0004, engine commit `5f992263`, tree `cacc843b`).** `LLAMA_TEMPORAL_SHEXP_FIRST=1`, off by
default. `llm_graph_context::tm_after_router` is a node that `build_moe_ffn` expands right after the window op
and the gate weights, before the resident sub-pass; with the switch on, `glm4-moe.cpp` builds the shared
expert first and hands it over as that node. Same ops, same add (`routed + shared`), as CUDA patch 0007. It
is keyed on the architecture, not on the routing mode, so the trained rule, Skliar, our decode-time rule and
the stock path all get it. The shared expert is 1,152 wide against 576 for a routed expert, so it adds about
two experts' worth of compute to the 17 residents that already ran under the read.

**Skliar's bias.** S1-5 ran lambda 0.7207 at 1.029 loads per decoded token-layer. Lambda was moved until the
loads on the timed texts and protocol (five texts, 3 x 128 decoded tokens, 24,960 token-layers) were within
0.5 % of 1.000, counted on the phone with every expert in RAM (routing does not depend on where the experts
are, gate below): 0.7275 gives 1.0062, 0.7294 gives 1.0050, 0.7370 gives 0.9815, **0.7310 gives 1.0022**
(25,014 loads). The timed streamed runs reproduce 25,014 exactly, switch off and on. Per text the loads still
range from 0.957 to 1.137.

**Gates, switch on, timing binary, lambda 0.7310** (`results/pixel_skliar/run_gates_shexp.sh`, tags `g3on`,
`g3off`; these runs were made hot and are not timing rows):

| check | Skliar on the plain MoE | trained model |
|---|---|---|
| pool on flash equals every expert in RAM, 64-bit hash of each decoded token's logits, 385 tokens x 5 texts | equal on all | equal on all |
| switch on equals switch off, pool on flash, same hashes | equal on all | equal on all |
| a second streamed run equals the first | equal on all | not run |
| routing equals `cache_bias.py` | texts 1, 3, 4: 5,005 of 5,005 decoded steps by whole-sequence replay; texts 2 and 5: per-step check 4,991 of 4,992 each, the other one an exact logit tie, loads equal | not applicable (trained rule, unchanged path) |

The perplexity form of the flash-equals-RAM gate was not run (the earlier attempt in this ledger was too slow and too hot on
this phone); bit-identical logits on every decoded token imply it.

**Timing** (21:39 to 22:40, `results/pixel_skliar/run_timing_shexp.sh`, rows tagged `sx0` = off and `sx1` =
on). Protocol of S1-5: context 1,024, the five texts, 128 tokens, 3 repetitions, cool gate and 60 s rest
before every run, one binary. Per text the three arms run in turn, each off and on back to back, and the
off/on order flips on every other text. 30 runs, none void: big-core cap at rating in all, no swap, skin at
most 34.0 C at the start, slowest token 61 ms. The trained arm here is the trained model on the same five
real texts as Skliar (S1-5 timed it on llama-bench's random tokens).

| arm | switch | tok/s, mean of 5 texts [min, max] | ms per token | loads per decoded token-layer | p95 token ms | slowest token ms | ms per token blocked on fetches |
|---|---|---|---|---|---|---|---|
| trained, pool on flash | off | 32.36 [31.64, 32.75] | 30.90 | 1.000 | 36.8 | 56.8 | 6.64 |
| trained, pool on flash | on | 33.67 [33.56, 33.90] | 29.70 | 1.000 | 35.3 | 50.4 | 5.65 |
| Skliar, pool on flash | off | 32.54 [32.04, 32.89] | 30.73 | 1.002 | 37.6 | 61.1 | 6.15 |
| Skliar, pool on flash | on | 33.30 [32.59, 34.12] | 30.03 | 1.002 | 36.7 | 61.3 | 5.64 |
| plain MoE, stock top-k, all in RAM | off | 48.09 [48.02, 48.18] | 20.79 | | 20.9 | 22.5 | 0 |
| plain MoE, stock top-k, all in RAM | on | 48.10 [48.01, 48.17] | 20.79 | | 20.9 | 23.4 | 0 |

p95, slowest and blocked time are over tokens after the first of each repetition. "Blocked on fetches" is the
time compute thread 0 spent waiting for an expert that was not yet in RAM.

| | text 1 | text 2 | text 3 | text 4 | text 5 | mean of paired differences [min, max] |
|---|---|---|---|---|---|---|
| trained, on against off | +3.9 % | +3.9 % | +2.9 % | +6.2 % | +3.3 % | **+4.05 % [+2.9, +6.2]** |
| Skliar, on against off | +0.1 % | +1.5 % | +3.8 % | +1.7 % | +4.6 % | **+2.32 % [+0.1, +4.6]** |
| all in RAM, on against off | -0.0 % | +0.3 % | -0.1 % | -0.2 % | +0.1 % | +0.02 % [-0.2, +0.3] |
| trained against Skliar, switch off | -0.8 % | -1.2 % | +1.0 % | -1.3 % | -0.5 % | **-0.56 % [-1.3, +1.0]** |
| trained against Skliar, switch on | +2.9 % | +1.2 % | +0.1 % | +3.1 % | -1.6 % | **+1.14 % [-1.6, +3.1]** |

Skliar's per-token line (token time against the token's loads over 13 layers, tokens after the first of each
repetition): off ms = 24.06 + 0.491 x loads, on ms = 23.41 + 0.485 x loads. The trained rule loads 13 experts
on every token, so it has no line; at 13 loads Skliar's lines give 30.44 ms off and 29.72 ms on, against 30.68
and 29.45 ms measured for the trained model.

Skliar's tokens by the largest number of experts one layer loaded (ms per token, off / on): 1: 28.41 / 27.91;
2: 30.28 / 29.15; 3: 31.72 / 31.14; 4: 33.52 / 33.03; 5+: 35.35 / 35.19.

- **Measured.** The switch makes the streamed trained model 4.1 % faster (1.2 ms per token) and streamed
  Skliar 2.3 % faster (0.7 ms). With everything in RAM it changes nothing, so the gain is overlap and not a
  cheaper graph. The time blocked on fetches falls by 0.99 ms per token for the trained model and 0.51 ms for
  Skliar.
- **Measured.** The cost per load did not fall: 0.491 to 0.485 ms. The intercept fell by 0.65 ms.
- **Measured.** Trained against Skliar at matched loads stays inside the spread over texts with either
  setting: -0.6 % off, +1.1 % on, and the sign changes between texts in both. The switch moved the comparison
  by 1.7 points toward the trained model. Five texts do not separate the two arms.
- **Derived.** Why the intercept and not the slope: the shared expert buys a fixed slice of time per layer
  that has something to wait for, not per expert loaded, and the number of such layers varies little from
  token to token. Why the trained model gains more: it waits in all 13 layers of every token, Skliar in about
  80 % of them (a fifth of its token-layers load nothing), and where a Skliar layer loads several experts the
  wait is longer than the shared expert. The per-class table agrees: the gain is largest on tokens whose
  worst layer loads one or two experts and near zero at five or more.
- **Measured.** The load is still not hidden on this phone at this size. With the switch on, thread 0 is
  blocked 5.6 ms per token (0.43 ms per layer), and the streamed arms take 29.7 to 30.0 ms per token against
  20.8 ms with everything in RAM.
- Sanity against S1-5: Skliar off at 1.002 loads 32.54 tok/s (S1-5 at 1.029 loads: 32.20); trained off on
  real text 32.36 (S1-5 on random tokens: 32.60; prior ledger 32.91); all in RAM 48.09 (prior ledger 48.11).

What went wrong and how it was caught:
- The host shortcut for matching lambda failed. The host build with every expert in RAM and no two-pass
  graph reported 0.089 loads per token-layer at lambda 0.7207, where the phone and the earlier host file
  (`results/pixel_skliar/host/`) give about 1.0. It was caught by comparing with the phone's own log line for
  the same text before any lambda was chosen. The cause was not found; lambda was matched on the phone instead
  (rows tagged `lam*-quick`), and the timed runs confirm the count.
- The whole-sequence routing replay reported FAIL on texts 2 and 5, as it did in S1-4 for other texts. The
  per-step check (`results/pixel_skliar/stepcheck_v2_g3on.json`, `_v5_`) shows one exact logit tie each and
  no real difference.
- The peak RSS of the trained model on real text is 848 MB, against 762 MB for the same model on random
  tokens in S1-5. Not investigated; it is the same with the switch off and on.

## S1-8 -- Why the gap to Skliar is small: per-layer trace (2026-11-17, 22:53)

One traced 48-token run per arm and switch setting (`LLAMA_TEMPORAL_TRACE=1`, text 1, no cool gate, tracing
overhead included, so these are structure and not benchmark rows; tags `tr0`, `tr1`). 572 layer-tokens per
arm after dropping the first five tokens. Page: `results/pixel_skliar/load_timeline.html` (built by
`make_load_timeline.py`; numbers in `load_timeline_stats.json`). The traces themselves (23 MB each) are not
committed.

| per layer, shared expert first | trained | Skliar |
|---|---|---|
| thread 0 computing the 17 residents | 0.21 ms | 0.22 ms |
| one expert's read, queued to last byte | 0.81 ms | 0.93 ms |
| stall, mean over all layers | 0.42 ms | 0.47 ms |
| stall by loads in the layer (share of layers) | 1: 0.42 (100 %) | 0: 0 (25 %), 1: 0.49 (60 %), 2: 0.89 (9 %), 3: 1.13 (4 %), 4+: 2.04 (2 %) |

- **Measured.** One load does not fit under the layer's routed compute: the read takes about four times as
  long, so every layer that loads stalls, with one load or several.
- **Measured.** The stall is roughly proportional to the loads in the layer (1.8 times one load for two, 2.3
  for three), and a quarter of Skliar's layers load nothing. At equal average loads the average stall is then
  about equal: 0.47 against 0.42 ms per layer, 0.65 ms per token (0.45 ms with the shared expert after the wait).
- **Measured, not explained.** That remainder is not the bursts: a single load stalls Skliar's run 0.49 ms
  against 0.42 ms, because its individual reads take longer (475 against 423 us). Different model files.
- **Derived.** A cap of one load per layer buys throughput only where one load hides under the layer's
  compute and a second does not. On this phone at 1.26 B that condition is not met. The rest of the layer
  (1.3 ms of attention and norms) comes before the router and cannot hide the read.

## S1-9 -- What sets the last byte of a load, and two attempts to move it (2026-11-17, 23:10 to 23:35)

Exploration only: quick runs on text 1, 128 tokens, one repetition, no cool gate (Thermal Status 0 and the
big-core cap at rating throughout; skin 31.7 to 35.1 C), tags `fx-*`, `fy-*`, `fz-*`, and three more traced
runs. Not timing rows.

From the S1-8 trace of the trained model (572 loads, six reads each, `LLAMA_TEMPORAL_SPLIT=2`): every read
takes about the same time, 409 to 441 us by start order, but the reads start one after another, at 63, 93,
132, 200, 275 and 338 us after they are queued. The last byte lands at 812 us; the last read to finish is one
of the last two to start in 429 of 572 loads. The workers sleep on a condition variable between layers (the
polling loop after a job lasts 0.2 to 0.4 ms, the gap to the next layer's reads is 1.3 ms), and wake one by
one. Had all six started with the first, the last byte would land at 639 us.

| setting (trained model, pool on flash, shared expert first) | tok/s, quick runs | later tokens, ms | blocked on fetches, ms per token |
|---|---|---|---|
| `SPLIT=2`, six reads per expert (the protocol so far) | 33.73, 33.30, 34.37 (a first cold run: 29.61) | 29.11 | 5.23 |
| `SPLIT=1`, three reads per expert | 35.66, 35.00, 35.44; on the test binary 34.98, 35.50 | 27.85 | 4.19 |
| `SPLIT=2`, `SPINNERS=4` / `6` (more workers allowed the existing short poll) | 33.84 / 33.52 | 29.12 / 29.13 | 5.28 / 5.46 |
| `SPLIT=1`, three workers polling for 3 ms after a job (test build) | 26.53, 27.03 | 36.6 | 11.5 |
| the same, workers pinned to cores 0 to 3 | 26.93, 27.30 | 35.8 | 11.3 |
| `SPLIT=2`, six workers polling for 3 ms | 13.80, 14.25 | 70.8 | 40.5 |

| Skliar, lambda 0.7310, same runs | tok/s | later tokens, ms | blocked, ms per token |
|---|---|---|---|
| `SPLIT=2` | 33.98, 32.99, 32.77 | 29.31 | 5.46 |
| `SPLIT=1` | 33.36, 35.04, 33.87 | 28.30 | 4.35 |

- **Measured.** With three reads per expert a load's last byte lands at 742 us instead of 812 and the stall
  per layer falls from 421 to 318 us (traces). A read of twice the size takes 497 to 537 us instead of about
  420: most of a read is fixed latency, so fewer, larger reads finish sooner than more, smaller ones that
  queue for workers. `SPLIT=2` was chosen on the 11 B model, whose slices are 2.5 times larger.
- **Measured.** Keeping workers awake by polling made it much worse, pinned to the small cores or not. In the
  trace the reads then start later (90 to 197 us) and take longer (591 to 879 us). Cause not established.
  The change is not in the patches; the diff is kept as `results/pixel_skliar/rejected_worker_spin.diff`.
- `SPLIT=1` helps both arms (trained 1.26 ms per token, Skliar 1.01 ms). With it the trained model is 1.6 %
  ahead of Skliar on later-token time on this one text. These are quick runs, not cool-gated rows.

## S1-10 -- Three reads per expert against six, timed (2026-11-17, 23:33 to 23:47)

`results/pixel_skliar/run_timing_split.sh`, rows tagged `sp2` (`LLAMA_TEMPORAL_SPLIT=2`, the protocol so far)
and `sp1` (`SPLIT=1`). Kept short on request: texts 1 and 2 only, two arms, 3 repetitions, 128 tokens, context
1,024, shared expert first in every run, cool gate and 60 s rest, one binary (engine `5f992263`), the order of
the two settings flipped between the texts. 8 runs, none void: big-core cap at rating, no swap, skin at most
33.4 C at the start.

| arm | reads per expert | tok/s, text 1, text 2 | mean | ms per token | p95 token ms | slowest token ms | ms per token blocked on fetches |
|---|---|---|---|---|---|---|---|
| trained, pool on flash | six | 33.73, 33.32 | 33.52 | 29.83 | 35.3 | 41.0 | 5.44 |
| trained, pool on flash | three | 35.11, 35.46 | **35.28** | 28.34 | 31.9 | 38.9 | 4.35 |
| Skliar (1.002 loads over five texts; 0.984 and 0.957 on these two) | six | 33.65, 33.59 | 33.62 | 29.75 | 35.9 | 65.3 | 5.52 |
| Skliar | three | 34.91, 35.02 | **34.97** | 28.60 | 33.8 | 45.6 | 4.19 |

- **Measured.** Three reads per expert are faster for both arms: trained +4.1 % and +6.4 % on the two texts
  (+5.3 % on the means), Skliar +3.7 % and +4.3 % (+4.0 %). The time blocked on fetches falls by 1.1 and
  1.3 ms per token.
- **Measured.** Trained against Skliar: +0.2 % and -0.8 % with six reads, +0.6 % and +1.2 % with three. Two
  texts; on these two Skliar loads 3 % fewer experts than the trained model, so the comparison leans
  slightly toward Skliar.
- **Gate.** Logits with three reads are bit-identical to the six-read gate runs of S1-7 on every decoded token
  of text 1, both arms (385 tokens each; tag `g4s1`). Texts 2 to 5 were not checked.
- Against the all-in-RAM reference of S1-7 (20.79 ms per token), streaming now costs the trained model
  7.5 ms per token, down from 10.1 ms in S1-5's configuration (shared expert after the wait, six reads).

## Not verified

- The stalls of S1-3 are not fully explained. The timed rows are free of them, but only cool runs are.
- Skliar was timed at context 1,024 only, on five texts, 128 tokens each.
- No same-day stock all-resident row and no same-day row for our rule on the plain MoE (dropped by the scope
  change). The all-resident comparison leans on the 2026-11-10 rows of the trained model.
- The trained rows use llama-bench's random tokens, Skliar real text. The trained rule's load count is 1.0
  either way; its compute on real text was not timed.
- `LLAMA_TEMPORAL_TAIL` above 1 and `LLAMA_TEMPORAL_ROUTE=decode` were gated but not timed under the cool
  gate. Throttled exploration runs suggested four sub-passes may help Skliar (median token 30.2 ms against
  32 to 36 ms); that is not a result.
- Fetch workers were fixed at six. Whether more workers shorten Skliar's bursts was not tried.
- Decode-phase memory for Skliar (the 18-expert working set after the prefill is trimmed) was not measured;
  peak RSS includes the prefill.
- Accuracy is not part of this ledger (P1-10 has it); the engine was checked for routing equality and
  flash-equals-RAM only.
- S1-10: two texts, one run each. `SPLIT=1` was not timed with the shared expert after the wait, on the
  all-in-RAM arm (it reads nothing), on the 11 B model, or with other worker counts.
- S1-7: the switch was timed at context 1,024 on five texts only, one run per text and setting, on the 1.26 B
  models. The 11 B model was not touched: the switch is in `glm4-moe.cpp` only, and that model's architecture
  (`qwen3moe` by its file name) was not checked. The trained
  model's all-in-RAM arm, our decode-time rule and `trained_deploy` on random tokens were not re-timed with
  the switch. Placing the shared expert elsewhere (for example split around the resident sub-pass) was not
  tried. Why the host build reports 0.089 loads per token-layer for Skliar without the two-pass graph is open.
- `rows.jsonl` holds every run of the session, including smoke, exploration, diagnostic and gate runs made
  hot; only rows tagged `t2` (S1-5), `big1` (S1-6), `sx0` and `sx1` (S1-7), `sp2` and `sp1` (S1-10) are results.

## Files

`run_pixel_skliar.py` (driver: `time`, `gate`, `dump`), `pixel_skliar_stepcheck.py`, `pixel_skliar_ties.py`,
`pixel_skliar_summary.py`, `pixel_skliar_gate_ppl.py` (abandoned gate). Under `results/pixel_skliar/`:
`rows.jsonl`, `summary_t2.md`, `run_timing_t2.log`, `summary_shexp.md`, `run_timing_shexp.sh` and `.log`,
`run_gates_shexp.sh` and `.log` (S1-7; tables by `pixel_skliar_shexp.py`), `logs/` (full output per run), `toklog/` (per-token
times), `gates/`, `host/`, `tokens/`.

Note on the driver: after the timed runs, an edit to `run_pixel_skliar.py`'s docstring deleted its header block
(imports, constants, arm table). It was retyped from the session record and checked to import and list the
same arms; the rows were produced by the version before the slip. The run environment of every row is in
`rows.jsonl` (`env`) and in the first line of its log.
