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
- `rows.jsonl` holds every run of the session, including smoke, exploration, diagnostic and gate runs made
  hot; only rows tagged `t2` are results.

## Files

`run_pixel_skliar.py` (driver: `time`, `gate`, `dump`), `pixel_skliar_stepcheck.py`, `pixel_skliar_ties.py`,
`pixel_skliar_summary.py`, `pixel_skliar_gate_ppl.py` (abandoned gate). Under `results/pixel_skliar/`:
`rows.jsonl`, `summary_t2.md`, `run_timing_t2.log`, `logs/` (full output per run), `toklog/` (per-token
times), `gates/`, `host/`, `tokens/`.

Note on the driver: after the timed runs, an edit to `run_pixel_skliar.py`'s docstring deleted its header block
(imports, constants, arm table). It was retyped from the session record and checked to import and list the
same arms; the rows were produced by the version before the slip. The run environment of every row is in
`rows.jsonl` (`env`) and in the first line of its log.
