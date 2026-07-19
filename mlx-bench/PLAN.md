# Mac replication of the A6000 decode serving benchmark (MLX q4) — execution plan

Goal: replicate the `llamacpp-bench/` **decode** benchmark (ceiling / no-swap control / temporal
deploy / vanilla-offload floor curve) on this Mac (Apple M4 Pro, 24 GB, macOS 26.5.1) in **MLX with
4-bit quantization**. Decode only. Prefill is out of scope until decode replicates well.

**What "replicate" means here.** Same model architectures, same protocol, same four setups, same
machinery semantics (resident slots + real cold→hot byte copies), same correctness discipline
(exactness proof of the swap/remap infra). Absolute tok/s and ratios are *expected to differ*
(unified memory ≈ 6× the tier-crossing bandwidth of PCIe; slower GPU) — we report the Mac curve
honestly next to the A6000 curve with the physics (bytes/token, effective copy bandwidth) explicit.

## 1. Ground truth being replicated (from `results/ablations/serving_benchmarks.csv`)

Decode @ context depth 1024, n=128 decode tokens, B=1, r=8. tok/s, higher better.

| setup | fine 18-of-192 | coarse 6-of-64 |
|---|---|---|
| a) ceiling (all experts resident) | 200.8 | 251.0 |
| b) our kernel, no swap (p=0) | 176 → n0 row 186.6 | 217 → n0 row 232.4 |
| c) deploy (R=k, ≤1 swap/layer, overlapped) | 165 | 128 |
| d) router-early variant | 173 | 143 |
| floor n=1 (== deploy swap rate, sync) | 121.3 (boost-variance) | 127.5 |
| floor TARGET n=round(0.8k) / n=round(k(1−k/E)) | 43.1 (n14) / 38.7 (n16) | 42.0 (n5, both round here) |
| floor all-miss n=k | 35.1 (n18) | 36.1 (n6) |

Floor N sweep: fine {0,1,2,4,8,14,16,18}, coarse {0,1,2,4,5,6}. Eviction: lowest-index
non-selected slot (NOT LRU). Copy sources cycled to defeat caching. Low-N rows are
compute/launch-bound (noisy); high-N TARGET rows are bandwidth-bound (stable).

## 2. A6000 → Mac mapping (design decisions, fixed up front)

| dimension | A6000 (llama.cpp CUDA fork) | Mac port (MLX) |
|---|---|---|
| engine | llama.cpp fork @0badc06, custom `temporal.cu` | vendored MLX model (from mlx-lm `qwen3_moe`) + our temporal module; custom decode loop |
| model | random-weight Qwen3-MoE, H=1024 L=45 heads 8/4 head_dim 128, Qwen3 vocab, tied emb, no shared expert; fine E=192/k=18/ff=384, coarse E=64/k=6/ff=1152, seed 0 | identical configs, random weights generated layer-streamed directly in MLX (no GGUF, no torch needed) |
| quant | Q4_K_M (~4.85 bpw; fine expert ≈ 840 KiB, coarse ≈ 2.5 MiB) | MLX affine q4 g64 (4.5 bpw; fine expert = 648 KiB, coarse = 1.944 MiB). Bytes differ ~23% → always report bytes/token alongside tok/s |
| tiers | CPU pool (host-mapped) → R VRAM slots over PCIe (~25 GB/s), SM copy kernel | cold pool arrays (never on compute path) → R hot-slot arrays, real copies in unified memory (~100–200 GB/s; measured in Phase 0). Tier boundary is emulated; bytes are real |
| expert GEMM | `mul_mat_id` hook, id-remap | `mx.gather_qmm` over hot slots with remapped ids |
| overlap (deploy) | copies issued at gate op, per-GEMM CUDA events, bit-identical | copies issued right after router on a second MLX stream (`mx.async_eval` / `mx.Stream`), GEMM depends on copy outputs; exactness gate below |
| memory metric | `nvidia-smi` peak (peak_vram_mib) | `mx.get_peak_memory()` + analytic hot-path bytes; unified-memory caveat noted in CSV header |
| correctness oracle | `llama-perplexity -ub 1` PPL to 4 dp, streamed ≡ ceiling | stronger: direct logit equality tests (we own the implementation) — see gates G2a/G2b |
| protocol | `-d 1024 -n 128 -r 8 -fa 1 -ub 1 -b 1` | prefill 1024 random ids (untimed, `mx.eval` + `mx.synchronize`), time 128 greedy decode steps; 1 warmup rep + 8 timed reps, mean±std |

Why MLX and not llama.cpp-Metal: the fork's kernels are CUDA-only, and MLX q4 is the user-chosen
target; MLX gives direct control over residency/overlap.

## 3. Deliverables

- `mlx-bench/model.py` — vendored Qwen3-MoE (attention w/ QK-norm, RoPE θ=1e6, norm_topk_prob) + `TemporalMoE` block (hot slots / cold pool / remap / swap / pinned-N floor driver)
- `mlx-bench/gen_random_qwen3moe_mlx.py` — deterministic layer-streamed q4 model builder (seed 0), saves safetensors+config (~6.5 GB each) under `mlx-bench/models/`
- `mlx-bench/bench_decode.py` — protocol runner; `--setup {ceiling,noswap,deploy,floor_n=N}`; emits CSV rows + per-section timers (attn / router / copies / expert GEMM / head) + bytes/token + peak mem
- `mlx-bench/tests/` — gates G0–G3 as runnable scripts (each ≤2 min)
- `results/ablations/serving_benchmarks_mac.csv` — same column schema as the A6000 CSV, header documents chip/RAM/macOS/mlx versions + all protocol deviations
- `mlx-bench/RESULTS.md` — side-by-side table vs Section 1, ratios to each platform's own ceiling, physics table (bytes/token, effective copy bandwidth). **No paper edits.**

## 4. Phases (each ends at a hard gate; do not proceed on a failed gate)

### Phase 0 — Environment + physics probes (owner: me; ~15 min)
1. `python3 -m venv mlx-bench/.venv && pip install mlx mlx-lm numpy` (mlx.core is NOT currently
   installed; only an orphan `mlx-metal` wheel is). Check ≥20 GB free disk.
2. Probe script `tests/g0_probes.py`: (i) q4 quantize + `mx.gather_qmm` smoke; (ii) measured
   cold→hot copy bandwidth at expert-sized slices (648 KiB and 1.94 MiB) — this is the floor's
   physics constant; (iii) two-stream overlap smoke (`mx.async_eval` on a second stream while the
   default stream computes) proving copies aren't elided by lazy eval (bytes audit: timed copy of
   X MB must scale with X).
- **Gate G0:** mlx imports on this python (3.13.9), gather_qmm correct vs dequantized matmul, copy
  bandwidth number in hand. Fail → fix env before anything else.

### Phase 1 — Model builder + ceiling (setup a) (owner: opus agent; gate reviewed by me)
1. Vendor mlx-lm's `qwen3_moe` model file into `mlx-bench/model.py`; strip to what we need; config
   from Section 2 (vocab from `len(AutoTokenizer("Qwen/Qwen3-0.6B"))` fetched once with the conda
   python that already has transformers; fallback constant if offline, noted in CSV).
2. `gen_random_qwen3moe_mlx.py`: build per-layer fp16 → quantize q4 g64 immediately → free
   (24 GB RAM: never materialize the ~21 GB fp16 model). Save both variants.
3. `bench_decode.py` ceiling path: standard forward, KV cache, protocol from Section 2.
   Tiny-config (L=2, H=128, heads 2/1, E=8, k=3, ff=64, vocab 512) smoke first — seconds, catches
   shape/RoPE/QK-norm bugs before the 6.5 GB build.
- **Gate G1:** fine + coarse ceilings measured, rep std < 3%, decode produces varying tokens (not
  degenerate constants — random weights make text meaningless but ids should still vary).
  Sanity band: expect roughly 100–200 tok/s fine (bandwidth roofline ≈ 0.71 GB active/token ÷
  measured bandwidth). **< 50 tok/s after basic care → STOP, escalate to Phase 4 owner (me) before
  any sweeps** — an inefficient ceiling poisons every ratio.
- Optional anchor (run only if G1 is suspicious): stock llama.cpp-Metal ceiling on the same-recipe
  GGUF (build via existing `llamacpp-bench/build_models.sh`; conda has torch+transformers). An
  independent engine's ceiling bounds how much perf MLX is leaving on the table.

### Phase 2 — Temporal machinery + exactness proof (owner: opus agent; diff reviewed by me)
1. `TemporalMoE`: per-layer quantized cold pool `[E,…]` (3 tensors: gate/up/down) + hot slots
   `[R,…]`, R=k; resident-id table; forward = router top-k → residency mask/remap → swap decision →
   `gather_qmm` on hot slots. Policies matched to the fork exactly: evict lowest-index non-selected
   slot; deploy swaps ≤1/layer/token at p=1.0 (top-ranked missing expert).
2. Setups: `noswap` (b analog: machinery on, 0 swaps, all selected remapped to resident),
   `deploy` (c), `floor_n=N` (free top-k; N pinned cold-miss copies/layer/token, sources cycled;
   copies complete before the expert GEMM).
- **Gate G2a (tiny config, seconds):** temporal path with R=E + identity remap ≡ stock forward,
  exact (or ≤1e-5 with the op-order difference recorded).
- **Gate G2b (full fine model, 8 decode tokens, ~1 min):** (i) lazy-full-top-k path (fetch all
  missing, no masking) ≡ ceiling logits — the fork's "NOFORCE1 bit-identical" proof that
  load/swap/remap/GEMM infra is exact; (ii) deploy path ≡ a slow reference emulator of the same
  masked-routing semantics computed from the full pool — identical token ids, ~0 logit delta.
- Tests are written against the tiny config first so iteration is seconds, then re-run at full size.

### Phase 3 — Floor curve + noswap + sync numbers (owner: sonnet agents; runs STRICTLY sequential)
1. Run protocol for: ceiling, noswap, floor N sweep (both granularities, N lists from Section 1).
   One bench process at a time, plugged in, no other load (A6000 warning: shared-GPU contention
   silently halved a number; thermal/boost variance is the Mac analog — report std per row).
2. Instrumented audits per run: measured copied bytes/token == N × 45 × expert_bytes exactly;
   per-section timer sum within ~5% of wall.
- **Gate G3:** (i) bytes audit exact; (ii) floor tok/s monotone non-increasing in N (within std)
  and high-N slope consistent with G0 copy bandwidth within ~2× (else copies are being elided or
  cached — fix before recording); (iii) floor n0 ≈ noswap within noise.
- Output rows appended to `serving_benchmarks_mac.csv` as they land.

### Phase 4 — Deploy overlap + speed work (owner: ME / fable subagent — this is the hard part)
1. Deploy (c): same-token overlapped fetch — issue the 3 slot copies right after the router on a
   second stream; expert GEMM consumes their outputs (dependency = correctness; G2b re-run after
   every change). Start unoptimized-correct, then optimize.
2. Speed levers, in expected-value order, each verified by per-section timers + re-gated:
   `mx.compile` the per-token step (45-layer B=1 python/launch overhead is the likely #1 cost;
   compile with mutable residency state + KV cache is finicky — this is why fable owns it);
   CPU-stream memcpy for the copies (unified memory: CPU can move bytes while GPU computes — the
   Mac analog of the fork's "small copy grid frees SMs" finding); batching the 3 copies; avoiding
   re-quantization or layout churn on hot slots.
3. Optional stretch after c is solid: d) router-early variant (route on pre-attention input,
   overlap copy with attention) — architectural variant, same-token semantics, included in the
   A6000 table; measure only, half a day cap.
- **Gate G4:** deploy ≥ floor_n1 (overlap must beat the same swap rate done synchronously);
  exactness gates still green; then record deploy rows.

### Phase 5 — Consolidation (owner: sonnet agent; I verify numbers against Section 1 by hand)
1. Final CSV + `RESULTS.md`: Mac vs A6000 side by side (each normalized to its own ceiling),
   physics table (expert bytes, bytes/token at each N, measured copy bandwidth both platforms —
   PCIe ~25 GB/s vs measured unified), protocol deviations list (bpw difference, r=8+warmup
   definition, memory metric).
2. Replication verdict = qualitative invariants, stated explicitly: floor monotone and
   bandwidth-bound at high N; n1 consistent with deploy swap rate; coarse pays more per swap than
   fine (granularity asymmetry direction); deploy beats sync-n1. Expected divergence: unified
   memory raises the floor (cheaper per byte) while the ceiling is compute/launch-bound lower —
   report as finding, not failure.

## 5. Orchestration rules (context hygiene)

- Sub-tests run in **sonnet** or **opus** agents (Agent tool, `model:` set per phase above); every
  brief demands back: verdict (PASS/FAIL vs the named gate) + key numbers + file paths, never file
  dumps. Gating phases run `run_in_background: false`.
- I (fable) hold: this plan, gate verdicts, the running results table, Phase 2 diff review, and all
  of Phase 4. If Phase 4 needs fan-out (e.g., trying 3 overlap schemes), spawn fable subagents with
  a tight single-question brief each.
- Benchmarks NEVER run concurrently with each other or with builds. Code/test writing may proceed
  in parallel with runs.
- Fail fast ordering inside every phase: tiny-config test → full-model short run (n=32, r=2 smoke)
  → full protocol. Nothing runs the full protocol before its gate's smoke passes.
- Any gate failure: stop the phase, report the failing observable (not a guess), fix, re-gate.
  Two consecutive unexplained gate failures in a phase → stop and reassess the approach (the
  fork's README "Warnings" list is the checklist: don't assume costs, measure; don't trust
  rooflines over the real path; don't time on a loaded machine).

## 6. Risks & pre-decided responses

| risk | response |
|---|---|
| MLX lazy eval elides/reorders the floor copies | G0(iii)/G3 bytes-vs-time audit catches it; force with explicit dependencies or `mx.eval` per layer (floor is sync by definition; only deploy needs async) |
| Python/launch overhead swamps B=1 decode (ceiling ≪ roofline) | G1 kill threshold 50 tok/s → Phase 4 owner takes over early; `mx.compile`; optional llama.cpp-Metal anchor bounds the gap |
| `mx.compile` breaks with mutable residency/KV state | keep an uncompiled reference path forever; every compiled variant must re-pass G2a/G2b |
| RAM: fp16 fine model ≈ 21 GB > budget during generation | layer-streamed build (generate→quantize→free per layer); assert process RSS < 16 GB during gen |
| Thermal/boost variance (Mac analog of A6000 boost-variance rows) | r=8 + warmup, report std per row, sequential runs, plugged in; flag rows with std > 5% |
| mlx-lm qwen3_moe missing/different | vendored copy is the source of truth; worst case write the ~200-line model directly (arch fully specified in Section 2) |
| vocab fetch offline | fallback constant recorded in CSV header (latency effect ~0; only head GEMM size matters and it's fixed by the constant) |

## 7. Out of scope (do not do)

- Prefill port (expert-major streaming) — only after decode replicates well, per the request.
- SSD/NVMe cold tier (FINAL_TOUCHES row 12 — separate item), pinned resident slots, lag-1 /
  early-loading semantics (settled: same-token swap with overlapped fetch only), trained weights,
  perplexity/quality evals, paper edits, committing results to git (user decides).
