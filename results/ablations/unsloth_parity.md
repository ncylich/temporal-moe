# Unsloth grouped_mm path vs our stock path — numerical parity (Qwen3-30B)

Producer: `analysis/ple/check_unsloth_kernels.py` (arms run separately because unsloth
patches transformers at import; dumps under `/workspace/qwen3moe-adapt/results/unsloth_check/`).
Model: Qwen3-30B-A3B base bf16 from `/dev/shm/qwen3-30b`. Data: the 16-sequence held-out
BPB slice (65,520 scored positions). Env: `/workspace/venv_fla`, torch 2.13.0+cu130,
transformers 5.12.1, unsloth 2026.8.4 + unsloth_zoo 2026.8.3. Measured 2026-08-05.

BPB = bits per byte (lower is better); deltas are (candidate − stock-in-this-env). In
**floor** rows the BPB column shows the torch-2.4 value and the delta is (stock@2.13 −
stock@2.4) — the same-math cross-environment churn the candidates are judged against.
"top-1 agree" is the fraction of positions where both configurations predict the same
argmax token.

## Residency OFF — plan step (a): kernel equivalence

| comparison | BPB | delta | top-1 |
|---|---|---|---|
| stock (residency_qwen, off) | 0.615517 | — | — |
| unsloth base | 0.615392 | −1.25e-04 | 0.9775 |
| unsloth + zero-init LoRA r32 | 0.615392 | −1.25e-04 | 0.9769 |
| peft wrapper alone (unsloth vs +LoRA) | — | +1.17e-07 | 0.9778 |
| **floor**: same stock code, torch 2.4 vs 2.13 | 0.615371 | +1.46e-04 | 0.9776 |

The bs1-vs-bs2 floor used by check_fused_kernels is dead in this env — torch 2.13's stock
path is batch-size invariant (delta −4e-09, agree 1.0000) — so the operative floor is the
cross-environment one. Unsloth's whole kernel difference is smaller than the churn between
two torch versions running identical math. **ACCEPT.**

## Residency ON (R=k=8, all layers, min_logit) — plan step (c): patch correctness

| comparison | BPB | delta | top-1 |
|---|---|---|---|
| stock (residency_qwen) | 0.733465 | — | — |
| unsloth + residency_unsloth | 0.734032 | +5.67e-04 | 0.7826 |
| unsloth + residency_unsloth + LoRA | 0.734020 | +5.54e-04 | 0.7826 |
| **floor**: same stock code, torch 2.4 vs 2.13 | 0.733527 | −6.17e-05 | 0.7894 |
| same-kernel trajectory noise: torch 2.4 bs1 vs bs2 | 0.734947 | +1.42e-03 | — |

Under the constraint the system is chaotic: an epsilon perturbation (torch version, batch
size, the peft wrapper's casts at BPB +1e-07) flips swap decisions and ~21% of argmaxes.
The BPB tolerance is therefore the measured same-kernel trajectory noise (1.42e-03), not
the residency-off threshold; unsloth's +5.54e-04 sits inside it, and its top-1 (0.7826) is
at the same-math floor (0.7894, guard −0.01). **ACCEPT.**

`residency_unsloth` was separately validated bit-exact against `residency_qwen` on a tiny
random Qwen3MoeForCausalLM (max|Δlogit| = 0, argmax identical, constraint effect 3.8e-02).

## Qwen3.5 — residency OFF (measured 2026-08-06, same protocol, 16-seq qwen slice)

| comparison | BPB | delta | top-1 |
|---|---|---|---|
| stock (residency_qwen, off) | 0.625072 | — | — |
| unsloth base | 0.625280 | +2.08e-04 | 0.9786 |
| unsloth + zero-init LoRA r32 | 0.625152 | +7.94e-05 | 0.9783 |
| peft wrapper alone | — | −1.28e-04 | 0.9779 |

No same-math floor exists for this model — the torch 2.4 venv cannot load it (no `fla`) and
the bs floor saturates (agree 1.0000) — so these are judged against the qwen3-measured
floors: BPB ±1.5e-04, top-1 0.9776. The full training config sits at +7.94e-05 / 0.9783:
**ACCEPT**. Two notes: (1) unlike qwen3, the peft wrapper here is not bit-exact (−1.28e-04)
— their param-wrapper re-lays-out the 3-D expert Parameters on this arch, another
reduction-order change, same churn scale; (2) both arms run the GDN layers on transformers'
torch fallback (`causal-conv1d` not installed), identically — matched, but a throughput
item, not a numerics one.

## Qwen3.5 — residency ON (R=k=8, all layers, min_logit)

| comparison | BPB | delta | top-1 |
|---|---|---|---|
| stock (residency_qwen) | 0.679850 | — | — |
| unsloth + residency_unsloth | 0.679504 | −3.47e-04 | 0.8704 |
| unsloth + residency_unsloth + LoRA | 0.680022 | +1.72e-04 | 0.8693 |
| peft wrapper alone | — | +5.18e-04 | 0.8711 |

Judged against the qwen3 floors (trajectory noise 1.4e-03 BPB; constraint-chaos top-1 floor
0.79 at E=128 — at E=256 the resident fraction is half, so 0.87 agreement being *higher* is
the expected direction): +1.72e-04 / 0.8693. **ACCEPT.** Sanity: the stock constraint cost,
0.679850 − 0.625072 = +0.0548 BPB, reproduces the known training-free E=256 cost (~0.052 on
its original slice) on this slice.

## Throughput — plan step (d), Qwen3-30B

Producers: `analysis/ple/probe_expert_lora_cost.py` (ours) and `analysis/ple/probe_unsloth_cost.py`
(unsloth), same venv, same trainable surface (expert LoRA r32 + attn LoRA r32 + router + RMSNorm,
1297.6M vs 1297.8M — the 0.2M gap is router/norm tensor counting), residency ON in both, fused
AdamW in both, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in both, bf16 adapters in both
(unsloth defaults to fp32 adapters, which OOMs this surface at mb2 — `--adapter-dtype`).

| arm | config | tok/s | s/micro-step | peak GB |
|---|---|---|---|---|
| ours (stock expert loop) | mb2 × seq2048 | 487 (foreach) / **516** (fused AdamW) | 7.94 | 78.4 |
| unsloth (grouped_mm) | mb2 × seq2048 | **5,429** | 0.75 | 77.7 |
| unsloth | mb4, fused AdamW | OOM in backward (needs ~4.6 GB more) | — | — |
| **unsloth, best achievable** | mb4, adamw8bit | **6,072** | 1.35 | 76.5 |

**Matched: mb2, 10.5× end-to-end. Best achievable: mb4 with 8-bit Adam, 6,072 tok/s.**
15M tokens: 41–46 min vs 8.1 h; a 5-point LR sweep: 3.4–3.8 h vs 40 h. This is consistent with the profiling record —
the stock path spends 89.5% of a MoE layer in the Python expert loop (`torch.where` +
`index_add_`), which is precisely what grouped_mm replaces. Caveat for the sweep, not the
comparison: bf16 adapters were forced for memory parity; unsloth's fp32-adapter default
would need mb1. Adapter precision is a training-config decision to make explicitly, and at
step 0 it is numerically irrelevant (LoRA B = 0).

## Throughput — plan step (d), Qwen3.5

Same protocol and trainable surface as Qwen3-30B, bf16 CE in both arms (the fp32 logit copy
alone is 1.9 GB at mb1 on this 248k vocab). All tok/s are **steady-state**: two untimed
warmup steps absorb model load, JIT, and autotune, so one-off startup (~5–10 min: load +
fla autotune warm + triton JIT) is an additive constant to a run's wall-clock, never a
throughput factor.

**r32 fits — but only with 8-bit Adam.** The text-only model is 69.3 GB on GPU (its
checkpoint is ForConditionalGeneration; `AutoModelForCausalLM` loads text-only — the vision
tower was ruled out as the cause). With bf16 Adam states the r32 surface needs ~15 GB and
misses the 85 (decimal) GB card by ~2–3 GB; bitsandbytes `AdamW8bit` (block-quantised
1-byte states — unsloth's own LoRA default) returns 3.7 GB and closes it. An earlier
version of this note called r32 "a hardware fact" — that was wrong; it was a fact of
2-byte optimizer states.

| arm | config | tok/s | s/micro-step | peak GB (decimal, card = 85.0) |
|---|---|---|---|---|
| ours (stock expert loop) | r32 mb1, adamw8bit | 215 | 9.51 | 82.2 |
| **unsloth, matched (target config)** | r32 mb1, adamw8bit | **3,305** | 0.62 | 82.6 |
| unsloth | r32 mb2, adamw8bit | OOM (−1.9 GiB) | — | — |
| ours | r16 mb1, fused AdamW | 218–231 (two runs) | 8.9–9.4 | 79.4 |
| unsloth | r16 mb1 | 3,889 | 0.53 | 79.3 |
| unsloth | r16 mb2 | 4,576 | 0.90 | 83.4 |

**At the target config (r32): matched speedup ~15.4×** (3,305 vs 215), 15M tokens in 76 min
vs ~19 h. r16 rows retained as alternates (~17×/~20×). The optimizer is a training-dynamics
choice, not a kernel: step-0 guards are unaffected, but the sweep must use one optimizer
consistently across arms and models — never difference runs that used different optimizers.

**Optimizer menu for r32, all measured (unsloth arm, mb1):** CCE (exact fused CE, logits
never materialised, ~2 GB) stacks with any of these but substitutes for none — fused
AdamW + CCE still OOMs by ~1.5 GiB.

| optimizer | states | fits | tok/s | note |
|---|---|---|---|---|
| fused AdamW | bf16 on-GPU | no | — | misses by ~1.5 GiB even with CCE |
| AdamW8bit | 8-bit on-GPU | yes | 3,305 | unsloth's LoRA default; recommended (+ CCE for headroom) |
| PagedAdamW32bit + CCE | fp32, unified-mem paged | yes | 1,255 probe / ~2,800 est. | probe steps the optimizer every micro-step (worst case); at acc8 the ~1.0 s/step paging cost amortises 8× → est. ~15% below 8-bit, with exact fp32 Adam dynamics |

Operational requirements on this model, both memory- not time-motivated:
- `UNSLOTH_COMPILE_DISABLE=1` — unsloth's compiled CUDA-graph pools permanently hold
  ~7.4 GB of VRAM outside the torch allocator, which is exactly the OOM margin. The 3,889
  therefore *understates* what compile-on unsloth would do with more VRAM headroom.
- `UNSLOTH_MOE_DISABLE_AUTOTUNE=1` + a one-off fla-cache warm run (light process, base
  model fwd+bwd at each micro-batch shape) — both autotuners bench at peak memory and OOM
  a full card; with warm caches they are skipped. Same understatement direction.
- fused AdamW, bf16 adapters, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — as on
  Qwen3-30B, both arms.
- GDN layers run transformers' torch fallback in both arms (`causal-conv1d` not installed).

## Consequence for use — no cross-arm comparisons, ever

Implementations carry O(1e-03) BPB offsets under the constraint. Every downstream
comparison — trained vs null vs baseline, recovery percentages — must be measured within
ONE implementation. A number produced on the unsloth path must never be differenced
against one produced on the stock path.
