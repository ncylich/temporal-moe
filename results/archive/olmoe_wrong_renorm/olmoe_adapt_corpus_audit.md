# OLMoE adaptation — Stage 1 corpus audit (in progress)

Provenance for the OLMoE residency-adaptation corpus (plan: `docs/research/olmoe-adaptation-plan.md`).

## Unconsumed-tail check (Stage 1 step 1) — RESULT: no unconsumed tail → parent-pool path

**Finding:** OLMoE-1B-7B(-0125) was pretrained on **~5.1T tokens**, while the public pretraining
mix `allenai/OLMoE-mix-0924` contains **~4.07T tokens**. Training therefore consumed *more* tokens
than the public tokenized mix contains (≈1.25 passes / plus the dolmino-mix-1124 anneal), so there
is **no index-identifiable unconsumed tail** in the public mix. The plan's Stage-1 option-2
**parent-pool sampling** path is required (and is the fully-specified fallback in the plan's risk
table).

Sources read:
- allenai/OLMoE-1B-7B-0125 model card — https://huggingface.co/allenai/OLMoE-1B-7B-0125
- OLMoE paper (arXiv:2409.02060) — 5.1T-token pretraining — https://arxiv.org/html/2409.02060v2
- allenai/OLMoE-mix-0924 dataset (~4.07T tokens, 6 domains) — https://huggingface.co/datasets/allenai/OLMoE-mix-0924

**Corpus strategy (consequence):** parent-pool sampling — fresh shards excluded by the OLMoE-mix-0924
public file manifest, from the DCLM-baseline pool + Dolma components (peS2o, StarCoder, arXiv,
OpenWebMath, Wikipedia), reweighted to published proportions, with a ~30% fraction of unseen
dolmino-mix-1124 shards by the same manifest-exclusion logic. Build 1B-token finetune corpus
(5B-capable recipe) + ~100M-token held-out BPB slice; n-gram dedup corpus vs BPB slice and lm-eval
task data. [BUILD PENDING]

## BPB divisor (Stage 1 step 4) — byte-derived, recorded here

Provisional Stage-0 impose used D = ln(2)·bytes/token = **3.1550** on wikitext-103-raw-v1 validation
under the OLMoE tokenizer. The FINAL divisor will be re-derived from the audited BPB slice's actual
byte counts under the OLMoE tokenizer and recorded in the BPB CSV header. (House rule: never inherit
a divisor; 2.7568/2.9780 are FLAME-tokenizer values and would be a bug here.)

## Router-finetuning budget priors (Stage-2 calibration; 10-min reading task)

- **ReMoE** (arXiv:2605.27081): router-ONLY fine-tuning of released MoE checkpoints (DeepSeek-V2-Lite,
  Qwen1.5-MoE-A2.7B) for expert-reuse/cache-locality; **~33M tokens** (2000 steps × grad-accum 8 ×
  seq 2048), AdamW LR 5e-5, 200-step warmup, bf16.
- **MELINOE** (arXiv:2602.11192): cache-aware fine-tuning of OLMoE / Phi-3.5-MoE / Mixtral-8x7B on
  small instruction sets (Dolly15K + GSM8K, ~order 10M tokens); updates router + gate projections
  **plus LoRA r=32 on expert up/down projections** (i.e. NOT router-only — an expert-side adapter,
  matching our plan's Stage-2b LoRA fallback).

**Calibration takeaway:** both literature priors router-tune at a *small* budget (~10-33M tokens),
but both only *nudge* an already-good free router toward locality; ours imposes a HARD residency
constraint from scratch (a +7.3 CE / +2.3 BPB gap to recover), so the plan's 0.5-5B range is
justified as generous-but-appropriate, and MELINOE's expert-side LoRA is direct precedent for our
Stage-2b decision gate if router-only stalls.

## Held-out BPB slice — BUILT

- Source: `allenai/dolmino-mix-1124` `data/dclm/024*/*.json.zst` (streamed, high-index shard range).
- Size: 24,414 × 4096 = **99,999,744 packed tokens** (~100M raw), 448,531,092 UTF-8 bytes.
- **Byte-derived divisor D = ln(2)·bytes/token = 3.1089** (bytes/token = 4.485, OLMoE tokenizer).
- Exclusion logic: dolmino-mix-1124 is a DISTINCT dataset from the pretraining manifest
  `allenai/OLMoE-mix-0924` (2903 shards, all excluded/seen); the 0125 anneal consumed only a
  SUBSAMPLE of dolmino, so these shards are in-distribution and largely unseen. (Rigor caveat:
  without AllenAI's exact anneal shard-list the non-membership is by dataset-identity + high-index
  heuristic, not per-shard proof; n-gram dedup vs the lm-eval task data is the downstream guard and
  is PENDING for the final finetune corpus.)
- Restated impose on this slice (256×4096 subsample): base CE 2.0915 / BPB 0.6727; +mask (R=8) CE
  8.5517 / BPB 2.7507; **gap +6.46 CE / +2.08 BPB** — consistent with the provisional wikitext number.

## Finetune corpus (1B) — recipe specified, materialization PENDING
Parent-pool sampling (DCLM-baseline + Dolma peS2o/StarCoder/arXiv/OpenWebMath/Wikipedia manifest-
excluded from OLMoE-mix-0924, reweighted to published proportions + ~30% unseen dolmino shards),
packed at 4096, n-gram dedup vs BPB slice + lm-eval task data. Deferred behind the Stage-0/1 gate
deliverables above (Stage 2 finetune is not yet authorized).

## Finetune corpus (1B) — MATERIALIZED (Stage 2 Step 1)
- **Total: 244,141 packs × 4096 = 1,000,001,536 tokens** (5B-capable recipe; 1B materialized).
- 70% DCLM-baseline (mlfoundations/dclm-baseline-1.0, global-shard_05): 700,002,304 tokens, 7,680 dedup-dropped.
- 30% dolmino-mix-1124 (dclm 00*/01* + math + pes2o + stackexchange + wiki; disjoint from the BPB slice's dclm/024*): 299,999,232 tokens, 5,296 dedup-dropped.
- **n-gram dedup (32-tok windows, stride 16) vs the 100M BPB slice + lm-eval task text: 12,976 packs dropped** (the science-critical corpus∩BPB-slice=∅ guard). DCLM-baseline shares the parent web pool with the slice's provenance, hence the higher DCLM drop count.
- Sources are distinct datasets / shard ranges from the held-out slice; dedup is the content-level guard (rigor caveat re: exact anneal manifest as noted above).

## Stage-2 router-only finetune — throughput & dispatch root cause (LR sweep, per 0059)
- **Measured throughput: 19.7k tok/s steady-state** (identical across all three LR arms; MB=16 packs
  of 4096, grad-checkpointing use_reentrant=False, bf16 compute + fp32 router master, ~71GB/80GB).
- **Root cause = expert dispatch, NOT the residency scan.** transformers 5.12.1
  `OlmoeExperts.forward` routes with a **Python `for expert_idx in expert_hit:` loop** over the (up to
  64) hit experts per layer — each iteration does a gather (`hidden_states[token_idx]`), two
  `nn.functional.linear` matmuls (gate_up then down), and an `index_add_`, serialized on the host.
  With top-8 of 64 experts across 16 layers this host-serialized loop dominates wall-clock. The
  FLAME residency scan (triton accel path, verified == pure-torch reference in verify23_scan.py) is
  negligible beside it. Grad-checkpointing recomputes this loop on the backward, so MB=16 is
  loop-limited, not memory-limited — raising MB would not raise tok/s.
- **Consequence for Stage 2b:** all bake-off arms inherit this ~19.7k tok/s (0.25B ≈ 3.5h/arm);
  arm D (self-distill) runs two forwards/step (teacher free-routing no-grad + student R=8) so ≈ 2×
  slower; arm E (LoRA) adds two small matmuls inside the same per-expert loop (modest slowdown).
