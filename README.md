# Temporal Mixture-of-Experts

A sparse MoE computes only `k` experts per token but still needs the whole expert pool in
fast memory to serve. Temporal MoE trains the model under a rolling-residency constraint
that keeps only the `k` active experts of each layer resident and swaps at most one expert
per token, so the incoming expert streams in behind the compute of the ones already there.
Serving memory then scales with active parameters instead of total parameters.

The constraint is trained in rather than applied at inference, so the router reorganises
around it. Expert selection moves off token identity and onto surrounding context, which
is why quality holds up.

[Paper](paper/main.pdf) ·
[Talk slides](https://docs.google.com/presentation/d/1AwLRFUdAcEJ-jtT-_qNqcqennXaoVzB_GslD9otz6rg/edit?usp=sharing)

## Results

Trained from scratch on isoFLOP sweeps from 10^16 to 10^19 FLOPs, at 6-of-64 and
18-of-192 expert granularity.

* Retains 72-82% of the MoE-over-dense quality gain at compute-optimal sizes.
* Holds 18 of 192 experts resident, cutting whole-model weight memory 5.7x.
* Serves an 11B-scale model in llama.cpp using 5.1x less memory, with a 17% decode
  slowdown on an RTX A6000 and 30% on a Pixel 10a (Tensor G4), compared to the baseline
  where all experts are in memory.

## Layout

| Path | Contents |
|---|---|
| `paper/` | Write-up |
| `temporal/` | Rolling-residency router |
| `docs/` | Design docs, mechanism analyses, ablations, evaluation methodology |
| `results/` | Measured results |
| `llamacpp-bench/`, `mlx-bench/`, `androidbench/` | Serving benchmarks on A6000, Apple Silicon, and Android |
| `experiments/`, `configs/`, `scripts/`, `analysis/` | Training and figures |

`Megatron-LM`, `TransformerEngine`, `apex`, and `lm-evaluation-harness` are submodules
from the training platform. Clone with `--recursive`.

The Android harness defaults to the handset it was developed on. Set `ANDROID_SERIAL` for
your own device. `HW_MAX` in `androidbench/bench.py` holds per-core clock ratings for that
handset and needs updating for other hardware.

## Credit

Built on [FLAME-MoE](https://github.com/cmu-flame/FLAME-MoE)
([paper](https://www.arxiv.org/abs/2505.20225)) by the CMU FLAME team. This is a research
fork, not affiliated with or endorsed by them.
