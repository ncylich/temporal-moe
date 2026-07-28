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

## Quickstart

Reproducing the figures needs no accelerator. Timings are measured on a fresh clone, not estimated.

**1. Clone.**

```bash
git clone https://github.com/ncylich/temporal-moe.git && cd temporal-moe   # 17 s
git submodule update --init --recursive                                    # training only, ~3 min, 2.1 GiB
```

The submodules are only needed to train, or to read raw checkpoints. Skip them for analysis.

**2. Set up the analysis environment.** No GPU, no torch, no CUDA. numpy, pandas and matplotlib only.

```bash
scripts/setup.sh analysis      # 170 s, creates .venv
. scripts/env.sh               # exports ROOT, PY, TMOE_ROOT, DATA_DIR, ...
```

**3. Fetch artifacts.** Checkpoints, router traces and the tokenized corpus live in four public
Hugging Face repositories; `results/MANIFEST.csv` maps all 1,352 files with sizes and sha256.
`artifacts.py` pulls into the layout the analysis scripts expect, verifying each file.

```bash
scripts/artifacts.py pull --glob 'ablations/*.csv'                     # result tables, 4.5 MiB
scripts/artifacts.py pull --glob 'g3_moe_s?_1e17/train.log' \
                          --glob 'g3_moe_s?_1e17/run.meta'             # three runs' logs, 1.0 MiB
$PY analysis/summarize.py g3_moe                                       # -> s1 1.2861, s2 1.2723, s3 1.2830
```

The full set is 214 GiB, so take a subset. Filters compose: `--repo`, `--run`, `--cited`, `--glob`
and `--max-bytes`. Add `--dry-run` to see the size first — `--run g3_moe_s2_1e17` on its own is
757 MiB, because it includes the checkpoint. Downloads use the standard library alone, so no token
and no `huggingface_hub`.

**4. Replot.** Nothing to download; the CSVs behind the figures are committed.

```bash
for f in analysis/plots/*.py; do $PY "$f"; done          # 44 s
```

All 11 scripts run, writing 42 PNGs to `results/phase0/figures/`: 31 distinct figures (isoFLOP
panels, loss curves, residency and swap-rate analyses, the de-lexicalization locus scatter, serving
sweeps) plus caption-free variants for the paper, which `--no-caption` produces on their own.

**Bare clone to 42 figures: about 3.9 minutes, no accelerator.**

## What a fresh clone can and cannot do

Analysis and figures work completely, with no accelerator, no submodules and no downloads beyond the
clone. Training works but is not one command: it needs the submodules initialised, a GPU, the
tokenized corpus pulled or rebuilt, and a TransformerEngine built from source against the pinned
torch and CUDA. `requirements.lock.txt` is a record of a working environment, not an install target,
and `scripts/setup.sh train` initialises the submodules and then says exactly that rather than
pretending otherwise. The overlap-architecture variants additionally need
`overlap_arch/overlap_variants_megatron.patch` applied, since the submodule pins vanilla Megatron-LM.
Serving is not automated at all: `llamacpp-bench/` and `mlx-bench/` target an RTX A6000 and Apple
silicon and are documented as manual builds, and `androidbench/` is pinned to one handset. So do not
expect to clone this and train or serve without setup work. Do expect every published number and
figure to be reproducible on a laptop. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for pinned
versions, the environment contract and the full artifact workflow.

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
from the training platform, needed for training but not for analysis. See the quickstart.

The Android harness defaults to the handset it was developed on. Set `ANDROID_SERIAL` for
your own device. `HW_MAX` in `androidbench/bench.py` holds per-core clock ratings for that
handset and needs updating for other hardware.

## Credit

Built on [FLAME-MoE](https://github.com/cmu-flame/FLAME-MoE)
([paper](https://www.arxiv.org/abs/2505.20225)) by the CMU FLAME team. This is a research
fork, not affiliated with or endorsed by them.
