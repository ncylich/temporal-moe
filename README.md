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

## Train a temporal MoE

**1. Prerequisites.** Longest first, so you know what you're in for:

```bash
git submodule update --init --recursive     # ~3 min, 2.1 GiB
scripts/setup.sh train                      # installs deps, then states what it cannot install
```

Then TransformerEngine built from source against the pinned torch and CUDA (~45 min) and apex
(~20 min). `requirements.lock.txt` is a record of a working environment, not an install target —
a resolver that swaps torch will break the driver match. Build order and pinned versions are in
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). You need a GPU from here on.

**2. Smoke run.** The smallest shape at a token budget small enough to finish in minutes — 41
iterations. It exercises the whole path including a checkpoint write, so if this passes, the setup
above is sound:

```bash
SHAPE=sm1 TARGET_FLOPS=1e14 TEMPORAL=1 bash experiments/run.sh
```

**3. Real runs.** The same shape and budget trains three ways, which is how every comparison in
the paper was produced:

```bash
SHAPE=s2 TARGET_FLOPS=1e17 TEMPORAL=1 GRAIN=3  bash experiments/run.sh   # rolling residency
SHAPE=s2 TARGET_FLOPS=1e17            GRAIN=3  bash experiments/run.sh   # free-routing MoE
SHAPE=s2 TARGET_FLOPS=1e17 DENSE=1             bash experiments/run.sh   # dense floor
```

MoE is the default; `TEMPORAL=1` adds the residency constraint on top of it and `DENSE=1` drops
the experts entirely for the isoFLOP floor.

`run.sh` derives `train_iters` so that `C = 6ND` hits `TARGET_FLOPS`, so a shape and a budget fully
specify a run — the three above are 3,917 iterations each. Iteration count follows from the budget
and is not set directly. Shapes `sm1` and `s0`–`s6` span 96 to 512 hidden. Checkpoints, `train.log`
and a `run.meta` recording the exact geometry land in `results/phase0/runs/$RUN_NAME/`.

The constraint knobs:

| variable | default | what it does |
|---|---|---|
| `TEMPORAL` | `0` | trains under rolling residency: only `k` experts per layer resident, at most one swap per token |
| `TEMPORAL_EVICT` | `min_logit` | which resident expert leaves when a new one is admitted |
| `TEMPORAL_RHO` | — | selection margin a challenger must clear to trigger a swap |
| `TEMPORAL_EMA_BETA` | — | smoothing on the demand signal the eviction decision reads |
| `GRAIN` | `1` | expert granularity: `1` is 6-of-64, `3` is 18-of-192, `5` is 30-of-320 |

The 1e18 and 1e19 results came through `experiments/scale_1e18_1e19/`, which pins the published
geometries instead of deriving them from a FLOP budget, and so takes `TRAIN_ITERS` directly. It
also spells the paradigms differently — `MOE_FULL=1` for free-routing MoE, where `run.sh` treats
that as the default. The overlap-architecture variants additionally need
`overlap_arch/overlap_variants_megatron.patch` applied, since the submodule pins vanilla
Megatron-LM.

## Reproduce the published results

No GPU, no torch, no submodules.

```bash
scripts/setup.sh analysis                    # creates .venv: numpy, pandas, matplotlib
. scripts/env.sh                             # exports ROOT, PY, TMOE_ROOT, DATA_DIR, ...
for f in analysis/plots/*.py; do $PY "$f"; done
```

All 12 scripts run and write 44 PNGs to `results/phase0/figures/` — 32 distinct figures (isoFLOP
panels, loss curves, residency and swap-rate analyses, the de-lexicalization locus scatter, serving
sweeps) plus caption-free variants for the paper, which `--no-caption` produces on their own. The
CSVs behind them are committed, so nothing downloads.

For anything backed by raw run artifacts, `results/MANIFEST.csv` maps all 1,352 published files to
four public Hugging Face repositories with sizes and sha256, and `artifacts.py` fetches and
verifies them:

```bash
scripts/artifacts.py pull --glob 'g3_moe_s?_1e17/train.log' \
                          --glob 'g3_moe_s?_1e17/run.meta'    # 1.0 MiB
$PY analysis/summarize.py g3_moe                              # -> s1 1.2861, s2 1.2723, s3 1.2830
```

The full set is 214 GiB, so filter. `--repo`, `--run`, `--cited`, `--glob` and `--max-bytes`
compose, and `--dry-run` sizes a selection first — `--run g3_moe_s2_1e17` alone is 757 MiB because
it includes the checkpoint. Downloads use the standard library, so no token and no
`huggingface_hub`.

## What this repository will not do for you

Training is not one command, and the prerequisites above are the reason. Serving is not automated
at all: `llamacpp-bench/` and `mlx-bench/` target an RTX A6000 and Apple silicon as manual builds,
and `androidbench/` is pinned to one handset.

Every published number and figure is reproducible on a laptop. A 1e17 cell retrained from a clean
checkout lands within measured seed variance of its published value, and a checkpoint pulled from
Hugging Face reproduces its evaluation exactly.

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
from the training platform, needed for training but not for analysis.

The Android harness defaults to the handset it was developed on. Set `ANDROID_SERIAL` for
your own device. `HW_MAX` in `androidbench/bench.py` holds per-core clock ratings for that
handset and needs updating for other hardware.

## Contributing

Work happens on a branch and lands through a pull request. Nothing is committed to `main`
directly, including small fixes and documentation.

```bash
git switch -c <topic>          # branch off main
git push -u origin <topic>     # then open a PR against main
```

One setup step per clone, before the first commit:

```bash
git config core.hooksPath .githooks
```

That enables `.githooks/commit-msg`, which rejects any message naming an AI assistant or
vendor. Git will not run hooks out of a tracked directory without being told to, so the
config line is required in every clone — a fresh checkout, or a remote GPU pod. History
here was scrubbed of those references once already, and a force-push cannot undo the
attribution once a commit carrying one has been pushed: the commit stays reachable by SHA
and keeps counting.

## Credit

Built on [FLAME-MoE](https://github.com/cmu-flame/FLAME-MoE)
([paper](https://www.arxiv.org/abs/2505.20225)) by the CMU FLAME team. This is a research
fork, not affiliated with or endorsed by them.
