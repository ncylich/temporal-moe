# Environment

The versions every published number in this repository was produced on, plus the contract the
scripts use to find things. Recorded so a rerun can reproduce the toolchain, not just the code.

## First-time setup, required

A fresh clone cannot train. All four vendored dependencies are git submodules and are **empty**
until initialised, so `experiments/run.sh` builds its arguments correctly and then dies at
`torchrun` with `can't open file '.../Megatron-LM/pretrain_gpt.py'`. This is the single biggest gap
between "cloned" and "working".

```bash
git submodule update --init --recursive
```

| submodule | pinned commit | upstream |
|---|---|---|
| `Megatron-LM` | `cbaf684` | `github.com/yuzc19/Megatron-LM`, branch `multi-nodes` |
| `TransformerEngine` | `fc03478` | `github.com/NVIDIA/TransformerEngine`, branch `release_v1.11` |
| `apex` | `c02c6c8` | `github.com/NVIDIA/apex` |
| `lm-evaluation-harness` | `0c8c0d8` | `github.com/yuzc19/lm-evaluation-harness`, branch `megatron` |

The `TransformerEngine` pin `fc03478` is the same build hash recorded in the installed version
string, `1.11.0+fc034785`, so the submodule and the pinned wheel agree.

Analysis and probe scripts that only read checkpoints still need `Megatron-LM` present, because the
distributed-checkpoint metadata pickle references megatron classes. See `analysis/probes/ckpt_read.py`.

### Overlap-architecture parity, verified

On `main` today, `EXTRA_MODEL_ARGS` is written by `parity_overlap.sh` and `overlap_v1_1e18.sh` and
read by no launcher, so the overlap flags cannot reach argparse from a clone of this repository.
The hook that read it existed on the working branch the 1e18 overlap runs were launched from and
was never merged, so **those historical runs were valid and their published numbers stand**. What
was lost is the ability to reproduce them from `main`. Restoring the hook lets the parity test run
here for the first time. Four arms, G3 temporal, mb32, 10 iterations, seed 1234, same data:

| arm | iter 5 | iter 10 | final test CE |
|---|---|---|---|
| patched, flags off | 10.650930 | 10.067500 | 9.767243 |
| patched, `--overlap-early-router` | 10.644200 | 10.062540 | 9.760985 |
| patched, `--overlap-parallel-ffn` | 10.653240 | 10.064900 | 9.765186 |
| unpatched Megatron, flags off | 10.650930 | 10.067510 | 9.767256 |

**Parity holds.** Patched-with-flags-off matches the unpatched baseline exactly at iteration 5
(delta 0.00e+00) and to 1.0e-5 at iteration 10, a relative difference of 0.0001%. That residual is
ordinary GPU nondeterminism, so the patch is inert when its flags are off. The harness comment
asking for bit-for-bit equality is stricter than the hardware allows; equality within run-to-run
noise is the achievable bar.

**Both flags are active.** They shift iteration-5 loss by -6.7e-3 and +2.3e-3 respectively, roughly
673 times the patched-versus-unpatched residual, so each one demonstrably changes the forward path.

This verifies the code path as it stands on `main` after the fix. It does not cast doubt on the
published overlap results, which were produced with the hook present.

### Getting from a fresh clone to a training step

Validated end to end on a clean checkout. `scripts/setup.sh train` now does the dependency work,
but the two source builds are still yours to run and they are the slow part.

```bash
git submodule update --init --recursive        # 200 s, 2.1 GiB
python3 -m venv --system-site-packages .venv   # reuses the pinned system torch, no 2.5 GB download
scripts/setup.sh train                         # submodules + runtime deps + python3-config shim
# TransformerEngine, from source, against the pinned torch:
CUDA_HOME=/usr/local/cuda NVTE_FRAMEWORK=pytorch MAX_JOBS=$(nproc) \
  .venv/bin/pip install --no-build-isolation ./TransformerEngine     # 2617 s / 43.6 min
# apex, with the CUDA extensions Megatron uses:
CUDA_HOME=/usr/local/cuda MAX_JOBS=32 .venv/bin/pip install --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
  ./apex                                                             # 1094 s / 18.2 min
```

Measured on one H100 80GB, driver 580.126.09, nvcc 12.4 V12.4.131, torch 2.4.1+cu124. The TE build
produced `1.11.0+fc034785`, an exact match for `requirements.lock.txt`.

Two things to know:

- **`apex` is not in `requirements.lock.txt`.** It builds and is required, but `pip freeze` records
  it as version `0.1` and it was omitted. The lockfile is not a complete record of the environment.
- **`git submodule status` is worth reading after the init.** A leading `+` means a submodule is not
  at its pinned commit. One observed instance left `lm-evaluation-harness` on `main` instead of its
  pin; a second `git submodule update` corrected it. Not reproducible in isolation, so check rather
  than assume.

If training fails at startup, the causes are almost always in this list, all of which
`scripts/setup.sh train` now handles:

| symptom | cause |
|---|---|
| `.../torchrun: No such file or directory` | torch not installed into `$PY`'s prefix; launchers use `"$PY" -m torch.distributed.run` |
| `No module named 'regex'` | Megatron tokenizer dependency |
| `pybind11/pybind11.h: No such file or directory` | pybind11 include not on `CPLUS_INCLUDE_PATH`; `env.sh` derives it |
| `make: python3-config: No such file or directory` | `python3 -m venv` ships no shim, unlike `virtualenv` |
| `No module named 'megatron.core.datasets.helpers_cpp'` | consequence of the previous two |

### Reproducing the overlap-architecture runs

The submodule checks out **vanilla** Megatron-LM at `cbaf684`. The overlap-architecture variants were
produced with four locally modified Megatron files that were never committed upstream, so a fresh
clone cannot run them as-is. The diff is preserved in this repository:

```bash
git -C Megatron-LM apply ../overlap_arch/overlap_variants_megatron.patch
```

It touches `megatron/core/transformer/moe/moe_layer.py`,
`megatron/core/transformer/transformer_config.py`,
`megatron/core/transformer/transformer_layer.py` and `megatron/training/arguments.py`, and adds the
two flags `--overlap-early-router` and `--overlap-parallel-ffn`.

This patch was verified equivalent to the live uncommitted diff on the machine the runs were
produced on: identical file list, and identical raw patch text (123 lines, 52 `+` lines including
headers). Applying it to a pristine `cbaf684` gives `4 files changed, 48 insertions(+), 7
deletions(-)`, and `git apply --check` passes. Without it, a fresh checkout has zero occurrences of
`overlap-early-router` in `arguments.py`, and any launcher passing those flags fails at argument
parsing.

Everything else, including both smoke paths in `experiments/run.sh` and
`experiments/scale_1e18_1e19/`, runs on the unpatched submodule.

## Getting the artifacts

The repository holds code and result tables. Checkpoints, router traces and the tokenized corpus
live in four public Hugging Face repositories, and `results/MANIFEST.csv` maps all 1,352 files to
their origin path, size and sha256.

```bash
scripts/setup.sh analysis                                  # CPU only, ~3 min
. scripts/env.sh
scripts/artifacts.py pull --glob 'ablations/*.csv'         # result tables, ~5 MiB
scripts/artifacts.py pull --run g3_moe_s2_1e17             # one run's checkpoint + logs
scripts/artifacts.py pull --cited --repo extras            # everything a published number needs
scripts/artifacts.py verify                                # check what is already on disk
```

Every file is verified against its recorded sha256; a file that fails is deleted rather than left
partial, so re-running retries only what failed. `--repo`, `--run`, `--cited`, `--glob` and
`--max-bytes` all narrow the selection, because the full set is 214 GiB.

### What the analysis-only environment can reproduce

**All eleven** scripts in `analysis/plots/` run under `setup.sh analysis` with no GPU, no
submodules, no torch and no downloads, in both default and `--no-caption` paper mode, entirely from
the CSVs committed in `results/ablations/`.

They did not always. `plot_mechinterp.py` and `plot_probe.py` read from a
`results/phase0/figure_data/` directory that no longer exists. Every file they wanted had been
consolidated into `results/ablations/` under a different name, and the scripts were never updated:

| the script asked for | where it actually lives | selected by |
|---|---|---|
| `expert_selection_per_token_{8,15,38}M_model.csv` | `expert_selection_per_token.csv` | `active_params_M` in {8, 15, 38} |
| `mechinterp_softmax_locus.csv` | `mechinterp_locus.csv` | `label` = `s0_SOFTMAX_BASELINE` |
| `mechinterp_locus_kfull.csv` | `mechinterp_locus.csv` | `label` in {`s2_FULL`, `s0_TEMPORAL`, `s2_TEMPORAL`} |
| `mechinterp_softmax_floors.csv` | `mechinterp_floors.csv` | `model` column, same five labels |
| `learned_locality_vs_scale.csv`, `rsweep.csv`, `mechinterp_floors.csv` | same names in `results/ablations/` | — |

Nothing needed regenerating and nothing needed downloading.

`analysis/paths.py` used to export a `FIGDATA` constant naming that dead directory, which is how the
staleness survived. It has been removed in favour of `ABLATIONS`, and `probe_replay.py` now writes
its CSVs there too, alongside the ten of its twelve outputs that are already committed.

`plot_probe.py` falls back to the committed CSVs whenever `router_log.pt` is absent, rather than
only in paper mode, so a plain run no longer dies on `import torch`. With the raw logs present it
still uses them, and `graphs_BC()` runs only in that case, since it genuinely needs them:

```bash
scripts/artifacts.py pull --glob 'run_captures/*/router_log.pt'   # 22 files, 4.56 GiB, needs torch
```

## Reproduction status

`g3_tmoe_s2_1e17` was retrained end to end on a fresh clone against the published corpus, and the
published checkpoints were re-evaluated in the same environment.

| | test CE | test BPB |
|---|---|---|
| published | 3.553032 | 1.2873 |
| retrained from scratch, same config and seed | 3.562365 | 1.2907 |
| delta | +0.009433 | **+0.0034** |

**This is a pass.** +0.0034 sits at roughly the 40th percentile of this codebase's own measured
run-to-run variability. From `results/ablations/seed_replicates.csv`, ten same-box seed pairs give a
mean absolute delta of **0.0044 BPB** (median 0.0043, sd 0.0028, range 0.0004 to 0.0089). Six of the
ten published seed pairs differ by *more* than this reproduction does. A frequently quoted figure of
~0.002 for this family is about half the spread the replicate table actually shows.

Training was clean: exit 0, zero NaN, zero skipped iterations, and the loss tracked the original to
2.9e-5 at iteration 10.

**The artifact chain reproduces exactly.** The published checkpoint, pulled with
`scripts/artifacts.py` and evaluated in this fresh environment, returns CE 3.553074 against the
published 3.553032: +0.000042 CE, +0.0000 BPB. `g3_tmoe_sm1_1e16` reproduces its published 1.4976
as 1.4982. The evaluation pipeline, the corpus and the published weights are all sound.

### Why retrained numbers move, and what was ruled out

`run.sh` builds `--data-path` with `find`, which returns filesystem order rather than sorted order,
and Megatron's `--split 90,5,5` partitions the concatenated corpus by index. Two machines therefore
train and evaluate on differently composed splits: 10 of 12 shards sat in different positions
between these two runs.

- **Evaluation-split composition is ruled out as the cause.** Forcing the original shard order at
  evaluation time, on the same checkpoint, moved the result by +0.0000 BPB.
- **Training-split composition remains the plausible mechanism**, and the replicate table already
  contains direct evidence: `g3_moe_s0_1e16_sigmoid_seed2` was run twice at the *same seed* on
  different boxes and differs by 0.0031 BPB on test and 0.0060 on validation, annotated in the
  record as "different val split". That is the same magnitude as this reproduction.

The shard ordering is left as-is deliberately. Sorting it is a one-line change and is more correct,
but it silently redefines which documents every published number was measured on.

## Environment contract

`scripts/env.sh` is the single source of truth. Source it first from any launcher:

```bash
. scripts/env.sh          # from the repo root
cd "$ROOT"
```

| variable | default | meaning |
|---|---|---|
| `ROOT` | `git rev-parse --show-toplevel`, else the parent of `scripts/` | repo root |
| `TMOE_ROOT` | `$ROOT` | same value, under the name `analysis/paths.py` reads |
| `PY` | `$ROOT/.venv/bin/python` if present, else `python3` | interpreter for every call site |
| `DATA_DIR` | `$ROOT/data/dclm_tokenized` | tokenized `*_text_document.{bin,idx}` shards |
| `TOKENIZER_MODEL` | `EleutherAI/pythia-12b` | HF tokenizer id, or a local directory such as `$ROOT/data/tok16k` |
| `CKPT_ROOT` | `$ROOT/results/phase0/runs` | parent directory for run outputs |
| `NV` | derived at runtime from `import nvidia` | nvidia pip package directory, used for the cudnn/cublas loader path |

Every one is `${VAR:-default}`, so all of them can be overridden from the caller's environment
without editing a script. Nothing is tied to a checkout location.

Python analysis code uses `analysis/paths.py`, which resolves `ROOT` from `$TMOE_ROOT`, then git,
then its own location, and exposes `ROOT`, `RUNS`, `CACHE`, `ABLATIONS`.

Running from a checkout somewhere else, against data that lives elsewhere:

```bash
. scripts/env.sh
export DATA_DIR=/mnt/corpus/tok16k_full
export TOKENIZER_MODEL=/mnt/corpus/tok16k
SHAPE=s2 TARGET_FLOPS=1e17 ./experiments/run.sh
```

## Pinned versions

Training and analysis ran on a single NVIDIA H100 80GB HBM3.

| component | version |
|---|---|
| Python | 3.11.10 |
| torch | 2.4.1+cu124 |
| torch CUDA runtime | 12.4 |
| TransformerEngine | 1.11.0+fc034785 |
| flash-attn | 2.6.3 |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |
| numpy | 1.26.4 |
| huggingface-hub | 1.20.1 |
| triton | 3.0.0 |
| datasets | 2.21.0 |
| nvidia-cudnn-cu12 | 9.1.0.70 (torch reports cuDNN 90100) |
| nvidia-cublas-cu12 | 12.4.2.65 |
| CUDA toolkit (nvcc) | 12.4, V12.4.131 |
| NVIDIA driver | 580.126.09 |

`requirements.lock.txt` at the repo root is the full `pip freeze` of that environment, 265 packages.
It is a record, not an install target: TransformerEngine and apex were built from source against
this exact torch and CUDA pair, so `pip install -r` will not reproduce them by itself. Build TE
before installing the rest, and keep torch pinned, since a resolver that replaces torch will break
the driver match.

## llama.cpp serving benchmarks

`llamacpp-bench/` builds on a llama.cpp CUDA fork at base commit **`0badc06`**, sm_86 (RTX A6000),
`Release`, `GGML_CUDA=ON`. `llamacpp-bench/systems_bench.patch` is the full diff against that
commit, 5 files. See `llamacpp-bench/README.md` for the build steps.

`mlx-bench/` is a separate Apple-silicon path, a vendored MLX model from mlx-lm `qwen3_moe` plus the
temporal module and a custom decode loop. See `mlx-bench/PLAN.md`.
