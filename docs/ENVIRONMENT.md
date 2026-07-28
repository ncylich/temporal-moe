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
then its own location, and exposes `ROOT`, `RUNS`, `CACHE`, `FIGDATA`.

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
