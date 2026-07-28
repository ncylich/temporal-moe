#!/usr/bin/env bash
# Single environment contract for the temporal-moe pipeline.
#
# Source this FIRST from experiments/run.sh and every scale/sweep launcher:
#
#   . "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"   # from experiments/<group>/foo.sh
#   . "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"      # from experiments/run.sh
#
# Every value is ${VAR:-default}, so any of them can be overridden from the caller's
# environment without editing a script. Nothing here is specific to a checkout location.
#
# Contract:
#   ROOT             repo root
#   PY               python interpreter used for every call site
#   DATA_DIR         directory of tokenized *_text_document.{bin,idx} shards
#   TOKENIZER_MODEL  HF tokenizer id or a local tokenizer directory
#   CKPT_ROOT        parent directory that run directories are written under
#   NV               directory of the installed nvidia pip packages (cudnn, cublas, ...)

# ---- ROOT: git first, then this file's own location (works in a tarball or a detached copy) ----
if [ -z "${ROOT:-}" ]; then
    _env_sh_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    ROOT=$(git -C "$_env_sh_dir" rev-parse --show-toplevel 2>/dev/null) \
        || ROOT=$(cd "$_env_sh_dir/.." && pwd)
    unset _env_sh_dir
fi
export ROOT
# TMOE_ROOT is the same value, exported under the name analysis/paths.py looks for, so a
# shell launcher and a python probe always agree on the root.
export TMOE_ROOT="${TMOE_ROOT:-$ROOT}"
# DEPRECATED alias. TEMPORAL_MOE_ROOT was the name the salvaged probe suite used before those
# probes were converted to analysis/paths.py. Nothing in this repository reads it any more, except
# flame1e18_downstream.sh, which prefers TMOE_ROOT and falls back to this. It is still exported so
# that anything outside the repository which sets or reads it keeps working. Set TMOE_ROOT instead.
export TEMPORAL_MOE_ROOT="${TEMPORAL_MOE_ROOT:-$ROOT}"

# ---- interpreter ----
# Prefer the in-tree venv when present, else whatever python3 is on PATH.
if [ -z "${PY:-}" ]; then
    _venv_py="$ROOT/.venv/bin/python"
    if [ -x "$_venv_py" ]; then PY="$_venv_py"; else PY=$(command -v python3); fi
    unset _venv_py
fi
export PY

# ---- data and tokenizer ----
export DATA_DIR="${DATA_DIR:-$ROOT/data/dclm_tokenized}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-EleutherAI/pythia-12b}"

# ---- checkpoints / run outputs ----
export CKPT_ROOT="${CKPT_ROOT:-$ROOT/results/phase0/runs}"

# ---- nvidia pip package directory, derived at runtime ----
# TransformerEngine needs cudnn/cublas from the nvidia-* wheels on the loader path. Ask the
# interpreter where they actually are instead of hardcoding a dist-packages literal, which
# breaks on a different python version, a venv, or a conda prefix.
if [ -z "${NV:-}" ]; then
    NV=$("$PY" -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null) || NV=""
fi
export NV

# Loader path for TE. Only set when NV resolved, so a CPU-only checkout still sources cleanly.
if [ -n "$NV" ]; then
    export CUDNN_PATH="${CUDNN_PATH:-$NV/cudnn}"
    export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cublas/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
fi

# Megatron builds datasets/helpers_cpp with `make` calling bare python3/python3-config,
# so the chosen interpreter has to come first on PATH for pybind11 includes to resolve.
export PATH="$(dirname "$PY"):$PATH"
