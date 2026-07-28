#!/usr/bin/env bash
# Environment setup, by mode.
#
#   scripts/setup.sh analysis    CPU only. numpy, pandas, matplotlib. Replot every figure from the
#                                committed CSVs, and run scripts/artifacts.py. No GPU, no torch,
#                                no submodules. This is what most readers of the paper need.
#
#   scripts/setup.sh train       Thin wrapper over the documented steps: initialise the four
#                                submodules and report what still has to be built by hand. It does
#                                NOT install the training stack, see the note it prints.
#
# Creates $ROOT/.venv, which scripts/env.sh already prefers for $PY, so sourcing env.sh afterwards
# picks it up with no further configuration.
#
# See docs/ENVIRONMENT.md for the pinned versions and the overlap-architecture patch.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"

MODE="${1:-}"
VENV="$ROOT/.venv"

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }
[ -z "$MODE" ] && usage

start=$SECONDS

case "$MODE" in
  analysis)
    echo "=== setup: analysis (CPU only) ==="
    BASE=$(command -v python3)
    echo "  base interpreter: $BASE ($("$BASE" -V 2>&1))"
    if [ ! -x "$VENV/bin/python" ]; then
      echo "  creating venv at $VENV"
      "$BASE" -m venv "$VENV"
    else
      echo "  reusing existing venv at $VENV"
    fi
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    # Deliberately only these three. Anything heavier belongs to the train path.
    "$VENV/bin/python" -m pip install --quiet numpy pandas matplotlib
    echo "  installed:"
    "$VENV/bin/python" - <<'PY'
import numpy, pandas, matplotlib
print(f"    numpy      {numpy.__version__}")
print(f"    pandas     {pandas.__version__}")
print(f"    matplotlib {matplotlib.__version__}")
import importlib.util
print(f"    torch      {'PRESENT (unexpected)' if importlib.util.find_spec('torch') else 'absent, as intended'}")
PY
    echo
    echo "  next:"
    echo "    . scripts/env.sh                                       # \$PY now points at .venv"
    echo "    scripts/artifacts.py pull --glob 'ablations/*.csv'     # fetch result tables"
    echo "    \$PY analysis/plots/<figure>.py"
    ;;

  train)
    echo "=== setup: train (thin wrapper over docs/ENVIRONMENT.md) ==="
    # This mode installs packages and links a python3-config next to the interpreter, so it must
    # never run against a system interpreter: pip would land in system site-packages and the
    # symlink would target /usr/bin. Only root would get away with that, and silently. analysis
    # mode creates $ROOT/.venv; train needs one too, and needs the pinned torch visible, hence
    # --system-site-packages rather than a bare venv.
    case "$PY" in
      "$ROOT"/*) ;;
      *)
        if [ -x "$VENV/bin/python" ]; then
          echo "  \$PY was outside the repo ($PY); switching to $VENV"
        else
          echo "  \$PY is outside the repo ($PY); creating $VENV --system-site-packages"
          python3 -m venv --system-site-packages "$VENV" || {
            echo "  ERROR: could not create $VENV"; exit 1; }
        fi
        PY="$VENV/bin/python"; export PY
        export PATH="$(dirname "$PY"):$PATH"
        ;;
    esac
    "$PY" -c 'import torch' >/dev/null 2>&1 || {
      echo "  ERROR: torch is not importable from $PY"
      echo "    train mode needs the pinned torch. If torch lives in the system interpreter,"
      echo "    recreate the venv so it can see it:"
      echo "      rm -rf \"$VENV\" && python3 -m venv --system-site-packages \"$VENV\""
      exit 1; }
    echo "  interpreter: $PY (inside the repo, nothing is written outside it)"
    echo
    echo "  initialising submodules (this is the mandatory step a fresh clone always needs)"
    git submodule update --init --recursive
    git submodule status | sed 's/^/    /'
    echo
    echo "  toolchain present here:"
    printf "    nvcc   : %s\n" "$(command -v nvcc >/dev/null && nvcc --version | tail -2 | head -1 || echo 'not found')"
    printf "    driver : %s\n" "$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo 'no nvidia-smi')"
    printf "    python : %s\n" "$($PY -V 2>&1)"
    # Pure-python runtime deps Megatron imports. Installed under a torch constraint so a
    # resolver can never swap the pinned build out from under the driver. Every one of these was
    # a separate hard failure on a fresh clone: regex (tokenizer import), pybind11 (the runtime
    # helpers_cpp compile), the rest on first use.
    echo
    echo "  installing Megatron's pure-python runtime deps (torch pinned, never upgraded)"
    _constraint=$(mktemp -t torch-constraint.XXXXXX)
    printf 'torch==%s\n' "$("$PY" -c 'import torch;print(torch.__version__.split("+")[0])' 2>/dev/null || echo 2.4.1)" > "$_constraint"
    "$PY" -m pip install -q --constraint "$_constraint" \
        regex sentencepiece transformers tokenizers einops nltk tiktoken pybind11 || {
        echo "  WARNING: dependency install failed, training will not start"; }
    rm -f "$_constraint"; unset _constraint
    "$PY" -c 'import torch; print("    torch after install:", torch.__version__)'

    # Megatron compiles megatron/core/datasets/helpers_cpp at first run with a Makefile that
    # shells out to a bare `python3-config`. `python3 -m venv` does not provide one (virtualenv
    # does), so the compile fails with "make: python3-config: No such file or directory" and then
    # ModuleNotFoundError. Link the versioned one when it is missing.
    _pycfg_dir=$(dirname "$PY")
    if [ ! -e "$_pycfg_dir/python3-config" ]; then
        _real=$(command -v python3.11-config || command -v python3-config || true)
        if [ -n "$_real" ]; then
            ln -sf "$_real" "$_pycfg_dir/python3-config"
            echo "    linked python3-config -> $_real"
        else
            echo "    WARNING: no python3-config found; install python3.11-dev or helpers_cpp will not build"
        fi
    fi
    unset _pycfg_dir _real

    echo
    echo "  NOT DONE by this script, on purpose:"
    echo "    requirements.lock.txt is a record of a working environment, not an install target."
    echo "    TransformerEngine and apex were built from source against torch 2.4.1+cu124 and"
    echo "    CUDA 12.4. 'pip install -r' will not reproduce them, and a resolver that replaces"
    echo "    torch breaks the driver match. Build TE first, keep torch pinned, then install the"
    echo "    rest. See docs/ENVIRONMENT.md."
    echo
    echo "    For the overlap-architecture runs also apply:"
    echo "      git -C Megatron-LM apply ../overlap_arch/overlap_variants_megatron.patch"
    ;;

  serve)
    echo "'serve' is not implemented."
    echo
    echo "  The serving benchmarks build a llama.cpp CUDA fork at base commit 0badc06 for sm_86"
    echo "  (RTX A6000), plus a separate Apple-silicon MLX path. Both target hardware this script"
    echo "  cannot verify against, and neither is a wrapper over anything ENVIRONMENT.md already"
    echo "  automates, so a 'serve' mode here would be untested guesswork."
    echo "  Follow llamacpp-bench/README.md and mlx-bench/PLAN.md directly."
    exit 1
    ;;

  *) usage ;;
esac

echo
echo "=== $MODE setup done in $((SECONDS-start))s ==="
