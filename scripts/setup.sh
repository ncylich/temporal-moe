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
    echo "  initialising submodules (this is the mandatory step a fresh clone always needs)"
    git submodule update --init --recursive
    git submodule status | sed 's/^/    /'
    echo
    echo "  toolchain present here:"
    printf "    nvcc   : %s\n" "$(command -v nvcc >/dev/null && nvcc --version | tail -2 | head -1 || echo 'not found')"
    printf "    driver : %s\n" "$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo 'no nvidia-smi')"
    printf "    python : %s\n" "$($PY -V 2>&1)"
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
