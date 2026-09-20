set -x
cd /workspace/temporal-moe
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version | tail -2
P=/workspace/temporal-moe/.venv/bin/python
printf 'torch==2.4.1\n' > /workspace/c_train.txt
$P -m pip install -q -c /workspace/c_train.txt ninja cmake packaging setuptools wheel pybind11

echo "=== TransformerEngine (expect ~45 min) ==="
NVTE_FRAMEWORK=pytorch MAX_JOBS=$(nproc) \
  $P -m pip install --no-build-isolation -c /workspace/c_train.txt ./TransformerEngine
echo "TE_EXIT=$?"

echo "=== apex (expect ~20 min) ==="
MAX_JOBS=32 $P -m pip install --no-build-isolation -c /workspace/c_train.txt \
  --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex
echo "APEX_EXIT=$?"

$P - <<'PY'
import torch; print("torch", torch.__version__)
try:
    import transformer_engine, transformer_engine.pytorch
    print("transformer_engine", transformer_engine.__version__)
except Exception as e: print("TE import FAILED:", type(e).__name__, e)
try:
    import apex; from apex import optimizers; print("apex OK")
except Exception as e: print("apex import FAILED:", type(e).__name__, e)
PY
echo "TE_APEX_DONE"
