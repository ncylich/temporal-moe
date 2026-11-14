set -x
cd /workspace/temporal-moe
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDNN_PATH=/usr
export CUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu
export CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu:$CPLUS_INCLUDE_PATH
P=/workspace/temporal-moe/.venv/bin/python
NVTE_FRAMEWORK=pytorch MAX_JOBS=$(nproc) \
  $P -m pip install --no-build-isolation -c /workspace/c_train.txt ./TransformerEngine
echo "TE_EXIT=$?"
$P -c "
import torch; print('torch', torch.__version__)
import transformer_engine, transformer_engine.pytorch
print('transformer_engine', transformer_engine.__version__)
import apex; print('apex OK')
"
echo "TE_DONE"
