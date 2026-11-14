set -x
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
python3 -m pip install -q --upgrade uv 2>/dev/null || pip install -q uv
export PATH=$HOME/.local/bin:$PATH
uv python install 3.12
rm -rf /workspace/venv_vllm312
uv venv --python 3.12 /workspace/venv_vllm312
VP=/workspace/venv_vllm312/bin/python
uv pip install --python $VP vllm==0.27.1
uv pip install --python $VP "transformers==5.12.1" ninja cmake hf_transfer
$VP -c "
import sys, torch, vllm, transformers, flashinfer
print('python', sys.version.split()[0])
print('torch', torch.__version__, '| vllm', vllm.__version__, '| transformers', transformers.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'x', torch.cuda.device_count())
import flashinfer.comm; print('flashinfer.comm import OK')
"
echo "VLLM312_DONE"
