set -x
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
P=/workspace/venv_fla/bin/python
$P -m pip uninstall -y torch torchvision torchaudio
$P -m pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
$P - <<'PY'
import unsloth, torch, transformers
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("transformers", transformers.__version__, "| unsloth", unsloth.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), "x", torch.cuda.device_count())
print("_grouped_mm", hasattr(torch,"_grouped_mm"))
a=torch.randn(1024,1024,device="cuda",dtype=torch.bfloat16); print("matmul finite", (a@a).float().sum().isfinite().item())
PY
echo "FLA_CU130B_DONE"
