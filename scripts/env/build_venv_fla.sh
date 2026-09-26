set -x
python3 -m venv /workspace/venv_fla
P=/workspace/venv_fla/bin/python
$P -m pip install -q --upgrade pip
# recorded env: torch 2.13.0 (needed for torch._grouped_mm / unsloth fused MoE).
# cu129 build instead of the recorded cu130: this pod's driver 570.195.03 caps at CUDA 12.8,
# and cu12x wheels run under CUDA minor-version compatibility. cu130 would need driver >=580.
$P -m pip install torch==2.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
$P - <<'PY'
import torch
print("torch", torch.__version__, "| cuda_avail", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
a=torch.randn(512,512,device="cuda",dtype=torch.bfloat16); b=torch.randn(512,512,device="cuda",dtype=torch.bfloat16)
print("matmul ok", (a@b).float().sum().isfinite().item())
print("has _grouped_mm:", hasattr(torch,"_grouped_mm"))
PY
echo "FLA_TORCH_DONE"
