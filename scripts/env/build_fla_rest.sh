set -x
P=/workspace/venv_fla/bin/python
printf 'torch==2.13.0\ntorchvision==0.28.0\ntorchaudio==2.11.0\n' > /workspace/c_fla.txt
$P -m pip install -c /workspace/c_fla.txt \
  transformers==5.12.1 unsloth==2026.8.4 unsloth_zoo==2026.8.3 \
  peft accelerate datasets trl safetensors huggingface_hub hf_transfer sentencepiece
$P - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), "| _grouped_mm", hasattr(torch,"_grouped_mm"))
import unsloth_zoo, unsloth
print("unsloth", unsloth.__version__, "| zoo", unsloth_zoo.__version__)
from transformers import Qwen3MoeForCausalLM
print("Qwen3MoeForCausalLM import OK")
from unsloth_zoo.temporary_patches import qwen3_moe as qm
print("zoo qwen3_moe patch module OK:", [n for n in dir(qm) if "sparse_moe" in n][:3])
PY
echo "FLA_DONE"
