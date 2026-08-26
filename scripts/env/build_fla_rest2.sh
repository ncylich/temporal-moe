set -x
P=/workspace/venv_fla/bin/python
C=/workspace/c_fla.txt
# 1. everything except unsloth, under the torch pin
$P -m pip install -c $C transformers==5.12.1 peft accelerate datasets trl \
   safetensors huggingface_hub hf_transfer sentencepiece protobuf tyro
# 2. unsloth itself: its metadata caps torch<2.12, but the recorded environment ran
#    unsloth 2026.8.4 on torch 2.13.0 (results/ablations/unsloth_parity.md). Install without
#    dependency resolution so the pin holds, exactly as that environment must have been built.
$P -m pip install --no-deps unsloth==2026.8.4 unsloth_zoo==2026.8.3
$P - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
import unsloth_zoo, unsloth
print("unsloth", unsloth.__version__, "| zoo", unsloth_zoo.__version__)
from unsloth_zoo.temporary_patches import qwen3_moe as qm
names=[n for n in dir(qm) if "moe" in n.lower()]
print("zoo qwen3_moe factory present:", "_make_qwen_moe_sparse_moe_block_forward" in dir(qm), names[:4])
PY
echo "FLA2_DONE"
