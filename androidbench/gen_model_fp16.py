#!/usr/bin/env python3
"""Generate the random-weight Qwen3-MoE HF checkpoint directly in fp16.

Same recipe as llamacpp-bench/gen_random_qwen3moe.py (identical config, seed 0), but
constructs the model with the default dtype already set to fp16 instead of building in
fp32 and casting afterwards. That halves peak host RAM (~42 GB -> ~21 GB), which is the
difference between fitting and not fitting on a 24 GB Mac.

DEVIATION TO RECORD: random init happens in fp16 rather than fp32-then-cast, so the
weight VALUES differ from the A6000 checkpoint. The architecture, parameter count, and
quantization are identical. This is already the situation between the A6000 (torch) and
Mac (MLX) generators, which also produce different random weights for the same shape --
weights are irrelevant to a latency benchmark, correctness is checked via perplexity.
"""
import argparse, os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llamacpp-bench"))
from gen_random_qwen3moe import (  # noqa: E402
    HIDDEN, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM, TOKENIZER_ID, VARIANTS,
)
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, AutoTokenizer  # noqa: E402


def build(variant: str, out_dir: str):
    v = VARIANTS[variant]
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    cfg = Qwen3MoeConfig(
        vocab_size=len(tok), hidden_size=HIDDEN, intermediate_size=HIDDEN * 4,
        moe_intermediate_size=v["moe_ff"], num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS, num_key_value_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        num_experts=v["num_experts"], num_experts_per_tok=v["top_k"],
        decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True,
        shared_expert_intermediate_size=0, max_position_embeddings=4096,
        rope_theta=1_000_000.0, tie_word_embeddings=True, torch_dtype="float16",
    )
    print(f"[{variant}] vocab={len(tok)} E={v['num_experts']} top_k={v['top_k']} "
          f"moe_ff={v['moe_ff']} H={HIDDEN} L={N_LAYERS}", flush=True)

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float16)          # <- construct in fp16, never fp32
    model = Qwen3MoeForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)

    n = sum(p.numel() for p in model.parameters())
    print(f"[{variant}] total params = {n/1e9:.3f} B "
          f"(~{n*0.5625/1024**2:.0f} MiB @ Q4_K_M weights)", flush=True)
    assert next(model.parameters()).dtype == torch.float16, "model is not fp16"

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(out_dir)
    print(f"[{variant}] saved HF checkpoint -> {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    build(a.variant, os.path.join(a.out_root, f"qwen3moe-rand-{a.variant}"))
