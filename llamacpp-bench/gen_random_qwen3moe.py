#!/usr/bin/env python3
"""Generate the random-weight Qwen3-MoE GGUF benchmark models for the temporal-MoE
serving benchmark (llamacpp-bench). Closes the "models were local-only, never committed"
hole: this script deterministically reconstructs both benchmark models from the README recipe.

Two granularities of the SAME total size and active FLOPs (E * moe_ff is invariant, and
top_k * moe_ff is invariant), differing only in expert granularity:
  fine   = 192 experts, top-18, moe_ff 384   (~840 KiB / expert swap at Q4_K_M)
  coarse =  64 experts, top-6,  moe_ff 1152   (~2.5 MiB / expert swap at Q4_K_M)

Weights are random: a latency/VRAM benchmark only needs a valid kernel on a realistic
architecture (correctness is checked via llama-perplexity PPL, not text).

Output: an HF checkpoint dir per variant; convert + quantize to Q4_K_M is done by
build_models.sh (which calls llama.cpp convert_hf_to_gguf.py + llama-quantize).
"""
import argparse, os, torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, AutoTokenizer

# Shared backbone (identical across granularities so total params + active FLOPs match).
# Depth pinned from the original model's KV-cache slope in serving_benchmarks.csv (context sweep):
# ~89 KiB/token => n_layer * n_kv_heads * head_dim ~= 22.8k => L=45 at kv_heads=4, head_dim=128.
# hidden 1024 keeps total ~10.5B / all-resident VRAM ~7.4 GiB (matches the recorded 7672).
HIDDEN       = 1024
N_LAYERS     = 45
N_HEADS      = 8
N_KV_HEADS   = 4
HEAD_DIM     = 128
TOKENIZER_ID = "Qwen/Qwen3-0.6B"   # only tokenizer files are pulled; sets vocab_size

VARIANTS = {
    "fine":   dict(num_experts=192, top_k=18, moe_ff=384),
    "coarse": dict(num_experts=64,  top_k=6,  moe_ff=1152),
}


def build(variant: str, out_dir: str):
    v = VARIANTS[variant]
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    vocab = len(tok)
    cfg = Qwen3MoeConfig(
        vocab_size=vocab,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 4,      # dense FFN size (unused when all layers are MoE)
        moe_intermediate_size=v["moe_ff"],
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_experts=v["num_experts"],
        num_experts_per_tok=v["top_k"],
        decoder_sparse_step=1,             # every layer is MoE
        mlp_only_layers=[],
        norm_topk_prob=True,
        shared_expert_intermediate_size=0, # Qwen3-MoE has no shared expert
        max_position_embeddings=4096,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
        torch_dtype="float16",
    )
    print(f"[{variant}] vocab={vocab} E={v['num_experts']} top_k={v['top_k']} "
          f"moe_ff={v['moe_ff']} H={HIDDEN} L={N_LAYERS}")
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).to(torch.float16)
    n = sum(p.numel() for p in model.parameters())
    print(f"[{variant}] total params = {n/1e9:.3f} B  (~{n*0.5625/1024**2:.0f} MiB @ Q4_K_M weights)")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"[{variant}] saved HF checkpoint -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS) + ["both"], default="both")
    ap.add_argument("--out-root", default="/workspace/models")
    args = ap.parse_args()
    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    for var in variants:
        build(var, os.path.join(args.out_root, f"qwen3moe-rand-{var}"))
