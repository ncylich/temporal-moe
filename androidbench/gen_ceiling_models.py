#!/usr/bin/env python3
"""Conservative same-shape ceiling models, sized to fit fully resident even at the worst
tested context depth (4096), leaving a large margin so background-app memory creep
(confirmed today: e112 swapped 510MB at ctx=1024 despite fitting fine overnight) doesn't
void the run. Since fully-resident decode is empirically E-independent on this device
(measured: E=18 dense and E=112 sparse read the same speed -- decode only touches the K
active experts), shrinking E from 112/141 to 80/100 does not compromise what's being
measured, it only buys headroom.

e80:  fine shape  (K=18, ff=384) -- was e112
e100n: k24 shape   (K=24, ff=288) -- was e141n
"""
import os, sys, torch

sys.path.insert(0, os.path.expanduser("~/Documents/temporal-moe/llamacpp-bench"))
from gen_random_qwen3moe import HIDDEN, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM, TOKENIZER_ID
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, AutoTokenizer

VARIANTS = {
    "e80":   dict(num_experts=80,  top_k=18, moe_ff=384),
    "e100n": dict(num_experts=100, top_k=24, moe_ff=288),
}


def build(variant, out_dir):
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
    print(f"[{variant}] E={v['num_experts']} top_k={v['top_k']} moe_ff={v['moe_ff']}", flush=True)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float16)
    model = Qwen3MoeForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)
    assert next(model.parameters()).dtype == torch.float16
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(out_dir)
    print(f"[{variant}] saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    for v in ("e80", "e100n"):
        build(v, f"./models/qwen3moe-rand-{v}")
