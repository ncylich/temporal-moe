#!/usr/bin/env python3
"""Deterministic (seed 0) random-weight q4 Qwen3-MoE builder (see PLAN.md Section 2).

RAM constraint (24 GB Mac): never materialize the ~21 GB fp16 model. Each tensor
is generated in fp16 normal(0, 0.02), immediately quantized (affine q4 g64, or q8
for the router gate to match mlx-lm), evaluated, and the fp16 freed before moving
on. Norms stay fp16 (set to ones, the RMSNorm default). Saves model.safetensors +
an mlx-lm-style config.json under models/qwen3moe-rand-{fine,coarse}-q4/.

Usage: python gen_random_qwen3moe_mlx.py {fine,coarse,both}
"""
import gc
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent

SEED = 0
GROUP = 64
BITS = 4
GATE_BITS = 8  # router gate quantized at 8-bit, matching mlx-lm's qwen3_moe
STD = 0.02
VOCAB = 151669  # len(AutoTokenizer("Qwen/Qwen3-0.6B")); offline constant

# Shared config (Section 2)
BASE = dict(
    model_type="qwen3_moe",
    hidden_size=1024,
    num_hidden_layers=45,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=128,
    vocab_size=VOCAB,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    rope_theta=1e6,
    tie_word_embeddings=True,
    norm_topk_prob=True,
    decoder_sparse_step=1,
    mlp_only_layers=[],
)

VARIANTS = {
    "fine": dict(num_experts=192, num_experts_per_tok=18, moe_intermediate_size=384),
    "coarse": dict(num_experts=64, num_experts_per_tok=6, moe_intermediate_size=1152),
}


def rss_gib():
    # macOS ru_maxrss is in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def quant(weights, key, shape, bits):
    """Generate fp16 normal(0, STD) of `shape`, quantize, store, free the fp16."""
    w = (mx.random.normal(shape) * STD).astype(mx.float16)
    wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=bits, mode="affine")
    mx.eval(wq, sc, bi)
    weights[f"{key}.weight"] = wq
    weights[f"{key}.scales"] = sc
    weights[f"{key}.biases"] = bi
    del w


def norm(weights, key, dim):
    weights[f"{key}.weight"] = mx.ones((dim,), dtype=mx.float16)


def build(variant):
    cfg = {**BASE, **VARIANTS[variant]}
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    nh, nkv, hd = (
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
    )
    E, ff = cfg["num_experts"], cfg["moe_intermediate_size"]

    out_dir = ROOT / "models" / f"qwen3moe-rand-{variant}-q4"
    out_dir.mkdir(parents=True, exist_ok=True)

    mx.random.seed(SEED)
    t0 = time.perf_counter()
    weights = {}

    # embeddings (tied -> also the lm_head)
    quant(weights, "model.embed_tokens", (cfg["vocab_size"], H), BITS)

    gate_overrides = {}
    for li in range(L):
        p = f"model.layers.{li}"
        # attention projections
        quant(weights, f"{p}.self_attn.q_proj", (nh * hd, H), BITS)
        quant(weights, f"{p}.self_attn.k_proj", (nkv * hd, H), BITS)
        quant(weights, f"{p}.self_attn.v_proj", (nkv * hd, H), BITS)
        quant(weights, f"{p}.self_attn.o_proj", (H, nh * hd), BITS)
        norm(weights, f"{p}.self_attn.q_norm", hd)
        norm(weights, f"{p}.self_attn.k_norm", hd)
        # layer norms
        norm(weights, f"{p}.input_layernorm", H)
        norm(weights, f"{p}.post_attention_layernorm", H)
        # router gate (q8) + expert stacks (q4, [E, out, in])
        quant(weights, f"{p}.mlp.gate", (E, H), GATE_BITS)
        gate_overrides[f"{p}.mlp.gate"] = {"group_size": GROUP, "bits": GATE_BITS}
        quant(weights, f"{p}.mlp.switch_mlp.gate_proj", (E, ff, H), BITS)
        quant(weights, f"{p}.mlp.switch_mlp.up_proj", (E, ff, H), BITS)
        quant(weights, f"{p}.mlp.switch_mlp.down_proj", (E, H, ff), BITS)

        if li % 10 == 0 or li == L - 1:
            gc.collect()
            print(
                f"  [{variant}] layer {li + 1}/{L}  RSS={rss_gib():.2f} GiB", flush=True
            )

    norm(weights, "model.norm", H)

    # ---- measured per-expert bytes (feeds Phase 3) ----
    ebytes = 0
    for n in ["gate_proj", "up_proj", "down_proj"]:
        for suf in ["weight", "scales", "biases"]:
            ebytes += weights[f"model.layers.0.mlp.switch_mlp.{n}.{suf}"].nbytes
    per_expert = ebytes // E

    total = sum(a.nbytes for a in weights.values())

    # config.json (mlx-lm-style; loader in model.py reads it directly)
    config = dict(cfg)
    config["intermediate_size"] = ff  # unused; kept for ModelArgs parity
    config["quantization"] = {"group_size": GROUP, "bits": BITS, **gate_overrides}
    config["_build"] = {
        "seed": SEED,
        "std": STD,
        "per_expert_bytes": per_expert,
        "total_quant_bytes": total,
        "note": "random weights; vocab from Qwen/Qwen3-0.6B tokenizer (offline constant)",
    }

    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    dt = time.perf_counter() - t0
    print(
        f"[{variant}] E={E} k={cfg['num_experts_per_tok']} ff={ff}  "
        f"total={total / 1e9:.3f} GB  per_expert={per_expert} B  "
        f"peakRSS={rss_gib():.2f} GiB  build={dt:.1f}s  -> {out_dir}",
        flush=True,
    )
    del weights
    gc.collect()
    return dict(
        variant=variant,
        total_gb=total / 1e9,
        per_expert_bytes=per_expert,
        build_s=dt,
        peak_rss_gib=rss_gib(),
    )


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    variants = ["fine", "coarse"] if which == "both" else [which]
    results = [build(v) for v in variants]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
