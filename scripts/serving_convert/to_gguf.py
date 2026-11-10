#!/usr/bin/env python3
"""Step 3: write the extracted Temporal MoE as a GGUF for the engine's `glm4moe` architecture.

No engine changes are needed. Mapping, each point checked against src/models/glm4-moe.cpp and
build_moe_ffn in src/llama-graph.cpp of the fork:

  Megatron                                   GGUF (glm4moe)
  linear_qkv.layer_norm_weight               blk.i.attn_norm            (pre-attention RMSNorm)
  linear_qkv.weight, rows [q_h|k_h|v_h]/head blk.i.attn_q / attn_k / attn_v  (de-interleaved)
  linear_proj.weight                         blk.i.attn_output
  fc1.layer_norm_weight | pre_mlp_layernorm  blk.i.post_attention_norm  (the engine's pre-FFN norm)
  layer 0 fc1 [gate|up], fc2                 blk.0.ffn_gate / ffn_up / ffn_down
  router.weight                              blk.i.ffn_gate_inp
  (none)                                     blk.i.exp_probs_b = 0      (required tensor)
  experts fc1 [E, gate|up, H], fc2           blk.i.ffn_gate_exps / ffn_up_exps / ffn_down_exps
  shared_experts fc1 [gate|up], fc2          blk.i.ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp

  RoPE: the engine uses NEOX pairing for glm4moe (dim i with i + head_dim/2), same as Megatron's
  non-interleaved RoPE, so q/k need no permutation.
  Gating: softmax over all experts, then expert_weights_norm=true renormalises over the selected
  set. That equals Megatron's softmax over the resident (unmasked) logits.

Zero padding (exact, since silu(0)*0 = 0 and padded down-columns are zero):
  routed width  -> multiple of 32 (Q4_0 needs row length % 32 == 0 on ffn_down_exps)
  dense width   -> multiple of 32
  shared width  -> multiple of the padded routed width (the engine requires
                   n_ff_shexp = n_ff_exp * expert_shared_count), which is also a multiple of 32
  vocab         -> the checkpoint's embedding rows (50304); ids past the tokenizer are UNUSED

    python scripts/serving_convert/to_gguf.py --llama-cpp ~/Documents/llama.cpp-android \
        --weights $WORK/fine_1e19.safetensors --topk 18 --out $WORK/fine_1e19-f16.gguf
"""
import argparse, json, os, sys

import numpy as np
import torch
from safetensors.torch import load_file


def ceil_to(n, m):
    return -(-n // m) * m


def pad(t, dim, size):
    if t.shape[dim] == size:
        return t
    shape = list(t.shape)
    shape[dim] = size - t.shape[dim]
    return torch.cat([t, torch.zeros(shape, dtype=t.dtype)], dim=dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", required=True, help="engine checkout (for its gguf-py)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, required=True, help="18 for fine, 6 for coarse")
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--pad", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ftype", default="f16", choices=["f16", "f32"])
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-12b")
    ap.add_argument("--name", default="temporal-moe")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.expanduser(a.llama_cpp), "gguf-py"))
    import gguf
    from huggingface_hub import hf_hub_download

    sd = {k: v.float() for k, v in load_file(a.weights).items()}
    V, H = sd["embedding.word_embeddings.weight"].shape
    L = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("decoder.layers."))
    nh = H // a.head_dim
    E, ff2, _ = sd["decoder.layers.1.mlp.experts.experts.linear_fc1.weight"].shape
    ff, ffp = ff2 // 2, ceil_to(ff2 // 2, a.pad)
    dn = sd["decoder.layers.0.mlp.linear_fc2.weight"].shape[1]
    dnp = ceil_to(dn, a.pad)
    sh = sd["decoder.layers.1.mlp.shared_experts.linear_fc2.weight"].shape[1]
    shp = ceil_to(sh, ffp)
    print(f"H={H} L={L} heads={nh} V={V} E={E} topk={a.topk} "
          f"routed {ff}->{ffp} dense {dn}->{dnp} shared {sh}->{shp} ({shp // ffp} x routed)")

    w = gguf.GGUFWriter(a.out, "glm4moe")
    w.add_name(a.name)
    w.add_file_type(1 if a.ftype == "f16" else 0)
    w.add_context_length(a.ctx)
    w.add_embedding_length(H)
    w.add_block_count(L)
    w.add_feed_forward_length(dnp)
    w.add_head_count(nh)
    w.add_head_count_kv(nh)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
    w.add_rope_dimension_count(a.head_dim)
    w.add_leading_dense_block_count(1)
    w.add_expert_count(E)
    w.add_expert_used_count(a.topk)
    w.add_expert_shared_count(shp // ffp)
    w.add_expert_feed_forward_length(ffp)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
    w.add_expert_weights_norm(True)
    w.add_vocab_size(V)

    # --- tokenizer: byte-level BPE, GPT-2 style pre-tokenizer ---
    tj = json.load(open(hf_hub_download(a.tokenizer, "tokenizer.json"), encoding="utf-8"))
    vocab = dict(tj["model"]["vocab"])
    added = {t["id"]: t for t in tj["added_tokens"]}
    for t in added.values():
        vocab[t["content"]] = t["id"]
    by_id = {i: s for s, i in vocab.items()}
    assert len(by_id) == max(by_id) + 1 <= V, (len(by_id), V)
    tokens, types = [], []
    for i in range(V):
        if i in added:
            tokens.append(added[i]["content"])
            types.append(gguf.TokenType.CONTROL if added[i]["special"] else gguf.TokenType.USER_DEFINED)
        elif i in by_id:
            tokens.append(by_id[i]); types.append(gguf.TokenType.NORMAL)
        else:
            tokens.append(f"<|unused{i}|>"); types.append(gguf.TokenType.UNUSED)
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    eos = vocab["<|endoftext|>"]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(eos)
    w.add_eos_token_id(eos)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    # --- tensors ---
    wt = np.float16 if a.ftype == "f16" else np.float32

    def add(name, t):
        arr = t.contiguous().numpy()
        w.add_tensor(name, arr.astype(np.float32 if arr.ndim == 1 else wt))

    def add_swiglu(fc1, fc2, width, gate, up, down):
        g, u = fc1.chunk(2, dim=-2)                      # rows [gate | up]
        add(gate, pad(g, -2, width)); add(up, pad(u, -2, width)); add(down, pad(fc2, -1, width))

    add("token_embd.weight", sd["embedding.word_embeddings.weight"])
    add("output_norm.weight", sd["decoder.final_layernorm.weight"])
    add("output.weight", sd["output_layer.weight"])
    for i in range(L):
        p, b = f"decoder.layers.{i}.", f"blk.{i}."
        add(b + "attn_norm.weight", sd[p + "self_attention.linear_qkv.layer_norm_weight"])
        qkv = sd[p + "self_attention.linear_qkv.weight"].view(nh, 3, a.head_dim, H)
        for j, n in enumerate("qkv"):
            add(b + f"attn_{n}.weight", qkv[:, j].reshape(nh * a.head_dim, H))
        add(b + "attn_output.weight", sd[p + "self_attention.linear_proj.weight"])
        if i == 0:
            add(b + "post_attention_norm.weight", sd[p + "mlp.linear_fc1.layer_norm_weight"])
            add_swiglu(sd[p + "mlp.linear_fc1.weight"], sd[p + "mlp.linear_fc2.weight"], dnp,
                       b + "ffn_gate.weight", b + "ffn_up.weight", b + "ffn_down.weight")
            continue
        add(b + "post_attention_norm.weight", sd[p + "pre_mlp_layernorm.weight"])
        add(b + "ffn_gate_inp.weight", sd[p + "mlp.router.weight"])
        add(b + "exp_probs_b.bias", torch.zeros(E))
        add_swiglu(sd[p + "mlp.experts.experts.linear_fc1.weight"],
                   sd[p + "mlp.experts.experts.linear_fc2.weight"], ffp,
                   b + "ffn_gate_exps.weight", b + "ffn_up_exps.weight", b + "ffn_down_exps.weight")
        add_swiglu(sd[p + "mlp.shared_experts.linear_fc1.weight"],
                   sd[p + "mlp.shared_experts.linear_fc2.weight"], shp,
                   b + "ffn_gate_shexp.weight", b + "ffn_up_shexp.weight", b + "ffn_down_shexp.weight")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e9:.3f} GB)")


if __name__ == "__main__":
    main()
