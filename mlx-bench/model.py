# Vendored Qwen3-MoE for the Mac MLX serving benchmark (see PLAN.md Section 2).
#
# Adapted from mlx-lm's models/qwen3_moe.py (Apple Inc., MIT). Stripped to the
# decode path we need: tied embeddings, norm_topk_prob (softmax over ALL experts
# -> top-k -> renorm), decoder_sparse_step=1 (every layer MoE), no shared expert,
# QK-norm, RoPE theta=1e6. The MoE block keeps mlx-lm's SwitchGLU / gather_qmm
# path. This file owns the model class; mlx-lm is used only as a parts bin for the
# stable low-level ops (KV cache, attention mask, SDPA, SwitchGLU, swiglu).
#
# `load(dir)` reads our local safetensors + config.json directly (no HF repo
# assumptions), quantizes the float skeleton to match the saved q4/q8 tensors,
# and loads them.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs:
    model_type: str = "qwen3_moe"
    hidden_size: int = 1024
    num_hidden_layers: int = 45
    intermediate_size: int = 384  # unused (all layers are MoE), kept for parity
    num_attention_heads: int = 8
    num_experts: int = 192
    num_experts_per_tok: int = 18
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    moe_intermediate_size: int = 384
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151669
    num_key_value_heads: int = 4
    head_dim: int = 128
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 4096
    norm_topk_prob: bool = True
    disk_experts: bool = False  # xl model: experts live on disk (see xl.py)

    @classmethod
    def from_dict(cls, params):
        import inspect

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        # disk_experts models keep experts on disk (xl.py attaches block.xl) and
        # never allocate the E-expert switch_mlp in RAM.
        if not args.disk_experts:
            self.switch_mlp = SwitchGLU(
                args.hidden_size, args.moe_intermediate_size, args.num_experts
            )

    def route(self, x):
        """HF Qwen3-MoE routing: softmax over ALL experts -> top-k -> renorm."""
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        return inds, scores

    def __call__(self, x):
        # Temporal mode (PLAN.md Phase 2) fires only on single-token decode
        # (L==1); prefill (L>1) always runs the full-MoE ceiling path. See
        # temporal.py.
        t = getattr(self, "temporal", None)
        if t is not None and x.shape[1] == 1:
            return t(x)
        xl = getattr(self, "xl", None)
        if xl is not None:  # disk-expert path (xl.py) handles prefill + decode
            return xl(x)
        inds, scores = self.route(x)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = Qwen3MoeSparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        # ROUTER-EARLY variant (temporal deploy_early, regime-2 disk tier only):
        # route + decide residency + ISSUE the expert fetch on the PRE-attention
        # input, run attention while the fetch is in flight, then run the experts
        # POST-attention with the pre-attention routing decision. See temporal.py
        # route_issue/expert_finish. A trained model would need to be trained
        # this way (routing on the pre-attention input selects different experts).
        t = getattr(self.mlp, "temporal", None)
        if t is not None and getattr(t.ctrl, "router_early", False) and x.shape[1] == 1:
            if t.ctrl.disk_fd is None:
                raise ValueError(
                    "router_early (deploy_early) requires the regime-2 disk tier "
                    "(set TEMPORAL_DISK_POOL); it has no RAM-tier / floor path.")
            # r_in: the MoE's own norm applied to the pre-attention residual --
            # our documented reading of "route on the pre-attention input".
            r_in = self.post_attention_layernorm(x)
            eff, scores, futures, nb = t.route_issue(r_in)
            h = x + self.self_attn(self.input_layernorm(x), mask, cache)
            mx.eval(h)                    # GPU runs attention while preads fly
            y = t.expert_finish(self.post_attention_layernorm(h),
                                eff, scores, futures, nb)
            return h + y
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3MoeDecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3MoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in range(self.args.num_hidden_layers)]


def load(model_dir):
    """Load a locally-generated q4 model directory (config.json + model.safetensors).

    disk_experts models load only nonexpert.safetensors (embeddings/attention/
    norms/router gates); their MoE blocks have no in-RAM switch_mlp and get an
    XLLayer attached later (xl.XLController). experts_flat.bin stays on disk."""
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    args = ModelArgs.from_dict(config)
    model = Model(args)

    st = "nonexpert.safetensors" if config.get("disk_experts") else "model.safetensors"
    weights = mx.load(str(model_dir / st))
    q = config["quantization"]

    def class_predicate(p, m):
        # Per-path override (e.g. the 8-bit router gate)
        if p in q and isinstance(q[p], dict):
            return q[p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights

    nn.quantize(
        model,
        group_size=q["group_size"],
        bits=q["bits"],
        mode=q.get("mode", "affine"),
        class_predicate=class_predicate,
    )

    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()
    return model, config
