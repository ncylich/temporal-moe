#!/usr/bin/env python3
"""Step 2: plain-PyTorch reference forward for the extracted Temporal MoE (no Megatron import).

Architecture (read off the checkpoint shapes and the Megatron source, all bias-free):
  x = embed[tok]
  per layer:  x += proj(attn(rmsnorm(x)));  x += ffn(rmsnorm(x))
  attention:  fused qkv, per-head interleaved rows [q_h | k_h | v_h]; RoPE on q,k with
              half-split pairing (dim i with i + head_dim/2), base 10000; scale 1/sqrt(head_dim)
  layer 0:    dense SwiGLU, fc1 rows = [gate | up], out = fc2(silu(gate) * up)
  layers 1+:  shared SwiGLU expert (ungated, added) + routed experts
              logits = router(x); residents = rolling-residency mask (temporal_router);
              gates = softmax over the resident logits; out = sum_e gate_e * expert_e(x)
  logits = output_layer(rmsnorm(x))

Routing modes:
  temporal  training-time rule, compute_resident_mask(k, evict) per sequence (the scored model)
  topk      unconstrained top-k per token, gates = softmax over the selected k. This is what
            stock llama.cpp computes with softmax gating and expert_weights_norm=true.

    python scripts/serving_convert/ref_model.py --weights $WORK/fine_1e19.safetensors \
        --tokens $WORK/test_tokens.npz --n 512 --routing temporal
"""
import argparse, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from temporal.temporal_router import compute_resident_mask  # noqa: E402

BPB_DIVISOR = 2.9780   # ln2 * 4.296 bytes/token, pythia 50k on dclm (docs/EVALUATION_METHODOLOGY.md)


def rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def swiglu(x, fc1, fc2):
    g, u = (x @ fc1.T).chunk(2, dim=-1)
    return (F.silu(g) * u) @ fc2.T


class _AsDouble(dict):
    """Weights upcast to float64 on access (keeps memory at the float32 footprint)."""

    def __getitem__(self, k):
        return super().__getitem__(k).double()


class RefModel:
    def __init__(self, path, topk, head_dim=16, evict="min_logit", device="cpu"):
        sd = {k: v.float().to(device) for k, v in load_file(path).items()}
        self.sd = sd
        self.H = sd["embedding.word_embeddings.weight"].shape[1]
        self.L = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("decoder.layers."))
        self.hd, self.nh = head_dim, self.H // head_dim
        self.topk, self.evict, self.device = topk, evict, device
        self.E = sd["decoder.layers.1.mlp.router.weight"].shape[0]

    def attention(self, x, i):
        B, S, H = x.shape
        p = f"decoder.layers.{i}.self_attention."
        h = rmsnorm(x, self.sd[p + "linear_qkv.layer_norm_weight"])
        qkv = (h @ self.sd[p + "linear_qkv.weight"].T).view(B, S, self.nh, 3 * self.hd)
        q, k, v = (t.transpose(1, 2) for t in qkv.split(self.hd, dim=-1))      # [B, nh, S, hd]
        inv = 1.0 / (10000 ** (torch.arange(0, self.hd, 2, device=x.device, dtype=x.dtype) / self.hd))
        ang = torch.outer(torch.arange(S, device=x.device, dtype=x.dtype), inv)
        cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()

        def rope(t):
            t1, t2 = t.chunk(2, dim=-1)
            return t * cos + torch.cat([-t2, t1], -1) * sin

        o = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        return o.transpose(1, 2).reshape(B, S, H) @ self.sd[p + "linear_proj.weight"].T

    def route(self, logits, routing):
        """logits [B,S,E] -> (mask [B,S,E] bool, gates [B,S,E])."""
        if routing == "temporal":
            m = compute_resident_mask(logits.transpose(0, 1).cpu(), self.topk, self.evict)
            mask = m.transpose(0, 1).to(logits.device)
        elif routing == "topk":
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, logits.topk(self.topk, dim=-1).indices, True)
        else:
            raise ValueError(routing)
        return mask, torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)

    def moe(self, h, i, routing):
        B, S, H = h.shape
        p = f"decoder.layers.{i}.mlp."
        out = swiglu(h, self.sd[p + "shared_experts.linear_fc1.weight"],
                     self.sd[p + "shared_experts.linear_fc2.weight"])
        logits = h @ self.sd[p + "router.weight"].T
        mask, gates = self.route(logits, routing)
        flat, fout = h.reshape(B * S, H), out.reshape(B * S, H)
        mflat, gflat = mask.reshape(B * S, -1), gates.reshape(B * S, -1)
        fc1, fc2 = self.sd[p + "experts.experts.linear_fc1.weight"], self.sd[p + "experts.experts.linear_fc2.weight"]
        for e in range(self.E):
            idx = mflat[:, e].nonzero().squeeze(-1)
            if idx.numel():
                fout.index_add_(0, idx, swiglu(flat[idx], fc1[e], fc2[e]) * gflat[idx, e:e + 1])
        return fout.view(B, S, H), mask

    @torch.no_grad()
    def forward(self, tokens, routing="temporal", return_masks=False, double=False):
        if double:                       # float64 control for rounding-sensitivity checks
            keep, self.sd = self.sd, _AsDouble(self.sd)
            try:
                return self.forward(tokens, routing, return_masks)
            finally:
                self.sd = keep
        sd = self.sd
        x = sd["embedding.word_embeddings.weight"][tokens]
        masks = []
        for i in range(self.L):
            x = x + self.attention(x, i)
            p = f"decoder.layers.{i}."
            if i == 0:
                h = rmsnorm(x, sd[p + "mlp.linear_fc1.layer_norm_weight"])
                x = x + swiglu(h, sd[p + "mlp.linear_fc1.weight"], sd[p + "mlp.linear_fc2.weight"])
            else:
                o, m = self.moe(rmsnorm(x, sd[p + "pre_mlp_layernorm.weight"]), i, routing)
                x = x + o
                masks.append(m)
        logits = rmsnorm(x, sd["decoder.final_layernorm.weight"]) @ sd["output_layer.weight"].T
        return (logits, masks) if return_masks else logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokens", required=True, help="npz from test_tokens.py (tokens [N, 2049])")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=18)
    ap.add_argument("--evict", default="min_logit", help="as logged by the run's router banner")
    ap.add_argument("--routing", default="temporal", choices=["temporal", "topk"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-logits", help="write logits of the first sequence to this .npy")
    ap.add_argument("--save-losses", help="write per-token losses [n, seq] to this .npy")
    a = ap.parse_args()

    model = RefModel(a.weights, a.topk, evict=a.evict, device=a.device)
    toks = torch.from_numpy(np.load(a.tokens)["tokens"][a.start:a.start + a.n].astype(np.int64))
    losses, t0 = [], time.time()
    for b in range(0, len(toks), a.batch):
        chunk = toks[b:b + a.batch, :a.seq + 1].to(a.device)
        logits = model.forward(chunk[:, :-1], a.routing)
        if b == 0 and a.save_logits:
            np.save(a.save_logits, logits[0].cpu().numpy())
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1),
                             reduction="none").view(chunk.shape[0], -1)
        losses.append(ce.cpu())
        done = torch.cat(losses)
        print(f"[{len(done)}/{len(toks)}] loss {done.mean():.5f}  {time.time()-t0:.0f}s", flush=True)
    per = torch.cat(losses)
    if a.save_losses:
        np.save(a.save_losses, per.numpy())
    mean = per.mean().item()
    se = per.mean(1).std().item() / math.sqrt(len(per)) if len(per) > 1 else float("nan")
    print(f"routing={a.routing} evict={a.evict} k={a.topk} samples={len(per)} tokens={per.numel()}")
    print(f"loss {mean:.5f} nats/token (SE over samples {se:.5f})  "
          f"bpb {mean/BPB_DIVISOR:.5f} (SE {se/BPB_DIVISOR:.5f})")


if __name__ == "__main__":
    main()
