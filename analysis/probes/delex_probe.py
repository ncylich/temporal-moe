#!/usr/bin/env python3
"""De-lexicalization capture probe (delex-1e19 Part 1). Megatron entry, same style as
activation_probe.py. Accumulates over the first N_MB micro-batches (env N_MB, default 8; at 1e19
mb=8 -> 64 sequences, the paper's fixed batch of 64x2048 = 131k tokens) and records, from the
trained model on that fixed batch:

  emb        [S, B, H]                  token input embeddings E[x_t] (decoder input; RoPE is in
                                        attention so this is the word embedding stream)
  per MoE layer L:
    logits   [S, B, E] fp16             raw router gating logits (-> gates=softmax, firing=top-k)
    mask     [S, B, E] bool | None      resident set (temporal cells only; firing there = mask&top-k)
    out_sum  [E, H] fp32, out_cnt [E]   sum & count of expert OUTPUT vectors over routed tokens
                                        (-> data-weighted logit lens v_e = out_sum/out_cnt)

Saved to $DELEX_OUT (default /tmp/delex_capture.pt). Weights/unembedding are read separately with
ckpt_read.py. TEMPORAL=1 installs the residency router (mask reflects the real served set).
Run via experiments/run.sh ACTPROBE-style (we reuse that scaffold with ENTRY overridden).
"""
import os, sys, torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from megatron.core.transformer.moe.router import TopKRouter

_TEMPORAL = os.environ.get("TEMPORAL", "0") == "1"
_EVICT = os.environ.get("TEMPORAL_EVICT", "min_logit")
N_MB = int(os.environ.get("N_MB", "8"))

R = {}                    # layer -> {"logits":[...], "mask":[...], "out_sum":Tensor, "out_cnt":Tensor}
EMB = []                  # list of [S,B,H] cpu fp16, one per micro-batch
_fwd = {"emb": 0}


_orig_router = TopKRouter.forward
def _router_forward(self, input):
    input = self.apply_input_jitter(input)
    logits = self.gating(input)
    k = int(self.config.moe_router_topk)
    ln = int(getattr(self, "layer_number", -1))
    d = R.setdefault(ln, {"logits": [], "mask": [], "k": k, "out_sum": None, "out_cnt": None})
    d["k"] = k
    mask = None
    if _TEMPORAL:
        from temporal.temporal_router import compute_resident_mask_accel
        with torch.no_grad():
            mask = compute_resident_mask_accel(logits, k, evict=_EVICT)
    if len(d["logits"]) < N_MB:
        d["logits"].append(logits.detach().to(torch.float16).cpu())
        d["mask"].append(mask.detach().cpu() if mask is not None else None)
    routed = logits.masked_fill(~mask, float("-inf")) if mask is not None else logits
    return self.routing(routed)
TopKRouter.forward = _router_forward


def _register(model):
    import re
    def layer_of(name):
        m = re.search(r"layers\.(\d+)\.", name); return int(m.group(1)) if m else -1
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        ln = layer_of(name)
        if name.endswith("mlp.experts") and ("GroupedMLP" in cls or cls == "SequentialMLP"):
            d = R.setdefault(ln, {"logits": [], "mask": [], "k": None, "out_sum": None, "out_cnt": None})
            st = {"tpe": None, "n": 0}
            def pre(m, args, st=st):
                st["tpe"] = args[1].detach().to(torch.int64).cpu()
            def post(m, args, out, d=d, st=st):
                if st["n"] >= N_MB:
                    return
                st["n"] += 1
                o = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()  # [T,H]
                tpe = st["tpe"]; E = tpe.numel(); H = o.shape[-1]
                if d["out_sum"] is None:
                    d["out_sum"] = torch.zeros(E, H); d["out_cnt"] = torch.zeros(E)
                b = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), tpe]), 0)
                for e in range(E):
                    a, c = int(b[e]), int(b[e + 1])
                    if c > a:
                        d["out_sum"][e] += o[a:c].sum(0); d["out_cnt"][e] += (c - a)
            mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)
        # token embeddings: capture the decoder input (== word-embedding stream)
        if name.endswith(".embedding") or cls == "LanguageModelEmbedding":
            def emb_hook(m, args, out):
                if _fwd["emb"] < N_MB:
                    _fwd["emb"] += 1
                    o = out[0] if isinstance(out, tuple) else out       # [S,B,H]
                    EMB.append(o.detach().to(torch.float16).cpu())
            mod.register_forward_hook(emb_hook)


def _wrap_provider():
    import pretrain_gpt
    orig = pretrain_gpt.model_provider
    def patched(*a, **k):
        model = orig(*a, **k)
        _register(model[0] if isinstance(model, list) else model)
        return model
    pretrain_gpt.model_provider = patched


def _dump():
    out = os.environ.get("DELEX_OUT", "/tmp/delex_capture.pt")
    layers = {}
    for ln, d in R.items():
        if not d["logits"]:
            continue
        layers[ln] = {
            "logits": torch.cat(d["logits"], dim=1),                       # [S, B_total, E]
            "mask": (torch.cat([m for m in d["mask"]], dim=1) if d["mask"][0] is not None else None),
            "k": d["k"], "out_sum": d["out_sum"], "out_cnt": d["out_cnt"],
        }
    emb = torch.cat(EMB, dim=1) if EMB else None                            # [S, B_total, H]
    torch.save({"temporal": _TEMPORAL, "evict": _EVICT, "n_mb": N_MB, "emb": emb, "layers": layers}, out)
    ex = layers[sorted(layers)[0]]
    print(f"[delex] saved {out}: {len(layers)} MoE layers, emb {tuple(emb.shape) if emb is not None else None}, "
          f"logits {tuple(ex['logits'].shape)}, temporal={_TEMPORAL}")


if __name__ == "__main__":
    # NOTE: do NOT call temporal_router.install() — it would overwrite our TopKRouter.forward patch.
    # _router_forward already replicates the residency routing via compute_resident_mask_accel
    # (same mask as training), exactly like router_probe.py.
    from megatron.training import pretrain
    from megatron.core.enums import ModelType
    import pretrain_gpt
    _wrap_provider()
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    try:
        pretrain(pretrain_gpt.train_valid_test_datasets_provider, pretrain_gpt.model_provider,
                 ModelType.encoder_or_decoder, pretrain_gpt.forward_step,
                 args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    finally:
        if R:
            _dump()
