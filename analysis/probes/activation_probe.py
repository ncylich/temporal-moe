#!/usr/bin/env python3
"""PART C probe entry: one forward pass over the fixed eval batch, capturing AGGREGATE activation
stats (no raw tensors). Mirrors router_probe.py's invocation style (Megatron entry, TEMPORAL env
applies the resident routing for temporal cells).

Captures on the FIRST micro-batch forward:
  per MoE layer / expert (stability_activations.csv):
    out_l2_mean   mean L2 of expert output over its routed tokens
    sel_count     # tokens routed to the expert
    interm_max    max |FFN intermediate| (post-fc1) over routed tokens
    interm_kurt   excess kurtosis of FFN intermediate entries
    rlogit_mean/std          router logit for this expert (over all tokens)
    rlogit_mean_res/nonres   router logit split by resident vs non-resident (temporal only; else NaN)
    gate_mean/std            applied gate value when the expert is selected (0 if never)
  trunk (stability_trunk.csv), per layer:
    attn_out_in_ratio, mlp_out_in_ratio   block output/input L2 ratio
    per head: attn_max_logit              max pre-softmax QK logit (causal, post-RoPE), head rows
  plus one residual-stream row (layer=-1): top-64 |activation| dims + overall mean/max at final layer.

Dump: $ACT_LOG_OUT (default /tmp/act_log.pt).  Convert to CSVs with stability_activations_to_csv.py.
"""
import os, sys, torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from megatron.core.transformer.moe.router import TopKRouter

_TEMPORAL = os.environ.get("TEMPORAL", "0") == "1"
_EVICT = os.environ.get("TEMPORAL_EVICT", "min_logit")
S = {}          # captured aggregate stats, keyed structure below
_done = {"flag": False}


def _kurt(x):
    x = x.reshape(-1).to(torch.float64)
    if x.numel() < 2:
        return float("nan")
    m = x.mean(); d = x - m; v = (d * d).mean()
    return ((d ** 4).mean() / (v * v) - 3.0).item() if v > 0 else float("nan")


# ---- router: capture first forward per layer, reduce to per-expert scalars inline ----
_orig_router = TopKRouter.forward
def _router_forward(self, input):
    input = self.apply_input_jitter(input)
    logits = self.gating(input)                        # [S,B,E]
    k = int(self.config.moe_router_topk)
    ln = int(getattr(self, "layer_number", -1))
    if ln not in S.setdefault("router", {}):
        with torch.no_grad():
            lg = logits.detach().float()               # [S,B,E]
            E = lg.shape[-1]
            flat = lg.reshape(-1, E)                    # [N, E]
            mask = None
            if _TEMPORAL:
                from temporal.temporal_router import compute_resident_mask_accel
                mask = compute_resident_mask_accel(logits, k, evict=_EVICT)
                mflat = mask.reshape(-1, E)
            used = flat.masked_fill(~mflat, float("-inf")) if mask is not None else flat
            # selection + applied gate (softmax over the selected k), per token
            topv, topi = used.topk(k, dim=-1)          # [N,k]
            gates = torch.softmax(topv, dim=-1)         # [N,k]
            selmask = torch.zeros_like(flat, dtype=torch.bool).scatter_(-1, topi, True)
            gatefull = torch.zeros_like(flat).scatter_(-1, topi, gates)
            selc = selmask.sum(0)                       # [E]
            gsum = gatefull.sum(0); gsq = (gatefull * gatefull * selmask).sum(0)
            gmean = torch.where(selc > 0, gsum / selc.clamp(min=1), torch.zeros_like(gsum))
            gvar = torch.where(selc > 0, gsq / selc.clamp(min=1) - gmean * gmean, torch.zeros_like(gsum))
            rec = {"k": k, "E": E,
                   "rlogit_mean": flat.mean(0).cpu(), "rlogit_std": flat.std(0).cpu(),
                   "sel_count": selc.cpu(), "gate_mean": gmean.cpu(),
                   "gate_std": gvar.clamp(min=0).sqrt().cpu()}
            if mask is not None:
                mf = mflat.float()
                cnt_r = mf.sum(0); cnt_n = (1 - mf).sum(0)
                sum_r = (flat * mf).sum(0); sum_n = (flat * (1 - mf)).sum(0)
                rec["rlogit_mean_res"] = (sum_r / cnt_r.clamp(min=1)).cpu()
                rec["rlogit_mean_nonres"] = (sum_n / cnt_n.clamp(min=1)).cpu()
            S["router"][ln] = rec
        routed = logits.masked_fill(~mask, float("-inf")) if mask is not None else logits
        return self.routing(routed)
    return _orig_router(self, input)
TopKRouter.forward = _router_forward


def _register(model):
    import re
    def layer_of(name):
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else -1

    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        ln = layer_of(name)
        # expert grouped MLP: capture tokens_per_expert (input) + output; fc1 intermediate via child hook
        if name.endswith("mlp.experts") and ("GroupedMLP" in cls or cls == "SequentialMLP"):
            st = S.setdefault("experts", {}).setdefault(ln, {})
            def pre(m, args, st=st):
                if "tpe" not in st:
                    st["tpe"] = args[1].detach().to(torch.int64).cpu()
            def post(m, args, out, st=st):
                if "done" in st:
                    return
                st["done"] = True
                o = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()  # [T,H]
                tpe = st.get("tpe"); interm = st.pop("_interm", None)                   # [T, fc1_out]
                if interm is None and hasattr(m, "weight1"):
                    # Legacy GroupedMLP (--moe-use-legacy-grouped-gemm, the fast sync-free path)
                    # has no linear_fc1 child to hook, so recompute the pre-activation fc1
                    # output per expert from the permuted input and the packed weights, in the
                    # model dtype, exactly what TE's linear_fc1 hook returns on the other path.
                    with torch.no_grad():
                        x = args[0].detach()
                        E_ = tpe.numel()
                        w1 = m.weight1.view(E_, x.shape[-1], -1)
                        b_ = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), tpe]), 0)
                        parts = [x[int(b_[e]):int(b_[e + 1])] @ w1[e] for e in range(E_)]
                        interm = torch.cat(parts, 0).float().cpu()
                E = tpe.numel()
                bounds = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), tpe]), 0)
                rownorm = o.norm(dim=-1)                                                 # [T]
                out_l2 = torch.zeros(E); imax = torch.zeros(E); ikurt = torch.full((E,), float("nan"))
                for e in range(E):
                    a, b = int(bounds[e]), int(bounds[e + 1])
                    if b > a:
                        out_l2[e] = rownorm[a:b].mean()
                        if interm is not None:
                            seg = interm[a:b]
                            imax[e] = seg.abs().max()
                            ikurt[e] = _kurt(seg)
                st["out_l2_mean"] = out_l2; st["interm_max"] = imax; st["interm_kurt"] = ikurt
                st["sel_count"] = tpe.clone()
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)
        if name.endswith("mlp.experts.linear_fc1"):
            st = S.setdefault("experts", {}).setdefault(ln, {})
            def fc1(m, args, out, st=st):
                if "_interm" not in st and "done" not in st:
                    o = out[0] if isinstance(out, tuple) else out
                    st["_interm"] = o.detach().float().cpu()
            mod.register_forward_hook(fc1)
        # attention & mlp block I/O ratio
        if name.endswith("self_attention") and "Attention" in cls:
            st = S.setdefault("attn", {}).setdefault(ln, {})
            def apre(m, args, st=st):
                if "in" not in st:
                    st["in"] = float(args[0].detach().float().norm())
            def apost(m, args, out, st=st):
                if "out" not in st:
                    o = out[0] if isinstance(out, tuple) else out
                    st["out"] = float(o.detach().float().norm())
            mod.register_forward_pre_hook(apre); mod.register_forward_hook(apost)
        if (name.endswith(".mlp") and ln >= 0):
            st = S.setdefault("mlp", {}).setdefault(ln, {})
            def mpre(m, args, st=st):
                if "in" not in st:
                    st["in"] = float(args[0].detach().float().norm())
            def mpost(m, args, out, st=st):
                if "out" not in st:
                    o = out[0] if isinstance(out, tuple) else out
                    st["out"] = float(o.detach().float().norm())
            mod.register_forward_pre_hook(mpre); mod.register_forward_hook(mpost)
        # core attention: capture post-RoPE q,k for max logit (subsample batch)
        if name.endswith("core_attention"):
            st = S.setdefault("core", {}).setdefault(ln, {})
            def cpre(m, args, kwargs=None, st=st):
                if "done" in st:
                    return
                q, k = args[0].detach(), args[1].detach()   # [S, B, H, D]
                st["done"] = True
                with torch.no_grad():
                    q = q.float().cpu()[:, :2]; k = k.float().cpu()[:, :2]   # first 2 seqs, on CPU
                    Sl, B, H, D = q.shape
                    mx = torch.full((H,), float("-inf"))
                    causal = torch.triu(torch.ones(Sl, Sl, dtype=torch.bool), 1)
                    for b in range(B):
                        for h in range(H):
                            sc = (q[:, b, h] @ k[:, b, h].transpose(0, 1)) / (D ** 0.5)
                            sc = sc.masked_fill(causal, float("-inf"))
                            mx[h] = torch.maximum(mx[h], sc.max().cpu())
                    st["maxlogit"] = mx
            mod.register_forward_pre_hook(cpre, with_kwargs=True)
        # residual stream at final layernorm input
        if name.endswith("final_layernorm") or name.endswith("decoder.final_norm"):
            st = S.setdefault("resid", {})
            def rpre(m, args, st=st):
                if "x" not in st:
                    x = args[0].detach().float()               # [S,B,H]
                    st["x_absmean_perdim"] = x.abs().mean((0, 1)).cpu()
            mod.register_forward_pre_hook(rpre)


def _wrap_provider():
    import pretrain_gpt
    orig = pretrain_gpt.model_provider
    def patched(*a, **k):
        model = orig(*a, **k)
        m = model[0] if isinstance(model, list) else model
        _register(m)
        return model
    pretrain_gpt.model_provider = patched


def _dump():
    out = os.environ.get("ACT_LOG_OUT", "/tmp/act_log.pt")
    torch.save({"temporal": _TEMPORAL, "evict": _EVICT, "stats": S}, out)
    print(f"[actprobe] saved {out}: MoE layers={sorted(S.get('experts', {}))}, "
          f"attn layers={sorted(S.get('attn', {}))}, temporal={_TEMPORAL}")


if __name__ == "__main__":
    from megatron.training import pretrain
    from megatron.core.enums import ModelType
    import pretrain_gpt
    _wrap_provider()
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    try:
        pretrain(
            pretrain_gpt.train_valid_test_datasets_provider,
            pretrain_gpt.model_provider,
            ModelType.encoder_or_decoder,
            pretrain_gpt.forward_step,
            args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        )
    finally:
        if S:
            _dump()
