#!/usr/bin/env python3
"""Expert output similarity: how alike are a layer's experts as functions, and does the answer
explain the substitution-tolerance result?

Runs in-process on one checkpoint (same startup as substitution_eval.py), caches one test
micro-batch, samples N token positions, and for every MoE layer evaluates EVERY routed expert on
the same sampled inputs: y_e(x) = W2_e * glu(W1_e x), ungated, no shared expert. From those it
records, per layer:

  all-pairs      mean over expert pairs and tokens of cos(y_e, y_f) and of ||y_e - y_f|| / ||y_f||
  within-topk    the same over the k experts the router actually selected for that token
                 (native regime: residency mask applied, so a temporal model's selection is what
                 it was trained with)
  router-facing  cos and relative error between each selected expert and the substitute the
                 substitution experiment would draw: next-best unselected by raw logit, a random
                 unselected expert, and the previous token's expert that is not selected now
  layer-level    relative change of the gated layer output when ONE selected expert's output is
                 replaced by the substitute's at the displaced expert's gate (pure function
                 swap, gate rule held fixed), for the same three substitutes
  weights        mean pairwise cosine of the flattened expert weights, the paper's weight metric

A random-init checkpoint of the same shape (no --load match) calibrates the "near zero" level.
The predictions this is meant to test: (1) temporal experts are closer in function than full-MoE
experts at matched budget and seed, (2) the gap closes at 1e19 as the substitution gap did, and
(3) the depth pattern matches: no advantage or a reversal in the first MoE layers at 1e19.

    EXPSIM_N=2048 EXPSIM_OUT=results/ablations/expert_similarity/<run>.npz \
        $PY analysis/probes/expert_similarity.py <megatron args>
"""
import hashlib
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

_CACHE = {"batch": None}
_REC = {}          # layer_number -> dict(x=[S,B,H], raw=[S,B,E], mask=[S,B,E], sel=[S,B,E], gates=[S,B,E])
_INPUT = {}        # layer_number -> router input (from the module forward hook)


def _post_hook(router, raw_logits, mask, probs, routing_map):
    ln = router.layer_number
    x = _INPUT.get(ln)
    if x is not None and ln not in _REC:
        S, B, E = raw_logits.shape
        _REC[ln] = {"x": x.detach(), "raw": raw_logits.detach().float(),
                    "mask": (mask if mask is not None else torch.ones_like(routing_map)).reshape(S, B, E).detach(),
                    "sel": routing_map.reshape(S, B, E).detach(), "gates": probs.reshape(S, B, E).detach().float()}
    return probs, routing_map


def _cos_pairs(y):
    """y: [E, N, H] -> per-token mean over pairs e<f of cosine, averaged over tokens; and relerr."""
    E, N, H = y.shape
    yn = torch.nn.functional.normalize(y.float(), dim=-1)                    # [E,N,H]
    cos_sum = torch.zeros((), device=y.device); rel_sum = torch.zeros((), device=y.device); n = 0
    norms = y.float().norm(dim=-1)                                            # [E,N]
    for e in range(E):
        c = (yn[e].unsqueeze(0) * yn[e + 1:]).sum(-1)                        # [E-e-1, N]
        d = (y[e].float().unsqueeze(0) - y[e + 1:].float()).norm(dim=-1) / norms[e + 1:].clamp(min=1e-6)
        cos_sum += c.sum(); rel_sum += d.sum(); n += c.numel()
    return float(cos_sum / max(n, 1)), float(rel_sum / max(n, 1))


def _analyse(layer, rec, N, seed, k):
    import numpy as np
    dev = rec["raw"].device
    S, B, E = rec["raw"].shape
    H = rec["x"].shape[-1]
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    # sample token positions with a predecessor in the same sequence (s >= 1)
    flat = torch.arange(S * B)
    cand = flat[(flat // B) >= 1]
    pos = cand[torch.randperm(len(cand), generator=g)[:N]].sort().values.to(dev)
    prev = pos - B
    x = rec["x"].reshape(S * B, H)[pos]                                       # [N,H]
    raw = rec["raw"].reshape(S * B, E)[pos]
    sel = rec["sel"].reshape(S * B, E)[pos]
    gates = rec["gates"].reshape(S * B, E)[pos]
    sel_prev = rec["sel"].reshape(S * B, E)[prev]
    experts = layer.mlp.experts
    w1 = experts.weight1.view(E, H, -1)
    w2 = experts.weight2.view(E, -1, H)
    with torch.no_grad():
        h = torch.einsum("nh,ehf->enf", x.to(w1.dtype), w1)                # [E,N,2F]
        h = experts.activation_func(h)
        y = torch.einsum("enf,efh->enh", h, w2)                              # [E,N,H]
        y32 = y.float()
        out = {}
        out["cos_all"], out["relerr_all"] = _cos_pairs(y)
        # within top-k and router-facing
        yn = torch.nn.functional.normalize(y32, dim=-1)
        norms = y32.norm(dim=-1)                                              # [E,N]
        gen = torch.Generator(device=dev); gen.manual_seed(seed + 1)
        r = torch.rand(N, E, device=dev, generator=gen)
        nextbest = raw.masked_fill(sel, float("-inf")).argmax(1)             # [N]
        random_u = r.masked_fill(sel, -1.0).argmax(1)
        stale_c = sel_prev & ~sel
        has_stale = stale_c.any(1)
        stale = r.masked_fill(~stale_c, -1.0).argmax(1)
        n_idx = torch.arange(N, device=dev)
        def sim(a_idx, b_idx, valid=None):
            c = (yn[a_idx, n_idx] * yn[b_idx, n_idx]).sum(-1)
            d = (y32[a_idx, n_idx] - y32[b_idx, n_idx]).norm(dim=-1) / norms[b_idx, n_idx].clamp(min=1e-6)
            if valid is not None:
                c, d = c[valid], d[valid]
            return float(c.mean()) if c.numel() else float("nan"), float(d.mean()) if d.numel() else float("nan")
        # one displaced expert per token: random among selected (as in the substitution experiment)
        disp = r.masked_fill(~sel, -1.0).argmax(1)
        out["cos_sel_nextbest"], out["relerr_sel_nextbest"] = sim(disp, nextbest)
        out["cos_sel_random"], out["relerr_sel_random"] = sim(disp, random_u)
        out["cos_sel_stale"], out["relerr_sel_stale"] = sim(disp, stale, has_stale)
        out["frac_has_stale"] = float(has_stale.float().mean())
        # within the selected set: mean pairwise cos over the k selected experts
        sel_idx = sel.float().topk(k, dim=1).indices                          # [N,k]
        ys = yn[sel_idx, n_idx.unsqueeze(1)]                                   # [N,k,H]
        gram = torch.einsum("nkh,nlh->nkl", ys, ys)
        iu = torch.triu_indices(k, k, 1)
        out["cos_within_topk"] = float(gram[:, iu[0], iu[1]].mean())
        # layer-level: gated output change from a pure function swap of the displaced expert
        gy = (gates.t().unsqueeze(-1) * y32).sum(0)                           # [N,H] = sum_e g_e y_e
        gnorm = gy.norm(dim=-1).clamp(min=1e-6)
        gd = gates[n_idx, disp].unsqueeze(-1)                                 # displaced gate
        for name, sub, valid in (("nextbest", nextbest, None), ("random", random_u, None), ("stale", stale, has_stale)):
            delta = (gd * (y32[sub, n_idx] - y32[disp, n_idx])).norm(dim=-1) / gnorm
            out[f"layer_relchange_{name}"] = float(delta[valid].mean()) if valid is not None else float(delta.mean())
        out["layer_relchange_zero"] = float(((gd * y32[disp, n_idx]).norm(dim=-1) / gnorm).mean())
        # weights: mean pairwise cosine of flattened expert weights (paper's metric)
        wf = torch.cat([w1.reshape(E, -1).float(), w2.reshape(E, -1).float()], 1)
        wn = torch.nn.functional.normalize(wf, dim=-1)
        wg = wn @ wn.t()
        out["cos_weights"] = float(wg[torch.triu(torch.ones(E, E, dtype=torch.bool, device=dev), 1)].mean())
        out["mean_gate_displaced"] = float(gd.mean())
        out["mean_out_norm"] = float(norms.mean())
    return out


def _install():
    import numpy as np
    import megatron.training.training as T
    from megatron.training import get_args
    from temporal import temporal_router

    orig = T.evaluate_and_print_results
    orig_load = T.load_checkpoint

    def lenient_load(*a, **kw):
        kw["strict"] = False
        return orig_load(*a, **kw)

    T.load_checkpoint = lenient_load

    def patched(prefix, forward_step_func, data_iterator, model, iteration,
                process_non_loss_data_func, config, verbose=False, write_to_tensorboard=True, **kw):
        if "test" not in str(prefix).lower():
            return orig(prefix, forward_step_func, data_iterator, model, iteration,
                        process_non_loss_data_func, config, verbose, write_to_tensorboard, **kw)
        args = get_args()
        it = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
        b = next(it)
        mdl = model[0] if isinstance(model, list) else model
        mdl.eval()
        # find MoE layers and hook their routers to capture the router input
        core = mdl
        while hasattr(core, "module"):
            core = core.module
        layers = {}
        for lyr in core.decoder.layers:
            if hasattr(lyr.mlp, "router"):
                ln = lyr.mlp.router.layer_number
                layers[ln] = lyr
                # pre-hook: the post-routing hook fires inside the router's forward, so the
                # input must already be recorded when it runs
                lyr.mlp.router.register_forward_pre_hook(
                    lambda m, inp, ln=ln: _INPUT.__setitem__(ln, inp[0]))
        temporal_router.POST_ROUTING_HOOK = _post_hook
        with torch.no_grad():
            forward_step_func(iter([b]), mdl)
        temporal_router.POST_ROUTING_HOOK = None
        toks = b["tokens"].cpu().numpy().astype(np.int32)
        N = int(os.environ.get("EXPSIM_N", "2048")); seed = int(os.environ.get("EXPSIM_SEED", "1234"))
        k = args.moe_router_topk
        res = {}
        for ln in sorted(_REC):
            res[ln] = _analyse(layers[ln], _REC[ln], N, seed, k)
            print(f"[expsim] L{ln:2d} cos_all={res[ln]['cos_all']:+.4f} relerr_all={res[ln]['relerr_all']:.3f} "
                  f"topk={res[ln]['cos_within_topk']:+.4f} nextbest={res[ln]['cos_sel_nextbest']:+.4f} "
                  f"random={res[ln]['cos_sel_random']:+.4f} stale={res[ln]['cos_sel_stale']:+.4f} "
                  f"lay_next={res[ln]['layer_relchange_nextbest']:.3f} lay_rand={res[ln]['layer_relchange_random']:.3f} "
                  f"lay_zero={res[ln]['layer_relchange_zero']:.3f} w={res[ln]['cos_weights']:+.4f}", flush=True)
        out = os.environ.get("EXPSIM_OUT")
        if out:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            keys = sorted({kk for v in res.values() for kk in v})
            np.savez_compressed(out, run=os.environ.get("RUN_NAME", "unknown"),
                                regime=os.environ.get("EXPSIM_REGIME", "unknown"), k=k,
                                E=args.num_experts, N=N, layers=np.array(sorted(res)), metrics=np.array(keys),
                                values=np.array([[res[ln][kk] for kk in keys] for ln in sorted(res)]),
                                tokens_sha256=hashlib.sha256(toks.tobytes()).hexdigest())
            print(f"[expsim] wrote {out}", flush=True)

    T.evaluate_and_print_results = patched


if __name__ == "__main__":
    from megatron.training import pretrain
    from megatron.core.enums import ModelType
    import pretrain_gpt
    from temporal import temporal_router

    temporal_router.install()
    _install()
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    pretrain(pretrain_gpt.train_valid_test_datasets_provider, pretrain_gpt.model_provider,
             ModelType.encoder_or_decoder, pretrain_gpt.forward_step,
             args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
