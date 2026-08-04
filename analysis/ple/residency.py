#!/usr/bin/env python3
"""Stage 0 mask port: impose FLAME rolling-residency (R = k = 8 of 64) on a loaded OLMoE-1B-7B-0125
via a module-swap on transformers' OLMoE routing — no transformers fork.

The residency scan reuses `temporal/temporal_router.py::compute_resident_mask` from the FLAME repo
VERBATIM (pure torch, Megatron-free). `OlmoeTopKRouter.forward` sees flattened [B*S, E] logits with
no sequence context, so `OlmoeSparseMoeBlock.forward` (which knows batch_size/sequence_length)
records the pack shape into a per-router attribute; the patched router reshapes to [S, B, E],
runs the scan, `masked_fill(~mask, -inf)`, then hands the masked logits to the ORIGINAL softmax/top-k.

Residency cold-fills at t=0 of each packed sequence (per-sequence scan), matching FLAME training.
enable_residency(model, R=8) / disable_residency(model) toggle at runtime; R = E reproduces base.
"""
import os, sys, types, torch
import torch.nn.functional as F

# reuse the FLAME scan verbatim; prefer the triton accel path (verified == torch reference in
# verify23_scan.py) for eval speed, fall back to the pure-torch reference.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from temporal.temporal_router import compute_resident_mask, compute_resident_mask_accel

from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter, OlmoeSparseMoeBlock

_CFG = {"on": False, "R": 8, "evict": "min_logit", "accel": True, "collect_aux": False,
        "collect_telem": False, "free_layers": 0, "free_set": None}
# free_layers: leave the first N MoE layers UNCONSTRAINED (ordinary free routing) while the rest run
# under rolling residency. This relaxes the constraint rather than adapting to it, so a cell using it
# is NOT comparable to a full-residency number without stating the cost: every expert of a freed
# layer must stay resident, which is 64 instead of 8 per layer.
# per-forward accumulators for masked-distribution aux + z-loss (router-only finetune)
AUX = {"aux": None, "z": None, "n": 0}
# telemetry accumulators (eval-time, R=8): swap-rate/layer + expert-usage entropy
TELEM = {"swaps": 0.0, "tokens": 0, "usage": None, "n_layers": 0}


def reset_telem():
    TELEM.update(swaps=0.0, tokens=0, usage=None, n_layers=0)


def telem_summary(E):
    """mean swaps/token/layer + expert-usage entropy (normalized by ln E) over accumulated evals."""
    swap_rate = TELEM["swaps"] / max(1, TELEM["tokens"])          # already per-layer averaged below
    u = TELEM["usage"]
    if u is None:
        return swap_rate, float("nan")
    p = u / u.sum().clamp(min=1)
    import math as _m
    ent = float(-(p * (p.clamp(min=1e-12)).log()).sum() / _m.log(E))
    return swap_rate, ent


def _accum_telem(mask):
    """mask [S,B,E] bool (resident/served set). swaps at t = experts newly resident vs t-1."""
    S, B, E = mask.shape
    added = (mask[1:] & ~mask[:-1]).sum(-1).float()               # [S-1,B]
    TELEM["swaps"] += float(added.sum()) / max(1, B)              # sum over time, per-sequence
    TELEM["tokens"] += (S - 1)                                    # per sequence (B folded above)
    usage = mask.float().sum((0, 1))                             # [E] firing counts (resident=served)
    TELEM["usage"] = usage if TELEM["usage"] is None else TELEM["usage"] + usage
    TELEM["n_layers"] += 1
_orig_block_forward = OlmoeSparseMoeBlock.forward
_orig_router_forward = OlmoeTopKRouter.forward


def reset_aux():
    AUX["aux"] = None; AUX["z"] = None; AUX["n"] = 0


def _accum_aux(used, E, R):
    """Switch load-balance aux + router z-loss on the MASKED distribution (differentiable in logits).
    used: [N, E] masked logits (-inf on non-resident). f_e = resident fraction; P_e = mean masked prob."""
    probs = torch.softmax(used, dim=-1)                 # [N,E], 0 on non-resident
    resident = torch.isfinite(used).float()             # [N,E], 1 on the R resident experts
    f = resident.mean(0)                                # [E] fraction of tokens each expert is resident
    P = probs.mean(0)                                   # [E] mean prob mass
    aux = E * (f * P).sum()
    z = (torch.logsumexp(used, dim=-1) ** 2).mean()     # logsumexp over resident (finite) logits
    AUX["aux"] = aux if AUX["aux"] is None else AUX["aux"] + aux
    AUX["z"] = z if AUX["z"] is None else AUX["z"] + z
    AUX["n"] += 1


def _block_forward(self, hidden_states):
    # record the pack shape so the router can reshape flattened logits to [S, B, E]
    b, s, _ = hidden_states.shape
    self.gate._resid_shape = (b, s)
    return _orig_block_forward(self, hidden_states)


def _router_forward(self, hidden_states):
    if not _CFG["on"]:
        return _orig_router_forward(self, hidden_states)
    _li = getattr(self, "_layer_idx", None)
    if _li is not None:
        _fs = _CFG.get("free_set")
        if _fs is not None:
            if _li in _fs:
                return _orig_router_forward(self, hidden_states)   # explicitly free
        elif _li < _CFG.get("free_layers", 0):
            return _orig_router_forward(self, hidden_states)       # first-N free
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)                 # [N, E]
    N, E = router_logits.shape
    R = _CFG["R"]
    b, s = getattr(self, "_resid_shape", (1, N))
    lg = router_logits.view(b, s, E).transpose(0, 1).contiguous()        # [S, B, E]
    forced = _CFG.get("forced")
    if forced is not None:                                               # O-2: use a precomputed schedule mask
        mask = forced[self._layer_idx].to(lg.device)                    # [S,B,E] bool for this layer
    else:
        with torch.no_grad():
            scan = compute_resident_mask_accel if (lg.is_cuda and _CFG.get("accel", True)) else compute_resident_mask
            mask = scan(lg.float(), R, evict=_CFG["evict"])              # [S,B,E] bool, R True/token
    if _CFG["collect_telem"]:
        _accum_telem(mask)                                       # swap-rate + usage (eval, R=8)
    mask_flat = mask.transpose(0, 1).reshape(N, E)
    used = router_logits.masked_fill(~mask_flat, float("-inf"))
    if _CFG["collect_aux"] and self.training:
        _accum_aux(used.float(), E, R)                  # aux+z on masked dist (keeps router grad)
    # existing softmax/top-k, unchanged
    router_probs = torch.nn.functional.softmax(used, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


def tag_layers(model):
    """Give each router its layer index so forced O-2 schedule masks can be routed per layer."""
    for i, m in enumerate(mod for mod in model.modules() if isinstance(mod, OlmoeTopKRouter)):
        m._layer_idx = i


def set_free_layers(indices):
    """Explicit set of layer indices to leave UNCONSTRAINED; None restores the free_layers rule.

    Generalises free_layers so any subset can be freed, which is what the per-layer damage ablation
    needs: constrain exactly one layer by freeing the other fifteen.
    """
    _CFG["free_set"] = set(indices) if indices is not None else None


def set_forced(masks):
    """masks: list of [S,B,E] bool per layer (or None to disable). Router serves exactly the True experts."""
    _CFG["forced"] = masks


def install_patch():
    OlmoeSparseMoeBlock.forward = _block_forward
    OlmoeTopKRouter.forward = _router_forward


def enable_residency(R=8, evict="min_logit", free_layers=None):
    _CFG.update(on=True, R=R, evict=evict)
    if free_layers is not None:
        _CFG["free_layers"] = free_layers


def assert_aux_live(out, aux, aux_c, _done=[]):
    """Fail loudly if the load-balancing term is not actually reaching the loss.

    Every way this can go wrong is silent. The run trains, the curve looks ordinary, and the router
    is simply unregularised:

      * `output_router_logits=True` not passed -> `out.router_logits` is None and there is nothing to
        compute an aux from. This is the flag that defaults to False in OlmoeForCausalLM.
      * `labels` passed to the model as well -> HF adds `router_aux_loss_coef * aux_loss` internally
        (modeling_olmoe.py, gated on `labels is not None`), on top of ours. Two load-balancing losses
        over different quantities: HF's over top-k selection, ours over residency. Not an error the
        loss curve would show.
      * `aux` detached from the router weights -> added to the loss, contributes no gradient.
      * the coefficient set to 0.

    Checked once per process, on the first step, because the failure is configuration and not data.
    """
    if _done:
        return
    _done.append(True)
    rl = getattr(out, "router_logits", None)
    if not rl:
        raise RuntimeError(
            "[aux] out.router_logits is empty: the forward was called without "
            "output_router_logits=True, so no load-balancing loss can be computed. That argument "
            "defaults to False in OlmoeForCausalLM.")
    if getattr(out, "loss", None) is not None:
        raise RuntimeError(
            "[aux] the model returned a loss, which means `labels` was passed. HF then already "
            "added router_aux_loss_coef * aux_loss internally, and the trainer's own AUX_C term "
            "would double-count it -- over a different quantity, HF balancing top-k selection and "
            "this code balancing residency. Compute the LM loss outside the model, or drop AUX_C.")
    if not aux_c:
        raise RuntimeError(f"[aux] AUX_C is {aux_c!r}: the router is unregularised.")
    if not torch.isfinite(aux):
        raise RuntimeError(f"[aux] aux is {aux.item()}, not finite.")
    if not aux.requires_grad:
        raise RuntimeError(
            "[aux] aux does not require grad, so adding it to the loss changes no weights. The "
            "router logits were detached somewhere between the forward and here.")
    print(f"[aux] live: {len(rl)} router-logit tensors, aux={aux.item():.4f} x {aux_c}, "
          f"grad reaches the router, model added none of its own", flush=True)


def aux_z_from_router_logits(router_logits_tuple, B, S, R, evict="min_logit"):
    """Compute Switch aux + router z-loss on the MASKED distribution from a forward's raw per-layer
    router_logits (output_router_logits=True). Differentiable in the router weights; computed once
    post-forward so it is gradient-checkpointing safe. Each rl: [B*S, E]."""
    aux_t = z_t = None; n = 0
    _free = _CFG.get("free_layers", 0)
    _fset = _CFG.get("free_set")
    for _li, rl in enumerate(router_logits_tuple):
        N, E = rl.shape
        if (_li in _fset) if _fset is not None else (_li < _free):
            # unconstrained layer: ordinary Switch aux/z on the UNMASKED distribution
            probs = torch.softmax(rl.float(), dim=-1)
            P = probs.mean(0)
            aux = E * (P * P).sum()          # importance loss; uniform P minimises it
            z = (torch.logsumexp(rl.float(), dim=-1) ** 2).mean()
            aux_t = aux if aux_t is None else aux_t + aux
            z_t = z if z_t is None else z_t + z; n += 1
            continue
        lg = rl.view(B, S, E).transpose(0, 1).contiguous()          # [S,B,E]
        with torch.no_grad():
            scan = compute_resident_mask_accel if lg.is_cuda else compute_resident_mask
            mask = scan(lg.float(), R, evict=evict).transpose(0, 1).reshape(N, E)
        used = rl.masked_fill(~mask, float("-inf")).float()
        probs = torch.softmax(used, dim=-1)
        f = torch.isfinite(used).float().mean(0)
        aux = E * (f * probs.mean(0)).sum()
        z = (torch.logsumexp(used, dim=-1) ** 2).mean()
        aux_t = aux if aux_t is None else aux_t + aux
        z_t = z if z_t is None else z_t + z; n += 1
    return aux_t / max(1, n), z_t / max(1, n)


def router_params(model):
    """The trainable router linears (OlmoeTopKRouter.weight per layer, ~16 x 2048x64 = 2.1M)."""
    ps = [m.weight for m in model.modules() if isinstance(m, OlmoeTopKRouter)]
    return ps


def norm_params(model):
    """All learnable RMSNorm gains (arm C: router + norm gains). OLMoE RMSNorm has a weight param."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm
    return [m.weight for m in model.modules() if isinstance(m, OlmoeRMSNorm)]


def load_c_adapted(model, path=None):
    """Apply an arm-C adaptation delta (router linears + RMSNorm gains) to a loaded base model.

    The bake-off saved these as `router.{i}.weight` and `extra.{i}` in module order, which is the
    same order router_params()/norm_params() return. Layout comes from the adaptation program's
    train_bakeoff.py::save_delta.
    """
    import os as _os
    from safetensors.torch import load_file
    if path is None:
        _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        path = _os.path.join(_root, "results", "ablations", "adapt_ckpts", "router_bake_C.safetensors")
    sd = load_file(path)
    rp, npar = router_params(model), norm_params(model)
    with torch.no_grad():
        for i, p in enumerate(rp):
            p.data.copy_(sd[f"router.{i}.weight"].to(p.dtype).to(p.device))
        for i, p in enumerate(npar):
            p.data.copy_(sd[f"extra.{i}"].to(p.dtype).to(p.device))
    return len(rp), len(npar), path


# ---- arm E: LoRA on every expert's gate_up / down projection (bf16), starts as a no-op ----
_LORA = {"scale": 2.0}
_orig_experts_forward = None


def _experts_forward_lora(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        em = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
    s = _LORA["scale"]
    for e in hit:
        e = e[0]
        if e == self.num_experts:
            continue
        pos, tid = torch.where(em[e])
        cs = hidden_states[tid]
        gu = F.linear(cs, self.gate_up_proj[e]) + s * F.linear(F.linear(cs, self.lora_gu_A[e]), self.lora_gu_B[e])
        gate, up = gu.chunk(2, dim=-1)
        ch = self.act_fn(gate) * up
        dn = F.linear(ch, self.down_proj[e]) + s * F.linear(F.linear(ch, self.lora_dn_A[e]), self.lora_dn_B[e])
        dn = dn * top_k_weights[tid, pos, None]
        final.index_add_(0, tid, dn.to(final.dtype))
    return final


def add_lora(model, r=32, alpha=64):
    """Attach per-expert LoRA (A: kaiming-ish normal, B: zero -> delta=0 at init) and patch the expert
    forward. Returns the LoRA parameter list (~15M). down/up factored: gate_up [E,2I,H], down [E,H,I]."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
    global _orig_experts_forward
    _LORA["scale"] = alpha / r
    ps = []
    for m in model.modules():
        if isinstance(m, OlmoeExperts):
            E, twoI, H = m.gate_up_proj.shape
            _, Hd, I = m.down_proj.shape
            dev = m.gate_up_proj.device

            def mk(shape, zero):
                t = torch.zeros(shape, device=dev, dtype=torch.bfloat16)
                if not zero:
                    torch.nn.init.normal_(t, std=1.0 / r)
                return torch.nn.Parameter(t)
            m.lora_gu_A = mk((E, r, H), False); m.lora_gu_B = mk((E, twoI, r), True)
            m.lora_dn_A = mk((E, r, I), False); m.lora_dn_B = mk((E, Hd, r), True)
            ps += [m.lora_gu_A, m.lora_gu_B, m.lora_dn_A, m.lora_dn_B]
    if _orig_experts_forward is None:
        _orig_experts_forward = OlmoeExperts.forward
    OlmoeExperts.forward = _experts_forward_lora
    return ps


def _lora_linear_hook(mod, inp, out):
    """out + scale * (x A^T) B^T, added to whatever the wrapped Linear produced."""
    x = inp[0]
    return out + mod._lora_scale * F.linear(F.linear(x, mod._lora_A), mod._lora_B)


def add_lora_attn(model, r=32, alpha=64, targets=("q_proj", "k_proj", "v_proj", "o_proj")):
    """Attach LoRA to the attention projections and return the parameter list.

    Everything the adaptation program has adapted so far lives in or after the router: router
    linears, RMSNorm gains, and per-expert LoRA. Attention has been frozen in every arm including
    F', the full-parameter finetune -- so "the constraint price is irreducible" was established
    without ever asking whether attention can absorb any of it. Residency restricts which experts a
    token may reach; attention decides what the token's representation contains when it gets there,
    and that is a different lever.

    Implemented as a forward hook rather than by swapping the Linear out, so the module tree, the
    checkpoint key names and `norm_params`/`router_params` see exactly what they saw before. B is
    zero-initialised, so the branch is an exact no-op at step 0 and flag-off parity is preserved by
    construction, the same property the expert LoRA and the PLE table rely on.
    """
    ps = []
    for m in model.modules():
        if type(m).__name__ != "OlmoeAttention":
            continue
        for name in targets:
            lin = getattr(m, name, None)
            if lin is None:
                continue
            dev, dt = lin.weight.device, lin.weight.dtype
            A = torch.zeros(r, lin.in_features, device=dev, dtype=dt)
            torch.nn.init.normal_(A, std=1.0 / r)
            lin._lora_A = torch.nn.Parameter(A)
            lin._lora_B = torch.nn.Parameter(
                torch.zeros(lin.out_features, r, device=dev, dtype=dt))
            lin._lora_scale = alpha / r
            lin.register_forward_hook(_lora_linear_hook)
            ps += [lin._lora_A, lin._lora_B]
    return ps


def freeze_all_but_router(model):
    rp = set(id(p) for p in router_params(model))
    n_tr = 0
    for p in model.parameters():
        p.requires_grad = id(p) in rp
        if p.requires_grad:
            n_tr += p.numel()
    return n_tr


def disable_residency():
    _CFG["on"] = False


def load_model(path=None, device="cuda"):
    if path is None:
        from olmoe_paths import MODEL_DIR
        path = MODEL_DIR
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    install_patch()
    tag_layers(model)          # _layer_idx is required by the free_layers check
    return model, tok


def enable_grad_checkpointing(model):
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
