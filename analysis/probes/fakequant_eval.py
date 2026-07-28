#!/usr/bin/env python3
"""PART E probe entry: fake-quantize routed-expert weights (per-group RTN) then run test-set eval.

Wraps pretrain_gpt.model_provider to install a one-shot forward_pre_hook that RTN-fake-quantizes
every routed-expert weight (decoder.layers.L.mlp.experts.experts.linear_fc{1,2}.weight) in place
before the first eval forward. Symmetric per-group RTN, group size 128 along the input features,
bit-width from env QUANT_BITS (8/4/3). Only routed-expert weights are touched (router, shared,
attention, embeddings untouched). Run with --skip-train so Megatron only loads + evaluates.

The run's train.log-style "on test set" line carries the quantized test CE; convert to BPB with
divisor 2.9780 (50k pythia vocab).
"""
import os, sys, re, torch
sys.path.insert(0, os.getcwd())

BITS = int(os.environ.get("QUANT_BITS", "0"))
GROUP = int(os.environ.get("QUANT_GROUP", "128"))
_done = {"flag": False}


def rtn_quant_(w, bits, group):
    """In-place symmetric per-group RTN along the last (input-feature) axis."""
    if bits <= 0 or bits >= 16:
        return
    qmax = 2 ** (bits - 1) - 1
    orig_shape = w.shape
    flat = w.reshape(-1, orig_shape[-1]).float()          # [rows, in]
    rows, cin = flat.shape
    out = torch.empty_like(flat)
    for s in range(0, cin, group):
        g = flat[:, s:s + group]
        scale = g.abs().amax(dim=-1, keepdim=True) / qmax   # [rows,1]
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.clamp(torch.round(g / scale), -qmax, qmax)
        out[:, s:s + group] = q * scale
    w.copy_(out.reshape(orig_shape).to(w.dtype))


def _is_routed_expert_weight(name):
    return ("experts" in name and "shared_experts" not in name
            and (".linear_fc1.weight" in name or ".linear_fc2.weight" in name
                 or re.search(r"\.linear_fc[12]\.weight\d+$", name)))


def _quantize_model(model, verbose=True):
    n = 0
    sample = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "expert" in name and len(sample) < 8:
                sample.append(f"{name} {tuple(p.shape)}")
            if _is_routed_expert_weight(name):
                rtn_quant_(p.data, BITS, GROUP)
                n += 1
    if verbose:
        if os.environ.get("QUANT_DEBUG"):
            print("[fakequant-debug] expert-param sample:\n  " + "\n  ".join(sample))
        print(f"[fakequant] QUANT_BITS={BITS} group={GROUP}: quantized {n} routed-expert weight tensors")


def _wrap_provider():
    import pretrain_gpt
    orig = pretrain_gpt.model_provider

    def patched(*a, **k):
        model = orig(*a, **k)
        m = model[0] if isinstance(model, list) else model

        # Quantize once, on the first EVAL-mode forward (mod.training == False). Deferring past the
        # training forwards means it lands AFTER the last optimizer step's FP32-master->bf16 resync
        # (which would otherwise undo it), and it runs under eval's no_grad so the in-place .data edit
        # is safe. Evaluation has no further optimizer step, so the quant persists through the test eval.
        def prehook(mod, args):
            if not _done["flag"] and not mod.training:
                _done["flag"] = True
                _quantize_model(mod)
        m.register_forward_pre_hook(prehook)
        return model
    pretrain_gpt.model_provider = patched


if __name__ == "__main__":
    if os.environ.get("TEMPORAL", "0") == "1":          # temporal cells: install residency router
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from temporal import temporal_router
        temporal_router.install()
    from megatron.training import pretrain
    from megatron.core.enums import ModelType
    import pretrain_gpt
    _wrap_provider()
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        pretrain_gpt.train_valid_test_datasets_provider,
        pretrain_gpt.model_provider,
        ModelType.encoder_or_decoder,
        pretrain_gpt.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
