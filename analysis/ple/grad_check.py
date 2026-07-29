#!/usr/bin/env python3
"""Verify the PLE table receives gradient THROUGH gradient checkpointing.

The C recipe enables gradient checkpointing, so every decoder layer's forward is recomputed during
backward. PLE reads the token ids from module state set by the model forward; if that state is
restored when the forward returns, the recompute sees nothing and the PLE add is silently missing
from the recomputed graph. The loss curve looks entirely normal and the table never moves.

This checks the property that matters rather than the mechanism:
  1. every PLE tensor that should get gradient does, with checkpointing ON;
  2. only the rows of tokens actually in the batch get gradient (the sparsity the zero property
     depends on);
  3. the gradient with checkpointing ON matches the gradient with it OFF.
"""

import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES     # noqa: E402
import ple as PLE           # noqa: E402


def grads_for(model, ple, ids, checkpointing):
    if checkpointing:
        RES.enable_grad_checkpointing(model)
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = False
    for p in ple.parameters():
        p.grad = None
    out = model(ids, output_router_logits=False)
    loss = torch.nn.functional.cross_entropy(
        out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float(), ids[:, 1:].reshape(-1))
    loss.backward()
    return {k: (v.grad.detach().clone() if v.grad is not None else None)
            for k, v in ple.named_parameters()}, float(loss)


def main():
    model, _ = RES.load_model()
    RES.enable_residency(R=8)
    RES.freeze_all_but_router(model)
    for p in RES.norm_params(model):
        p.requires_grad = True
    ple = PLE.install(model, 32, device="cuda")
    model.train()

    torch.manual_seed(0)
    uniq = torch.tensor([5, 11, 23, 47, 101, 233, 599, 1213], device="cuda")
    ids = uniq.repeat(4).unsqueeze(0)          # [1, 32], only 8 distinct token ids

    g_on, loss_on = grads_for(model, ple, ids, checkpointing=True)
    g_off, loss_off = grads_for(model, ple, ids, checkpointing=False)

    print(f"loss checkpointing on = {loss_on:.6f}   off = {loss_off:.6f}")
    ok = True
    for k in g_on:
        a, b = g_on[k], g_off[k]
        has_a = a is not None and bool((a != 0).any())
        has_b = b is not None and bool((b != 0).any())
        agree = (a is not None and b is not None
                 and torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-6))
        print(f"  {k:4s} grad nonzero  ckpt-on={has_a}  ckpt-off={has_b}  match={agree}")
        if k == "U":
            if not has_a:
                ok = False
            rows = torch.nonzero((a != 0).any(-1)).flatten()
            expect = torch.unique(uniq).sort().values
            got = rows.sort().values
            same = got.numel() == expect.numel() and bool((got == expect).all())
            print(f"       rows with gradient: {got.tolist()}")
            print(f"       expected (batch token ids): {expect.tolist()}   match={same}")
            if not same:
                ok = False
        if not agree:
            ok = False
    print(f"\nGRAD_CHECK {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
