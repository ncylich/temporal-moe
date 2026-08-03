#!/usr/bin/env python3
"""Measure where training memory actually goes, per micro-batch and per PLE rank.

The full-rank rung's shortfall blocks Phase 1's FIRST cell, because the plan runs full and 512
before the low rungs on purpose. The remedy depends on which term dominates, so this measures the
terms rather than estimating them:

  weights          the loaded model, before any training state
  C-surface state  fp32 masters + AdamW moments for routers and norm gains (tiny, ~2.2M params)
  PLE table        fp32 parameters
  PLE gradient     autograd materialises a DENSE gradient for an indexed Parameter, the full size
                   of the table, regardless of how few rows the batch touched
  PLE optimizer    8-bit Adam states, 2 bytes per parameter
  activations      everything else: the peak minus the above, which under gradient checkpointing is
                   layer-boundary tensors plus recompute and kernel workspace

The fp32 master is NOT a candidate for removal. A zero-initialised table takes very small early
updates, which is exactly where bf16 underflows: below ~6e-8 relative, an update to a zero row
rounds away entirely and the row never leaves zero.
"""

import argparse, gc, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES     # noqa: E402
import ple as PLE           # noqa: E402

GiB = 2 ** 30


def probe(model, rank, mb, seq=4096, steps=2):
    PLE.uninstall()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    rp = RES.router_params(model)
    norm_ps = RES.norm_params(model)
    for p in norm_ps:
        p.requires_grad = True
    train_params = rp + norm_ps
    masters = [p.detach().float().clone().requires_grad_(True) for p in train_params]
    opt = torch.optim.AdamW(masters, lr=1e-9, betas=(0.9, 0.95), weight_decay=0.0)

    ple_mod = opt_ple = None
    tbl_bytes = 0
    if rank != "off":
        ple_mod = PLE.install(model, rank, device="cuda")
        tbl_bytes = sum(p.numel() * p.element_size() for p in ple_mod.parameters())
        from bitsandbytes.optim import AdamW8bit
        opt_ple = AdamW8bit([{"params": ple_mod.table_params(), "weight_decay": 0.0},
                             {"params": ple_mod.basis_params(), "weight_decay": 0.0}],
                            lr=1e-9, betas=(0.9, 0.95))
    after_state = torch.cuda.memory_allocated()

    model.train()
    ids = torch.randint(0, model.config.vocab_size, (mb, seq), device="cuda").long()
    peak = 0
    for _ in range(steps):
        out = model(ids, output_router_logits=True)
        lm = torch.nn.functional.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float(), ids[:, 1:].reshape(-1))
        aux, z = RES.aux_z_from_router_logits(out.router_logits, mb, seq, 8)
        (lm + 0.01 * aux + 0.001 * z).backward()
        grad_bytes = (sum(p.grad.numel() * p.grad.element_size()
                          for p in ple_mod.parameters() if p.grad is not None) if ple_mod else 0)
        for m, p in zip(masters, train_params):
            m.grad = p.grad.float() if p.grad is not None else None
        opt.step()
        for m, p in zip(masters, train_params):
            p.data.copy_(m.data.to(p.dtype)); p.grad = None
        opt.zero_grad(set_to_none=True)
        if opt_ple is not None:
            opt_ple.step(); opt_ple.zero_grad(set_to_none=True)
        peak = max(peak, torch.cuda.max_memory_allocated())
        del out, lm, aux, z

    opt_bytes = 0
    if opt_ple is not None:
        for st in opt_ple.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    opt_bytes += v.numel() * v.element_size()
    res = {"rank": str(rank), "micro_batch": mb,
           "weights_GiB": round(base / GiB, 3),
           "ple_table_GiB": round(tbl_bytes / GiB, 3),
           "ple_grad_GiB": round(grad_bytes / GiB, 3),
           "ple_optimizer_GiB": round(opt_bytes / GiB, 3),
           "state_total_GiB": round((after_state - base) / GiB, 3),
           "peak_GiB": round(peak / GiB, 3),
           "activations_GiB": round((peak - after_state - grad_bytes) / GiB, 3)}
    PLE.uninstall()
    del masters, opt, opt_ple, ple_mod
    gc.collect(); torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="off:16,off:8,off:4,full:4,full:2,512:16")
    A = ap.parse_args()
    model, _ = RES.load_model()
    RES.enable_residency(R=8)
    RES.enable_grad_checkpointing(model)
    RES.freeze_all_but_router(model)
    total = torch.cuda.get_device_properties(0).total_memory / GiB
    print(f"device total {total:.1f} GiB\n")
    out = []
    for case in A.cases.split(","):
        r, mb = case.split(":")
        rank = r if r in ("off", "full") else int(r)
        try:
            res = probe(model, rank, int(mb))
        except torch.OutOfMemoryError:
            res = {"rank": r, "micro_batch": int(mb), "peak_GiB": "OOM"}
            gc.collect(); torch.cuda.empty_cache()
        out.append(res); print(json.dumps(res), flush=True)
    print("\n" + json.dumps(out))
