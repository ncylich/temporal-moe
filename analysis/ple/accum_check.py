#!/usr/bin/env python3
"""Verify gradient accumulation: mb4 x accum4 against mb16, on the same 16 sequences.

Run flash-off, which is a correctness check rather than a training cell, so the standing rule in
PLE_PLAN.md §10 does not apply and the comparison is not swamped by attention-backward atomics.

WHAT IS AND IS NOT EXPECTED TO MATCH.

The LM term matches up to floating-point reassociation: cross_entropy with reduction='mean' over
four equal-sized micro-batches, each divided by 4 and summed, is the same quantity as one mean over
all 16 sequences, but summed in a different order.

The aux and z terms do NOT match exactly, and this is mathematical rather than numerical. Switch
aux is `E * sum_e(f_e * P_e)` where `f_e` and `P_e` are both means over the tokens in the batch. A
mean of products is not a product of means, so averaging four micro-batch aux values differs from
computing aux once over all 16 sequences by the across-micro-batch covariance of f and P. The z
term is `mean(logsumexp^2)`, likewise nonlinear in the batch.

So the honest question is not "is it bitwise identical" but "is the residual small against the
things it could confound". This script reports both terms separately so the answer is legible.
"""

import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES     # noqa: E402
from olmoe_paths import DATA_DIR   # noqa: E402

AUX_C, Z_C = 0.01, 0.001


def run(model, seqs, mb, train_params):
    for p in train_params:
        p.grad = None
    n = seqs.shape[0]
    accum = n // mb
    tot = {"lm": 0.0, "aux": 0.0, "z": 0.0}
    for i in range(accum):
        b = seqs[i * mb:(i + 1) * mb]
        out = model(b, output_router_logits=True)
        lm = torch.nn.functional.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float(), b[:, 1:].reshape(-1))
        aux, z = RES.aux_z_from_router_logits(out.router_logits, b.shape[0], b.shape[1], 8)
        ((lm + AUX_C * aux + Z_C * z) / accum).backward()
        tot["lm"] += float(lm) / accum
        tot["aux"] += float(aux) / accum
        tot["z"] += float(z) / accum
        del out, lm, aux, z
    return tot, [p.grad.detach().float().clone() for p in train_params]


def trajectory(model, train_params, masters_src, seqs_all, mb, steps, lr, accum):
    """Run `steps` optimizer steps, mirroring train_ple.py exactly, and return the loss sequence.

    Mirrors the three things that silently break accumulation:
      - the loss IS divided by accum inside the micro-loop
      - there is no LR scheduler to step at the wrong cadence (constant-LR AdamW)
      - clipping is applied ONCE to the accumulated gradient, after the micro-loop
    """
    masters = [m.detach().clone().requires_grad_(True) for m in masters_src]
    with torch.no_grad():
        for p, m in zip(train_params, masters):
            p.data.copy_(m.data.to(p.dtype))
    opt = torch.optim.AdamW(masters, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    losses, gnorms = [], []
    for s in range(steps):
        for p in train_params:
            p.grad = None
        acc = 0.0
        for micro in range(accum):
            b = seqs_all[s][micro * mb:(micro + 1) * mb]
            out = model(b, output_router_logits=True)
            lm = torch.nn.functional.cross_entropy(
                out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float(), b[:, 1:].reshape(-1))
            aux, z = RES.aux_z_from_router_logits(out.router_logits, b.shape[0], b.shape[1], 8)
            ((lm + AUX_C * aux + Z_C * z) / accum).backward()
            acc += float(lm) / accum
            del out, lm, aux, z
        for m, p in zip(masters, train_params):
            m.grad = p.grad.float() if p.grad is not None else None
        gn = torch.nn.utils.clip_grad_norm_(masters, 1.0)
        opt.step()
        with torch.no_grad():
            for m, p in zip(masters, train_params):
                p.data.copy_(m.data.to(p.dtype)); p.grad = None
        opt.zero_grad(set_to_none=True)
        losses.append(acc); gnorms.append(float(gn))
    return losses, gnorms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--eff", type=int, default=16, help="effective batch")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    A = ap.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(0)

    model, _ = RES.load_model()
    RES.enable_residency(R=8)
    RES.enable_grad_checkpointing(model)
    RES.freeze_all_but_router(model)
    rp = RES.router_params(model)
    nps = RES.norm_params(model)
    for p in nps:
        p.requires_grad = True
    train_params = rp + nps
    model.train()

    g = torch.Generator().manual_seed(7)
    seqs = torch.randint(0, model.config.vocab_size, (A.eff, A.seq), generator=g).to("cuda").long()

    t16, g16 = run(model, seqs, A.eff, train_params)
    t4, g4 = run(model, seqs, A.eff // 4, train_params)

    print(f"{'term':6s} {'mb'+str(A.eff):>16s} {'mb'+str(A.eff//4)+' x4':>16s} {'abs diff':>12s} {'rel':>10s}")
    for k in ("lm", "aux", "z"):
        d = abs(t16[k] - t4[k])
        print(f"{k:6s} {t16[k]:16.10f} {t4[k]:16.10f} {d:12.3e} {d/max(abs(t16[k]),1e-30):10.2e}")

    gr16 = torch.cat([x.reshape(-1) for x in g16])
    gr4 = torch.cat([x.reshape(-1) for x in g4])
    d = (gr16 - gr4).abs()
    cos = torch.nn.functional.cosine_similarity(gr16.reshape(1, -1), gr4.reshape(1, -1)).item()
    print(f"\ngradient over {gr16.numel()} trained params")
    print(f"  bitwise identical : {bool(torch.equal(gr16, gr4))}")
    print(f"  max |diff|        : {float(d.max()):.4e}   (max |grad| {float(gr16.abs().max()):.4e})")
    print(f"  relative max      : {float(d.max()/gr16.abs().max()):.3e}")
    print(f"  cosine similarity : {cos:.12f}")
    print(f"  relative L2       : {float(d.norm()/gr16.norm()):.3e}")

    # ---- multi-step trajectory: the check that catches the silent failures ----
    masters_src = [p.detach().float().clone() for p in train_params]
    gg = torch.Generator().manual_seed(11)
    seqs_all = [torch.randint(0, model.config.vocab_size, (A.eff, A.seq), generator=gg).to("cuda").long()
                for _ in range(A.steps)]
    l16, n16 = trajectory(model, train_params, masters_src, seqs_all, A.eff, A.steps, A.lr, 1)
    l4, n4 = trajectory(model, train_params, masters_src, seqs_all, A.eff // 4, A.steps, A.lr, 4)
    with torch.no_grad():
        for p, m in zip(train_params, masters_src):
            p.data.copy_(m.data.to(p.dtype))

    print(f"\n{A.steps} optimizer steps, identical data, lr={A.lr}")
    print(f"  {'step':>4s} {'mb'+str(A.eff)+' loss':>16s} {'mb'+str(A.eff//4)+'x4 loss':>16s} "
          f"{'diff':>11s} {'|g| mb16':>10s} {'|g| mb4x4':>10s}")
    worst = 0.0
    for i, (a, b_, ga, gb) in enumerate(zip(l16, l4, n16, n4)):
        worst = max(worst, abs(a - b_))
        print(f"  {i:>4d} {a:16.10f} {b_:16.10f} {abs(a-b_):11.3e} {ga:10.4f} {gb:10.4f}")
    print(f"\n  max |loss diff| over the sequence: {worst:.3e}")
    print("  A loss undivided by accum, a per-micro-batch scheduler, or clipping inside the "
          "micro-loop would all show as a gross divergence here, not a small one.")
    print(f"  grad-norm agreement also matters: clipping per micro-batch would make |g| mb4x4 "
          f"differ grossly from mb16.")


if __name__ == "__main__":
    main()
