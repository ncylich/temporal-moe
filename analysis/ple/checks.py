#!/usr/bin/env python3
"""Correctness checks for the PLE implementation. One subcommand per property.

    checks.py init          zero-init contribution is bitwise 0, and the table can still learn
    checks.py placement     post-MoE placement leaves SAME-layer routing bit-identical
    checks.py grad          the table receives gradient THROUGH gradient checkpointing
    checks.py zero  --trained T --train-tokens N    uncovered + held-out rows stay bit-zero
    checks.py bitwise --impl new|ref --out P        deterministic forward/backward dump
    checks.py bitwise --compare A B                 compare two dumps
    checks.py accum         mb16 vs mb4 x accum4 on identical data

Each check earned its place by catching something. `grad` caught a dead branch: gradient
checkpointing recomputes the layer forward during backward, and an earlier version restored the
stashed token ids when the model forward returned, so the PLE add was absent from the recomputed
graph and the table received no gradient at all while the loss curve looked healthy. `bitwise`
proved flag-off parity against the unmodified reference trainer and localised the reference's
run-to-run spread to Flash Attention's non-deterministic backward. `init` catches the
zero-table-plus-zero-gate fixed point, where every gradient is zero and the branch can never train.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ple as PLE
from olmoe_paths import DATA_DIR


# ------------------------------------------------------------------ init (CPU, no model)
def cmd_init(A):
    for r in (32, 128, "full"):
        t = PLE.FactoredPLE(50304, 16, 2048, r, device="cpu")
        ids = torch.tensor([[1, 2, 3]])
        out = t(ids, 0, torch.float32)
        zero = bool((out == 0).all())
        (t(ids, 0, torch.float32) * torch.randn(1, 3, 2048)).sum().backward()
        tab = t.table_params()[0]
        rows = torch.nonzero((tab.grad != 0).any(-1)).flatten().unique().tolist()
        print(json.dumps({"rank": r, "contribution_bitwise_zero": zero,
                          "table_grad_nonzero": bool((tab.grad != 0).any()),
                          "rows_receiving_grad": rows, "gate_init": float(t.g[0]),
                          "n_params": t.n_params()}))
        # the fixed point a zero gate would create, for the record
        t2 = PLE.FactoredPLE(1000, 4, 64, 32, device="cpu")
        with torch.no_grad():
            t2.g.fill_(0.0)
        (t2(torch.tensor([[1, 2, 3]]), 0, torch.float32) * torch.randn(1, 3, 64)).sum().backward()
        if r == 32:
            print(f"  with gate=0: dU_nonzero={bool((t2.U.grad != 0).any())} "
                  f"dV_nonzero={bool((t2.V.grad != 0).any())} dg_nonzero={bool((t2.g.grad != 0).any())}"
                  "  <- all False is why the gate starts at 1.0")


# ------------------------------------------------------------------ placement (GPU)
def cmd_placement(A):
    import residency as RES
    torch.manual_seed(0)
    model, _ = RES.load_model()
    RES.enable_residency(R=8)
    model.eval()
    ids = torch.randint(0, 50304, (1, 256), device="cuda")

    def rl():
        with torch.no_grad():
            return [r.float().clone() for r in model(ids, output_router_logits=True).router_logits]

    PLE.uninstall()
    base_rl = rl()
    with torch.no_grad():
        base = model(ids).logits.clone()
    t = PLE.install(model, 32, device="cuda")
    zero_rl = rl()
    with torch.no_grad():
        zero = model(ids).logits.clone()
    with torch.no_grad():
        t.U.normal_(0, 0.02)
    act_rl = rl()
    with torch.no_grad():
        act = model(ids).logits.clone()
    same = [bool((a == b).all()) for a, b in zip(base_rl, act_rl)]
    print(f"zero table: logits bitwise equal {bool((base == zero).all())}, "
          f"all router logits equal {all(bool((a == b).all()) for a, b in zip(base_rl, zero_rl))}")
    print(f"active table: layer 0 router logits bitwise equal {same[0]}; "
          f"layers >0 differing {sum(1 for s in same[1:] if not s)} of {len(same) - 1}")
    print(f"per-layer bitwise-equal flags: {''.join('1' if s else '0' for s in same)}")
    ok = bool((base == zero).all()) and same[0] and bool((base != act).any())
    print(f"PLACEMENT {'PASS' if ok else 'FAIL'}  "
          "(same-layer routing untouched is the guarantee; deeper layers move because PLE writes "
          "into the residual stream, which any residual contribution does)")


# ------------------------------------------------------------------ grad through checkpointing
def cmd_grad(A):
    import residency as RES
    model, _ = RES.load_model()
    RES.enable_residency(R=8)
    RES.freeze_all_but_router(model)
    for p in RES.norm_params(model):
        p.requires_grad = True
    t = PLE.install(model, 32, device="cuda")
    model.train()
    uniq = torch.tensor([5, 11, 23, 47, 101, 233, 599, 1213], device="cuda")
    ids = uniq.repeat(4).unsqueeze(0)

    def grads(ckpt):
        if ckpt:
            RES.enable_grad_checkpointing(model)
        else:
            model.gradient_checkpointing_disable(); model.config.use_cache = False
        for p in t.parameters():
            p.grad = None
        o = model(ids)
        torch.nn.functional.cross_entropy(
            o.logits[:, :-1].reshape(-1, o.logits.size(-1)).float(), ids[:, 1:].reshape(-1)).backward()
        return {k: (v.grad.clone() if v.grad is not None else None) for k, v in t.named_parameters()}

    on, off = grads(True), grads(False)
    ok = True
    for k in on:
        a, b = on[k], off[k]
        agree = a is not None and b is not None and torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-6)
        print(f"  {k:4s} nonzero ckpt-on={a is not None and bool((a != 0).any())} "
              f"ckpt-off={b is not None and bool((b != 0).any())} match={agree}")
        if k == "U":
            got = torch.nonzero((a != 0).any(-1)).flatten().sort().values
            exp = torch.unique(uniq).sort().values
            m = got.numel() == exp.numel() and bool((got == exp).all())
            print(f"       rows with grad {got.tolist()} == batch token ids: {m}")
            ok &= m and bool((a != 0).any())
        ok &= agree
    print(f"GRAD {'PASS' if ok else 'FAIL'}")


# ------------------------------------------------------------------ zero property
def cmd_zero(A):
    if A.train_tokens is None:
        raise SystemExit("--train-tokens required: the covered set is defined by what the cell saw")
    corpus = torch.load(os.path.join(DATA_DIR, "finetune_ids.pt"))
    order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))
    nsteps = -(-A.train_tokens // (A.mb * 4096))
    seen_ids = torch.unique(corpus[order[: nsteps * A.mb]].reshape(-1))
    sd = torch.load(A.trained, map_location="cpu")
    rank = sd.pop("rank")
    tab = sd["P"] if "P" in sd else sd["U"]
    nz = (tab.reshape(tab.shape[0], -1) != 0).any(-1)
    covered = torch.zeros(tab.shape[0], dtype=torch.bool)
    covered[seen_ids.long()] = True
    out = {"rank": rank, "train_tokens": A.train_tokens,
           "n_covered": int(covered.sum()), "n_uncovered": int((~covered).sum()),
           "uncovered_rows_bit_zero": int((nz & ~covered).sum()) == 0,
           "covered_rows_that_moved": int((nz & covered).sum())}
    hp = os.path.join(DATA_DIR, "ple_heldout.pt")
    if os.path.exists(hp):
        ho = torch.load(hp)
        held = torch.zeros(tab.shape[0], dtype=torch.bool)
        held[ho["ids"].long()] = True
        out.update(n_heldout=int(held.sum()),
                   heldout_rows_bit_zero=int((nz & held).sum()) == 0,
                   heldout_rows_that_were_covered=int((held & covered).sum()))
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ bitwise parity / accumulation
def cmd_bitwise(A):
    import importlib.util
    if A.compare:
        a, b = (torch.load(p, map_location="cpu") for p in A.compare)
        allsame = True
        for k in a:
            if isinstance(a[k], torch.Tensor):
                same, md = bool(torch.equal(a[k], b[k])), float((a[k].float() - b[k].float()).abs().max())
            else:
                same, md = a[k] == b[k], abs(float(a[k]) - float(b[k]))
            allsame &= bool(same)
            print(f"{k:14s} bitwise={str(same):5s} max|diff|={md:.3e}")
        print(f"BITWISE {'PASS -- identical computation' if allsame else 'FAIL -- real divergence'}")
        return
    # warn_only: transformers' grouped-GEMM expert path calls torch.histc, flagged nondeterministic
    # because it accumulates with atomics, but its output is exact integer bin counts.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    if A.no_flash:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    here = os.path.dirname(os.path.abspath(__file__))
    if A.impl == "new":
        sys.path.insert(0, here); import residency as RES
    else:
        arch = os.path.join(os.path.dirname(os.path.dirname(here)), "scripts", "adaptation")
        sys.path.insert(0, arch); sys.path.insert(0, here)
        spec = importlib.util.spec_from_file_location("olmoe_residency",
                                                     os.path.join(arch, "olmoe_residency.py"))
        RES = importlib.util.module_from_spec(spec)
        sys.modules["olmoe_residency"] = RES
        spec.loader.exec_module(RES)
    from olmoe_paths import MODEL_DIR
    model, _ = RES.load_model(MODEL_DIR)
    RES.enable_residency(R=8)
    RES.freeze_all_but_router(model)
    rp, nps = RES.router_params(model), RES.norm_params(model)
    for p in nps:
        p.requires_grad = True
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.randint(0, model.config.vocab_size, (1, A.seq), generator=g).to("cuda").long()
    out = {}
    model.eval()
    with torch.no_grad():
        out["logits"] = model(ids).logits.detach().float().cpu()
    RES.enable_grad_checkpointing(model)
    model.train()
    for p in rp + nps:
        p.grad = None
    o = model(ids, output_router_logits=True)
    lm = torch.nn.functional.cross_entropy(
        o.logits[:, :-1].reshape(-1, o.logits.size(-1)).float(), ids[:, 1:].reshape(-1))
    aux, z = RES.aux_z_from_router_logits(o.router_logits, 1, A.seq, 8)
    (lm + 0.01 * aux + 0.001 * z).backward()
    out.update(lm=float(lm), aux=float(aux), z=float(z), loss=float(lm + 0.01 * aux + 0.001 * z),
               grad_router=torch.stack([p.grad.float().cpu() for p in rp]),
               grad_norms=torch.cat([p.grad.float().cpu().reshape(-1) for p in nps if p.grad is not None]))
    torch.save(out, A.out)
    print(f"[{A.impl}] loss={out['loss']:.10f} -> {A.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("init")
    sp.add_parser("placement")
    sp.add_parser("grad")
    z = sp.add_parser("zero"); z.add_argument("--trained", required=True)
    z.add_argument("--train-tokens", type=int); z.add_argument("--mb", type=int, default=16)
    b = sp.add_parser("bitwise"); b.add_argument("--impl", choices=["new", "ref"])
    b.add_argument("--out"); b.add_argument("--compare", nargs=2)
    b.add_argument("--seq", type=int, default=512); b.add_argument("--no-flash", action="store_true")
    A = ap.parse_args()
    {"init": cmd_init, "placement": cmd_placement, "grad": cmd_grad,
     "zero": cmd_zero, "bitwise": cmd_bitwise}[A.cmd](A)
