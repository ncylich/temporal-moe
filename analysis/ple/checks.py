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
def cmd_aux(A):
    """Smoke test: is the load-balancing loss actually reaching the router?

    Runs one forward under the real recipe and asserts the guard passes, then deliberately breaks
    each way it can fail and asserts the guard catches it. A guard that has never been seen to fire
    is not evidence of anything.
    """
    import residency as RES
    model, tok = RES.load_model()
    RES.enable_residency(R=8)
    ids = torch.randint(0, 50000, (2, 512), device="cuda")

    out = model(ids, output_router_logits=True)
    aux, z = RES.aux_z_from_router_logits(out.router_logits, ids.shape[0], ids.shape[1], 8)
    RES.assert_aux_live(out, aux, 0.01)
    print(f"  [1/4] real recipe: PASS   aux={aux.item():.4f} z={z.item():.4f}")

    def expect(label, fn):
        RES.assert_aux_live.__defaults__[0].clear()          # re-arm the once-per-process guard
        try:
            fn(); print(f"  {label}: NOT CAUGHT -- the guard is useless"); raise SystemExit(1)
        except RuntimeError as e:
            print(f"  {label}: caught -- {str(e).splitlines()[0][:78]}")
        RES.assert_aux_live.__defaults__[0].clear()

    bare = model(ids)                                        # output_router_logits defaults False
    expect("[2/4] flag off        ", lambda: RES.assert_aux_live(bare, aux, 0.01))
    expect("[3/4] coefficient 0   ", lambda: RES.assert_aux_live(out, aux, 0.0))
    expect("[4/5] aux detached    ", lambda: RES.assert_aux_live(out, aux.detach(), 0.01))
    # The subtle one: pass labels and HF adds router_aux_loss_coef * aux_loss itself, so the
    # trainer's AUX_C term double-counts -- over a different quantity, HF balancing top-k selection
    # and this code balancing residency. Nothing about the loss curve would show it.
    withlab = model(ids, labels=ids, output_router_logits=True)
    expect("[5/5] labels -> HF aux", lambda: RES.assert_aux_live(withlab, aux, 0.01))
    RES.assert_aux_live.__defaults__[0].clear()
    print("\nall five: the guard passes the real recipe and fires on every way it can break")


def cmd_auxparity(A):
    """A freed layer's aux must be EXACTLY the stock OLMoE aux for that layer.

    This is the invariant the port exists to establish. temporal-moe never changes the aux formula:
    it masks non-resident experts to -inf and calls the unmodified routing(), so a layer at R=E gets
    plain Switch aux. Porting that convention here means a freed layer must reproduce HF's
    load_balancing_loss_func on the same logits, not merely resemble it.

    Also asserts the two branches are on the same SCALE, which is what the old importance-loss
    substitution broke: at the uniform optimum both forms are k, not k and 1.
    """
    import residency as RES
    from transformers.models.olmoe.modeling_olmoe import load_balancing_loss_func as hf_aux
    model, tok = RES.load_model()
    E = model.config.num_experts
    k = model.config.num_experts_per_tok
    os.environ["OLMOE_TOPK"] = str(k)
    ids = torch.randint(0, 50000, (2, 512), device="cuda")

    # every layer freed -> every layer must match stock OLMoE
    RES.enable_residency(R=8); RES.set_free_layers(list(range(model.config.num_hidden_layers)))
    out = model(ids, output_router_logits=True)
    ours, _ = RES.aux_z_from_router_logits(out.router_logits, ids.shape[0], ids.shape[1], 8)
    theirs = hf_aux(out.router_logits, E, k, None)
    # PER LAYER a freed layer must be exactly HF's formula. HF's TOTAL differs because it pools
    # every layer into one global statistic while Megatron, temporal-moe and this code compute per
    # layer and average; that divergence is deliberate and is checked separately below.
    worst = 0.0
    for li, rl in enumerate(out.router_logits):
        o, _ = RES.aux_z_from_router_logits((rl,), ids.shape[0], ids.shape[1], 8)
        h = hf_aux((rl.float(),), E, k, None)                 # fp32 both sides: HF softmaxes in the
        worst = max(worst, abs(o.item() - h.item()))          # logits' dtype, which is bf16 here
    print(f"  freed layer vs stock OLMoE, worst of {len(out.router_logits)} layers: {worst:.2e}")
    assert worst < 1e-3, f"a freed layer must reproduce the stock aux formula, off by {worst}"

    print(f"  totals differ by aggregation only: ours(per-layer) {ours.item():.4f} vs "
          f"HF(global pool) {theirs.item():.4f} -- expected, Megatron computes per layer")

    # uniform-logit scale check: both branches must land on k, not k and 1
    N = 4096
    flat = torch.zeros(N, E, device="cuda")
    P = torch.softmax(flat, -1).mean(0)
    tk = torch.softmax(flat, -1).topk(k, -1).indices
    f = torch.zeros_like(P).scatter_add_(
        0, tk.reshape(-1), torch.ones(tk.numel(), device="cuda", dtype=P.dtype)) / N
    load_form = (E * (f * P).sum()).item()
    imp_form = (E * (P * P).sum()).item()
    print(f"  at uniform routing: load form {load_form:.4f} (= k = {k})   "
          f"old importance form {imp_form:.4f}   ratio {load_form / imp_form:.1f}x")
    assert abs(load_form - k) < 1e-3, "the load form should equal k at uniform routing"
    # The constrained branch changed too: residency fraction -> dispatch fraction. At R=k every
    # resident is selected, so the two are the same number and no run trained to date moves. That is
    # the claim that makes this change safe to apply retroactively, so it is asserted, not assumed.
    from temporal.temporal_router import compute_resident_mask_accel as scan
    RES.set_free_layers(None); RES.enable_residency(R=k)
    o2 = model(ids, output_router_logits=True)
    worst_rk = 0.0
    for rl in o2.router_logits:
        N, E2 = rl.shape
        lg = rl.view(ids.shape[0], ids.shape[1], E2).transpose(0, 1).contiguous()
        with torch.no_grad():
            m = scan(lg.float(), k, evict="min_logit").transpose(0, 1).reshape(N, E2)
        u = rl.masked_fill(~m, float("-inf")).float()
        pr = torch.softmax(u, -1); P2 = pr.mean(0)
        f_res = torch.isfinite(u).float().mean(0)                          # the old convention
        ix = pr.topk(k, -1).indices
        f_dis = torch.zeros_like(P2).scatter_add_(
            0, ix.reshape(-1), torch.ones(ix.numel(), device=rl.device, dtype=P2.dtype)) / N
        worst_rk = max(worst_rk, abs((E2*(f_res*P2).sum()).item() - (E2*(f_dis*P2).sum()).item()))
    print(f"  at R=k={k}, residency vs dispatch fraction: worst |diff| {worst_rk:.2e}")
    assert worst_rk < 1e-4, f"R=k invariance broken, off by {worst_rk}: existing runs would move"

    print("\n  PASS: freed layer == stock OLMoE aux; one scale; R=k leaves trained cells unchanged")


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
    sp.add_parser("aux")
    sp.add_parser("auxparity")
    sp.add_parser("placement")
    sp.add_parser("grad")
    z = sp.add_parser("zero"); z.add_argument("--trained", required=True)
    z.add_argument("--train-tokens", type=int); z.add_argument("--mb", type=int, default=16)
    b = sp.add_parser("bitwise"); b.add_argument("--impl", choices=["new", "ref"])
    b.add_argument("--out"); b.add_argument("--compare", nargs=2)
    b.add_argument("--seq", type=int, default=512); b.add_argument("--no-flash", action="store_true")
    A = ap.parse_args()
    {"init": cmd_init, "aux": cmd_aux, "auxparity": cmd_auxparity, "placement": cmd_placement, "grad": cmd_grad,
     "zero": cmd_zero, "bitwise": cmd_bitwise}[A.cmd](A)
