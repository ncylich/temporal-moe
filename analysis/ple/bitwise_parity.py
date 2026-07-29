#!/usr/bin/env python3
"""Bitwise no-op test: is the flag-off trainer the same COMPUTATION as the reference?

The training-run comparison answers a code question with statistics, and replicates can only ever
show a difference is small -- never that it is zero. This answers it directly. One deterministic
forward and one backward, identical input, identical weights, both implementations. A true no-op
must be bitwise identical, not close.

Run once per implementation, then compare the dumps:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python bitwise_parity.py --impl new --out /tmp/bw_new.pt
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python bitwise_parity.py --impl ref --out /tmp/bw_ref.pt
    python bitwise_parity.py --compare /tmp/bw_new.pt /tmp/bw_ref.pt

--impl new  imports analysis/ple/residency.py   (the working module)
--impl ref  imports scripts/adaptation/olmoe_residency.py (the verbatim archive of what ran)

Separate processes on purpose: both modules monkey-patch the same transformers classes, so
importing them into one process would have the second patch win and the test would compare a thing
against itself.

Captured: the eval-mode logits, the training-mode loss and its three components, and the gradient
of every trained parameter (routers + norm gains) after a single backward on the same batch.
"""

import argparse, os, sys
import torch


def build(impl):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    if impl == "new":
        sys.path.insert(0, here)
        import residency as RES
    else:
        # the archive is verbatim, so it hardcodes /workspace paths and expects to find
        # temporal/ on sys.path itself. Import it by location without altering the file.
        arch = os.path.join(root, "scripts", "adaptation")
        sys.path.insert(0, arch)
        sys.path.insert(0, here)          # so olmoe_paths is importable if needed
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "olmoe_residency", os.path.join(arch, "olmoe_residency.py"))
        RES = importlib.util.module_from_spec(spec)
        sys.modules["olmoe_residency"] = RES
        spec.loader.exec_module(RES)
    return RES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["new", "ref"])
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--no-flash", action="store_true",
                    help="disable the flash SDPA backend. Its BACKWARD is non-deterministic "
                         "(atomics over the sequence dim); the math/mem-efficient backends are not. "
                         "Diagnostic only -- it changes the kernel, so it is not what the published "
                         "runs used.")
    A = ap.parse_args()

    if A.compare:
        a = torch.load(A.compare[0], map_location="cpu")
        b = torch.load(A.compare[1], map_location="cpu")
        print(f"{'quantity':28s} {'bitwise':>8s}  {'max|diff|':>12s}")
        allsame = True
        for k in a:
            if isinstance(a[k], torch.Tensor):
                same = bool(torch.equal(a[k], b[k]))
                md = float((a[k].float() - b[k].float()).abs().max())
            else:
                same = a[k] == b[k]
                md = abs(float(a[k]) - float(b[k]))
            allsame &= bool(same)
            print(f"{k:28s} {str(same):>8s}  {md:12.3e}")
        print(f"\nBITWISE_PARITY {'PASS -- identical computation' if allsame else 'FAIL -- real divergence'}")
        return

    # warn_only=True, deliberately. transformers' grouped-GEMM expert path calls torch.histc, which
    # torch flags as having no deterministic implementation because it accumulates with atomics.
    # Its OUTPUT is exact integer bin counts, so the value is reproducible regardless of
    # accumulation order -- the flag is about the kernel, not the result. Erroring here would block
    # the test over an operation that cannot actually differ. Everything else stays strict.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    if A.no_flash:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)   # its backward uses atomics too
        torch.backends.cuda.enable_math_sdp(True)

    RES = build(A.impl)
    from olmoe_paths import MODEL_DIR
    model, _ = RES.load_model(MODEL_DIR)
    RES.enable_residency(R=8)
    RES.freeze_all_but_router(model)
    rp = RES.router_params(model)
    norm_ps = RES.norm_params(model)
    for p in norm_ps:
        p.requires_grad = True
    train_params = rp + norm_ps

    g = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.randint(0, model.config.vocab_size, (1, A.seq), generator=g).to("cuda").long()

    out = {}
    # ---- eval-mode forward, no grad ----
    model.eval()
    with torch.no_grad():
        out["logits"] = model(ids).logits.detach().float().cpu()

    # ---- training-mode forward + backward, gradient checkpointing on, as the trainers run ----
    RES.enable_grad_checkpointing(model)
    model.train()
    for p in train_params:
        p.grad = None
    o = model(ids, output_router_logits=True)
    labels = ids[:, 1:].reshape(-1)
    lm = torch.nn.functional.cross_entropy(
        o.logits[:, :-1].reshape(-1, o.logits.size(-1)).float(), labels)
    aux, z = RES.aux_z_from_router_logits(o.router_logits, ids.shape[0], ids.shape[1], RES._CFG["R"])
    loss = lm + 0.01 * aux + 0.001 * z
    loss.backward()

    out["train_logits"] = o.logits.detach().float().cpu()
    out["lm"] = float(lm)
    out["aux"] = float(aux)
    out["z"] = float(z)
    out["loss"] = float(loss)
    gr = [p.grad.detach().float().cpu() for p in rp]
    gn = [p.grad.detach().float().cpu() for p in norm_ps if p.grad is not None]
    out["grad_router"] = torch.stack(gr)
    out["grad_norms"] = torch.cat([x.reshape(-1) for x in gn])
    torch.save(out, A.out)
    print(f"[{A.impl}] loss={loss:.10f} lm={lm:.10f} aux={aux:.10f} z={z:.10f} -> {A.out}")


if __name__ == "__main__":
    main()
