#!/usr/bin/env python3
"""Fully training-free stack: calibrated RMSNorm gains, then a calibrated PLE table on top.

How much of the rolling-residency damage can be undone with ZERO gradient steps?

Stage 1 reproduces Cal-0: record per-channel input RMS at all 65 RMSNorm sites under free routing
and under R=8, then set g' = g * clamp(RMS_free / RMS_masked, lo, hi). The router is left at base
weights, exactly as Cal-0 did -- there is no closed-form router calibration in this program, and the
router's damage under residency is a selection effect rather than a moment mismatch, so rescaling
cannot address it. That is a limitation of the stack, not an oversight.

Stage 2 writes the calibrated surface in the trainer's own checkpoint format (masters = router
params then norm params, 81 tensors, matching the non-LoRA layout) so calibrate.py --resume-c and
eval_table.py --csurf can consume it without new machinery.

Stage 3 (driven by the caller) captures Delta against that surface and installs a PLE table.

Reference points, recovery of the impose gap (base free 0.6727 = 100%, imposed untrained
2.7507 = 0%):
    Cal-0 clipped norms alone         31.5%   (prior program)
    calibrated PLE alone, base model  37.40%  (full rank, this program)
    router-only TRAINING              70.7%
    recipe C TRAINING                 90.07%
"""

import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "residency"))  # sibling dir (2026-08 split)
import residency as RES               # noqa: E402
from olmoe_paths import DATA_DIR      # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS           # noqa: E402

BASE, IMPOSE = 0.6727, 2.7507
ACC, CNT, REC = {}, {}, {"on": False}


def rec(b):
    return 1.0 - (b - BASE) / (IMPOSE - BASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-packs", type=int, default=512, help="~2M tokens of RMS statistics")
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--clip-lo", type=float, default=0.5)
    ap.add_argument("--clip-hi", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ple-table", default=None,
                    help="install this PLE table BEFORE calibrating norms, reversing the order: "
                         "norms are then fitted to whatever scale error PLE leaves behind. The "
                         "free-routing RMS reference is taken with PLE DISABLED, since base free "
                         "routing is the behaviour being matched and PLE is a correction for "
                         "damage that does not exist there.")
    ap.add_argument("--tag", default="cal0")
    A = ap.parse_args()

    from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm
    D = json.load(open(os.path.join(DATA_DIR, "bpb_slice_meta.json")))["divisor_D"]
    model, _ = RES.load_model()
    model.eval()

    PLE_MOD = None
    if A.ple_table:
        import ple as PLE
        _sd = torch.load(A.ple_table, map_location="cuda")
        _rank = _sd.pop("rank")
        PLE_MOD = PLE.install(model, _rank if _rank == "full" else int(_rank), device="cuda")
        with torch.no_grad():
            for _k, _v in _sd.items():
                getattr(PLE_MOD, _k).copy_(_v.to("cuda"))
        print(f"[cal-stack] PLE installed first from {os.path.basename(A.ple_table)} rank={_rank}",
              flush=True)

    norms = [m for m in model.modules() if isinstance(m, OlmoeRMSNorm)]
    base_g = [n.weight.detach().float().clone() for n in norms]
    print(f"[cal-stack] {len(norms)} RMSNorm sites", flush=True)

    _orig = OlmoeRMSNorm.forward

    def rec_fwd(self, x):
        if REC["on"]:
            i = self._cal_idx
            xf = x.detach().float().reshape(-1, x.shape[-1])
            s = xf.pow(2).sum(0)
            ACC[i] = s if i not in ACC else ACC[i] + s
            CNT[i] = CNT.get(i, 0) + xf.shape[0]
        return _orig(self, x)

    OlmoeRMSNorm.forward = rec_fwd
    for i, m in enumerate(norms):
        m._cal_idx = i

    ids = torch.load(os.path.join(DATA_DIR, "bpb_slice_ids.pt"))
    stat_ids = ids[:A.stats_packs].long()

    def rms_pass(residency_on):
        ACC.clear(); CNT.clear(); REC["on"] = True
        RES.enable_residency(R=8) if residency_on else RES.disable_residency()
        if PLE_MOD is not None:
            # PLE off for the free reference, on for the constrained state being corrected.
            import ple as _P
            _P._STATE["ple"] = PLE_MOD if residency_on else None
        with torch.no_grad():
            for i in range(stat_ids.shape[0]):
                model(stat_ids[i:i + 1].to("cuda"))
        REC["on"] = False
        return [(ACC[i] / CNT[i]).sqrt() for i in range(len(norms))]

    print("[cal-stack] RMS pass: free routing", flush=True)
    rms_free = rms_pass(False)
    print("[cal-stack] RMS pass: residency R=8", flush=True)
    rms_masked = rms_pass(True)
    OlmoeRMSNorm.forward = _orig

    ratio = [rms_free[i] / rms_masked[i].clamp(min=1e-6) for i in range(len(norms))]
    gprime = [base_g[i] * ratio[i].clamp(A.clip_lo, A.clip_hi) for i in range(len(norms))]
    rr = torch.cat([r.reshape(-1) for r in ratio])
    print(f"[cal-stack] RMS_free/RMS_masked: median {float(rr.median()):.4f} "
          f"p5 {float(rr.quantile(0.05)):.4f} p95 {float(rr.quantile(0.95)):.4f} "
          f"clipped frac {float(((rr < A.clip_lo) | (rr > A.clip_hi)).float().mean()):.4f}", flush=True)

    def set_norms(gs):
        with torch.no_grad():
            for nmod, g in zip(norms, gs):
                nmod.weight.data.copy_(g.to(nmod.weight.dtype))

    sub = ids[torch.linspace(0, ids.shape[0] - 1, A.eval_n).long()].long()

    def eval_bpb():
        RES.enable_residency(R=8)
        tot = n = 0
        with torch.no_grad():
            for i in range(sub.shape[0]):
                x = sub[i:i + 1].to("cuda")
                lg = model(x).logits.float()
                tot += float(torch.nn.functional.cross_entropy(
                    lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="sum"))
                n += x[:, 1:].numel()
        return (tot / n) / D

    if PLE_MOD is not None:
        import ple as _P
        _P._STATE["ple"] = PLE_MOD
    set_norms(base_g)
    b_base = eval_bpb()
    print(f"[cal-stack] base norms   @R8 BPB={b_base:.6f} recovery={rec(b_base)*100:.2f}%", flush=True)
    set_norms(gprime)
    b_cal = eval_bpb()
    print(f"[cal-stack] calib norms  @R8 BPB={b_cal:.6f} recovery={rec(b_cal)*100:.2f}%", flush=True)

    # Write the calibrated surface in the trainer's checkpoint layout so the existing tools consume
    # it unchanged: masters = router params (base, uncalibrated) followed by norm params (calibrated).
    out = A.out or os.path.join(DATA_DIR, f"csurf_{A.tag}norms.pt")
    masters = [p.detach().float().cpu() for p in RES.router_params(model)] + \
              [g.detach().float().cpu() for g in gprime]
    torch.save({"masters": masters, "opt": {}, "seen": 0, "step": 0, "pos": 0,
                "hist": [], "lora": 0}, out)
    print(f"[cal-stack] wrote {out} ({len(masters)} master tensors: "
          f"{len(RES.router_params(model))} router + {len(gprime)} norms)", flush=True)

    path = os.path.join(ABLATIONS, "ple_cal_stack.csv")
    import csv as _csv
    _pt = f" + PLE {os.path.basename(A.ple_table)}" if A.ple_table else ""
    rows = [{"stage": f"base norms, base router, R=8{_pt}", "bpb": round(b_base, 6),
             "recovery_pct": round(rec(b_base) * 100, 2), "trained": "no"},
            {"stage": f"calibrated norms (clip {A.clip_lo}-{A.clip_hi}), base router, R=8{_pt}",
             "bpb": round(b_cal, 6), "recovery_pct": round(rec(b_cal) * 100, 2), "trained": "no"}]
    exists = os.path.exists(path)
    with open(path, "a" if exists else "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
