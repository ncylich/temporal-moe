#!/usr/bin/env python3
"""PLE training cell (PLE_PLAN.md §5), and the flag-off parity control (§4 item 3).

Trains the C surface -- router linears + RMSNorm gains -- optionally co-training a per-layer
embedding table on top. With --rank off the PLE branch is not constructed at all, and the
computation is intended to be the same as the adaptation program's `train_bakeoff.py` arm C:
same corpus, same data order (randperm seeded 0), same micro-batch, same aux/z coefficients,
same fp32-master AdamW, same R=8 eval on the audited slice.

Usage:
  train_ple.py --tag NAME --tokens 10000000 [--rank off|32|128|512|full] [--lr 3e-4]
               [--eval-every 10000000] [--table-wd 0.0] [--mb 16] [--adam8bit]

Reports BPB = CE_nats / D on the audited held-out slice, D re-read from bpb_slice_meta.json
rather than inherited as a literal (PLE_PLAN.md §10).
"""

import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                     # noqa: E402
import ple as PLE                           # noqa: E402
from olmoe_paths import DATA_DIR            # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--rank", default="off", help="off | 32 | 128 | 512 | full")
ap.add_argument("--tokens", type=int, default=10_000_000)
ap.add_argument("--lr", type=float, default=3e-4)          # bake-off winner LR, arm C
ap.add_argument("--ple-lr", type=float, default=None, help="defaults to --lr")
ap.add_argument("--eval-every", type=int, default=10_000_000)
ap.add_argument("--table-wd", type=float, default=0.0,
                help="weight decay on the PLE table. DECIDED: 0 on every rung, so rank is the only "
                     "regularizer and a null at low rank is attributable to rank rather than split "
                     "between rank and decay. Settable, but see row_norms.py for the diagnostic any "
                     "non-zero value should be set against.")
ap.add_argument("--mb", type=int, default=16)
ap.add_argument("--accum", type=int, default=1,
                help="gradient accumulation steps. The effective batch is --mb * --accum and must "
                     "be held at 16 to match the C recipe: the full-rank rung needs --mb 4 --accum 4 "
                     "because activations, not the table, dominate memory.")
ap.add_argument("--seed", type=int, default=1234, help="PLE basis init only; data order is seeded 0")
ap.add_argument("--adam8bit", action="store_true", help="8-bit Adam for the PLE table (§2)")
ap.add_argument("--lora", type=int, default=0,
                help="add per-expert LoRA of this rank to the trained surface, making the base CE "
                     "(router + norms + LoRA r32) instead of C. Default 0 = bare C surface, which "
                     "is what the rank ladder runs on so that rank is isolated.")
ap.add_argument("--ple-start", type=int, default=0,
                help="PHASE 3 (§7, sequential vs joint): withhold PLE updates until this many "
                     "tokens have been seen, so the run is router+norms alone and then router+"
                     "norms+PLE. No checkpoint/resume machinery is needed: the table is zero-init, "
                     "so while its optimizer is not stepped its contribution is exactly 0.0 and "
                     "the first leg is bit-identical to a flag-off run.")
ap.add_argument("--resume-c", default=None,
                help="resume the C surface (router + norm gains + their AdamW state) and the data "
                     "cursor from a csurf_*.pt written by an earlier cell. Lets the second leg of a "
                     "sequential run re-use a shared first leg instead of recomputing it, so two "
                     "second-leg variants differ ONLY in what is being compared.")
ap.add_argument("--calib-suffix", default="",
                help="which calibration table to load: '' = captured on the untrained base, "
                     "'_at50M' = captured against the C surface at 50M. Must match where the table "
                     "is installed, or the correction is measured against the wrong damage.")
ap.add_argument("--calib-init", action="store_true",
                help="initialize the PLE table from calib_table_r<rank>.pt instead of zeros. The "
                     "with/without comparison this enables is NOT covered by the prior program's "
                     "Cal-2 null: norm gains receive gradient every step so any init washes out, "
                     "whereas a PLE row is updated only when its token appears, so a calibrated "
                     "init persists exactly for the rare rows that are underdetermined.")
ap.add_argument("--free-layers", type=int, default=0,
                help="leave the first N MoE layers unconstrained (ordinary free routing) while the "
                     "rest run under rolling residency. This RELAXES the constraint, so such a cell "
                     "is not comparable to a full-residency number without stating the cost: a freed "
                     "layer must keep all 64 experts resident instead of 8, which is +43.8% resident "
                     "expert memory for one layer and +87.5% for two.")
ap.add_argument("--free-set", default="",
                help="explicit comma-separated layer indices to leave UNCONSTRAINED, e.g. 0,1,15. "
                     "Overrides --free-layers. Same relaxation and the same memory cost: each freed "
                     "layer keeps all 64 experts resident instead of 8.")
ap.add_argument("--heldout", action="store_true",
                help="withhold the token ids in ple_heldout.pt from the PLE lookup, so the "
                     "zero-property check tests rows that were eligible to train")
ap.add_argument("--out", default=None, help="defaults to $OLMOE data dir")
A = ap.parse_args()

RANK = A.rank if A.rank in ("off", "full") else int(A.rank)
OUT = A.out or DATA_DIR
SEQ, AUX_C, Z_C = 4096, 0.01, 0.001
IMPOSE_BPB = 2.7507

meta = json.load(open(f"{DATA_DIR}/bpb_slice_meta.json"))
D = meta["divisor_D"]                                        # re-read, never inherited

model, tok = RES.load_model()
RES.enable_residency(R=8, free_layers=A.free_layers)
RES.enable_grad_checkpointing(model)
_FREE = [int(x) for x in A.free_set.split(",") if x.strip() != ""]
if _FREE:
    RES.set_free_layers(_FREE)
    _slots = (16 - len(_FREE)) * 8 + len(_FREE) * 64
    print(f"[resid] layers {_FREE} UNCONSTRAINED, the rest under rolling residency R=8; "
          f"resident expert slots {_slots} vs 128 (+{_slots/128*100-100:.1f}% memory)", flush=True)
elif A.free_layers:
    print(f"[resid] first {A.free_layers} MoE layer(s) UNCONSTRAINED; layers {A.free_layers}-15 "
          f"under rolling residency R=8", flush=True)
RES.freeze_all_but_router(model)
rp = RES.router_params(model)
norm_ps = RES.norm_params(model)                             # arm C surface: + learnable RMSNorm gains
extra = norm_ps
for p in extra:
    p.requires_grad = True

if A.lora:
    # CE surface. Mechanisms ADD here: C and E tie as alternatives (91.44% each) but stacked reach
    # 93.16%, so the bare C surface the ladder runs on is deliberately weakened and cannot reach
    # §1's claim. add_lora starts as an exact no-op (B factors zero-init).
    extra = extra + RES.add_lora(model, r=A.lora, alpha=2 * A.lora)
    for p in extra:
        p.requires_grad = True

train_params = rp + extra
masters = [p.detach().float().clone().requires_grad_(True) for p in train_params]
opt = torch.optim.AdamW(masters, lr=A.lr, betas=(0.9, 0.95), weight_decay=0.0)

# ---- the PLE branch, entirely absent when --rank off ----
ple_mod = None
opt_ple = None
if RANK != "off":
    ple_mod = PLE.install(model, RANK, device="cuda", seed=A.seed)
    if A.calib_init:
        cpath = os.path.join(DATA_DIR, f"calib_table_r{RANK}{A.calib_suffix}.pt")
        cs = torch.load(cpath, map_location="cuda")
        with torch.no_grad():
            if RANK == "full":
                ple_mod.P.copy_(cs["P"].to("cuda"))
            else:
                ple_mod.U.copy_(cs["U"].to("cuda")); ple_mod.V.copy_(cs["V"].to("cuda"))
        tp = ple_mod.table_params()[0]
        print(f"[ple] calibrated init from {os.path.basename(cpath)}: "
              f"||table||={float(tp.norm()):.4f}, "
              f"{int((tp.reshape(tp.shape[0],-1)!=0).any(-1).sum())} rows nonzero", flush=True)
    ho_path = os.path.join(DATA_DIR, "ple_heldout.pt")
    if A.heldout and os.path.exists(ho_path):
        ho = torch.load(ho_path)
        ple_mod.set_heldout(ho["ids"])
        if A.calib_init:
            # A calibrated init would seed the held-out rows too, and the zero-property check
            # requires them bit-zero. They are masked out of the forward regardless, so zeroing
            # them changes nothing the model can see and keeps the invariant testable.
            with torch.no_grad():
                for t in ple_mod.table_params():
                    t[ho["ids"].long()] = 0.0
            print("[ple] zeroed held-out rows after calibrated init (masked in forward anyway)",
                  flush=True)
        print(f"[ple] held out {ho['ids'].numel()} token ids "
              f"({ho['loss_share']*100:.4f}% of eval loss) for the zero-property check", flush=True)
    groups = [
        {"params": ple_mod.table_params(), "weight_decay": A.table_wd},   # the table: decay wired here
        {"params": ple_mod.basis_params(), "weight_decay": 0.0},          # basis + gates: no decay
    ]
    if A.adam8bit:
        from bitsandbytes.optim import AdamW8bit
        opt_ple = AdamW8bit(groups, lr=A.ple_lr or A.lr, betas=(0.9, 0.95))
    else:
        opt_ple = torch.optim.AdamW(groups, lr=A.ple_lr or A.lr, betas=(0.9, 0.95))

n_ple = ple_mod.n_params() if ple_mod else 0
print(f"[ple] tag={A.tag} rank={RANK} lr={A.lr} "
      f"table_wd={'n/a (flag off)' if RANK == 'off' else A.table_wd} adam8bit={A.adam8bit} "
      f"trainable_C={sum(p.numel() for p in train_params)} "
      f"(router={sum(p.numel() for p in rp)} extra={sum(p.numel() for p in extra)} lora_r={A.lora}) "
      f"ple={n_ple} tokens={A.tokens} D={D:.7f}", flush=True)

corpus = torch.load(f"{DATA_DIR}/finetune_ids.pt")
order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))  # same order as the bake-off
bpb_ids = torch.load(f"{DATA_DIR}/bpb_slice_ids.pt")
eval_sub = bpb_ids[torch.linspace(0, bpb_ids.shape[0] - 1, 256).long()].to("cuda").long()
E_experts = model.config.num_experts


def eval_bpb_telem():
    model.eval(); RES.enable_residency(R=8, free_layers=A.free_layers)
    if _FREE:
        RES.set_free_layers(_FREE)
    RES.reset_telem(); RES._CFG["collect_telem"] = True
    tot = n = 0
    with torch.no_grad():
        for i in range(eval_sub.shape[0]):
            x = eval_sub[i:i + 1]
            out = model(x).logits.float()
            l = torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)),
                                                  x[:, 1:].reshape(-1), reduction="sum")
            tot += l.item(); n += x[:, 1:].numel()
    RES._CFG["collect_telem"] = False; model.train()
    swap, ent = RES.telem_summary(E_experts)
    return (tot / n) / D, swap, ent


seen = step = pos = 0
hist = []
if A.resume_c:
    _ck = torch.load(A.resume_c, map_location="cuda")
    with torch.no_grad():
        for _m, _s in zip(masters, _ck["masters"]):
            _m.data.copy_(_s.to("cuda"))
        for _m, _p in zip(masters, train_params):
            _p.data.copy_(_m.data.to(_p.dtype))
    # A synthetic surface (one written by cal_stack.py from closed-form calibration rather than by
    # training) carries no optimizer state. Resuming those means starting Adam fresh, which is
    # correct: there is no moment history to continue.
    if _ck.get("opt"):
        opt.load_state_dict(_ck["opt"])
    else:
        print("[resume] checkpoint carries no optimizer state; starting Adam fresh", flush=True)
    seen, step, pos = _ck["seen"], _ck["step"], _ck["pos"]
    hist = list(_ck["hist"])
    print(f"[resume] C surface from {os.path.basename(A.resume_c)}: seen={seen/1e6:.0f}M "
          f"step={step} pos={pos} evals={len(hist)}; data cursor continues, no repeat", flush=True)
# Detached scalars only, meaned at eval time. Kept as tensors so no per-step device sync is added:
# the training math is untouched, this is bookkeeping. The train/eval gap at each eval point is the
# generalization measure, and inferring it from the printed step lines afterwards is lossier.
lm_acc = []
_ple_live = [False]
model.train()
t0 = time.time()
while seen < A.tokens:
    RES._CFG["R"] = 8
    # One optimizer step over --accum micro-batches. Gradients accumulate in p.grad across the
    # inner loop; the master copy, clipping and step happen once, so the effective batch is
    # --mb * --accum regardless of how it is split.
    lm_last = None
    for _micro in range(A.accum):
        if pos + A.mb > corpus.shape[0]:
            pos = 0
        batch = corpus[order[pos:pos + A.mb]].to("cuda").long(); pos += A.mb
        labels = batch[:, 1:].reshape(-1)
        out = model(batch, output_router_logits=True)
        logits = out.logits
        lm = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels)
        aux, z = RES.aux_z_from_router_logits(out.router_logits, batch.shape[0], batch.shape[1],
                                              RES._CFG["R"])
        loss = (lm + AUX_C * aux + Z_C * z) / A.accum
        if not torch.isfinite(loss):
            print(f"[ABORT] non-finite loss step {step} micro {_micro}", flush=True); sys.exit(3)
        loss.backward()
        seen += batch.numel(); lm_last = lm
        lm_acc.append(lm.detach())
        del out, logits, loss
    lm = lm_last
    for m, p in zip(masters, train_params):
        m.grad = p.grad.float() if p.grad is not None else None
    torch.nn.utils.clip_grad_norm_(masters, 1.0)
    opt.step()
    for m, p in zip(masters, train_params):
        p.data.copy_(m.data.to(p.dtype)); p.grad = None
    opt.zero_grad(set_to_none=True)
    if opt_ple is not None:
        if seen >= A.ple_start:
            torch.nn.utils.clip_grad_norm_(list(ple_mod.parameters()), 1.0)
            opt_ple.step()
            if A.ple_start and not _ple_live[0]:
                print(f"[ple] table began updating at {seen/1e6:.1f}M tokens (--ple-start "
                      f"{A.ple_start/1e6:.0f}M)", flush=True)
                _ple_live[0] = True
        opt_ple.zero_grad(set_to_none=True)
    step += 1
    if step % 20 == 0:
        print(f"[step {step}] tok={seen/1e6:.1f}M lm={lm.item():.4f} "
              f"{seen/(time.time()-t0)/1e3:.1f}k tok/s", flush=True)
    if seen // A.eval_every > len(hist):
        b, swap, ent = eval_bpb_telem()
        train_lm = float(torch.stack(lm_acc).mean()) if lm_acc else float("nan")
        lm_acc = []
        hist.append({"tok": seen, "bpb": b, "swap_rate": swap, "usage_entropy": ent,
                     "train_lm": train_lm, "train_bpb": train_lm / D})
        print(f"[eval] {A.tag} rank={RANK} tok={seen/1e6:.0f}M BPB={b:.6f} "
              f"train_lm={train_lm:.6f} train_bpb={train_lm/D:.6f} gap={train_lm/D - b:+.6f} "
              f"swap={swap:.4f} ent={ent:.4f}", flush=True)
        # Table snapshot at every eval, so rare-row growth is a trajectory rather than one
        # post-hoc point. Pure I/O; nothing here feeds back into training.
        torch.save({"masters": [m.detach().cpu() for m in masters], "opt": opt.state_dict(),
                    "seen": seen, "step": step, "pos": pos, "hist": hist, "lora": A.lora},
                   f"{OUT}/csurf_{A.tag}_at{seen // 10**6}M.pt")
        if ple_mod is not None:
            torch.save({"rank": str(RANK),
                        **{k: v.detach().cpu() for k, v in ple_mod.state_dict().items()}},
                       f"{OUT}/ple_table_{A.tag}_at{seen // 10**6}M.pt")
        if b > IMPOSE_BPB:
            print(f"[ABORT] BPB {b:.4f} > impose {IMPOSE_BPB}", flush=True); sys.exit(4)

fb, fswap, fent = eval_bpb_telem()
res = {"tag": A.tag, "rank": str(RANK), "lr": A.lr, "table_wd": A.table_wd, "lora": A.lora,
       "ple_start": A.ple_start, "calib_init": A.calib_init, "free_layers": A.free_layers, "free_set": A.free_set,
       "calib_suffix": A.calib_suffix,
       "adam8bit": A.adam8bit, "mb": A.mb, "seed": A.seed,
       "train_tokens": seen, "steps": step, "ple_params": n_ple,
       "final_bpb": fb, "final_swap": fswap, "final_entropy": fent,
       "divisor": D, "divisor_source": "bpb_slice_meta.json (ln2 * bytes_per_token)",
       "curve": hist}
json.dump(res, open(f"{OUT}/ple_{A.tag}.json", "w"), indent=1)
if ple_mod is not None:
    torch.save({"rank": str(RANK),
                **{k: v.detach().cpu() for k, v in ple_mod.state_dict().items()}},
               f"{OUT}/ple_table_{A.tag}.pt")
print(f"[DONE] tag={A.tag} rank={RANK} final BPB={fb:.6f} swap={fswap:.4f} ent={fent:.4f}", flush=True)
