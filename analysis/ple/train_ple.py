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
ap.add_argument("--seed", type=int, default=1234, help="PLE basis init only; data order is seeded 0")
ap.add_argument("--adam8bit", action="store_true", help="8-bit Adam for the PLE table (§2)")
ap.add_argument("--out", default=None, help="defaults to $OLMOE data dir")
A = ap.parse_args()

RANK = A.rank if A.rank in ("off", "full") else int(A.rank)
OUT = A.out or DATA_DIR
SEQ, AUX_C, Z_C = 4096, 0.01, 0.001
IMPOSE_BPB = 2.7507

meta = json.load(open(f"{DATA_DIR}/bpb_slice_meta.json"))
D = meta["divisor_D"]                                        # re-read, never inherited

model, tok = RES.load_model()
RES.enable_residency(R=8)
RES.enable_grad_checkpointing(model)
RES.freeze_all_but_router(model)
rp = RES.router_params(model)
norm_ps = RES.norm_params(model)                             # arm C surface: + learnable RMSNorm gains
extra = norm_ps
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
      f"(router={sum(p.numel() for p in rp)} norms={sum(p.numel() for p in extra)}) "
      f"ple={n_ple} tokens={A.tokens} D={D:.7f}", flush=True)

corpus = torch.load(f"{DATA_DIR}/finetune_ids.pt")
order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))  # same order as the bake-off
bpb_ids = torch.load(f"{DATA_DIR}/bpb_slice_ids.pt")
eval_sub = bpb_ids[torch.linspace(0, bpb_ids.shape[0] - 1, 256).long()].to("cuda").long()
E_experts = model.config.num_experts


def eval_bpb_telem():
    model.eval(); RES.enable_residency(R=8); RES.reset_telem(); RES._CFG["collect_telem"] = True
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
model.train()
t0 = time.time()
while seen < A.tokens:
    if pos + A.mb > corpus.shape[0]:
        pos = 0
    RES._CFG["R"] = 8
    batch = corpus[order[pos:pos + A.mb]].to("cuda").long(); pos += A.mb
    labels = batch[:, 1:].reshape(-1)
    out = model(batch, output_router_logits=True)
    logits = out.logits
    lm = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels)
    aux, z = RES.aux_z_from_router_logits(out.router_logits, batch.shape[0], batch.shape[1], RES._CFG["R"])
    loss = lm + AUX_C * aux + Z_C * z
    if not torch.isfinite(loss):
        print(f"[ABORT] non-finite loss step {step}", flush=True); sys.exit(3)
    loss.backward()
    for m, p in zip(masters, train_params):
        m.grad = p.grad.float() if p.grad is not None else None
    torch.nn.utils.clip_grad_norm_(masters, 1.0)
    opt.step()
    for m, p in zip(masters, train_params):
        p.data.copy_(m.data.to(p.dtype)); p.grad = None
    opt.zero_grad(set_to_none=True)
    if opt_ple is not None:
        torch.nn.utils.clip_grad_norm_(list(ple_mod.parameters()), 1.0)
        opt_ple.step()
        opt_ple.zero_grad(set_to_none=True)
    seen += batch.numel(); step += 1
    if step % 20 == 0:
        print(f"[step {step}] tok={seen/1e6:.1f}M lm={lm.item():.4f} "
              f"{seen/(time.time()-t0)/1e3:.1f}k tok/s", flush=True)
    if seen // A.eval_every > len(hist):
        b, swap, ent = eval_bpb_telem()
        hist.append({"tok": seen, "bpb": b, "swap_rate": swap, "usage_entropy": ent})
        print(f"[eval] {A.tag} rank={RANK} tok={seen/1e6:.0f}M BPB={b:.6f} swap={swap:.4f} ent={ent:.4f}",
              flush=True)
        if b > IMPOSE_BPB:
            print(f"[ABORT] BPB {b:.4f} > impose {IMPOSE_BPB}", flush=True); sys.exit(4)

fb, fswap, fent = eval_bpb_telem()
res = {"tag": A.tag, "rank": str(RANK), "lr": A.lr, "table_wd": A.table_wd,
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
