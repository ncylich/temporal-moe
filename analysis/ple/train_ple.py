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
ap.add_argument("--lr", type=float, default=3e-4)          # bake-off winner LR, arm C --
# NOTE: arm C was a PLE bake-off, i.e. this LR was fitted for per-layer-embedding adaptation,
# not for residency, and the OLMoE runs it was carried into were under the gate-mass artifact.
# It has never been validated for this task on a correctly configured model.
# None = use the model's shipped router_aux_loss_coef. OLMoE declares 0.01, which is what every
# run to date used, so this is a no-op for OLMoE -- it matters for the Qwen models, which
# declare 0.001 and were nonetheless trained at 0.01.
ap.add_argument("--aux-c", type=float, default=None)
# Distillation (mirrors train_unsloth): KL to the pristine unconstrained base -- expert +
# attn LoRA disabled by zeroing their scales, router/norm C-surface swapped to its initial
# values for the teacher pass, residency off. Loss scaled by T^2 (gradient T-invariant).
ap.add_argument("--distill", action="store_true")
ap.add_argument("--distill-T", type=float, default=1.0)
ap.add_argument("--aux-scope", default="micro", choices=("micro", "global"))
ap.add_argument("--z-c", type=float, default=0.001)
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
ap.add_argument("--seed", type=int, default=1234, help="PLE basis init only; data order is --data-seed")
ap.add_argument("--data-seed", type=int, default=0,
                help="seed for the corpus permutation, i.e. WHICH packs a cell trains on and in what "
                     "order. Default 0 is the bake-off order every published cell used, so leaving it "
                     "alone reproduces them exactly. Changing it is how a replicate is taken: --seed "
                     "moves only the PLE basis init, so on a --rank off cell it moves nothing at all, "
                     "and a rerun would differ only by Flash Attention's non-deterministic backward "
                     "(~0.0024 BPB). A replicate that varies the data draw is the one that bears on "
                     "whether a margin against a published number is real.")
ap.add_argument("--adam8bit", action="store_true", help="8-bit Adam for the PLE table (§2)")
ap.add_argument("--lora", type=int, default=0,
                help="add per-expert LoRA of this rank to the trained surface, making the base CE "
                     "(router + norms + LoRA r32) instead of C. Default 0 = bare C surface, which "
                     "is what the rank ladder runs on so that rank is isolated.")
ap.add_argument("--lora-attn", type=int, default=0,
                help="add LoRA of this rank to the attention projections (q/k/v/o) on top of "
                     "whatever --lora does to the experts. Attention is frozen in every published "
                     "arm of this program, including F' the full-parameter finetune, so the "
                     "'irreducible constraint price' was measured without it ever being asked to "
                     "help. Residency restricts which experts a token may reach; attention decides "
                     "what the token carries when it gets there.")
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
if A.aux_c is None:
    A.aux_c = float(getattr(model.config, 'router_aux_loss_coef', 0.01))
    print(f"  [aux] using the model's shipped router_aux_loss_coef = {A.aux_c}", flush=True)
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

if A.lora_attn:
    # Attention is frozen in every arm of this program, F' included, so the constraint price was
    # established without testing whether attention can absorb any of it. Added last so the
    # parameter order is a prefix-extension of a run without it, which is what lets a checkpoint
    # from such a run be resumed into one with it.
    extra = extra + RES.add_lora_attn(model, r=A.lora_attn, alpha=2 * A.lora_attn)
    for p in extra:
        p.requires_grad = True

train_params = rp + extra
masters = [p.detach().float().clone().requires_grad_(True) for p in train_params]
opt = torch.optim.AdamW(masters, lr=A.lr, betas=(0.9, 0.95), weight_decay=0.0)

if A.distill:
    # Teacher machinery. rp (router linears + RMSNorm gains) train in-place, so the teacher
    # needs their INITIAL values (snapshotted here, before any step). The LoRA deltas are
    # additive with explicit scales, so zeroing the scales disables them exactly.
    _rp_pristine = [p.detach().clone() for p in rp]
    _attn_lora_lins = [lin for m in model.modules() if type(m).__name__ == "OlmoeAttention"
                       for lin in (getattr(m, n, None) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
                       if lin is not None and hasattr(lin, "_lora_scale")]

    def _teacher_logits(batch):
        saved_scale = RES._LORA["scale"]
        RES._LORA["scale"] = 0.0
        saved_attn = [lin._lora_scale for lin in _attn_lora_lins]
        for lin in _attn_lora_lins:
            lin._lora_scale = 0.0
        cur = [p.detach().clone() for p in rp]
        with torch.no_grad():
            for p, pr in zip(rp, _rp_pristine):
                p.data.copy_(pr)
        was_on = RES._CFG["on"]
        RES._CFG["on"] = False
        with torch.no_grad():
            lg_t = model(batch).logits.float()
        RES._CFG["on"] = was_on
        with torch.no_grad():
            for p, c in zip(rp, cur):
                p.data.copy_(c)
        for lin, s in zip(_attn_lora_lins, saved_attn):
            lin._lora_scale = s
        RES._LORA["scale"] = saved_scale
        return lg_t

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
      f"(router={sum(p.numel() for p in rp)} extra={sum(p.numel() for p in extra)} lora_r={A.lora} lora_attn={A.lora_attn}) "
      f"ple={n_ple} tokens={A.tokens} D={D:.7f}", flush=True)

corpus = torch.load(f"{DATA_DIR}/finetune_ids.pt")
order = torch.randperm(corpus.shape[0],
                       generator=torch.Generator().manual_seed(A.data_seed))  # 0 = the bake-off order
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
    # Per-layer effective expert count, on the first eval batch, in whatever regime this cell runs.
    # One extra forward with router logits: cheap next to the eval itself, and it is the quantity the
    # aux loss exists to hold up, so a cell that silently collapses its routing is visible here and
    # nowhere else in the recorded output.
    eff = RES.effective_experts(
        model(eval_sub[0:1], output_router_logits=True).router_logits, 1, eval_sub.shape[1], 8)
    RES._CFG["collect_telem"] = False; model.train()
    swap, ent = RES.telem_summary(E_experts)
    return (tot / n) / D, swap, ent, eff


seen = step = pos = 0
hist = []
if A.resume_c:
    _ck = torch.load(A.resume_c, map_location="cuda")
    # zip() stops at the shorter sequence, so a length mismatch here loads a prefix and says
    # nothing. That is the RIGHT behaviour when this run adds a mechanism the checkpoint predates
    # -- the new parameters stay at their zero init and start contributing from step 0 of the
    # second leg -- and the WRONG behaviour if the recipes simply disagree. Both look identical
    # without this, so state which it is.
    if len(_ck["masters"]) != len(masters):
        if len(_ck["masters"]) > len(masters):
            sys.exit(f"[abort] {os.path.basename(A.resume_c)} holds {len(_ck['masters'])} tensors, "
                     f"this run has {len(masters)}. The checkpoint was written by a LARGER recipe; "
                     f"resuming would silently drop what it trained.")
        print(f"[resume] checkpoint has {len(_ck['masters'])} tensors, this run has {len(masters)}: "
              f"the {len(masters) - len(_ck['masters'])} added parameters stay at their zero init "
              f"and begin training now. Verify this is a mechanism being ADDED, not a recipe "
              f"mismatch.", flush=True)
        for _i, (_a, _b) in enumerate(zip(masters, _ck["masters"])):
            if tuple(_a.shape) != tuple(_b.shape):
                sys.exit(f"[abort] tensor {_i} is {tuple(_a.shape)} here and {tuple(_b.shape)} in "
                         f"the checkpoint; the shared prefix does not line up, so this is a recipe "
                         f"mismatch rather than an extension.")
    with torch.no_grad():
        for _m, _s in zip(masters, _ck["masters"]):
            _m.data.copy_(_s.to("cuda"))
        for _m, _p in zip(masters, train_params):
            _p.data.copy_(_m.data.to(_p.dtype))
    # A synthetic surface (one written by cal_stack.py from closed-form calibration rather than by
    # training) carries no optimizer state. Resuming those means starting Adam fresh, which is
    # correct: there is no moment history to continue.
    if not _ck.get("opt"):
        print("[resume] checkpoint carries no optimizer state; starting Adam fresh", flush=True)
    elif len(_ck["masters"]) != len(masters):
        # AdamW's state_dict indexes its parameter group positionally, so it cannot be loaded into
        # an optimizer holding a different number of tensors -- it raises rather than misaligning,
        # which is the safe failure but still ends the run. Starting fresh loses the moment history
        # for the shared prefix; that cost is stated rather than hidden.
        print(f"[resume] optimizer state covers {len(_ck['masters'])} tensors but this run has "
              f"{len(masters)}; starting Adam fresh. The shared parameters resume at their trained "
              f"values but without their moment history.", flush=True)
    else:
        opt.load_state_dict(_ck["opt"])
    # `pos` is an index into `order`, which is rebuilt here from --data-seed. Resuming under a
    # different seed would carry the cursor into a different permutation: the run would silently
    # re-see packs the first leg already trained on and skip others, with nothing in the output
    # saying so. Same for the free set -- a checkpoint trained with layers 0,1,15 unconstrained is
    # not a starting point for a differently-constrained model.
    for _k, _mine in (("data_seed", A.data_seed), ("free_set", A.free_set)):
        _theirs = _ck.get(_k)
        if _theirs is not None and _theirs != _mine:
            sys.exit(f"[abort] {os.path.basename(A.resume_c)} was written with {_k}={_theirs!r}, "
                     f"this run has {_k}={_mine!r}. Pass --{_k.replace('_', '-')} {_theirs!r}.")
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
        if A.distill:
            lg_t = _teacher_logits(batch)
        out = model(batch, output_router_logits=True)
        logits = out.logits
        if A.distill:
            # Forward KL(teacher || student) at T, position-averaged, T^2-scaled.
            _T = A.distill_T
            _pt = torch.softmax(lg_t / _T, -1)
            lm = (_T * _T) * (
                -(_pt * torch.log_softmax(logits.float() / _T, -1)).sum(-1).mean()
                + (_pt * torch.log_softmax(lg_t / _T, -1)).sum(-1).mean())
            del lg_t, _pt
        else:
            lm = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels)
        aux, z = RES.aux_z_from_router_logits(out.router_logits, batch.shape[0], batch.shape[1],
                                              RES._CFG["R"])
        RES.assert_aux_live(out, aux, A.aux_c) if A.aux_c else None
        loss = (lm + A.aux_c * aux + A.z_c * z) / A.accum
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
        b, swap, ent, eff = eval_bpb_telem()
        train_lm = float(torch.stack(lm_acc).mean()) if lm_acc else float("nan")
        lm_acc = []
        hist.append({"tok": seen, "bpb": b, "swap_rate": swap, "usage_entropy": ent,
                     "eff_load": [e["eff_load"] for e in eff],
                     "eff_tok": [e["eff_tok"] for e in eff],
                     "train_lm": train_lm, "train_bpb": train_lm / D})
        print(f"[eval] {A.tag} rank={RANK} tok={seen/1e6:.0f}M BPB={b:.6f} "
              f"train_lm={train_lm:.6f} train_bpb={train_lm/D:.6f} gap={train_lm/D - b:+.6f} "
              f"swap={swap:.4f} ent={ent:.4f} "
              f"eff_load[min/med/max]={min(e['eff_load'] for e in eff):.1f}/"
              f"{sorted(e['eff_load'] for e in eff)[len(eff)//2]:.1f}/"
              f"{max(e['eff_load'] for e in eff):.1f}", flush=True)
        # Table snapshot at every eval, so rare-row growth is a trajectory rather than one
        # post-hoc point. Pure I/O; nothing here feeds back into training.
        torch.save({"masters": [m.detach().cpu() for m in masters], "opt": opt.state_dict(),
                    "seen": seen, "step": step, "pos": pos, "hist": hist, "lora": A.lora,
                    "data_seed": A.data_seed, "free_set": A.free_set,
                    "lora_attn": A.lora_attn},
                   f"{OUT}/csurf_{A.tag}_at{seen // 10**6}M.pt")
        if ple_mod is not None:
            torch.save({"rank": str(RANK),
                        **{k: v.detach().cpu() for k, v in ple_mod.state_dict().items()}},
                       f"{OUT}/ple_table_{A.tag}_at{seen // 10**6}M.pt")
        if b > IMPOSE_BPB:
            print(f"[ABORT] BPB {b:.4f} > impose {IMPOSE_BPB}", flush=True); sys.exit(4)

fb, fswap, fent, feff = eval_bpb_telem()
res = {"tag": A.tag, "rank": str(RANK), "lr": A.lr, "table_wd": A.table_wd, "lora": A.lora,
       "ple_start": A.ple_start, "calib_init": A.calib_init, "free_layers": A.free_layers, "free_set": A.free_set,
       "calib_suffix": A.calib_suffix,
       "adam8bit": A.adam8bit, "mb": A.mb, "seed": A.seed, "data_seed": A.data_seed,
       "lora_attn": A.lora_attn,
       "train_tokens": seen, "steps": step, "ple_params": n_ple,
       "final_bpb": fb, "final_swap": fswap, "final_entropy": fent,
       "final_eff_load": [e["eff_load"] for e in feff],
       "final_eff_tok": [e["eff_tok"] for e in feff],
       "final_eff_freed": [e["freed"] for e in feff],
       "divisor": D, "divisor_source": "bpb_slice_meta.json (ln2 * bytes_per_token)",
       "curve": hist}
json.dump(res, open(f"{OUT}/ple_{A.tag}.json", "w"), indent=1)
if ple_mod is not None:
    torch.save({"rank": str(RANK),
                **{k: v.detach().cpu() for k, v in ple_mod.state_dict().items()}},
               f"{OUT}/ple_table_{A.tag}.pt")
print(f"[DONE] tag={A.tag} rank={RANK} final BPB={fb:.6f} swap={fswap:.4f} ent={fent:.4f}", flush=True)
