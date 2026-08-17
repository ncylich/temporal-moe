#!/usr/bin/env python3
"""CE adaptation of gemma4-26B-A4B-IT to decode-time residency (R=k=8, response tokens only).

Plain cross-entropy on the model's OWN vLLM-generated responses (gemma's greedy outputs are
low-entropy, so hard labels ~ soft labels and distillation buys nothing). The constraint is
enforced exactly as served: prefill free (scan observes the prompt), R=8 from the first
response token, warm; with --micro-batch > 1 rows are length-sorted, padded, and the rule
is applied per row (scan batch columns are independent; trailing pads sit after each row's
response and touch nothing scored). Loss on response tokens only.

Trainable surface: attention LoRA r32 + router projections + RMSNorm gains, plus optional
per-expert LoRA on the 3D expert tensors (--expert-lora-r) via a grouped-GEMM forward
(torch._grouped_mm; 98 -> 2900 tok/s vs the stock eager expert loop at micro-batch 8).

Loads via unsloth when its patches keep our Gemma4TextRouter hook alive; every load asserts
constraint engagement. --smoke runs the full regression gauntlet (grouped-path parity vs the
eager loop, LoRA engagement/restore, batched-plumbing exactness, free/constrained batch
parity, gradient flow, timed steps, save/reload) and exits.

    train_gemma_ce.py --traj gemma4_train5k --tokens 3400000 \
        --expert-lora-r 16 --out .../gemma_ce_expert_adapter.pt
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_expert_lora(model, r):
    """Per-expert LoRA on the 3D expert tensors via torch._grouped_mm.

    The stock HF forward is a Python loop over hit experts (~1500 dispatches per
    layer pass -- overhead-bound, 98 tok/s measured). Here (token, expert) pairs
    are sorted by expert once per forward and each projection becomes ONE grouped
    GEMM over all experts; LoRA deltas are two more grouped GEMMs. Base expert
    weights are transposed IN PLACE to the grouped layout (E, in, out) -- they are
    frozen, our forward is the only consumer, and merge transposes back.
    B zero-init: delta starts at exactly 0."""
    import torch.nn as nn
    patched = 0
    for mod in model.modules():
        gu = getattr(mod, "gate_up_proj", None)
        dp = getattr(mod, "down_proj", None)
        if not (isinstance(gu, nn.Parameter) and gu.dim() == 3
                and isinstance(dp, nn.Parameter) and dp.dim() == 3):
            continue
        E, twoI, H = gu.shape                  # stored (E, 2I, H)
        _, H2, I = dp.shape                    # stored (E, H, I)
        with torch.no_grad():                  # -> grouped layout (E, in, out)
            gu.data = gu.data.transpose(1, 2).contiguous()      # (E, H, 2I)
            dp.data = dp.data.transpose(1, 2).contiguous()      # (E, I, H)
        dev, dt = gu.device, gu.dtype
        mod.elora_gu_A = nn.Parameter(torch.randn(E, H, r, device=dev, dtype=dt) / r)
        mod.elora_gu_B = nn.Parameter(torch.zeros(E, r, twoI, device=dev, dtype=dt))
        mod.elora_dp_A = nn.Parameter(torch.randn(E, I, r, device=dev, dtype=dt) / r)
        mod.elora_dp_B = nn.Parameter(torch.zeros(E, r, H2, device=dev, dtype=dt))
        mod.elora_scale = 2.0                  # alpha/r with alpha = 2r

        def fwd(self, hidden_states, top_k_index, top_k_weights):
            T, k = top_k_index.shape
            Emax = self.num_experts
            flat_e = top_k_index.reshape(-1)
            keep = flat_e < Emax               # sentinel guard (reference loop skips E)
            if not bool(keep.all()):
                flat_e = flat_e[keep]
            order = torch.argsort(flat_e, stable=True)
            src_idx = (torch.arange(T * k, device=flat_e.device)[keep]
                       if not bool(keep.all())
                       else torch.arange(T * k, device=flat_e.device))[order]
            tok = src_idx // k
            x = hidden_states.index_select(0, tok)
            offs = torch.bincount(
                flat_e[order], minlength=Emax).cumsum(0).to(torch.int32)
            s = self.elora_scale
            g = torch._grouped_mm
            gate_up = g(x, self.gate_up_proj, offs=offs) + s * g(
                g(x, self.elora_gu_A, offs=offs), self.elora_gu_B, offs=offs)
            gate, up = gate_up.chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            down = g(h, self.down_proj, offs=offs) + s * g(
                g(h, self.elora_dp_A, offs=offs), self.elora_dp_B, offs=offs)
            w = top_k_weights.reshape(-1)[src_idx].unsqueeze(1)
            # deterministic combine: one unique slot per (token, expert-slot) pair
            # (a single index_add_ over duplicate token ids would use atomics --
            # nondeterministic run-to-run; the smoke restore-gate caught this)
            contrib = torch.zeros(T * k, hidden_states.shape[1],
                                  device=down.device, dtype=down.dtype)
            contrib = contrib.index_copy(0, src_idx, down * w)
            return contrib.view(T, k, -1).sum(1).to(hidden_states.dtype)

        import types
        mod.forward = types.MethodType(fwd, mod)
        patched += 1
    assert patched, "no 3D expert tensors found to patch"
    print(f"[gce] expert LoRA r={r} (grouped_mm path) on {patched} layers", flush=True)
    return patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/gemma4-26b-it")
    ap.add_argument("--traj", default="gemma4_train5k")
    ap.add_argument("--tokens", type=int, default=3_400_000, help="response-token budget")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--extra-lr-div", type=float, default=5.0,
                    help="router/norm full-weight lr = lr / this")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--out", default="/workspace/olmoe-adapt/data/gemma_ce_adapter.pt")
    ap.add_argument("--no-constraint", action="store_true",
                    help="CONTROL: identical run with residency OFF during training "
                         "(isolates constraint-aware adaptation from plain self-SFT)")
    ap.add_argument("--eval-only", action="store_true",
                    help="load --out adapter, score frozen-500 self-CE (free and R8), exit")
    ap.add_argument("--expert-lora-r", type=int, default=0,
                    help="per-expert LoRA rank on the 3D expert tensors (0 = off); "
                         "delta applied per HIT expert inside the loop -- never "
                         "materialises the full-tensor delta")
    ap.add_argument("--max-seq", type=int, default=1024,
                    help="loader max sequence length; think-on trajectories need "
                         "2048 (prompt 512 + think+answer 1024+)")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="rows per forward; rows are length-sorted into chunks and "
                         "padded to the chunk max. Constraint applied per row via "
                         "CFG batch/enforce_from-vector. 16 rows per optimizer "
                         "step regardless (accum adjusts)")
    ap.add_argument("--precompute-kl", default=None,
                    help="forward-only pass storing the BASE model's top-50 free-"
                         "routing logprobs per response token over the trajectory "
                         "set; feeds --kl-anchor. Run on the UNADAPTED model")
    ap.add_argument("--kl-anchor", default=None,
                    help="path from --precompute-kl: adds kl-weight * KL(student "
                         "free-mode || base top-50) on response tokens (anti-"
                         "forgetting anchor)")
    ap.add_argument("--kl-weight", type=float, default=0.1)
    ap.add_argument("--precompute-tokw", default=None,
                    help="forward-only pass over all trajectories computing per-"
                         "response-token CE under free and R8 routing; saves "
                         "{row_index: (ce_free fp16, ce_r8 fp16)} to PATH, then "
                         "exits. Feeds --tok-weights")
    ap.add_argument("--tok-weights", default=None,
                    help="path from --precompute-tokw: weight response-token CE by "
                         "w = 1 + 2*clip(ce_R8 - ce_free, 0, 3) (constraint-"
                         "disagreement weighting)")
    ap.add_argument("--smoke", action="store_true",
                    help="engagement checks + 2 timed steps + save/reload, then exit")
    ap.add_argument("--merge-scale", type=float, default=1.0,
                    help="scale the adapter delta at merge: LoRA B tensors are "
                         "multiplied by s; full-weight tensors (router/norm) are "
                         "interpolated base*(1-s)+ckpt*s. 1.0 = full adapter")
    ap.add_argument("--merge-out", default=None,
                    help="after loading the adapter, save the merged model to this dir and exit")
    A = ap.parse_args()

    assert not (A.expert_lora_r and A.out.endswith("gemma_ce_adapter.pt")), \
        "expert-LoRA run would overwrite the attention-only adapter; pass --out"
    rows = torch.load(f"/workspace/instruct-traj/{A.traj}.pt", weights_only=False)["rows"]
    print(f"[gce] {len(rows)} trajectories", flush=True)

    import granularity_ladder as GL
    use_unsloth = True
    try:
        from unsloth import FastModel
        model, tok = FastModel.from_pretrained(A.model, max_seq_length=A.max_seq,
                                               dtype=torch.bfloat16, load_in_4bit=False,
                                               full_finetuning=False)
        tok = getattr(tok, "tokenizer", tok)
        model = FastModel.get_peft_model(model, r=32, lora_alpha=64, lora_dropout=0.0,
                                         use_gradient_checkpointing=True)
    except Exception as e:
        print(f"[gce] unsloth path failed ({type(e).__name__}: {e}); falling back to HF+peft",
              flush=True)
        use_unsloth = False
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        tok = AutoTokenizer.from_pretrained(A.model)
        model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16).to("cuda")
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    GL.patch_gemma4()
    GL.tag_gemma4(model)
    PAD = tok.pad_token_id or 0

    def make_batch(rs, ridx=None):
        S = max(r["ids"].shape[0] for r in rs)
        B = len(rs)
        ids = torch.full((B, S), PAD, dtype=torch.long)
        am = torch.zeros((B, S), dtype=torch.long)
        tgt = torch.full((B, S), -100, dtype=torch.long)
        plens, ntok = [], 0
        for b, r_ in enumerate(rs):
            L = r_["ids"].shape[0]
            pl = int(r_["prompt_len"])
            ids[b, :L] = r_["ids"]
            am[b, :L] = 1
            tgt[b, pl:L] = r_["ids"][pl:L]
            plens.append(pl)
            ntok += L - pl
        return (ids.to("cuda"), am.to("cuda"), tgt.to("cuda"), plens, ntok)

    KLREF = None
    if A.kl_anchor:
        KLREF = torch.load(A.kl_anchor, weights_only=False)
        print(f"[gce] KL anchor loaded for {len(KLREF)} rows "
              f"(weight {A.kl_weight})", flush=True)

    TOKW = None
    if A.tok_weights:
        TOKW = torch.load(A.tok_weights, weights_only=False)
        print(f"[gce] disagreement weights loaded for {len(TOKW)} rows "
              f"(w = 1 + 2*clip(ce_R8 - ce_free, 0, 3))", flush=True)

    def batch_ce(logits, tgt, scale, ridx=None):
        """Mean response-token CE over a padded batch; per-row float() keeps the
        [B,S,V] float materialisation bounded. With --tok-weights, tokens where
        the BASE model's constrained CE exceeds its free CE are upweighted."""
        targets = tgt[:, 1:]
        num = den = 0
        for b in range(targets.shape[0]):
            m_ = targets[b] != -100
            if not bool(m_.any()):
                continue
            ce = torch.nn.functional.cross_entropy(
                logits[b][m_].float(), targets[b][m_], reduction="none")
            if TOKW is not None and ridx is not None:
                cf, cr = TOKW[ridx[b]]
                assert cf.shape[0] == ce.shape[0], \
                    f"weight len {cf.shape[0]} vs resp len {ce.shape[0]} row {ridx[b]}"
                w = 1.0 + 2.0 * (cr.float() - cf.float()).clamp(0, 3)
                w = w.to(ce.device)
                num = num + (ce * w).sum()
                den += float(w.sum())
            else:
                num = num + ce.sum()
                den += int(m_.sum())
        return num / den / scale

    if A.precompute_kl:
        import shutil
        model.eval()
        mb = max(1, A.micro_batch)
        lidx = sorted(range(len(rows)), key=lambda i: rows[i]["ids"].shape[0])
        outk = {}
        with torch.no_grad():
            for c0 in range(0, len(lidx), mb):
                ridx = lidx[c0:c0 + mb]
                rs = [rows[i] for i in ridx]
                ids, am, tgt, plens, _ = make_batch(rs)
                GL.CFG.update(on=False, enforce_from=0, batch=len(rs))
                lg = model(ids, attention_mask=am).logits[:, :-1]
                targets = tgt[:, 1:]
                for b, ri in enumerate(ridx):
                    m_ = targets[b] != -100
                    lp = torch.log_softmax(lg[b][m_].float(), -1)
                    top = lp.topk(50, dim=-1)
                    outk[ri] = (top.indices.to(torch.int32).cpu(),
                                top.values.half().cpu())
                if (c0 // mb) % 100 == 0:
                    print(f"[gce-kl] {c0}/{len(lidx)}", flush=True)
        GL.CFG.update(batch=1)
        torch.save(outk, "/tmp/gce_kl_tmp.pt")
        shutil.move("/tmp/gce_kl_tmp.pt", A.precompute_kl)
        print(f"[gce-kl] saved {len(outk)} rows -> {A.precompute_kl} -- DONE",
              flush=True)
        return

    if A.precompute_tokw:
        import shutil
        model.eval()
        mb = max(1, A.micro_batch)
        lidx = sorted(range(len(rows)), key=lambda i: rows[i]["ids"].shape[0])
        outw = {}
        t0 = time.time()
        with torch.no_grad():
            for c0 in range(0, len(lidx), mb):
                ridx = lidx[c0:c0 + mb]
                rs = [rows[i] for i in ridx]
                ids, am, tgt, plens, _ = make_batch(rs)
                per = {}
                for on in (False, True):
                    GL.CFG.update(on=on, R=8, enforce_from=plens,
                                  batch=len(rs), cold_start=False)
                    lg = model(ids, attention_mask=am).logits[:, :-1]
                    targets = tgt[:, 1:]
                    for b, ri in enumerate(ridx):
                        m_ = targets[b] != -100
                        ce = torch.nn.functional.cross_entropy(
                            lg[b][m_].float(), targets[b][m_],
                            reduction="none").half().cpu()
                        per.setdefault(ri, []).append(ce)
                for ri, (cf, cr) in per.items():
                    outw[ri] = (cf, cr)
                if (c0 // mb) % 50 == 0:
                    print(f"[gce-tokw] {c0}/{len(lidx)} rows "
                          f"({(time.time()-t0):.0f}s)", flush=True)
        GL.CFG.update(batch=1)
        torch.save(outw, "/tmp/gce_tokw_tmp.pt")
        shutil.move("/tmp/gce_tokw_tmp.pt", A.precompute_tokw)
        up = sum(float(((cr.float()-cf.float()).clamp(0,3) > 0.1).float().mean())
                 for cf, cr in outw.values()) / len(outw)
        print(f"[gce-tokw] saved {len(outw)} rows -> {A.precompute_tokw}; "
              f"mean frac tokens upweighted {up:.3f} -- DONE", flush=True)
        return

    smoke_ref = None
    if A.expert_lora_r:
        if A.smoke:  # reference logits from the UNPATCHED forward, free routing
            _probe = rows[0]["ids"][:256].to("cuda").long().unsqueeze(0)
            with torch.no_grad():
                GL.CFG.update(on=False, enforce_from=0)
                smoke_ref = model(_probe).logits[:, -1].float()
        add_expert_lora(model, A.expert_lora_r)

    # router + norm gains trainable alongside the LoRA
    extra = []
    for n, p in model.named_parameters():
        if ("router" in n.lower() and "proj" in n) or n.endswith("norm.weight"):
            p.requires_grad_(True)
            extra.append(p)
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[gce] stack={'unsloth' if use_unsloth else 'hf+peft'} trainable="
          f"{sum(p.numel() for p in train_params)/1e6:.1f}M (extra router/norm "
          f"{sum(p.numel() for p in extra)/1e6:.1f}M)", flush=True)

    # CONSTRAINT-ENGAGEMENT GATE: constrained forward must differ from free, or the
    # loader's patches ate our hook and training would silently adapt to nothing.
    probe = rows[0]["ids"][:256].to("cuda").long().unsqueeze(0)
    plen = min(int(rows[0]["prompt_len"]), 128)
    with torch.no_grad():
        GL.CFG.update(on=False, enforce_from=0)
        lf = model(probe).logits[:, -1].float()
        GL.CFG.update(on=True, R=8, free_set=None, R_map=None, enforce_from=plen,
                      cold_start=False)
        lc = model(probe).logits[:, -1].float()
    d = float((lf - lc).abs().max())
    assert d > 1e-3, f"constraint NOT engaged under this loader (max logit delta {d:.2e})"
    print(f"[gce] constraint engaged (max logit delta {d:.3f})", flush=True)

    if A.smoke:
        # 1) expert-LoRA engagement: B is zero-init, so a forward must be identical
        #    to base; bumping one B must change logits; restoring must restore.
        assert A.expert_lora_r, "--smoke requires --expert-lora-r"
        emods = [m_ for m_ in model.modules() if hasattr(m_, "elora_gu_B")]
        # 0) PARITY: grouped path with B=0 must reproduce the unpatched forward
        with torch.no_grad():
            GL.CFG.update(on=False, enforce_from=0)
            lp = model(probe).logits[:, -1].float()
        pd = float((lp - smoke_ref).abs().max())
        # single-module parity vs the eager loop is 1e-6 in fp32 / 0.008 in bf16;
        # at model level bf16 accumulation reorder compounds over 30 layers, so the
        # gate is semantic (top-1 identical) plus a drift bound.
        top1_ok = bool((lp.argmax(-1) == smoke_ref.argmax(-1)).all())
        assert top1_ok and pd < 2.0, \
            f"grouped-path parity FAIL (top1_ok={top1_ok}, max logit diff {pd:.3f})"
        print(f"[gce-smoke] grouped-path parity OK (top-1 identical, max logit "
              f"drift {pd:.3f} bf16)", flush=True)
        with torch.no_grad():
            l0 = model(probe).logits[:, -1].float()
            emods[0].elora_gu_B.add_(0.05)
            l1 = model(probe).logits[:, -1].float()
            emods[0].elora_gu_B.zero_()
            l2 = model(probe).logits[:, -1].float()
        dd, dr = float((l0 - l1).abs().max()), float((l0 - l2).abs().max())
        assert dd > 1e-3, f"expert LoRA NOT engaged (delta {dd:.2e})"
        assert dr < 1e-4, f"expert LoRA restore failed (delta {dr:.2e})"
        print(f"[gce-smoke] expert LoRA engaged (bump delta {dd:.3f}, "
              f"restore {dr:.2e})", flush=True)
        # 1b) BATCH PARITY: per-row response-CE at mb1 vs one padded batch
        pr = [rows[i] for i in (0, 40, 200, 400)]

        def parity(on):
            def row_ce(r_):
                ids = r_["ids"].to("cuda").long().unsqueeze(0)
                pl = int(r_["prompt_len"])
                GL.CFG.update(on=on, R=8, enforce_from=pl, batch=1,
                              cold_start=False)
                with torch.no_grad():   # attention_mask=ones: same kernel path
                    lg_ = model(ids, attention_mask=torch.ones_like(ids)) \
                        .logits[0, pl - 1:-1].float()
                return float(torch.nn.functional.cross_entropy(
                    lg_, ids[0, pl:], reduction="mean"))
            ce1 = [row_ce(r_) for r_ in pr]
            ids, am, tgt, plens, _ = make_batch(pr)
            GL.CFG.update(on=on, R=8, enforce_from=plens, batch=len(pr),
                          cold_start=False)
            with torch.no_grad():
                lgb = model(ids, attention_mask=am).logits[:, :-1]
            ceb = []
            for b in range(len(pr)):
                m_ = tgt[b, 1:] != -100
                ceb.append(float(torch.nn.functional.cross_entropy(
                    lgb[b][m_].float(), tgt[b, 1:][m_], reduction="mean")))
            GL.CFG.update(batch=1)
            return ce1, ceb

        # (a0) EXACT plumbing invariant: one row through the batched path
        # (make_batch + CFG batch + enforce_from list) has identical shapes to
        # mb1, so logits must match to numerical identity.
        r0 = pr[1]
        ids0 = r0["ids"].to("cuda").long().unsqueeze(0)
        pl0 = int(r0["prompt_len"])
        GL.CFG.update(on=True, R=8, enforce_from=pl0, batch=1, cold_start=False)
        with torch.no_grad():
            la = model(ids0, attention_mask=torch.ones_like(ids0)).logits.float()
        idsb, amb, _, plb, _ = make_batch([r0])
        GL.CFG.update(on=True, R=8, enforce_from=plb, batch=1, cold_start=False)
        with torch.no_grad():
            lb = model(idsb, attention_mask=amb).logits.float()
        GL.CFG.update(batch=1)
        d0 = float((la - lb).abs().max())
        assert d0 < 1e-3, f"single-row batched-plumbing mismatch (diff {d0:.2e})"
        print(f"[gce-smoke] batched-plumbing exactness OK (single row diff "
              f"{d0:.2e})", flush=True)
        # (a) constraint OFF, B=4: only batch-SHAPE bf16 drift may remain.
        # Judge in absolute nats (baselines here are near zero).
        c1, cb = parity(on=False)
        ad = max(abs(a - b_) for a, b_ in zip(c1, cb))
        assert ad < 0.01, \
            f"FREE-mode batch parity FAIL (abs drift {ad:.4f} nats): {c1} vs {cb}"
        print(f"[gce-smoke] batch parity, constraint OFF: max abs drift "
              f"{ad:.4f} nats -- mechanics OK", flush=True)
        # (b) constraint ON: resident-set ties flip under bf16 batch-shape drift
        # and cascade discretely, so per-row CE cannot match exactly. Judge the
        # objective in aggregate, report per row.
        c1, cb = parity(on=True)
        m1, mbt = sum(c1) / len(c1), sum(cb) / len(cb)
        rel_on = abs(m1 - mbt) / m1
        print(f"[gce-smoke] batch parity, constraint ON: per-row mb1 "
              f"{['%.3f' % c for c in c1]} vs batched {['%.3f' % c for c in cb]}; "
              f"mean {m1:.4f} vs {mbt:.4f} ({rel_on*100:+.2f}%)", flush=True)
        assert rel_on < 0.02, f"batched constrained objective drifted: {m1} vs {mbt}"
        # 2) timed optimiser steps, batched exactly like training
        tp = [p for p in model.parameters() if p.requires_grad]
        print(f"[gce-smoke] trainable {sum(p.numel() for p in tp)/1e6:.1f}M "
              f"(expert {sum(p.numel() for m_ in emods for p in (m_.elora_gu_A, m_.elora_gu_B, m_.elora_dp_A, m_.elora_dp_B))/1e6:.1f}M)",
              flush=True)
        sopt = torch.optim.AdamW(tp, lr=1e-5)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        tok_seen, t0 = 0, time.time()
        steady_tok, steady_t = 0, 0.0
        mb_ = max(1, A.micro_batch)
        sl = sorted(range(400), key=lambda i: rows[i]["ids"].shape[0])
        for si in range(6):
            sopt.zero_grad(set_to_none=True)
            t_step, stoks = time.time(), 0
            for bi in range(max(1, 16 // mb_)):
                rs = [rows[i] for i in
                      sl[(si * 2 + bi) * mb_:(si * 2 + bi + 1) * mb_]]
                ids, am, tgt, plens, ntok = make_batch(rs)
                GL.CFG.update(on=True, R=8, enforce_from=plens, batch=len(rs),
                              cold_start=False)
                logits = model(ids, attention_mask=am).logits[:, :-1]
                loss = batch_ce(logits, tgt, max(1, 16 // mb_))
                loss.backward()
                tok_seen += ntok
                stoks += ntok
            GL.CFG.update(batch=1)
            # LoRA grad flow: at step 0 B (zero-init) gets grad through A while
            # A's grad is exactly 0 (= B^T g); after B's first update A must flow.
            gB = emods[0].elora_gu_B.grad
            assert gB is not None and float(gB.abs().max()) > 0, "no grad on expert B"
            if si == 1:
                gA = emods[0].elora_gu_A.grad
                assert gA is not None and float(gA.abs().max()) > 0,                     "no grad on expert A after B moved"
            sopt.step()
            torch.cuda.synchronize()
            st = time.time() - t_step
            print(f"[gce-smoke] step {si} loss {loss.item()*2:.4f} "
                  f"{st:.1f}s ({stoks/st:.0f} tok/s)", flush=True)
            if si >= 2:
                steady_tok += stoks
                steady_t += st
        dt_ = time.time() - t0
        print(f"[gce-smoke] STEADY (steps 2-5): {steady_tok/steady_t:.0f} tok/s | "
              f"total {tok_seen} toks {dt_:.1f}s | peak mem "
              f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        # 3) save/reload roundtrip
        named = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                 if p.requires_grad}
        torch.save({"tensors": named}, "/tmp/smoke_expert_adapter.pt")
        ck = torch.load("/tmp/smoke_expert_adapter.pt", map_location="cpu",
                        weights_only=False)
        allp = dict(model.named_parameters())
        miss = [n for n in ck["tensors"] if n not in allp]
        assert not miss, f"roundtrip mismatch: {miss[:3]}"
        n_e = sum(1 for n in ck["tensors"] if "elora" in n)
        print(f"[gce-smoke] roundtrip OK ({len(ck['tensors'])} tensors, "
              f"{n_e} expert-LoRA) -- SMOKE PASS", flush=True)
        return

    if A.eval_only or A.merge_out:
        ck = torch.load(A.out, map_location="cpu", weights_only=False)
        named = dict(model.named_parameters())
        miss = [n for n in ck["tensors"] if n not in named]
        assert not miss, f"{len(miss)} adapter tensors unmatched, e.g. {miss[:3]}"
        s = A.merge_scale
        with torch.no_grad():
            for n, t in ck["tensors"].items():
                t = t.to(named[n].data.device, named[n].dtype)
                if s != 1.0:
                    if "lora_B" in n or "elora_gu_B" in n or "elora_dp_B" in n:
                        t = t * s              # scales the low-rank delta linearly
                    elif "lora_A" not in n and "elora" not in n:
                        # full-weight tensors store absolutes: interpolate to base
                        t = named[n].data * (1 - s) + t * s
                named[n].data.copy_(t)
        if s != 1.0:
            print(f"[gce] adapter merged at scale {s}", flush=True)
        print(f"[gce] adapter loaded (seen={ck['seen']/1e6:.2f}M)", flush=True)
        if A.merge_out:
            with torch.no_grad():
                for mod in model.modules():
                    if hasattr(mod, "elora_gu_A"):
                        # weights live in grouped layout (E, in, out); fold then
                        # transpose back to the checkpoint layout (E, out, in)
                        mod.gate_up_proj.data += mod.elora_scale * torch.bmm(
                            mod.elora_gu_A.data, mod.elora_gu_B.data)
                        mod.down_proj.data += mod.elora_scale * torch.bmm(
                            mod.elora_dp_A.data, mod.elora_dp_B.data)
                        mod.gate_up_proj.data =                             mod.gate_up_proj.data.transpose(1, 2).contiguous()
                        mod.down_proj.data =                             mod.down_proj.data.transpose(1, 2).contiguous()
                        for nm in ("elora_gu_A", "elora_gu_B", "elora_dp_A",
                                   "elora_dp_B"):
                            delattr(mod, nm)
            m = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
            m.save_pretrained(A.merge_out, safe_serialization=True)
            tok.save_pretrained(A.merge_out)
            print(f"[gce] merged model -> {A.merge_out}", flush=True)
            return
        ev = torch.load("/workspace/instruct-traj/gemma4_instruct.pt",
                        weights_only=False)["rows"]
        model.eval()
        for label, on in (("free", False), ("R8", True)):
            tot = ntok = 0
            with torch.no_grad():
                for r in ev:
                    ids = r["ids"].to("cuda").long().unsqueeze(0)
                    plen = int(r["prompt_len"])
                    GL.CFG.update(on=on, R=8, enforce_from=plen if on else 0,
                                  cold_start=False, free_set=None, R_map=None)
                    lg = model(ids).logits[0].float()
                    tot += float(torch.nn.functional.cross_entropy(
                        lg[plen - 1:-1], ids[0, plen:], reduction="sum"))
                    ntok += ids.shape[1] - plen
            print(f"[gce-eval] {label} self-CE {tot/ntok:.4f} nats/tok "
                  f"(frozen 500, held out)", flush=True)
        return

    extra_ids = {id(p) for p in extra}
    lora_ps = [p for p in train_params if id(p) not in extra_ids]
    opt = torch.optim.AdamW([{"params": lora_ps, "lr": A.lr},
                             {"params": extra, "lr": A.lr / A.extra_lr_div}],
                            weight_decay=0.0)
    print(f"[gce] lr groups: lora {A.lr} | router/norm {A.lr / A.extra_lr_div}", flush=True)
    model.train()
    seen = step = 0
    t0 = time.time()
    mb = max(1, A.micro_batch)
    accum_batches = max(1, A.accum // mb)      # 16 rows per optimizer step
    lidx = sorted(range(len(rows)), key=lambda i: rows[i]["ids"].shape[0])
    chunks = [lidx[i:i + mb] for i in range(0, len(lidx), mb)]
    order = torch.randperm(len(chunks), generator=torch.Generator().manual_seed(0)).tolist()
    oi = 0

    def save():
        # torch.save straight onto the quota-limited network mount produced a
        # short write (inline_container pos mismatch) that killed the first
        # expert run at its step-400 checkpoint. Local write, atomic move.
        import shutil
        named = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                 if p.requires_grad}
        torch.save({"tensors": named, "seen": seen, "stack":
                    "unsloth" if use_unsloth else "hf+peft",
                    "R": 0 if A.no_constraint else 8, "expert_lora_r": A.expert_lora_r,
                    "elora_layout": "grouped(E,in,out)",
                    "traj": A.traj, "lr": A.lr}, "/tmp/gce_ckpt_tmp.pt")
        shutil.move("/tmp/gce_ckpt_tmp.pt", A.out)

    while seen < A.tokens:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum_batches):
            ridx = chunks[order[oi % len(order)]]
            rs = [rows[i] for i in ridx]
            oi += 1
            ids, am, tgt, plens, ntok = make_batch(rs)
            GL.CFG.update(on=not A.no_constraint, R=8, enforce_from=plens,
                          batch=len(rs), cold_start=False)
            logits = model(ids, attention_mask=am).logits[:, :-1]
            loss = batch_ce(logits, tgt, accum_batches, ridx=ridx)
            loss.backward()          # free the constrained graph BEFORE the KL
            if KLREF is not None:    # forward: two live graphs OOM'd at mb2/4096
                del logits
                GL.CFG.update(on=False, enforce_from=0, batch=len(rs))
                lg_free = model(ids, attention_mask=am).logits[:, :-1]
                targets = tgt[:, 1:]
                kl_num = kl_den = 0
                for b, ri in enumerate(ridx):
                    if ri not in KLREF:
                        continue
                    m_ = targets[b] != -100
                    tid, tlp = KLREF[ri]
                    tid = tid.to(lg_free.device).long()
                    tlp = tlp.to(lg_free.device).float()
                    lgb = lg_free[b][m_].float()
                    s_at = lgb.gather(1, tid) - torch.logsumexp(lgb, -1, keepdim=True)
                    p = tlp.exp()
                    p = p / p.sum(1, keepdim=True)      # renormalise top-50
                    kl_num = kl_num + (p * (tlp - s_at)).sum()
                    kl_den += int(m_.sum())
                if kl_den:
                    kl_loss = A.kl_weight * kl_num / kl_den / accum_batches
                    kl_loss.backward()
                    loss = loss.detach() + kl_loss.detach()
                GL.CFG.update(on=not A.no_constraint, R=8, enforce_from=plens,
                              batch=len(rs), cold_start=False)
            seen += ntok
        GL.CFG.update(batch=1)
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            print(f"[gce] step {step} seen {seen/1e6:.2f}M loss {loss.item()*accum_batches:.4f} "
                  f"({seen/(time.time()-t0):.0f} tok/s)", flush=True)
        if step % A.save_every == 0:
            save()
    save()
    print(f"[gce] DONE seen={seen/1e6:.2f}M -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
