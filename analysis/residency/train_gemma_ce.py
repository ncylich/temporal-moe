#!/usr/bin/env python3
"""CE adaptation of gemma4-26B-A4B-IT to decode-time residency (R=k=8, response tokens only).

Plain cross-entropy on the model's OWN vLLM-generated responses (gemma's greedy outputs are
low-entropy, so hard labels ~ soft labels and distillation buys nothing). The constraint is
enforced exactly as served: prefill free (scan observes the prompt), R=8 from the first
response token, warm. Loss on response tokens only. Surface: attention LoRA r32 + router
projections + RMSNorm gains -- the 3D expert tensors have no LoRA plumbing on this arch.

Loads via unsloth when its patches keep our Gemma4TextRouter hook alive (asserted before any
step: constrained logits must differ from free); otherwise plain HF + peft. Saves the
trainable tensors as gemma_ce_adapter.pt every save-every steps and at the end.

    train_gemma_ce.py --traj gemma4_train5k --tokens 3400000
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_expert_lora(model, r):
    """Per-expert LoRA on the fused 3D expert tensors (gate_up_proj (E,2I,H),
    down_proj (E,H,I)). Deltas are computed per HIT expert inside the forward loop
    ((2I,r)@(r,H) per hit) -- the full-tensor materialisation that produced the
    93 tok/s Qwen3.5 figure never happens. B zero-init: delta starts at 0."""
    import torch.nn as nn
    patched = 0
    for mod in model.modules():
        gu = getattr(mod, "gate_up_proj", None)
        dp = getattr(mod, "down_proj", None)
        if not (isinstance(gu, nn.Parameter) and gu.dim() == 3
                and isinstance(dp, nn.Parameter) and dp.dim() == 3):
            continue
        E, twoI, H = gu.shape
        _, H2, I = dp.shape
        dev, dt = gu.device, gu.dtype
        mod.elora_gu_A = nn.Parameter(torch.randn(E, r, H, device=dev, dtype=dt) / r)
        mod.elora_gu_B = nn.Parameter(torch.zeros(E, twoI, r, device=dev, dtype=dt))
        mod.elora_dp_A = nn.Parameter(torch.randn(E, r, I, device=dev, dtype=dt) / r)
        mod.elora_dp_B = nn.Parameter(torch.zeros(E, H2, r, device=dev, dtype=dt))
        mod.elora_scale = 2.0                     # alpha/r with alpha = 2r

        def fwd(self, hidden_states, top_k_index, top_k_weights):
            # per-HIT deltas beat a single bmm here: full materialisation costs
            # ~1.5 GiB of gradient traffic per layer per step (measured 52 vs
            # 98 tok/s), while per-hit slices keep backward small.
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = nn.functional.one_hot(
                    top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                e = expert_idx[0]
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[e])
                current_state = hidden_states[token_idx]
                w_gu = self.gate_up_proj[e] + self.elora_scale * (
                    self.elora_gu_B[e] @ self.elora_gu_A[e])
                w_dp = self.down_proj[e] + self.elora_scale * (
                    self.elora_dp_B[e] @ self.elora_dp_A[e])
                gate, up = nn.functional.linear(current_state, w_gu).chunk(2, dim=-1)
                cur = self.act_fn(gate) * up
                cur = nn.functional.linear(cur, w_dp)
                cur = cur * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx,
                                               cur.to(final_hidden_states.dtype))
            return final_hidden_states

        import types
        mod.forward = types.MethodType(fwd, mod)
        patched += 1
    assert patched, "no 3D expert tensors found to patch"
    print(f"[gce] expert LoRA r={r} on {patched} layers", flush=True)
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
    ap.add_argument("--smoke", action="store_true",
                    help="engagement checks + 2 timed steps + save/reload, then exit")
    ap.add_argument("--merge-out", default=None,
                    help="after loading the adapter, save the merged model to this dir and exit")
    A = ap.parse_args()

    rows = torch.load(f"/workspace/instruct-traj/{A.traj}.pt", weights_only=False)["rows"]
    print(f"[gce] {len(rows)} trajectories", flush=True)

    import granularity_ladder as GL
    use_unsloth = True
    try:
        from unsloth import FastModel
        model, tok = FastModel.from_pretrained(A.model, max_seq_length=1024,
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
    if A.expert_lora_r:
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
        # 2) two timed optimiser steps on real rows
        tp = [p for p in model.parameters() if p.requires_grad]
        print(f"[gce-smoke] trainable {sum(p.numel() for p in tp)/1e6:.1f}M "
              f"(expert {sum(p.numel() for m_ in emods for p in (m_.elora_gu_A, m_.elora_gu_B, m_.elora_dp_A, m_.elora_dp_B))/1e6:.1f}M)",
              flush=True)
        sopt = torch.optim.AdamW(tp, lr=1e-5)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        tok_seen, t0 = 0, time.time()
        steady_tok, steady_t = 0, 0.0
        for si in range(6):
            sopt.zero_grad(set_to_none=True)
            t_step, stoks = time.time(), 0
            for r in rows[si * 2:si * 2 + 2]:
                ids = r["ids"].to("cuda").long().unsqueeze(0)
                plen2 = int(r["prompt_len"])
                GL.CFG.update(on=True, R=8, enforce_from=plen2, cold_start=False)
                logits = model(ids).logits[:, :-1]
                targets = ids[:, 1:].clone()
                targets[:, : plen2 - 1] = -100
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    targets.reshape(-1), ignore_index=-100) / 2
                loss.backward()
                tok_seen += ids.shape[1] - plen2
                stoks += ids.shape[1] - plen2
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
        with torch.no_grad():
            for n, t in ck["tensors"].items():
                named[n].data.copy_(t.to(named[n].dtype))
        print(f"[gce] adapter loaded (seen={ck['seen']/1e6:.2f}M)", flush=True)
        if A.merge_out:
            with torch.no_grad():
                for mod in model.modules():
                    if hasattr(mod, "elora_gu_A"):
                        for e in range(mod.gate_up_proj.shape[0]):
                            mod.gate_up_proj[e] += mod.elora_scale * (
                                mod.elora_gu_B[e] @ mod.elora_gu_A[e])
                            mod.down_proj[e] += mod.elora_scale * (
                                mod.elora_dp_B[e] @ mod.elora_dp_A[e])
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
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).tolist()
    oi = 0

    def save():
        named = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                 if p.requires_grad}
        torch.save({"tensors": named, "seen": seen, "stack":
                    "unsloth" if use_unsloth else "hf+peft",
                    "R": 0 if A.no_constraint else 8, "expert_lora_r": A.expert_lora_r,
                    "traj": A.traj, "lr": A.lr}, A.out)

    while seen < A.tokens:
        opt.zero_grad(set_to_none=True)
        for _ in range(A.accum):
            r = rows[order[oi % len(order)]]
            oi += 1
            ids = r["ids"].to("cuda").long().unsqueeze(0)
            plen = int(r["prompt_len"])
            GL.CFG.update(on=not A.no_constraint, R=8, enforce_from=plen,
                          cold_start=False)
            logits = model(ids).logits[:, :-1]
            targets = ids[:, 1:].clone()
            targets[:, : plen - 1] = -100
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                ignore_index=-100) / A.accum
            loss.backward()
            seen += ids.shape[1] - plen
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            print(f"[gce] step {step} seen {seen/1e6:.2f}M loss {loss.item()*A.accum:.4f} "
                  f"({seen/(time.time()-t0):.0f} tok/s)", flush=True)
        if step % A.save_every == 0:
            save()
    save()
    print(f"[gce] DONE seen={seen/1e6:.2f}M -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
