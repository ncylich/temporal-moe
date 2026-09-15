#!/usr/bin/env python3
"""CE adaptation of Qwen3.5-35B-A3B-Instruct to decode-time residency (R=k=8, response
tokens only) -- the gemma recipe (train_gemma_ce.py) on the qwen unsloth stack
(train_unsloth.py machinery: FastModel bf16, unsloth checkpointing, fused residency).

Plain cross-entropy on the model's OWN vLLM-generated responses (sampled per its
generation_config, seed 1234). Constraint exactly as served: prefill free (scan observes
the prompt), R=8 from the first response token, warm. Loss on response tokens only.
Surface: attention LoRA r32 + router gates + RMSNorm gains.

    train_qwen_ce.py --traj qwen35_train5k --tokens 3400000
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/qwen35-35b-a3b-instruct")
    ap.add_argument("--traj", default="qwen35_train5k")
    ap.add_argument("--tokens", type=int, default=3_400_000, help="response-token budget")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--extra-lr-div", type=float, default=5.0,
                    help="router/norm full-weight lr = lr / this")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--out", default="/workspace/qwen35-adapt/data/qwen_ce_adapter.pt")
    ap.add_argument("--no-constraint", action="store_true",
                    help="CONTROL: identical run with residency OFF during training")
    ap.add_argument("--merge-out", default=None,
                    help="load --out adapter, save merged model to this dir, exit")
    A = ap.parse_args()

    # Same accommodation the gemma trainer makes: on this pod the cuDNN fused-attention
    # backend raises CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH out of
    # scaled_dot_product_attention and takes the unsloth path down before any work runs.
    # Numerics-neutral -- flash and mem-efficient SDPA compute the same attention -- and
    # "cuDNN SDP off" is already documented as part of the published qwen recipe.
    torch.backends.cuda.enable_cudnn_sdp(False)

    rows = torch.load(f"/workspace/instruct-traj/{A.traj}.pt", weights_only=False)["rows"]
    rows = [r for r in rows if len(r["ids"]) <= A.max_seq]
    print(f"[qce] {len(rows)} trajectories (<= {A.max_seq} tokens)", flush=True)

    from unsloth import FastModel
    import residency as RES
    import residency_unsloth as RU

    model, tok = FastModel.from_pretrained(A.model, max_seq_length=A.max_seq,
                                           dtype=torch.bfloat16, load_in_4bit=False,
                                           full_finetuning=False)
    tok = getattr(tok, "tokenizer", tok)
    for mod in model.modules():                       # drop the unused vision tower
        if getattr(mod, "visual", None) is not None and \
                "Vision" in type(mod.visual).__name__:
            mod.visual = None
            torch.cuda.empty_cache()
            break
    model = FastModel.get_peft_model(model, r=32, lora_alpha=64, lora_dropout=0.0,
                                     use_gradient_checkpointing="unsloth")
    nblk = RU.install(model)
    RES._CFG.update(on=True, R=8, evict="min_logit", swaps=1, R_map=None,
                    collect_telem=False, collect_aux=False, enforce_from=0)
    RES.set_free_layers(None)

    extra = []
    for name, p in model.named_parameters():
        if name.endswith(".gate.weight") or "norm" in name.split(".")[-2]:
            p.requires_grad_(True)
            extra.append(p)
    train_params = [p for p in model.parameters() if p.requires_grad]
    for p in train_params:
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
    print(f"[qce] blocks={nblk} trainable={sum(p.numel() for p in train_params)/1e6:.1f}M "
          f"(router/norm {sum(p.numel() for p in extra)/1e6:.1f}M)", flush=True)

    # CONSTRAINT-ENGAGEMENT GATE: constrained forward must differ from free.
    probe = rows[0]["ids"][:256].to("cuda").long().unsqueeze(0)
    plen = min(int(rows[0]["prompt_len"]), 128)
    with torch.no_grad():
        RES._CFG.update(on=False, enforce_from=0)
        lf = model(probe).logits[:, -1].float()
        RES._CFG.update(on=True, R=8, enforce_from=plen)
        lc = model(probe).logits[:, -1].float()
    d = float((lf - lc).abs().max())
    assert d > 1e-3, f"constraint NOT engaged (max logit delta {d:.2e})"
    print(f"[qce] constraint engaged (max logit delta {d:.3f})", flush=True)

    if A.merge_out:
        ck = torch.load(A.out, map_location="cpu", weights_only=False)
        named = dict(model.named_parameters())
        miss = [n for n in ck["tensors"] if n not in named]
        assert not miss, f"{len(miss)} adapter tensors unmatched, e.g. {miss[:3]}"
        with torch.no_grad():
            for n, t in ck["tensors"].items():
                named[n].data.copy_(t.to(named[n].dtype))
        m = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
        m.save_pretrained(A.merge_out, safe_serialization=True)
        tok.save_pretrained(A.merge_out)
        print(f"[qce] merged model -> {A.merge_out}", flush=True)
        return

    extra_ids = {id(p) for p in extra}
    lora_ps = [p for p in train_params if id(p) not in extra_ids]
    import bitsandbytes as bnb
    opt = bnb.optim.PagedAdamW8bit([{"params": lora_ps, "lr": A.lr},
                                    {"params": extra, "lr": A.lr / A.extra_lr_div}],
                                   weight_decay=0.0)
    print(f"[qce] lr groups: lora {A.lr} | router/norm {A.lr / A.extra_lr_div}", flush=True)
    model.train()
    seen = step = 0
    t0 = time.time()
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).tolist()
    oi = 0

    def save():
        named = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                 if p.requires_grad}
        os.makedirs(os.path.dirname(A.out), exist_ok=True)
        torch.save({"tensors": named, "seen": seen, "stack": "unsloth+RU",
                    "R": 0 if A.no_constraint else 8,
                    "traj": A.traj, "lr": A.lr}, A.out)

    while seen < A.tokens:
        opt.zero_grad(set_to_none=True)
        for _ in range(A.accum):
            r = rows[order[oi % len(order)]]
            oi += 1
            ids = r["ids"].to("cuda").long().unsqueeze(0)
            plen = int(r["prompt_len"])
            RES._CFG.update(on=not A.no_constraint, R=8, enforce_from=plen)
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
        if step % 25 == 0:
            print(f"[qce] step {step} seen {seen/1e6:.2f}M loss {loss.item()*A.accum:.4f} "
                  f"({seen/(time.time()-t0):.0f} tok/s)", flush=True)
        if step % A.save_every == 0:
            save()
    save()
    print(f"[qce] DONE seen={seen/1e6:.2f}M -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
