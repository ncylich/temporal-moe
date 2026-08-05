#!/usr/bin/env python3
"""Tokens/sec for a real training step on the Unsloth path, matched to probe_expert_lora_cost.py.

TRAINING_OPTIM_PLAN.md step (d). The 'ours' number comes from probe_expert_lora_cost.py run in
the same venv (venv_fla, torch 2.13); this measures the Unsloth arm with the same trainable
surface -- expert LoRA r32 + attention LoRA r32 + router gates + RMSNorm gains -- the same
micro-batch/seq, gradient checkpointing on, CE-only loss, forward+backward+optimiser.

Residency ON via residency_unsloth (accepted at step (c)); `--residency off` isolates the scan's
share. `--mb` exists because 'matched config' and 'best achievable' are two different numbers
that must never be conflated (the mb4-vs-mb2 conflation produced the bogus 'Qwen3.5 is 24.7%
slower').

    probe_unsloth_cost.py --model qwen3 --mb 2
"""
import argparse
import os
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("qwen3", "qwen3_5"))
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lora", type=int, default=32)
    ap.add_argument("--residency", default="on", choices=("on", "off"))
    ap.add_argument("--adapter-dtype", default="bf16", choices=("bf16", "fp32"))
    A = ap.parse_args()

    import unsloth  # noqa: F401  must precede any transformers import
    from unsloth import FastModel
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import residency as RES
    import train_qwen as TQ

    path = TQ.FAMILY[A.model]["model"]
    model, _ = FastModel.from_pretrained(
        path, max_seq_length=A.seq, dtype=torch.bfloat16,
        load_in_4bit=False, full_finetuning=False)
    # Drop Qwen3.5's unused ~8 GB vision tower (see probe_expert_lora_cost) BEFORE peft
    # wraps modules, so adapters and hooks never see it.
    for mod in model.modules():
        if getattr(mod, "visual", None) is not None \
                and "Vision" in type(mod.visual).__name__:
            mod.visual = None
            torch.cuda.empty_cache()
            break
    model = FastModel.get_peft_model(
        model, r=A.lora, lora_alpha=2 * A.lora, lora_dropout=0.0,
        use_gradient_checkpointing="unsloth")

    import residency_unsloth as RU
    nblk = RU.install(model)
    if A.residency == "on":
        RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False)
        RES.set_free_layers(None)
    else:
        RES._CFG.update(on=False, collect_telem=False)

    # peft freezes everything outside the adapters; the sweep's trainable surface also has
    # router gates and RMSNorm gains. Unfreeze them and give the optimiser every trainable.
    n_extra = 0
    for name, p in model.named_parameters():
        if name.endswith(".gate.weight") or "norm" in name.split(".")[-2]:
            p.requires_grad_(True)
            n_extra += 1
    params = [p for p in model.parameters() if p.requires_grad]
    ntr = sum(p.numel() for p in params)
    # Unsloth keeps adapters in fp32; ours are bf16. At 1.3B trainable that is ~+10 GB across
    # params+grads+Adam states -- the difference between fitting mb2 and OOM on 80 GB. Cast
    # to bf16 for dtype parity with probe_expert_lora_cost (--adapter-dtype fp32 to keep theirs).
    if A.adapter_dtype == "bf16":
        for p in params:
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)

    model.train()
    # fused: single-kernel step with no foreach temporaries; matched with probe_expert_lora_cost.
    opt = torch.optim.AdamW(params, lr=1e-5, fused=True)
    cfg = getattr(model.config, "text_config", model.config)
    V = cfg.vocab_size
    print(f"  {A.model} unsloth: blocks={nblk} residency={A.residency} mb={A.mb} seq={A.seq} "
          f"trainable={ntr/1e6:.1f}M (+{n_extra} router/norm tensors)", flush=True)

    ids = torch.randint(0, V, (A.mb, A.seq), device="cuda")
    def step():
        out = model(ids)
        lg = out.logits[:, :-1]
        # bf16 CE, matched with probe_expert_lora_cost -- see the comment there.
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        del out, lg

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(A.steps):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / A.steps
    tps = A.mb * A.seq / dt
    print(f"  {tps:,.0f} tok/s   {dt:.2f} s/micro-step   "
          f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"  -> 15M tokens = {15e6/tps/60:.0f} min; a 5-point sweep = {5*15e6/tps/3600:.1f} h")
    print("=== UNSLOTH COST PROBE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
