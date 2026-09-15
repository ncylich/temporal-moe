#!/usr/bin/env python3
"""Stage-4 (b): materialize the merged-CE HF model for GGUF. Bake CE's trained deltas into a CLEAN OLMoE
model — router + RMSNorm gains loaded as plain weights, LoRA merged EXACTLY into the expert weights
(W' = W + (alpha/r) B A per expert), LoRA modules removed. Identity-check the merged model reproduces
CE-final BPB 0.8149 at R=8 BEFORE saving. Saves to /workspace/olmoe-adapt/merged_ce_model."""
import sys, json, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES
from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

OUT = "/workspace/olmoe-adapt/data"; SAVE = "/workspace/olmoe-adapt/merged_ce_model"
D = json.load(open(f"{OUT}/bpb_slice_meta.json"))["divisor_D"]

model, tok = RES.load_model(); RES.enable_residency(R=8)
RES.freeze_all_but_router(model)
rp = RES.router_params(model); norm_ps = RES.norm_params(model)
lora_ps = RES.add_lora(model, r=32, alpha=64)
train_params = rp + norm_ps + lora_ps
ck = torch.load(f"{OUT}/ckpt_bake_CE.pt", map_location="cuda")
for p, m in zip(train_params, ck["masters"]):
    p.data.copy_(m.to("cuda").to(p.dtype))
print(f"[merge] loaded CE deltas (seen={ck['seen']/1e6:.0f}M)", flush=True)

bpb_ids = torch.load(f"{OUT}/bpb_slice_ids.pt")
eval_sub = bpb_ids[torch.linspace(0, bpb_ids.shape[0] - 1, 256).long()].to("cuda").long()


def eval_bpb():
    RES.enable_residency(R=8); tot = n = 0
    with torch.no_grad():
        for i in range(eval_sub.shape[0]):
            x = eval_sub[i:i + 1]; o = model(x).logits.float()
            tot += torch.nn.functional.cross_entropy(o[:, :-1].reshape(-1, o.size(-1)), x[:, 1:].reshape(-1), reduction="sum").item()
            n += x[:, 1:].numel()
    return (tot / n) / D


pre = eval_bpb()                                              # LoRA active
scale = RES._LORA["scale"]
with torch.no_grad():
    for mod in model.modules():
        if isinstance(mod, OlmoeExperts):
            for e in range(mod.num_experts):
                mod.gate_up_proj[e] += scale * (mod.lora_gu_B[e] @ mod.lora_gu_A[e])
                mod.down_proj[e] += scale * (mod.lora_dn_B[e] @ mod.lora_dn_A[e])
            for a in ("lora_gu_A", "lora_gu_B", "lora_dn_A", "lora_dn_B"):
                delattr(mod, a)                              # remove LoRA -> clean OLMoE model
OlmoeExperts.forward = RES._orig_experts_forward             # restore plain expert forward
RES._LORA["scale"] = 0.0
post = eval_bpb()
print(f"[merge] identity check: pre-merge(LoRA) BPB={pre:.4f}  post-merge(baked) BPB={post:.4f}  |delta|={abs(pre-post):.5f}", flush=True)
assert abs(pre - post) < 0.003, "merge not identity!"
assert abs(post - 0.8149) < 0.01, f"merged model {post:.4f} != CE-final 0.8149"
print(f"[merge] VERIFIED merged-CE reproduces CE-final ({post:.4f} ~= 0.8149) at R=8.", flush=True)

# save clean merged model (residency is runtime-only; the weights are a vanilla OLMoE)
RES.disable_residency()
model.half().save_pretrained(SAVE); tok.save_pretrained(SAVE)
print(f"[merge] saved merged-CE HF model -> {SAVE}", flush=True)
