import sys, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as R
model, tok = R.load_model()
E = model.config.num_experts
ids = tok("The temporal constraint on mixture-of-experts routing improves generalization because",
          return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    R.disable_residency()
    base = model(ids).logits.float()
    R.enable_residency(R=E)                     # R = E -> mask all-ones -> must equal base
    imp = model(ids).logits.float()
d = (base - imp).abs().max().item()
print(f"VERIFY1 R=E identity: max|base - R=E| = {d:.3e}  -> {'PASS' if d < 1e-4 else 'FAIL'}")
# also confirm R=8 actually CHANGES the output (mask is doing something)
with torch.no_grad():
    R.enable_residency(R=8)
    imp8 = model(ids).logits.float()
d8 = (base - imp8).abs().max().item()
print(f"        R=8 changes output: max|base - R=8| = {d8:.3e}  -> {'OK (nonzero)' if d8 > 1e-3 else 'SUSPECT (no effect)'}")
