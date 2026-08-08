import sys, math, torch
sys.path.insert(0,"/workspace/olmoe-adapt/scripts"); import olmoe_residency as R
from datasets import load_dataset
SEQ=4096
model, tok = R.load_model()
ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="validation")
text="\n\n".join(t for t in ds["text"] if t.strip())
enc=tok(text,return_tensors="pt").input_ids[0]
packs=enc[:2*SEQ].view(2,SEQ).to("cuda")
def ce(Rv):
    R.enable_residency(R=Rv) if Rv<64 else R.disable_residency()
    tot=n=0
    with torch.no_grad():
        for i in range(2):
            ids=packs[i:i+1]; out=model(ids).logits.float()
            l=torch.nn.functional.cross_entropy(out[:,:-1].reshape(-1,out.size(-1)),ids[:,1:].reshape(-1),reduction="sum")
            tot+=l.item(); n+=ids[:,1:].numel()
    return tot/n
base=ce(64)
print(f"R-SWEEP (2x4096, base CE={base:.4f}):")
for Rv in [48,32,16,8]:
    c=ce(Rv); print(f"  R={Rv:2d}: CE={c:.4f}  dCE={c-base:+.4f}")
