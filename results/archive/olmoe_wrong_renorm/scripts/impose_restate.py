import sys, json, torch
sys.path.insert(0,"/workspace/olmoe-adapt/scripts"); import olmoe_residency as R
meta=json.load(open("/workspace/olmoe-adapt/data/bpb_slice_meta.json")); D=meta["divisor_D"]
ids=torch.load("/workspace/olmoe-adapt/data/bpb_slice_ids.pt")   # [Nseq,4096] int32
NEVAL=int(sys.argv[1]) if len(sys.argv)>1 else 512
# evenly-spaced subsample across the slice for representativeness
idx=torch.linspace(0, ids.shape[0]-1, NEVAL).long()
sub=ids[idx].to("cuda").long()
model, tok = R.load_model()
def ce(masked):
    R.enable_residency(R=8) if masked else R.disable_residency()
    tot=n=0
    with torch.no_grad():
        for i in range(sub.shape[0]):
            x=sub[i:i+1]; out=model(x).logits.float()
            l=torch.nn.functional.cross_entropy(out[:,:-1].reshape(-1,out.size(-1)), x[:,1:].reshape(-1), reduction="sum")
            tot+=l.item(); n+=x[:,1:].numel()
    return tot/n
cb=ce(False); ci=ce(True)
print(f"RESTATE impose on AUDITED slice (dolmino-mix-1124 dclm, {NEVAL}x4096 tok, byte-divisor D={D:.4f}):")
print(f"  base  CE={cb:.4f}  BPB={cb/D:.4f}")
print(f"  +mask CE={ci:.4f}  BPB={ci/D:.4f}   (R=k=8 of 64)")
print(f"  IMPOSE GAP dCE={ci-cb:+.4f}  dBPB={(ci-cb)/D:+.4f}")
