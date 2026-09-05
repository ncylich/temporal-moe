import sys, math, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as R
from datasets import load_dataset

SEQ=4096; NSEQ=int(sys.argv[1]) if len(sys.argv)>1 else 8
model, tok = R.load_model()
# provisional PUBLIC held-out slice (wikitext-103 validation) — Stage 1 audited slice replaces this
ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="validation")
text = "\n\n".join(t for t in ds["text"] if t.strip())
enc = tok(text, return_tensors="pt").input_ids[0]
nbytes = len(text.encode("utf-8"))
ntok_total = enc.numel()
# byte-derived divisor for THIS tokenizer+slice: BPB = CE_nats / D, D = ln2 * bytes_per_token
D = math.log(2) * (nbytes / ntok_total)
# pack into NSEQ x SEQ
usable = (ntok_total // SEQ)
NSEQ = min(NSEQ, usable)
packs = enc[:NSEQ*SEQ].view(NSEQ, SEQ).to("cuda")
def mean_ce(masked):
    R.enable_residency(R=8) if masked else R.disable_residency()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(NSEQ):
            ids = packs[i:i+1]
            out = model(ids).logits.float()
            sh_logits = out[:, :-1]; sh_labels = ids[:, 1:]
            l = torch.nn.functional.cross_entropy(sh_logits.reshape(-1, sh_logits.size(-1)),
                                                  sh_labels.reshape(-1), reduction="sum")
            tot += l.item(); ntok += sh_labels.numel()
    return tot/ntok
ce_base = mean_ce(False); ce_imp = mean_ce(True)
print(f"IMPOSE (provisional, wikitext-103 val, {NSEQ}x{SEQ} tok, byte-divisor D={D:.4f}):")
print(f"  base  CE={ce_base:.4f}  BPB={ce_base/D:.4f}")
print(f"  +mask CE={ce_imp:.4f}  BPB={ce_imp/D:.4f}   (R=k=8 of 64)")
print(f"  IMPOSE GAP: dCE={ce_imp-ce_base:+.4f}  dBPB={(ce_imp-ce_base)/D:+.4f}")
