#!/usr/bin/env python3
"""Harness-validation eval (orch 0098/0099): base OLMoE-1B-7B-0125, FREE routing (NO residency), full
precision, on c4 en-validation with our standard perplexity path. Reports mean CE (nats/token), token
count, byte count, eval config. Purpose: cross-check units/tokenizer/harness against the OLMoE-0924
wandb eval/c4_en-validation/CrossEntropyLoss. Direct next-token CE == the perplexity path (ppl=exp(CE))."""
import sys, json, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

SEQ = 4096
TOK_BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
PATH = "/workspace/olmoe-adapt/model"
tok = AutoTokenizer.from_pretrained(PATH)
EOS = tok.eos_token_id
# full precision = fp32 weights, free routing (no residency patch installed at all)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32,
                                             attn_implementation="sdpa").to("cuda").eval()
print(f"[c4val] model loaded fp32, free top-8 routing (no residency). budget={TOK_BUDGET} tok, seq={SEQ}", flush=True)

ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
buf, packs, nbytes = [], [], 0
for ex in ds:
    t = ex.get("text", "")
    if not t:
        continue
    nbytes += len(t.encode("utf-8"))
    buf.extend(tok(t, add_special_tokens=False).input_ids + [EOS])
    while len(buf) >= SEQ:
        packs.append(buf[:SEQ]); buf = buf[SEQ:]
    if len(packs) * SEQ >= TOK_BUDGET:
        break
ids = torch.tensor(packs, dtype=torch.long)
print(f"[c4val] packed {ids.shape[0]} seqs = {ids.numel()} tokens from {nbytes} utf-8 bytes", flush=True)

tot_ce, n_pred = 0.0, 0
with torch.no_grad():
    for i in range(ids.shape[0]):
        x = ids[i:i + 1].to("cuda")
        logits = model(x).logits.float()
        ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                               x[:, 1:].reshape(-1), reduction="sum")
        tot_ce += ce.item(); n_pred += x[:, 1:].numel()
        if (i + 1) % 50 == 0:
            print(f"[c4val] {i+1}/{ids.shape[0]} seqs, running CE={tot_ce/n_pred:.4f} nats/tok", flush=True)

ce_nats = tot_ce / n_pred
bytes_per_tok = nbytes / ids.numel()
bpb = ce_nats / (0.6931471805599453 * bytes_per_tok)   # CE_nats / (ln2 * bytes/tok)
res = {"source": "c4val_ours", "model": "OLMoE-1B-7B-0125", "routing": "free_top8_no_residency",
       "precision": "fp32", "corpus": "allenai/c4 en-validation (streaming)", "seq": SEQ,
       "n_seqs": ids.shape[0], "n_tokens_total": ids.numel(), "n_pred_tokens": n_pred,
       "n_bytes_utf8": nbytes, "bytes_per_token": bytes_per_tok,
       "CE_nats_per_token": ce_nats, "perplexity": float(torch.tensor(ce_nats).exp()),
       "BPB": bpb, "note": "direct next-token CE == perplexity path; ppl=exp(CE)"}
json.dump(res, open("/workspace/olmoe-adapt/data/c4val_ours.json", "w"), indent=1)
print("[c4val] RESULT", json.dumps(res, indent=1), flush=True)
print(f"[c4val] DONE CE={ce_nats:.4f} nats/tok  ppl={res['perplexity']:.3f}  BPB={bpb:.4f}  "
      f"tokens={n_pred}  bytes={nbytes}", flush=True)
