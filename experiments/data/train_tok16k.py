#!/usr/bin/env python3
"""Train a 16k-vocab byte-level BPE tokenizer on dclm text (to cut the 50k-vocab embedding/logits
overhead). Saves a HF-loadable fast tokenizer at data/tok16k/ for Megatron HuggingFaceTokenizer.
"""
import json, glob, os
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

ROOT = os.environ.get("TMOE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
VOCAB = 16000
OUT = f"{ROOT}/data/tok16k"
SAMPLE_TXT = f"{ROOT}/data/tok16k_sample.txt"
os.makedirs(OUT, exist_ok=True)

# Build a text sample (~part00 + part01 = ~500k docs, plenty for a 16k BPE).
if not os.path.exists(SAMPLE_TXT):
    n = 0
    with open(SAMPLE_TXT, "w") as out:
        for pf in sorted(glob.glob(f"{ROOT}/data/dclm_parts/part0[0-1].jsonl")):
            for line in open(pf):
                try:
                    t = json.loads(line).get("text")
                except Exception:
                    continue
                if t:
                    out.write(t.replace("\n", " ") + "\n"); n += 1
    print(f"sample docs: {n}", flush=True)

tok = ByteLevelBPETokenizer()
tok.train(files=[SAMPLE_TXT], vocab_size=VOCAB, min_frequency=2,
          special_tokens=["<|endoftext|>"])
tok.save(f"{OUT}/tokenizer.json")

fast = PreTrainedTokenizerFast(
    tokenizer_file=f"{OUT}/tokenizer.json",
    eos_token="<|endoftext|>", bos_token="<|endoftext|>", unk_token="<|endoftext|>",
    pad_token="<|endoftext|>",
)
fast.save_pretrained(OUT)
print("saved", OUT, "vocab_size", fast.vocab_size, "eos_id", fast.eos_token_id, flush=True)
