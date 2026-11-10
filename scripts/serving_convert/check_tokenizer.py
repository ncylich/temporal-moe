#!/usr/bin/env python3
"""Step 3 check: the GGUF's tokenizer (run by the engine) against the Hugging Face tokenizer.

Decodes test-split tokens to text, then tokenizes that text with both. The round trip through
text can differ from the stored ids by a few tokens where a 2048-token sample cuts a document
mid-character, so the comparison that matters is engine ids == Hugging Face ids.

    python scripts/serving_convert/check_tokenizer.py --gguf $WORK/fine_1e19-f16.gguf \
        --bin $WORK/gguf_logits --tokens $WORK/test_tokens.npz --samples 64
"""
import argparse, os, subprocess, tempfile

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--bin", required=True)
ap.add_argument("--tokens", required=True)
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--tokenizer", default="EleutherAI/pythia-12b")
a = ap.parse_args()

tk = Tokenizer.from_file(hf_hub_download(a.tokenizer, "tokenizer.json"))
orig = np.load(a.tokens)["tokens"][:a.samples].reshape(-1).astype(np.int64)
text = tk.decode(orig.tolist(), skip_special_tokens=False)
tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, "t.txt"), "w", encoding="utf-8") as f:
    f.write(text)
hf = np.array(tk.encode(text, add_special_tokens=False).ids)
subprocess.run([a.bin, "tokenize", a.gguf, f"{tmp}/t.txt", f"{tmp}/t.i32"], check=True,
               stderr=subprocess.DEVNULL)
eng = np.fromfile(f"{tmp}/t.i32", dtype=np.int32)
same = len(hf) == len(eng) and bool((hf == eng).all())
print(f"text bytes {len(text.encode())}  stored ids {len(orig)}  hf ids {len(hf)}  engine ids {len(eng)}")
print(f"engine == hf: {same}")
if not same:
    n = min(len(hf), len(eng))
    i = int(np.argmax(hf[:n] != eng[:n]))
    print(f"first difference at {i}: hf {hf[i-3:i+5]} engine {eng[i-3:i+5]} text {tk.decode(hf[i-3:i+5].tolist())!r}")
    raise SystemExit(1)
