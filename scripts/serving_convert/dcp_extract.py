#!/usr/bin/env python3
"""Step 1: pull the model tensors out of a Megatron torch-DCP checkpoint on Hugging Face.

No Megatron import and no full download. The checkpoint's `.metadata` pickle maps every tensor
chunk to (shard file, byte offset, length); the two classes it references outside torch
(`pathlib.PosixPath`, Megatron's `MCoreSavePlan`) are stubbed at unpickle time. Each chunk is a
`torch.save`d tensor, so the model weights (about 2.5 GB of a 17.8 GB checkpoint, the rest is
optimizer state) are fetched with HTTP range requests and reassembled.

Output: <out>.safetensors (tensors as stored, bf16) and <out>.keys.txt (name, shape, dtype).

    python scripts/serving_convert/dcp_extract.py \
        --run temporal_fine_g3_1e19 --iter 4278 --out $WORK/fine_1e19
"""
import argparse, io, os, pickle
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from huggingface_hub import hf_hub_download, hf_hub_url
from safetensors.torch import save_file

GAP = 1 << 20          # merge ranges closer than this into one request
MAX_SPAN = 64 << 20    # cap on one merged request


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        if isinstance(st, dict):
            self.__dict__.update(st)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("torch", "collections", "builtins"):
            return super().find_class(module, name)
        return type(name, (_Stub,), {})


def read_metadata(path):
    with open(path, "rb") as f:
        return _Unpickler(f).load()


def is_weight(key, meta):
    return (not key.startswith("optimizer") and "_extra_state" not in key
            and getattr(meta, "size", None) is not None
            and getattr(getattr(meta, "properties", None), "dtype", None) is not None)


def plan_spans(items):
    """items: [(file, offset, length, tag)] -> [(file, start, end, [(offset, length, tag)])]."""
    spans = []
    for f, off, ln, tag in sorted(items, key=lambda x: (x[0], x[1])):
        if spans and spans[-1][0] == f and off - spans[-1][2] <= GAP \
                and off + ln - spans[-1][1] <= MAX_SPAN:
            spans[-1][2] = max(spans[-1][2], off + ln)
            spans[-1][3].append((off, ln, tag))
        else:
            spans.append([f, off, off + ln, [(off, ln, tag)]])
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ncylich/temporal-moe-ckpts")
    ap.add_argument("--run", default="temporal_fine_g3_1e19")
    ap.add_argument("--iter", type=int, default=4278)
    ap.add_argument("--out", required=True, help="output prefix (no extension)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    sub = f"{a.run}/ckpt/iter_{a.iter:07d}"
    md = read_metadata(hf_hub_download(a.repo, f"{sub}/.metadata"))
    weights = {k: m for k, m in md.state_dict_metadata.items() if is_weight(k, m)}
    out = {k: torch.empty(tuple(m.size), dtype=m.properties.dtype) for k, m in weights.items()}

    by_fqn = {}
    for idx, info in md.storage_data.items():
        if idx.fqn in weights:
            by_fqn.setdefault(idx.fqn, {})[tuple(idx.offset)] = info
    items = []
    for k, m in weights.items():
        for c in m.chunks:
            info = by_fqn[k][tuple(c.offsets)]
            items.append((info.relative_path, info.offset, info.length, (k, tuple(c.offsets), tuple(c.sizes))))
    spans = plan_spans(items)
    total = sum(s[2] - s[1] for s in spans)
    print(f"{len(weights)} tensors, {len(items)} chunks, {len(spans)} requests, {total/1e9:.2f} GB")

    sess = requests.Session()
    tok = os.environ.get("HF_TOKEN")
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}

    def fetch(span):
        f, start, end, parts = span
        url = hf_hub_url(a.repo, f"{sub}/{f}")
        for attempt in range(5):
            try:
                r = sess.get(url, headers={**hdr, "Range": f"bytes={start}-{end-1}"}, timeout=300)
                r.raise_for_status()
                buf = r.content
                assert len(buf) == end - start
                break
            except Exception as e:  # noqa: BLE001 - retry any transport error
                if attempt == 4:
                    raise
                print(f"retry {f}@{start}: {e}")
        for off, ln, (k, coff, csz) in parts:
            t = torch.load(io.BytesIO(buf[off - start: off - start + ln]), map_location="cpu",
                           weights_only=True)
            sl = tuple(slice(o, o + s) for o, s in zip(coff, csz))
            out[k][sl] = t.reshape(csz)
        return end - start

    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(fetch, spans)):
            done += n
            if i % 200 == 0:
                print(f"  {done/1e9:.2f}/{total/1e9:.2f} GB", flush=True)

    save_file(out, a.out + ".safetensors")
    with open(a.out + ".keys.txt", "w") as f:
        for k, t in out.items():
            f.write(f"{k}\t{tuple(t.shape)}\t{t.dtype}\n")
    n = sum(t.numel() for t in out.values())
    print(f"wrote {a.out}.safetensors: {len(out)} tensors, {n/1e9:.4f}B parameters")


if __name__ == "__main__":
    main()
