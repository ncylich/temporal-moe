#!/usr/bin/env python3
"""Reproduce the first N samples of the Megatron test split the 1e19 runs were scored on.

The run's final "validation loss ... on test set" covers 20 eval iters x 128 micro-batches x 8
= 20480 samples of 2048 tokens (global batch 1024), read in dataloader order from a
BlendedDataset over 22 tokenized dclm parts with --split 90,5,5 and seed 1234. This script
rebuilds that dataset with Megatron's own dataset code (pure Python, imports fine without CUDA or
TransformerEngine) and writes the first N samples, which are a strict prefix of what was scored.

Two things are substituted, neither changes the indices:
  * `helpers_cpp` (a pybind11 extension) is replaced by line-for-line Python ports of the two
    functions used here, `build_sample_idx` and `build_blending_indices`.
  * The .bin token files (21 GB) are created sparse and only the byte ranges the N samples touch
    are fetched from the `ncylich/temporal-moe-corpus` dataset repo. The .idx files are fetched
    whole (165 MB).

    python scripts/serving_convert/test_tokens.py --megatron Megatron-LM \
        --data $WORK/dclm_tokenized --n 512 --out $WORK/test_tokens.npz
"""
import argparse, os, sys, types

import numpy as np

# Blend order exactly as logged by the training run (train.log, "Building GPTDataset splits").
PARTS = [19, 17, 18, 21, 20, 15, 14, 13, 16, 12, 11, 4, 6, 10, 8, 2, 3, 7, 9, 5, 1, 0]
SIZES = (4380672, 40960, 20480)          # train, valid, test samples, as logged
REPO = "ncylich/temporal-moe-corpus"


def _build_sample_idx(dtype):
    def f(sizes, document_idx, seq_length, num_epochs, tokens_per_epoch,
          drop_last_partial_sequence=True, add_extra=1):
        total = num_epochs * tokens_per_epoch - add_extra
        num_samples = total // seq_length if drop_last_partial_sequence else -(-total // seq_length)
        out = np.zeros((num_samples + 1, 2), dtype=dtype)
        sizes = np.asarray(sizes).tolist()
        document_idx = np.asarray(document_idx).tolist()
        last = len(document_idx) - 1
        s, d, off = 1, 0, 0
        while s <= num_samples:
            remaining = seq_length + add_extra
            while remaining != 0:
                doc_len = sizes[document_idx[d]] - off
                remaining -= doc_len
                if remaining <= 0:
                    off += remaining + doc_len - add_extra
                    remaining = 0
                else:
                    if d == last:
                        assert s == num_samples
                        off = sizes[document_idx[d]] - add_extra
                        break
                    d += 1
                    off = 0
            out[s] = (d, off)
            s += 1
        return out
    return f


def _build_blending_indices(dataset_index, dataset_sample_index, weights, num_datasets, size, verbose):
    w = np.asarray(weights, dtype=np.float64)
    cur = np.zeros(num_datasets, dtype=np.int64)
    for i in range(size):
        err = w * max(float(i), 1.0) - cur
        j = int(np.argmax(err))          # first max, same as the C++ strict '>' scan
        dataset_index[i] = j
        dataset_sample_index[i] = cur[j]
        cur[j] += 1


def install_shim():
    m = types.ModuleType("megatron.core.datasets.helpers_cpp")
    m.build_sample_idx_int32 = _build_sample_idx(np.int32)
    m.build_sample_idx_int64 = _build_sample_idx(np.int64)
    m.build_blending_indices = _build_blending_indices
    sys.modules[m.__name__] = m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--megatron", required=True, help="path to the Megatron-LM checkout")
    ap.add_argument("--data", required=True, help="local dir for .idx and sparse .bin files")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    import requests

    os.makedirs(a.data, exist_ok=True)
    sizes = {f.path: f.size for f in HfApi().list_repo_tree(REPO, "dclm_tokenized", repo_type="dataset")}
    for p in PARTS:
        name = f"part{p:02d}_text_document"
        hf_hub_download(REPO, f"dclm_tokenized/{name}.idx", repo_type="dataset",
                        local_dir=os.path.dirname(os.path.abspath(a.data)))
        binp = os.path.join(a.data, name + ".bin")
        if not os.path.exists(binp):
            with open(binp, "wb") as f:
                f.truncate(sizes[f"dclm_tokenized/{name}.bin"])
    assert os.path.basename(os.path.abspath(a.data)) == "dclm_tokenized"

    sys.path.insert(0, a.megatron)
    install_shim()
    import logging, torch
    logging.basicConfig(level=logging.INFO)
    torch.distributed.get_rank = lambda: 0
    from megatron.core.datasets import indexed_dataset
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.utils import get_blend_from_list

    tok = types.SimpleNamespace(eod=0, pad=1, unique_identifiers={"class": "pythia"})
    blend = get_blend_from_list(sum((["1.0", os.path.join(a.data, f"part{p:02d}_text_document")]
                                     for p in PARTS), []))
    cfg = GPTDatasetConfig(
        random_seed=1234, sequence_length=2048, blend=blend, blend_per_split=None,
        split="90,5,5", num_dataset_builder_threads=1, path_to_cache=None, mmap_bin_files=True,
        tokenizer=tok, reset_position_ids=False, reset_attention_mask=False,
        eod_mask_loss=False, create_attention_mask=False)
    test = BlendedMegatronDatasetBuilder(GPTDataset, list(SIZES), lambda: True, cfg).build()[2]
    assert len(test) >= SIZES[2], len(test)   # the dataloader stops at SIZES[2]

    # Pass 1: record which byte ranges the first n samples read.
    wanted = []
    orig_read = indexed_dataset._MMapBinReader.read

    def rec(self, dtype, count, offset):
        wanted.append((self._bin_buffer_mmap.filename, offset, count * np.dtype(dtype).itemsize))
        return orig_read(self, dtype, count, offset)

    indexed_dataset._MMapBinReader.read = rec
    for i in range(a.n):
        test[i]
    indexed_dataset._MMapBinReader.read = orig_read

    def fetch(w):
        path, off, ln = w
        url = hf_hub_url(REPO, "dclm_tokenized/" + os.path.basename(path), repo_type="dataset")
        for attempt in range(5):
            try:
                r = requests.get(url, headers={"Range": f"bytes={off}-{off+ln-1}"}, timeout=120)
                r.raise_for_status()
                assert len(r.content) == ln
                break
            except Exception:  # noqa: BLE001 - retry any transport error
                if attempt == 4:
                    raise
        with open(path, "r+b") as f:
            f.seek(off)
            f.write(r.content)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch, wanted))
    print(f"fetched {len(wanted)} ranges, {sum(w[2] for w in wanted)/1e6:.1f} MB")

    # Pass 2: read the samples for real.
    toks = np.zeros((a.n, 2049), dtype=np.uint16)
    masked = 0
    for i in range(a.n):
        s = test[i]
        toks[i, :2048] = s["tokens"].numpy()
        toks[i, 2048] = s["labels"][-1].item()
        masked += int((s["loss_mask"] == 0).sum())
    np.savez(a.out, tokens=toks, dataset_index=np.asarray(test.dataset_index[:a.n]),
             dataset_sample_index=np.asarray(test.dataset_sample_index[:a.n]))
    print(f"wrote {a.out}: {toks.shape}, loss-masked positions: {masked}, "
          f"all-zero rows: {(toks.max(1) == 0).sum()}")


if __name__ == "__main__":
    main()
