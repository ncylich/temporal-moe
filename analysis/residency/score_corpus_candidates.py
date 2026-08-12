#!/usr/bin/env python3
"""Which public corpus is the least surprising to a Qwen base model?

Qwen3's pretraining data is proprietary, so "representative" cannot be matched directly and has to be
measured. Bits per byte is the right statistic because it is tokenizer-invariant: two corpora that
tokenize at different rates are still comparable, which raw cross-entropy per token would not be.

The incumbent is the OLMoE adaptation corpus (OLMo's Dolma mixture), chosen to match OLMoE rather
than Qwen. It is not free: OLMoE's unconstrained null lost 0.0224 BPB over 50M tokens of it, so about
a fifth of what gets attributed to the residency constraint is really the corpus and recipe.

Scored on EQUAL BYTES, not equal sequences. BPB = CE_nats / (ln2 * bytes_per_token); scoring the same
sequence count across corpora that tokenize differently would compare different amounts of text.

Residency is OFF -- this ranks corpora against the published model, not against the constraint.

    score_corpus_candidates.py --family qwen3 --bytes 8000000
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_qwen as RQ                                        # noqa: E402

CAND = "/workspace/corpus_candidates"
FAMILY = {
    "qwen3":   {"model": "/dev/shm/qwen3-30b", "sfx": "qwen3",
                "incumbent": "/workspace/qwen3moe-adapt/data"},
    "qwen3_5": {"model": "/workspace/qwen35-adapt/model", "sfx": "qwen",
                "incumbent": "/workspace/qwen35-adapt/data"},
}


@torch.no_grad()
def bpb(model, ids, divisor, n_seq):
    tot = ntok = 0
    for i in range(n_seq):
        b = ids[i:i + 1].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel()
        del lg
    return (tot / ntok) / divisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen3", choices=("qwen3", "qwen3_5"))
    ap.add_argument("--bytes", type=int, default=8_000_000, help="bytes scored per corpus")
    A = ap.parse_args()
    F = FAMILY[A.family]

    arms = []
    for p in sorted(glob.glob(f"{CAND}/*_{F['sfx']}.json")):
        m = json.load(open(p))
        arms.append((m["corpus"], p.replace(".json", ".pt"), m["divisor_D"], m["bytes_per_token"]))
    inc = json.load(open(f"{F['incumbent']}/finetune_meta_{F['sfx']}.json"))
    arms.append(("dolma-incumbent", f"{F['incumbent']}/finetune_ids_{F['sfx']}.pt",
                 0.6931471805599453 * inc["bytes_per_token"], inc["bytes_per_token"]))

    model, _ = RQ.load_model(path=F["model"], family=A.family)
    RES._CFG.update(on=False, collect_telem=False)      # residency off: this ranks corpora, not the constraint

    print(f"\n  {A.family}: BPB on {A.bytes/1e6:.0f}MB per corpus, residency OFF (lower = better fit)\n",
          flush=True)
    out = []
    for name, path, D, bpt in arms:
        ids = torch.load(path, weights_only=False)
        seq = ids.shape[1]
        n = max(1, min(len(ids), int(A.bytes / (seq * bpt))))   # equal BYTES, not equal sequences
        v = bpb(model, ids, D, n)
        out.append((name, v, n, seq, bpt))
        print(f"  {name:18} BPB {v:.6f}   ({n} x {seq} tok = {n*seq*bpt/1e6:.1f}MB, "
              f"{bpt:.4f} B/tok, D={D:.7f})", flush=True)
        del ids
        torch.cuda.empty_cache()

    out.sort(key=lambda r: r[1])
    base = dict((n, v) for n, v, *_ in out)["dolma-incumbent"]
    print(f"\n  ranked (best first), delta vs the incumbent Dolma corpus:")
    for name, v, *_ in out:
        print(f"    {name:18} {v:.6f}   {v-base:+.6f}")
    print(f"\n  Reference: OLMoE's unconstrained null degraded 0.0224 BPB over 50M tokens of the\n"
          f"  incumbent, which is ~20% of the gap otherwise charged to the residency constraint.",
          flush=True)
    print("=== CORPUS SCORE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
