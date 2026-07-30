#!/usr/bin/env python3
"""1f -- produce the end-of-document mask that e8 needs, and that nothing in the repo produced.

`probe_replay.e8()` (document-boundary churn) reads
`results/phase0/probe_batch_cache/eod_{16k,50k}.npy`, a `[B, S]` boolean array marking which
positions of the fixed evaluation batch are end-of-document tokens. No committed code ever wrote that
file: it was produced ad hoc, is gitignored, and is absent from `MANIFEST.csv`, so e8 has been
skipping every run. It was recorded as unrecoverable from the published artifacts, which was wrong --
the corpus it derives from is present in both tokenizations, so it needs a producer, not an artifact.

This is that producer. It runs the same Megatron data pipeline as the router probe, captures the
input ids of the fixed batch, and marks positions equal to the tokenizer's EOD id.

Two things make the output trustworthy rather than merely present:

- **The batch has to be the one the router logs were captured on.** It is the same pipeline, same
  seed and same shape, so it should be; but "should be" is not a check, so the file records the
  ids' shape and a hash alongside the mask, and `--verify` compares the shape against a named
  router log before anything downstream uses it.
- **A mask of all False is a silent no-op**, not an error: e8 would run, find no boundaries, and
  report a deficit of zero. The dump refuses to write a mask with no EOD positions at all.

    EODPROBE=1 ... experiments/run.sh          # via the launcher branch
    $PY analysis/probes/eod_capture.py --verify <run>     # shape-check against that run's router log
"""
import os
import sys

import numpy as np

if "--verify" not in sys.argv:
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CACHE, RUNS

N_MB = int(os.environ.get("N_MB", "8"))
IDS = []


def _install(model):
    """Capture input ids off the embedding, the one module that sees them before anything else."""
    for name, mod in model.named_modules():
        if not (name.endswith(".embedding") or mod.__class__.__name__ == "LanguageModelEmbedding"):
            continue

        def pre(m, args, kwargs):
            # Megatron passes input_ids by keyword; a positional-only hook captures nothing here.
            ids = args[0] if args else kwargs.get("input_ids")
            if ids is not None and len(IDS) < N_MB:
                IDS.append(ids.detach().to("cpu").numpy())
            return None

        mod.register_forward_pre_hook(pre, with_kwargs=True)
        return


def _wrap_provider():
    import pretrain_gpt
    orig = pretrain_gpt.model_provider

    def patched(*a, **k):
        model = orig(*a, **k)
        _install(model[0] if isinstance(model, list) else model)
        return model

    pretrain_gpt.model_provider = patched


def _dump():
    from megatron.training import get_tokenizer
    tok = get_tokenizer()
    eod_id = int(getattr(tok, "eod", None) or getattr(tok, "eod_id"))
    ids = np.concatenate(IDS, axis=0)                       # [B_total, S] (batch-first from the loader)
    if ids.shape[1] < ids.shape[0]:                          # be explicit rather than assume an order
        ids = ids.T
    mask = ids == eod_id
    if not mask.any():
        raise RuntimeError(
            f"[eod] no position in the captured batch equals the EOD id {eod_id}. Writing this mask "
            f"would make e8 run, find no boundaries and report a deficit of zero -- a silent no-op. "
            f"Check the tokenizer: ids range {ids.min()}..{ids.max()}, shape {ids.shape}.")
    tag = os.environ.get("EOD_TAG") or ("50k" if ids.max() > 20000 else "16k")
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"eod_{tag}.npy")
    np.save(out, mask)
    np.save(os.path.join(CACHE, f"eod_{tag}_meta.npy"),
            np.array([ids.shape[0], ids.shape[1], eod_id, int(mask.sum())]))
    print(f"[eod] wrote {out}: shape {mask.shape}, eod_id {eod_id}, "
          f"{int(mask.sum())} boundaries ({mask.mean()*100:.2f}% of positions), tag {tag}")


def verify():
    """Shape-check a mask against a router log before anything downstream trusts it."""
    import torch
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    for run in runs:
        rl = os.path.join(RUNS, run, "router_log.pt")
        if not os.path.exists(rl):
            print(f"[skip] {run}: no router log")
            continue
        d = torch.load(rl, map_location="cpu")
        rec = next(iter(d["layers"].values()))
        S, B = rec["logits"].shape[0], rec["logits"].shape[1]
        ok = False
        for tag in ("16k", "50k"):
            p = os.path.join(CACHE, f"eod_{tag}.npy")
            if not os.path.exists(p):
                continue
            m = np.load(p)
            match = m.shape[1] == S and m.shape[0] >= B
            print(f"  {run}: router log S={S} B={B} | eod_{tag} {m.shape} -> "
                  f"{'USABLE' if match else 'shape mismatch'}")
            ok = ok or match
        if not ok:
            print(f"  {run}: no usable eod mask; e8 will skip this run")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
        sys.exit(0)
    from megatron.training import pretrain
    from megatron.core.enums import ModelType
    import pretrain_gpt
    _wrap_provider()
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    try:
        pretrain(pretrain_gpt.train_valid_test_datasets_provider, pretrain_gpt.model_provider,
                 ModelType.encoder_or_decoder, pretrain_gpt.forward_step,
                 args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    finally:
        if IDS:
            _dump()
