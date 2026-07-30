#!/usr/bin/env python3
"""C8 / N6 -- causal token-versus-context substitution. A capture MODE, not an analysis script.

Every other probe here hooks one forward pass over one fixed batch and records what happens. C8 needs
the opposite: several forward passes over *constructed* batches, kept aligned token-by-token against an
unperturbed reference, so that the change in routing can be attributed to the thing that was changed.

Three arms, one invocation each (`CAUSAL_ARM`):

  ref      the batch untouched; the reference routing
  token    at each probe position t, replace the token with a FREQUENCY-MATCHED substitute,
           leaving every other position alone
  context  at each probe position t, replace the surrounding +-w window, leaving t itself alone

and for each arm we record, per MoE layer, the selected expert set **at the probe positions only**.
The metric is then per layer: Jaccard shift under the context arm divided by Jaccard shift under the
token arm. Above 1 = that layer's routing is moved more by its surroundings than by the token it is
processing.

Three design points that are not incidental:

**Score position t alone.** Substituting the token at t also changes the *context* of positions t+-w.
Scoring anything but t would leak token sensitivity into the context arm and inflate it.

**Frequency-match the substitute.** A uniformly random substitute changes the embedding norm and the
routing prior at the same time, which would inflate the token arm for a reason that has nothing to do
with token identity. Substitutes are drawn from tokens of near-identical frequency in this same batch.

**Space the probe positions.** Perturbing several positions per sequence is what makes this cheap, but
two probes closer than 2w would sit in each other's perturbed window. Spacing is enforced at >= 4w+2
and recorded in the output.

Substituting a token is implemented by overwriting the *embedding row* rather than the input id, which
is exactly equivalent here and needs no change to the data pipeline: every config uses RoPE, so the
`LanguageModelEmbedding` output is the word-embedding stream with no additive positional term (the same
property delex_probe.py relies on).

The arms run as separate invocations of the same fixed batch. That is only valid if the batch really is
identical across them, so each invocation records a hash of its input ids and the analysis refuses to
compare arms whose hashes differ.

    CAUSAL_ARM=ref     CAUSAL_OUT=... $PY analysis/probes/delex_causal.py <megatron args>
    CAUSAL_ARM=token   ...
    CAUSAL_ARM=context ...
    $PY analysis/probes/delex_causal.py --analyze <run>      # after all three
"""
import os
import sys

import numpy as np
import torch

if "--analyze" not in sys.argv:
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from megatron.core.transformer.moe.router import TopKRouter

ARM = os.environ.get("CAUSAL_ARM", "ref")
N_MB = int(os.environ.get("N_MB", "8"))
STRIDE = int(os.environ.get("CAUSAL_STRIDE", "0"))          # 0 -> derived from w
SEED = int(os.environ.get("CAUSAL_SEED", "1234"))
_TEMPORAL = os.environ.get("TEMPORAL", "0") == "1"
_EVICT = os.environ.get("TEMPORAL_EVICT", "min_logit")

R = {}                    # layer -> list of [S,B,E] bool selected-expert masks
EMB_IN = []               # captured input ids, one [S,B] per micro-batch
_state = {"plan": None, "w": None, "ids": None, "n_emb": 0}


def _probe_positions(S, w, stride):
    """Positions to perturb: evenly spaced, clear of the sequence edges and of each other."""
    lo, hi = w + 1, S - w - 1
    return np.arange(lo, hi, stride, dtype=np.int64)


def _freq_matched(ids_flat, vocab_hint, rng):
    """token id -> a substitute of near-identical frequency in this batch.

    Frequency is counted on the batch itself, which is a sample of corpus frequency. Ranking tokens by
    it and pairing each with its neighbour in that ranking is enough to hold the routing prior roughly
    fixed; the point is only to avoid pairing a common token with a near-unseen one.
    """
    uniq, cnt = np.unique(ids_flat, return_counts=True)
    order = np.argsort(cnt, kind="stable")
    ranked = uniq[order]
    sub = {}
    for i, t in enumerate(ranked):
        j = i + 1 if i + 1 < len(ranked) else i - 1          # nearest neighbour in frequency rank
        sub[int(t)] = int(ranked[j])
    return sub


if "--analyze" not in sys.argv:
    _orig_router = TopKRouter.forward

    def _router_forward(self, input):
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        k = int(self.config.moe_router_topk)
        ln = int(getattr(self, "layer_number", -1))
        d = R.setdefault(ln, [])
        mask = None
        if _TEMPORAL:
            from temporal.temporal_router import compute_resident_mask_accel
            with torch.no_grad():
                mask = compute_resident_mask_accel(logits, k, evict=_EVICT)
        routed = logits.masked_fill(~mask, float("-inf")) if mask is not None else logits
        if len(d) < N_MB:
            with torch.no_grad():
                sel = torch.zeros_like(routed, dtype=torch.bool)
                sel.scatter_(-1, routed.topk(k, dim=-1).indices, True)
                d.append(sel.cpu())
        return self.routing(routed)

    TopKRouter.forward = _router_forward


def _install(model):
    """Hook the embedding: capture input ids, and apply this arm's perturbation to its output."""
    for name, mod in model.named_modules():
        if not (name.endswith(".embedding") or mod.__class__.__name__ == "LanguageModelEmbedding"):
            continue

        def pre(m, args, kwargs=None):
            ids = args[0] if args else None
            if ids is not None and _state["n_emb"] < N_MB:
                EMB_IN.append(ids.detach().to(torch.int64).cpu())

        def post(m, args, out):
            if _state["n_emb"] >= N_MB:
                return out
            _state["n_emb"] += 1
            o = out[0] if isinstance(out, tuple) else out            # [S,B,H]
            if ARM == "ref":
                return out
            S, B, H = o.shape
            w = _state["w"]
            stride = STRIDE or (4 * w + 2)
            pos = _probe_positions(S, w, stride)
            ids = EMB_IN[-1]                                          # [S,B] or [B,S]
            if ids.shape[0] != S:
                ids = ids.T
            rng = np.random.default_rng(SEED)
            sub = _freq_matched(ids.numpy().ravel(), None, rng)
            emb_w = m.word_embeddings.weight if hasattr(m, "word_embeddings") else None
            new = o.clone()
            for t in pos:
                if ARM == "token":
                    for b in range(B):
                        s = sub.get(int(ids[t, b]))
                        if s is not None and emb_w is not None:
                            new[t, b] = emb_w[s].to(new.dtype)
                elif ARM == "context":
                    lo, hi = max(0, t - w), min(S, t + w + 1)
                    idx = [p for p in range(lo, hi) if p != t]
                    perm = rng.permutation(len(idx))
                    src = o[[idx[p] for p in perm]]
                    new[idx] = src.to(new.dtype)
            _state["plan"] = pos
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)


def _wrap_provider():
    import pretrain_gpt
    orig = pretrain_gpt.model_provider

    def patched(*a, **k):
        model = orig(*a, **k)
        m = model[0] if isinstance(model, list) else model
        _state["w"] = int(os.environ.get("CAUSAL_W", "0")) or _infer_w(m)
        _install(m)
        return model

    pretrain_gpt.model_provider = patched


def _infer_w(model):
    """Context half-width = top-k, one residency lifetime, matching the locus probes' w=k."""
    for mod in model.modules():
        if hasattr(mod, "config") and getattr(mod.config, "moe_router_topk", None):
            return int(mod.config.moe_router_topk)
    return 6


def _dump():
    out = os.environ.get("CAUSAL_OUT", f"/tmp/causal_{ARM}.pt")
    layers = {ln: torch.cat(v, dim=1) for ln, v in R.items() if v}     # [S, B_total, E]
    ids = torch.cat(EMB_IN, dim=1 if EMB_IN[0].shape[0] != EMB_IN[0].shape[1] else 0) if EMB_IN else None
    torch.save({"arm": ARM, "temporal": _TEMPORAL, "n_mb": N_MB, "w": _state["w"],
                "stride": STRIDE or (4 * _state["w"] + 2),
                "positions": None if _state["plan"] is None else np.asarray(_state["plan"]),
                "ids_hash": None if ids is None else int(
                    torch.tensor(np.frombuffer(ids.numpy().tobytes(), dtype=np.int64)).sum().item()),
                "layers": layers}, out)
    ex = layers[sorted(layers)[0]]
    print(f"[causal] saved {out}: arm={ARM} {len(layers)} MoE layers {sorted(layers)}, "
          f"sel {tuple(ex.shape)}, w={_state['w']}, temporal={_TEMPORAL}")


def analyze():
    """Compare the three arms and write the per-layer sensitivity ratio."""
    import csv
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import registry
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import ABLATIONS

    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows, summary = [], []
    for run in runs:
        r = registry.get(run)
        arms = {}
        for arm in ("ref", "token", "context"):
            p = os.path.join(registry.RUNS, run, f"causal_{arm}.pt")
            if not os.path.exists(p):
                print(f"[skip] {run}: missing {arm} arm ({p})")
                arms = None
                break
            arms[arm] = torch.load(p, map_location="cpu", weights_only=False)
        if not arms:
            continue
        hashes = {a: d["ids_hash"] for a, d in arms.items()}
        if len(set(hashes.values())) != 1:
            print(f"[FAIL] {run}: the three arms saw different batches {hashes} — not comparable")
            continue
        pos = arms["token"]["positions"]
        w = arms["ref"]["w"]
        layers = sorted(arms["ref"]["layers"])
        print(f"[run] {run} ({r.regime}, {r.grain_label}, {r.budget}) w={w} "
              f"stride={arms['token']['stride']} positions/seq={len(pos)} layers {layers[0]}-{layers[-1]}")
        for L in layers:
            ref = arms["ref"]["layers"][L][pos].numpy()                 # [P, B, E]
            jac = {}
            for arm in ("token", "context"):
                cur = arms[arm]["layers"][L][pos].numpy()
                inter = (ref & cur).sum(-1).astype(np.float64)
                union = (ref | cur).sum(-1).astype(np.float64)
                jac[arm] = float(np.mean(1.0 - inter / np.maximum(union, 1)))
            ratio = jac["context"] / jac["token"] if jac["token"] > 0 else float("nan")
            rows.append([run, r.budget, r.regime, r.grain_label, L, w,
                         arms["token"]["stride"], int(len(pos)), int(ref.shape[1]),
                         round(jac["token"], 5), round(jac["context"], 5), round(ratio, 4)])
            summary.append((run, L, ratio))
            print(f"    L{L:<3} token shift {jac['token']:.4f}  context shift {jac['context']:.4f}  "
                  f"ratio {ratio:.3f}")
    if rows:
        os.makedirs(ABLATIONS, exist_ok=True)
        p = os.path.join(ABLATIONS, "mechinterp_causal.csv")
        with open(p, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["run", "budget", "regime", "grain", "layer", "window_w", "stride",
                          "positions_per_seq", "n_sequences", "token_jaccard_shift",
                          "context_jaccard_shift", "context_over_token"])
            wtr.writerows(rows)
        print(f"\n[write] {p}: {len(rows)} rows")


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        analyze()
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
        if R:
            _dump()
