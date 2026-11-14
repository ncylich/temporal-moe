#!/usr/bin/env python3
"""Substitution tolerance: per-token CE change when an active expert is replaced at inference.

Runs in-process on one trained checkpoint, the same way analysis/probes/sweep_eval.py does: Megatron
starts once, the test-split evaluation is intercepted, a fixed set of test micro-batches is cached,
and every arm is scored on those identical tensors. The reference arm is the unperturbed model.
Every other arm installs `temporal_router.POST_ROUTING_HOOK`, which edits the selected expert set
AFTER the residency mask and top-k have run exactly as the model was trained, so one routing path
serves both regimes (a full MoE runs with TEMPORAL_RESIDENCY_R=E, the established protocol).

Per token and per perturbed MoE layer, `m` of the k active experts (a seeded random subset) are
displaced. Conditions decide what takes their place:

  random    a uniformly random unselected expert (the headline)
  nextbest  the highest-scoring unselected expert by the router's own raw logit
  stale     an expert that was selected at the previous token of the same sequence and is not
            selected now (what a refused swap forces; at R=k the resident set equals the selected
            set, so "a random resident" has no candidates and this is the realistic substitute)
  zero      no substitute at all, the displaced expert's contribution is dropped (the ceiling)

Gate weight of the substitute (SUBST_GATES, comma list of own,inherit):
  own       the weight the router itself would have assigned to the new set. Full MoE (mask all
            True): the substitute's own softmax-over-E probability, other gates untouched, which is
            what pre-softmax top-k does. Temporal (masked): softmax over the new selected set, which
            is what the masked router computes for its residents.
  inherit   the substitute takes the displaced expert's gate weight.

Arms cover all MoE layers at once at two replacement counts (matched fraction, one expert), plus
one layer at a time for the headline condition. Output: one .npz per run with a hash of the scored
token stream, the reference per-document CE sums and, per arm, the same sums, for bootstrap over documents in
analysis/residency/substitution_tolerance.py.

    SUBST_NSEQ=512 SUBST_OUT=results/ablations/substitution/<run>.npz \
        $PY analysis/probes/substitution_eval.py <megatron args>
"""
import hashlib
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

_CACHE = {"batches": None}


class SubstitutionHook:
    """Callable installed as temporal_router.POST_ROUTING_HOOK. Stateless across arms except for the
    per-arm configuration and a per-layer record of the previous micro-batch's selection (unused:
    "stale" is defined within a sequence, so the previous token comes from the same tensor)."""

    def __init__(self):
        self.cond = None
        self.m = 0
        self.gate = "own"
        self.layers = None          # set of layer numbers, or None for all MoE layers
        self.seed = 0
        self.calls = 0

    def configure(self, cond, m, gate, layers, seed):
        self.cond, self.m, self.gate, self.layers, self.seed = cond, int(m), gate, layers, int(seed)
        self.calls = 0
        self.n_sub = 0.0            # realized substitutions (or removals) summed over token-layers
        self.n_tok = 0.0            # token-layers seen by the hook

    def __call__(self, router, raw_logits, mask, probs, routing_map):
        if self.cond is None or (self.layers is not None and router.layer_number not in self.layers):
            return probs, routing_map
        self.calls += 1
        S, B, E = raw_logits.shape
        N = S * B
        raw = raw_logits.reshape(N, E).float()
        sel = routing_map.reshape(N, E).clone()
        gates = probs.reshape(N, E).clone()
        resident = mask.reshape(N, E) if mask is not None else torch.ones_like(sel)
        temporal = not bool(resident.all())
        k = int(sel[0].sum())
        m = min(self.m, k)
        gen = torch.Generator(device=raw.device)
        gen.manual_seed(int(hashlib.blake2b(
            f"{self.seed}:{router.layer_number}:{self.calls}".encode(), digest_size=8).hexdigest(), 16)
            % (2 ** 62))

        # Displaced experts: a random subset of size m of the selected set, per row.
        r = torch.rand(N, E, device=raw.device, generator=gen).masked_fill(~sel, -1.0)
        disp_idx = r.topk(m, dim=1).indices                                  # [N, m]
        disp = torch.zeros_like(sel).scatter_(1, disp_idx, True)

        if self.cond == "zero":
            sub = torch.zeros_like(sel)
        else:
            cand = ~sel
            if self.cond == "stale":
                prev = torch.zeros_like(sel)
                sel_sb = routing_map.reshape(S, B, E)
                prev.view(S, B, E)[1:] = sel_sb[:-1]                        # previous token, same row
                cand = cand & prev
            if self.cond == "nextbest":
                key = raw.masked_fill(~cand, float("-inf"))
            else:
                key = torch.rand(N, E, device=raw.device, generator=gen).masked_fill(~cand, -1.0)
            sub_idx = key.topk(m, dim=1).indices                             # [N, m]
            ok = torch.gather(cand, 1, sub_idx)                              # rows may lack candidates
            sub = torch.zeros_like(sel).scatter_(1, sub_idx, ok)
            # Only displace as many experts as we can substitute (keeps the count exact per token).
            n_sub = sub.sum(1, keepdim=True)
            keep_disp = torch.zeros_like(disp)
            rank = torch.cumsum(disp.int(), dim=1)
            keep_disp = disp & (rank <= n_sub)
            disp = keep_disp

        new_sel = (sel & ~disp) | sub
        # The stale condition can lack candidates (a rolling-residency model changes at most one
        # expert per token, so at most one stale expert exists), so record what was actually done.
        self.n_sub += float(disp.sum())
        self.n_tok += float(N)
        new_gates = gates.masked_fill(disp, 0.0)
        if self.cond != "zero":
            if self.gate == "inherit":
                # Pair displaced -> substitute in index order, hand over the gate weight.
                dvals = torch.gather(gates, 1, disp_idx)                     # [N, m]
                dmask = torch.gather(disp, 1, disp_idx)
                svals = (dvals * dmask).sum(1, keepdim=True) / dmask.sum(1, keepdim=True).clamp(min=1)
                # Equal split of the displaced mass across substitutes when m > 1 keeps the total
                # gate mass identical to the unperturbed token; with m == 1 it is exact inheritance.
                new_gates = new_gates + sub.to(gates.dtype) * svals.to(gates.dtype)
            elif temporal:
                sc = torch.softmax(raw.masked_fill(~new_sel, float("-inf")), dim=-1)
                new_gates = (sc * new_sel).to(gates.dtype)
            else:
                sc = torch.softmax(raw, dim=-1)
                new_gates = new_gates + (sc * sub).to(gates.dtype)
            scale = getattr(router.config, "moe_router_topk_scaling_factor", None)
            if scale and not temporal and self.gate == "own":
                new_gates = torch.where(sub, new_gates * scale, new_gates)
        return new_gates.reshape(probs.shape).to(probs.dtype), new_sel.reshape(routing_map.shape)


HOOK = SubstitutionHook()


def _arms(k, moe_layers):
    """(name, cond, m, gate, layers) for every arm. Matched fraction is one of six."""
    conds = os.environ.get("SUBST_CONDS", "random,nextbest,stale,zero").split(",")
    gates = os.environ.get("SUBST_GATES", "own,inherit").split(",")
    m_matched = max(1, round(k / 6))
    fracs = {"matched": m_matched}
    if m_matched != 1:
        fracs["one"] = 1
    out = []
    for fname, m in fracs.items():
        for c in conds:
            for g in (["-"] if c == "zero" else gates):
                out.append((f"{c}|{g}|{fname}|all", c, m, g, None))
    per_layer_cond = os.environ.get("SUBST_LAYER_COND", "random")
    per_layer_gate = os.environ.get("SUBST_LAYER_GATE", "own")
    if os.environ.get("SUBST_PER_LAYER", "1") != "0":
        for ln in moe_layers:
            out.append((f"{per_layer_cond}|{per_layer_gate}|matched|L{ln}", per_layer_cond,
                        m_matched, per_layer_gate, {ln}))
    return out


def _install():
    import numpy as np
    import megatron.training.training as T
    from megatron.training import get_args, get_tokenizer
    from temporal import temporal_router

    orig = T.evaluate_and_print_results

    # The 2025-07 checkpoints predate the layernorm `_extra_state` entries Transformer Engine 2.16
    # registers, so the strict model load rejects them. Every real weight is present (the
    # distributed load logs exactly which keys differ, run with --dist-ckpt-strictness log_all);
    # load non-strictly and let the reference CE against the published test CE confirm the weights.
    orig_load = T.load_checkpoint

    def lenient_load(*a, **kw):
        kw["strict"] = False
        return orig_load(*a, **kw)

    T.load_checkpoint = lenient_load

    def patched(prefix, forward_step_func, data_iterator, model, iteration,
                process_non_loss_data_func, config, verbose=False, write_to_tensorboard=True, **kw):
        if "test" not in str(prefix).lower():
            return orig(prefix, forward_step_func, data_iterator, model, iteration,
                        process_non_loss_data_func, config, verbose, write_to_tensorboard, **kw)
        args = get_args()
        nseq = int(os.environ.get("SUBST_NSEQ", "512"))
        if _CACHE["batches"] is None:
            it = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            n_mb = max(1, -(-nseq // args.micro_batch_size))
            _CACHE["batches"] = [next(it) for _ in range(n_mb)]
            print(f"[subst] cached {n_mb} test micro-batches x {args.micro_batch_size} x "
                  f"{args.seq_length} tokens; every arm scores these same tensors", flush=True)
        batches = _CACHE["batches"]
        mdl = model[0] if isinstance(model, list) else model
        mdl.eval()

        def run_arm():
            ces = []
            with torch.no_grad():
                for b in batches:
                    out = forward_step_func(iter([b]), mdl)
                    t = out[0] if isinstance(out, tuple) else out
                    ces.append(t.float().detach().cpu())
            return torch.cat(ces, 0)                                         # [nseq, S]

        # Token stream, loss mask and document ids (segments between EOD tokens, per sequence).
        toks = torch.cat([b["tokens"] for b in batches], 0).cpu()
        lmask = torch.cat([b["loss_mask"] for b in batches], 0).cpu().float()
        eod = get_tokenizer().eod
        doc = torch.cumsum((toks == eod).int(), dim=1)                       # [nseq, S]
        doc_id = (doc + torch.arange(doc.shape[0]).unsqueeze(1) * (doc.max() + 2)).numpy()
        uniq, doc_ix = np.unique(doc_id.reshape(-1), return_inverse=True)
        w = lmask.reshape(-1).numpy()
        ntok = np.bincount(doc_ix, weights=w, minlength=len(uniq))

        temporal_router.POST_ROUTING_HOOK = None
        ref = run_arm()
        ref_sum = np.bincount(doc_ix, weights=ref.reshape(-1).numpy() * w, minlength=len(uniq))
        ref_mean = float((ref * lmask).sum() / lmask.sum())
        print(f"[subst] reference lm_loss={ref_mean:.6f} over {int(lmask.sum())} tokens, "
              f"{len(uniq)} documents", flush=True)

        L = args.num_layers
        moe_layers = [ln for ln in range(1, L + 1) if args.moe_layer_freq == 1 or
                      (isinstance(args.moe_layer_freq, list) and args.moe_layer_freq[ln - 1])]
        k = args.moe_router_topk
        seed = int(os.environ.get("SUBST_SEED", "1234"))
        arms = _arms(k, moe_layers)
        res = {"arm": [], "mean_ce": [], "mean_delta": [], "doc_sum": [], "hook_calls": [],
               "subs_per_token": []}
        temporal_router.POST_ROUTING_HOOK = HOOK
        for i, (name, cond, m, gate, layers) in enumerate(arms):
            HOOK.configure(cond, m, gate, layers, seed + 7919 * i)
            ce = run_arm()
            d = (ce - ref)
            mean_ce = float((ce * lmask).sum() / lmask.sum())
            mean_delta = float((d * lmask).sum() / lmask.sum())
            res["arm"].append(name); res["mean_ce"].append(mean_ce); res["mean_delta"].append(mean_delta)
            res["doc_sum"].append(np.bincount(doc_ix, weights=ce.reshape(-1).numpy() * w,
                                              minlength=len(uniq)))
            res["hook_calls"].append(HOOK.calls)
            spt = HOOK.n_sub / max(HOOK.n_tok, 1.0)
            res["subs_per_token"].append(spt)
            print(f"[subst] {name:34s} m={m} lm_loss={mean_ce:.6f} delta={mean_delta:+.6f} "
                  f"subs/token-layer={spt:.3f} hook_calls={HOOK.calls}", flush=True)
            if HOOK.calls == 0:
                raise RuntimeError(f"substitution_eval: arm {name} never reached the router hook")
        temporal_router.POST_ROUTING_HOOK = None
        ref2 = run_arm()
        drift = float(((ref2 - ref).abs() * lmask).sum() / lmask.sum())
        print(f"[subst] replay check: mean |CE - CE_ref| on a repeated reference pass = {drift:.2e}",
              flush=True)
        if drift > 1e-4:
            raise RuntimeError("substitution_eval: repeated reference pass disagrees; batches not replayed")

        out = os.environ.get("SUBST_OUT")
        if out:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez_compressed(
                out, run=os.environ.get("RUN_NAME", "unknown"), k=k, E=args.num_experts,
                moe_layers=np.array(moe_layers), nseq=ref.shape[0], seq_len=ref.shape[1],
                regime=os.environ.get("SUBST_REGIME", "unknown"),
                residency_R=os.environ.get("TEMPORAL_RESIDENCY_R", "0"),
                tokens_sha256=hashlib.sha256(toks.numpy().astype(np.int32).tobytes()).hexdigest(),
                doc_ntok=ntok, ref_doc_sum=ref_sum, ref_mean=ref_mean, replay_drift=drift,
                arm=np.array(res["arm"]), mean_ce=np.array(res["mean_ce"]),
                mean_delta=np.array(res["mean_delta"]), doc_sum=np.stack(res["doc_sum"]),
                hook_calls=np.array(res["hook_calls"]),
                subs_per_token=np.array(res["subs_per_token"]))
            print(f"[subst] wrote {out}", flush=True)

    T.evaluate_and_print_results = patched


if __name__ == "__main__":
    from megatron.training import pretrain
    from megatron.core.enums import ModelType
    import pretrain_gpt
    from temporal import temporal_router

    temporal_router.install()
    print("[subst] temporal router installed", flush=True)
    _install()
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    pretrain(pretrain_gpt.train_valid_test_datasets_provider, pretrain_gpt.model_provider,
             ModelType.encoder_or_decoder, pretrain_gpt.forward_step,
             args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
