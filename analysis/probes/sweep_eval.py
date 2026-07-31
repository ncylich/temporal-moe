#!/usr/bin/env python3
"""In-process sweep evaluation: pay Megatron's startup once per sweep instead of once per arm.

Measured on this pod, a single eval arm costs about 7 minutes, of which roughly 4 is Megatron/TE
init, dataset index building and checkpoint load. Every arm is a separate process, so a 13-arm C3
sweep pays that four minutes thirteen times. Loading the model once and re-evaluating it under
different residency settings turns ~90 minutes into ~40.

What makes this possible without touching the model: `temporal_router.temporal_forward` reads
`TEMPORAL_RESIDENCY_R` and `TEMPORAL_R_SCHEDULE` from the environment on **every forward call**, not
at import. Mutating them between evaluations changes the residency regime with no reload.

**The batches must be identical across arms or the comparison is worthless.** Megatron's evaluation
advances its data iterator, so calling evaluate twice in one process would score the second arm on
different documents than the first -- which is exactly the kind of silent incomparability that looks
like a result. This caches the evaluation batches on the first pass and replays those same cached
tensors for every arm, which is a stronger guarantee than the per-process approach it replaces: there
each arm rebuilt the iterator and relied on the seed and shard order matching.

    SWEEP='native:0 dose_R12:12 dose_R24:24' $PY analysis/probes/sweep_eval.py <megatron args>
    SWEEP='L2:E@2 L3:E@3' ...        # <tag>:E@<layer> sets TEMPORAL_R_SCHEDULE=<layer>:E

Emits `[sweep] <tag> lm_loss=<x>` per arm and writes results/ablations/sweep_eval.csv.

Validate before trusting: a tag whose setting reproduces an arm already measured the old way must
return the same loss to eval precision. `SWEEP_SELFTEST=1` re-runs the first arm at the end and checks the
two agree exactly, which catches an iterator that was consumed rather than replayed.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_CACHE = {"batches": None}
_RESULTS = []


def _parse_sweep():
    """'tag:R' or 'tag:E@layer' -> [(tag, {env}), ...]. R may be an integer or the literal E."""
    spec = os.environ.get("SWEEP", "").split()
    out = []
    for item in spec:
        tag, _, val = item.partition(":")
        env = {}
        if "=" in val:
            # Generic form, tag:VAR=VAL;VAR2=VAL2 -- lets an arm set any knob the router reads from
            # the environment per forward call, not just residency R.
            for kv in val.split(";"):
                if "=" in kv:
                    k2, _, v2 = kv.partition("=")
                    env[k2] = v2
        elif "@" in val:
            r, _, layer = val.partition("@")
            env["TEMPORAL_R_SCHEDULE"] = f"{layer}:{r}"
            env["TEMPORAL_RESIDENCY_R"] = "0"
        else:
            env["TEMPORAL_R_SCHEDULE"] = ""
            env["TEMPORAL_RESIDENCY_R"] = "0" if val in ("", "0", "native") else val
        out.append((tag, env))
    return out


def _install():
    """Replace the test-set evaluation with a loop over the sweep, on cached batches."""
    import torch
    import megatron.training.training as T
    from megatron.training import get_args

    orig_eval_print = T.evaluate_and_print_results

    def patched(prefix, forward_step_func, data_iterator, model, iteration,
                process_non_loss_data_func, config, verbose=False, write_to_tensorboard=True,
                **kw):
        # Only take over the test-set pass; the validation pass runs once and is ignored.
        if "test" not in str(prefix).lower():
            return orig_eval_print(prefix, forward_step_func, data_iterator, model, iteration,
                                   process_non_loss_data_func, config, verbose,
                                   write_to_tensorboard, **kw)
        args = get_args()
        sweep = _parse_sweep()
        if not sweep:
            return orig_eval_print(prefix, forward_step_func, data_iterator, model, iteration,
                                   process_non_loss_data_func, config, verbose,
                                   write_to_tensorboard, **kw)

        # Cache the evaluation batches once, then replay the identical tensors for every arm.
        if _CACHE["batches"] is None:
            it = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            # The iterator yields MICRO-batches, and one eval iteration consumes a full global batch.
            # Caching args.eval_iters micro-batches would score on 1/32 of the reference data and give
            # a loss that matches nothing already measured.
            per_iter = max(1, args.global_batch_size // args.micro_batch_size)
            n_mb = args.eval_iters * per_iter
            _CACHE["batches"] = [next(it) for _ in range(n_mb)]
            print(f"[sweep] cached {len(_CACHE['batches'])} micro-batches "
                  f"({args.eval_iters} eval iters x {per_iter} micro-batches of {args.micro_batch_size} "
                  f"x {args.seq_length}); every arm is scored on these same tensors", flush=True)

        mdl = model[0] if isinstance(model, list) else model
        # Env var, not a CLI flag: everything on argv is parsed by Megatron, which
        # rejects arguments it does not know.
        selftest = os.environ.get("SWEEP_SELFTEST", "") not in ("", "0")
        order = sweep + ([sweep[0]] if selftest else [])
        for i, (tag, env) in enumerate(order):
            os.environ.update(env)
            mdl.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for b in _CACHE["batches"]:
                    out = forward_step_func(iter([b]), mdl)
                    loss = out[0] if isinstance(out, tuple) else out
                    if isinstance(loss, torch.Tensor):
                        tot += float(loss.float().mean()); n += 1
            lm = tot / max(n, 1)
            label = tag if i < len(sweep) else f"{tag}__selftest"
            print(f"[sweep] {label} lm_loss={lm:.6f} R={env.get('TEMPORAL_RESIDENCY_R')} "
                  f"sched='{env.get('TEMPORAL_R_SCHEDULE')}'", flush=True)
            _RESULTS.append((label, lm, env.get("TEMPORAL_RESIDENCY_R"),
                             env.get("TEMPORAL_R_SCHEDULE")))
        # Two independent checks, because the first one alone passed while the sweep was broken.
        if selftest and len(_RESULTS) >= 2:
            a, b = _RESULTS[0][1], _RESULTS[-1][1]
            # GPU reductions are not bitwise deterministic across repeated passes, so an exact
            # threshold is unmeetable. 1e-6 is far below any effect being measured (smallest real
            # dose step is ~0.045) while still catching a genuinely different batch, which would
            # move the loss by order 1e-3 or more.
            ok = abs(a - b) < 1e-6
            print(f"[sweep] SELFTEST replay {'PASS' if ok else 'FAIL'}: first arm {a:.6f} vs repeat "
                  f"{b:.6f} (delta {abs(a-b):.2e}). A mismatch means the arms did not see the same "
                  f"batches.", flush=True)
            if not ok:
                raise RuntimeError("sweep_eval: repeated arm disagreed; batches are not being replayed")
        # The knob must actually do something. Distinct R values that return the same loss mean the
        # residency setting is not reaching the model -- the exact failure the replay check missed,
        # and one that otherwise reads as a clean flat dose curve.
        distinct_R = {r[2] for r in _RESULTS[:len(sweep)]}
        distinct_loss = {round(r[1], 9) for r in _RESULTS[:len(sweep)]}
        if len(distinct_R) > 1 and len(distinct_loss) == 1:
            raise RuntimeError(
                f"sweep_eval: {len(distinct_R)} different R values all returned {_RESULTS[0][1]:.6f}. "
                f"The residency knob is not reaching the model -- check that the temporal router is "
                f"installed (TEMPORAL=1) before the model is built.")
        _dump()

    T.evaluate_and_print_results = patched


def _dump():
    import csv
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import ABLATIONS
    run = os.environ.get("RUN_NAME", "unknown")
    os.makedirs(ABLATIONS, exist_ok=True)
    p = os.path.join(ABLATIONS, "sweep_eval.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "tag", "lm_loss", "R", "schedule"])
        for tag, lm, r, sch in _RESULTS:
            w.writerow([run, tag, f"{lm:.6f}", r, sch])
    print(f"[sweep] wrote {len(_RESULTS)} rows to {p}", flush=True)


if __name__ == "__main__":
    from megatron.training import pretrain
    from megatron.core.enums import ModelType
    import pretrain_gpt

    # Patch TopKRouter.forward BEFORE the model is built, the same way temporal/pretrain_temporal.py
    # does. Omitting this was the whole bug: without it the model evaluates as a plain MoE, the
    # residency env vars are read by code that never runs, and every arm returns the same loss --
    # which looks like a working sweep whose knob has no effect.
    if os.environ.get("TEMPORAL", "0") not in ("0", ""):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from temporal import temporal_router
        temporal_router.install()
        print("[sweep] temporal router installed", flush=True)

    _install()
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    pretrain(pretrain_gpt.train_valid_test_datasets_provider, pretrain_gpt.model_provider,
             ModelType.encoder_or_decoder, pretrain_gpt.forward_step,
             args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
