# temporal — rolling-residency MoE router

A Mixture-of-Experts router that keeps a **rolling resident set** of experts across a sequence:
at each token it holds the top-`k` experts resident and swaps in at most one non-resident expert,
evicting the least-useful current resident. This makes expert selection temporally coherent (few
swaps/token) so the resident weights can be streamed from SSD/RAM rather than held all at once.

## Layout

| File | What it is |
| --- | --- |
| `temporal_router.py` | **Core, shipped mechanism.** `compute_resident_mask` (pure), the Triton/CUDA-graph accelerated scan (`compute_resident_mask_accel`), `temporal_forward`, and `install()` (patches Megatron's `TopKRouter.forward`). Eviction: `min_logit` (shipped) or `lru`. |
| `ablation_mechanisms.py` | **Default-off, negative-result knobs**, kept only for reproducibility of `results/ablations/*.csv` — demand momentum, aux-free trigger, coherence/anticipatory/bursty losses, nomination head. Not on any shipped path. |
| `pretrain_temporal.py` | Training entrypoint: `install()` then Megatron's normal GPT pretrain loop. Invoked by `experiments/run.sh` with `TEMPORAL=1`. |
| `__init__.py` | Re-exports the combined public surface of the two modules. |
| `tests/` | CPU/GPU pure-function TDD specs (`test_temporal_router.py` = core, `test_ablation_mechanisms.py` = ablations). |

## Running the tests

```
PYTHONPATH=Megatron-LM:. .venv/bin/python -m pytest temporal/tests analysis/probes -q
```

The router-vs-replay cross-checks import `analysis/probes/probe_replay.py`; co-collecting
`analysis/probes` puts it on the test path (running `temporal/tests` alone skips those 3 checks).

## Results

See `results/ablations/FINDINGS.md` for the verdicts these mechanisms produced and `paper/` for the
write-up.
