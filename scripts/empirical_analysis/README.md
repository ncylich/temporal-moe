# Upstream router-behaviour measurements (not runnable here)

Router saturation and expert co-activation measurements over the released FLAME-MoE 290M, 721M and
1.7B checkpoints, carried over from [FLAME-MoE](https://github.com/cmu-flame/FLAME-MoE). Kept for
attribution and because `docs/research/background/reading-list.md` cites these numbers as the
starting point for the rolling-residency work. **None of it runs outside CMU's cluster.**

- `router_saturation.ipynb` and `expert_coactivation.ipynb` hardcode `/home/haok/MoE-Research` in
  their first code cell.
- The `capture-*.sh`, `router_saturation-*.sh` and `expert_coactivation-*.sh` drivers are SLURM
  jobs for CMU's partitions, and read captured activations from their Google Cloud storage keyed on
  internal training job ids.
- They `source scripts/config.sh`, which is a stub here because those resources are not ours.

Read them as a record of how the upstream numbers were produced, not as something to execute.

The equivalent measurements on the models in this repository live in `analysis/probes/`, which
resolve paths through `analysis/paths.py` and run from a clean checkout. `results/ablations/`
carries the resulting CSVs, and `docs/research/FINDINGS.md` explains what they show.
