# Upstream FLAME-MoE analysis (not runnable here)

Analysis code carried over from [FLAME-MoE](https://github.com/cmu-flame/FLAME-MoE), the CMU
project this repository builds on. It is kept for attribution and as a record of the methodology
the temporal work started from. **None of it runs outside CMU's cluster**, including for the
maintainer of this fork.

Three separate reasons, any one of which is enough:

- `plot/expert_specialization.ipynb` and `plot/router_saturation.ipynb` hardcode
  `/home/haok/MoE-Research` in their first code cell.
- The driver scripts are SLURM jobs targeting CMU's partitions
  (`--partition=flame`, `--qos=flame-t1b_g1_qos`, 8 GPUs, 1536 GB).
- They read captured activations from CMU's Google Cloud storage, keyed on internal training job
  ids, via `scripts/config.sh` — which is a stub in this repository precisely because those
  resources are not ours.

The notebooks retain their original rendered outputs, so the figures and numbers are readable
without running anything. That is the intended way to use this directory.

If you want the equivalent analysis on the models in this repository, use `analysis/probes/` and
`analysis/plots/` instead. Those resolve paths through `analysis/paths.py`, read the published
artifacts, and run from a clean checkout with no accelerator — see the repository README.
