# Orchestration scripts (2026-08-26 rerun session)

Job scripts that drove the adapter rebuild, the benchmark surface, and the swap-rate
frontier. They were written at the terminal during the session and lived in `/workspace`;
committed here because the Python producers alone do not record HOW each arm was run
(budgets, arms, record names, matched references), and that is what makes a result
reproducible.

They assume this pod's layout: `/workspace/venv_vllm312` and `/workspace/venv_fla`,
gemma4 base staged at `/dev/shm/gemma4-26b-it`, qwen base at `/root/models/qwen35-35b-a3b`,
merged checkpoints under `/root/models/`, and trajectories in `/workspace/instruct-traj`.

## The data-regeneration chain, in order

1. `analysis/residency/build_d7_prompts.py` -- rebuild the D7 prompt pool (8,482 prompts,
   deterministic, 8-gram screened against four benchmark test sets)
2. `analysis/residency/build_codelane.py` -- the additional 2,500-prompt code lane, screen
   extended to MBPP
3. `analysis/residency/gen_traj_vllm.py` -- generate the model's own trajectories for a
   prompt file (`tmoe_codelane_traj.sh` is the wrapper used for the code lane)
4. `analysis/residency/cut_trajectories.py` / `cut_short_responses.py` -- length filters
5. `tmoe_d7code_chain.sh <seed>` -- KL precompute, train, merge, verify, full surface
   (`tmoe_seed_replicate.sh <seed>` is the same for the plain D7 recipe)
6. `tmoe_variant.sh <name> <train flags>` -- one recipe variant end to end

## Running many jobs

`tmoe_runner.sh <gpu>` is a standing queue worker: it pulls `.job` files from
`/workspace/tmoe_queue`, waits until its own device is free, runs one, and requeues a job
that died because the device was busy. One per GPU keeps all four fed without hand-queueing
between jobs. `tmoe_slot.sh` is the older flock-based single-shot wrapper it replaced.

Two traps worth knowing, both hit during the session:
- `nvidia-smi -i N` honours `CUDA_VISIBLE_DEVICES`, so probing a device from a process that
  has it set reads the wrong GPU. The runner unsets it for the probe.
- `pgrep -f <pattern>` matches the shell running it. Stop the runners with the pid files in
  `/workspace/tmoe_queue/pids/`, never with a pattern.
