# experiments — launchers, sweeps, and data pipeline

Everything used to produce the Phase-0 temporal-MoE results. Scripts assume they are launched from
the repo root; each `cd`s to the repo root itself before doing work.

## Layout

| Path | What it is |
| --- | --- |
| `run.sh` | The single env-parametrized launcher (one training/eval/probe run). All knobs are env vars — `SHAPE`, `TARGET_FLOPS`, `DENSE=1`, `TEMPORAL=1`, `PROBE=1`, `EVAL_ONLY=1`, … See `docs/EVALUATION_METHODOLOGY.md`. |
| `data/` | Corpus pipeline: download DCLM parts, build JSONL, train the BPE tokenizer, tokenize to Megatron `.bin`/`.idx` shards. |
| `isoflop_1e16_1e17/` | The 1e16–1e17 IsoFLOP sweeps: `drive.sh` (serial driver over a `NAME SHAPE FLOPS …` config file) plus the `*.txt` config files and matrix scripts. Backs **FINDINGS §2–6** and `results/ablations/phase0_isoflop_points.csv`. |
| `scale_1e18_1e19/` | The 1e18–1e19 scale-up runs (`flame_scale_run.sh`, `flame38m_run.sh`, and their sequencers). Backs **FINDINGS §7** and the `flame38m_*` / `flame512_*` / `flame192_*` / `t18_*` / `t19_*` CSVs under `results/ablations/`. |

## Note

The `*.txt` config files and per-sweep `*.sh` scripts are **frozen provenance** — the exact configs
that produced the committed results. Prefer adding a new config over editing an existing one.
