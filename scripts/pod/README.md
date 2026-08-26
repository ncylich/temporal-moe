# Pod backup

The container disk (`/`) is wiped when the pod stops. `/workspace` is a network volume and
survives. These two scripts keep everything irreplaceable on the durable side.

| script | what it does | when |
|---|---|---|
| `backup.sh` | low-priority daemon (`ionice -c3 nice -n19`), every 120s, mirrors the Claude Code transcripts and this repo to `/workspace/pod-snapshot/root`. Never uses `--delete`. | run in tmux for the whole session |
| `snapshot.sh save` | full copy of `/root` to the same tree, plus HF credentials saved separately | before stopping the pod |
| `snapshot.sh restore` | rebuilds `/root` from the snapshot, fixing permissions | after restarting the pod |

Overrides: `SNAPSHOT_DIR`, `SNAPSHOT_TARGET`, `BACKUP_DST`, `BACKUP_SRCS` (colon-separated),
`BACKUP_INTERVAL`. `SNAPSHOT_TARGET` exists so restore can be rehearsed without touching
`/root` -- do that before trusting it.

## What is NOT backed up

Model weights and merged checkpoints are excluded: they are large and reproducible. On this
pod that means `/root/models` and the Hugging Face cache. **HF credentials live inside the
excluded cache and ARE saved**, to `$SNAPSHOT_DIR/hf-credentials`, and restored with mode
600 -- losing the token is a real outage, losing the weights is a re-download.

Anything else you keep on the container disk and care about is your responsibility to add
to `BACKUP_SRCS`. A directory that exists under `/workspace/pod-snapshot/root` is not
evidence it is being backed up; check it is non-empty.

## Verified 2026-08-26

Both directions rehearsed against a synthetic tree with the overrides above, plus the
daemon against a scratch destination:

- exclusions honoured (models and the HF cache absent from the snapshot; `.venv`,
  `__pycache__`, `*.pyc` absent from the daemon copy -- 7.2G of source became 1.1G)
- credentials, SSH keys and HF tokens all present, contents byte-identical after restore
- permissions after restore: `.ssh` 700, private key 600, public key 644,
  `.git-credentials` 600, HF token 600
- the overwrite guard refuses a non-empty target without `--force`
- a canary file placed in the destination survived a re-sync, confirming that the daemon
  is a backup and not a mirror
