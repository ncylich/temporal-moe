# RunPod Setup, Communication & Workflow

This document collects the RunPod setup, remote-communication, and workflow
instructions carried over from the `matryoshka-distil` repo. It consolidates the
proven SSH patterns, deployment rules, monitoring protocol, and the hard-won
lessons from real pod launches. Read this before deploying anything to a fresh
RunPod pod.

---

## Run Monitoring Protocol

ALWAYS check runs for failures immediately after launch, then again after
1 minute, then again every 5–30 minutes depending on expected execution time.
Never assume a run is healthy without verifying the log.

- Check a run **30 seconds** after first remotely launching it, then loop-check
  it every **5 minutes** — do NOT use monitors.

---

## Required Run Sequence

Run all training and experiments on the RunPod machine:

1. `ssh <USER>@ssh.runpod.io -i ~/.ssh/runpod_key`
2. `cd /workspace/<repo>`
3. `source .venv/bin/activate`
4. Run your command (train/experiment/analysis/etc.)

No need to re-activate the venv if you are running multiple commands in the same
terminal session.

---

## Remote Code Deployment

- **NEVER** patch code directly on remote machines (no `python -c`, `sed`,
  heredoc writes, or any other in-place modification).
- Always: edit locally → commit → push → `git pull` on remote → run.
- No exceptions, even for one-line fixes. Direct patching creates invisible
  divergence and is error-prone through the TTY.
- Transfer code via GIT ONLY (commit+push from Mac, `git pull` / `reset --hard`
  on the pod).

---

## RunPod SSH techniques

RunPod's `ssh.runpod.io` gateway is a Go-based proxy with significant
limitations:

- **No exec mode**: `ssh user@ssh.runpod.io "command"` fails with "Your SSH
  client doesn't support PTY".
- **No scp/sftp/rsync**: subsystem requests fail ("subsystem request failed on
  channel 0"). rsync partially connects but the TTY escape codes corrupt the
  protocol ("unexpected tag 87").
- **Interactive shell only**: must use `-tt` flag and pipe commands via heredoc.
- The gateway **triple-echoes** commands (echo in heredoc, echo from TTY, echo
  from shell prompt) — output is noisy but commands do execute.

**Running commands through the gateway (proven pattern):**

```bash
ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no USER@ssh.runpod.io << 'EOF'
commands here
exit
EOF
```

- The `'EOF'` (quoted) heredoc prevents local variable expansion.
- Keep commands short — long heredocs are unreliable through the TTY proxy.
- `exit` at the end is required or the session hangs.

**Transferring code — preferred methods (ranked):**

1. **git clone + sed patches** (best for small changes): Clone repo on pod, apply
   fixes via `sed`. Keeps commands short and TTY-safe.
   ```bash
   ssh -tt ... USER@ssh.runpod.io << 'EOF'
   cd /workspace && git clone https://github.com/user/repo.git
   cd repo && sed -i 's/old_pattern/new_pattern/g' file.py
   exit
   EOF
   ```
   > Note: the strict deployment rule above forbids editing code on the pod. Use
   > git pull (method 2) for any real code change; reserve `sed` for throwaway
   > diagnostics only.
2. **Commit changes and pull on remote**: Push local changes to a branch, have
   the pod pull them. Cleanest for repeated deploys. **Preferred.**
3. **Base64 encode small tarballs** (<15KB): Works through the gateway but >15KB
   risks TTY corruption.
   ```bash
   tar czf /tmp/patch.tar.gz file1.py file2.py
   B64=$(base64 -i /tmp/patch.tar.gz)
   ssh -tt ... USER@ssh.runpod.io << EOF
   cd /workspace && echo '${B64}' | base64 -d | tar xz
   exit
   EOF
   ```
4. **Direct TCP SSH** (when available — supports full scp/rsync):
   ```bash
   # Get IP:port from pod Connect menu > TCP Port Mapping
   scp -P <PORT> -i ~/.ssh/id_ed25519 file root@<IP>:/workspace/
   rsync -avz -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" dir/ root@<IP>:/workspace/dir/
   ```
   Note: Direct TCP port may take minutes to become reachable after pod start,
   and can time out intermittently.

**Pod environment notes:**

- `tmux` is NOT pre-installed — install with
  `apt-get update -qq && apt-get install -y -qq tmux`.
- `git-lfs` is NOT pre-installed either —
  `apt-get install -y git-lfs && git lfs install`.
- Use tmux to keep jobs alive after SSH disconnect:
  `tmux new-session -d -s NAME 'command'`.
- Run heavy/long steps INSIDE tmux so an SSH disconnect can't kill them.
- Python is `python3` (3.11.x typically), pip packages install globally.
- Working directory: `/workspace/` (persists across pod restarts if using
  network volumes).
- `transformers` on RunPod pods is often outdated — install from source for new
  model support: `pip install git+https://github.com/huggingface/transformers.git`.
- Older `transformers` uses `use_auth_token`; newer uses `token` — check for
  `TypeError: unexpected keyword argument 'use_auth_token'`.

**Monitoring a running job:**

```bash
ssh -tt ... USER@ssh.runpod.io << 'EOF'
tail -30 /workspace/<repo>/<run>.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
exit
EOF
```

- `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader` is
  the reliable health signal: ~98% util + tens of GB used = training;
  0%/0 MiB = finished or dead (cross-check the run's sentinel/done file).
- Prefer the `/poll-log` skill over hand-rolled SSH+grep loops.
- If hand-rolling, keep the remote command to plain `echo`/`tail`/`cat`/`grep`
  with NO nested `$(...)` whose output you reparse locally — the gateway triple-
  echoes and corrupts captured substrings. Dump raw lines and parse on the Mac
  side, or just `tail` the log and eyeball it.

**Dependencies (baseline):**

```bash
pip install torch transformers datasets wandb
```

> `deploy_runpod.sh` (where present) handles rsync + dependency install +
> tmux-wrapped launch. Only works with direct TCP SSH (not the `ssh.runpod.io`
> gateway).

---

## Waiting on a pod-side condition (don't poll a static file)

- A backgrounded Bash task (`run_in_background: true`) sends a completion
  notification on its own. After launching it, STOP — do not poll its output
  file. Re-reading an unchanged file burns tool calls for nothing.
- To wait on a pod-side condition, run ONE foreground SSH
  `until [ cond ]; do sleep 15; done; <dump state>` with a generous `timeout`
  (up to 600s). Let it block this turn and return the answer. Do not chain short
  sleeps; do not re-read.
- A long SSH call may get pushed to the background — that is fine; just wait for
  its single notification.

---

## Pod setup fast path & lessons learned

These correspond to mistakes that actually cost time on a real launch. Don't
repeat them.

### Token verification UP FRONT (cheap, ~5s)

Verify the HF token has the RIGHT access before installing anything — both the
model gate AND any private dataset. HF returns **404 (masked), not 401**, when an
authenticated user lacks access to a private repo, so "not found" means
"no access", not "wrong URL".

```bash
# whoami: confirm the user and that the expected org is in orgs
curl -s -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2 \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('name'),[o.get('name') for o in d.get('orgs',[])])"
# dataset access: MUST be 200, not 404
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/datasets/<ORG>/<DATASET>
```

Wire the token for LFS (git, since LFS pulls over https):

```bash
git config --global credential.helper store
printf 'protocol=https\nhost=huggingface.co\nusername=<hfuser>\npassword=<hf_token>\n\n' | git credential approve
```

If you previously stored a bad token, `git credential reject` it first, or just
overwrite `~/.git-credentials`. If a submodule clone left a half-state, reset it
before retry:

```bash
git submodule deinit -f <submodule/path>
rm -rf .git/modules/<submodule/path>
git submodule update --init <submodule/path>
cd <submodule/path> && git lfs pull
```

Always VERIFY pulled LFS files are real, not stubs (~130 bytes). If a file is
<10 KB, the token lacked access → BLOCKED. Don't burn time on workarounds; ask
for a correctly-scoped token.

### Reuse the pod's system torch — do NOT reinstall it

The biggest time sink: reinstalling torch onto network storage via
`pip install --target /workspace/pylibs ... torch` pulls a redundant multi-GB
CUDA stack and writes it to slow RunPod network storage (~25 min apparent hang).

The pod ALREADY has a working system torch. Check first and reuse it:

```bash
python3 -c "import torch; print('system torch', torch.__version__, torch.cuda.is_available())"
```

Install ONLY what's missing into the SYSTEM env (no `--target`, no
`PYTHONPATH=/workspace/pylibs` indirection):

```bash
pip install --cache-dir /workspace/pip_cache git+https://github.com/huggingface/transformers.git datasets accelerate
```

`transformers`-from-source is needed for newer model support; its wheel build
takes ~2–3 min (normal). `datasets`/`accelerate` are small. Ignore any brief that
tells you to `--target /workspace/pylibs` + reinstall torch.

Diagnostic to tell "slow-but-fine" from "stuck": compare pod `date` to the setup
log mtime, and watch `du -sh` of the install target grow + the pip/git PID's CPU
time tick up. If the target dir is growing, it's working.

### Launchers that self-detach

A launcher may `setsid`/`nohup` the training process and write to a fixed log
path plus a `*_done.txt` sentinel with `exit_code=N`. Consequences:

- The run is doubly protected: tmux (your wrapper) + setsid (the launcher).
  Either dying won't kill training.
- Your tmux wrapper may exit/return before the log file first appears; watch for
  the sentinel and the log, not the tmux session living.
- `logs/` may contain committed artifacts from prior runs — don't confuse those
  with this run's fresh log.

---

## Working Style

- Match existing structure and naming.
- Prefer simple, readable implementations.
- Avoid style-only refactors unless requested.
- Keep changes narrowly scoped to the task.
- Do not use comments except for extremely important and otherwise unclear
  sections.
- Minimize git diffs: prefer small in-place edits over rewriting functions from
  scratch. Parameterize existing code rather than duplicating it.
- Reuse existing code when modifying or adding to files. Extract shared helpers
  from existing functions rather than writing parallel implementations. Check for
  similar logic nearby before writing new code.
- Validate behavior with the smallest relevant command/test.
- To understand the code to edit, it is often better to look at the changed code
  from commit history first, then if still confused look at many files.
- Do not catch errors unless they're expected to occur: letting code silently
  fail is far more dangerous than it crashing.
- NEVER use "quantization error" or "compression error" as the reasoning for why
  something doesn't work. More often than not, it's not the real reason — it's a
  convenient excuse to avoid really investigating the root problem. If you really
  think compression is the problem, design detailed and thorough experiments to
  capture exactly WHERE the compression becomes problematic and why.

---

## Git Safety

- Do not revert unrelated local changes.
- Do not use destructive git commands unless explicitly requested.
- Keep diffs focused and reviewable.
- NEVER cite yourself or any other agents in commits or PRs.

