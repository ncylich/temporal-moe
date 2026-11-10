# Session prompt: laptop benchmark of the trained Temporal MoE

Paste everything below the line into a fresh agent session started from Windows PowerShell on
the laptop, the same way as `laptopbench/SESSION_PROMPT.md`.

---

You are running on Mohsen's laptop (i7-11370H, 16 GB LPDDR4x, Samsung PM981a NVMe, Windows 11).
Earlier sessions here (ledger entries `L1-*` and `L2-*` in `laptopbench/LEDGER.md`) measured an
11B random-weight model whose expert swaps were prescribed. This session measures the model we
actually trained, with its own router choosing the experts, and reports quality, peak memory and
decode speed for that one model. Reviewers asked for exactly this, so correctness of the routing
matters more than speed.

## Inputs

- Repo: `https://github.com/ncylich/temporal-moe.git`, branch `laptop-trained-model`. It is the
  `laptop` branch plus `scripts/serving_convert/` (converter, checks, engine patch) and this
  prompt. Update the existing clone at `C:\tmoe\temporal-moe`.
- Engine: `https://github.com/ncylich/llama.cpp.git`, branch `temporal-moe-win` at `43bec5f3`
  (the revision the earlier laptop runs used). Create a local branch there and apply
  `scripts/serving_convert/engine_patch/0001-*.patch` with `git am`, as its `README.md` says.
  Before building, fetch branch `trained-model-pixel` of the temporal-moe repo and use its copy
  of `scripts/serving_convert/engine_patch/` if it is newer: the phone session maintains the
  patch there. Do not push anything to the engine repo.
- Model weights: Hugging Face `https://huggingface.co/ncylich/temporal-moe-ckpts`, run
  `temporal_fine_g3_1e19`, iteration 4278 (fine model, 18 of 192 experts per layer, 13 MoE
  layers plus one dense layer, 1.26B parameters). The checkpoint is in a training format, so
  convert it here by following `scripts/serving_convert/README.md` steps 1, 3 and 4 inside WSL
  (about 2.5 GB download, CPU only, no Megatron needed for these steps). The README's commands
  were written on a Mac, so adapt the compiler line and library names to Linux.
- Expected result of the conversion: `fine_1e19-q4_0.gguf`, 744,281,312 bytes, sha256
  `c34d3445064a9b14cadacee1de9ec74d568995f078de6da0d8aef767b412c8ff`. If the size matches and
  the hash does not, record both and continue. If the size differs, stop and report.

## Read first

1. `scripts/serving_convert/README.md`, `RESULTS.md` and `engine_patch/README.md`.
2. `androidbench/TRAINED_MODEL_LEDGER.md`: the phone session's steps and what it found, including
   how it checked the engine's routing against the PyTorch reference.
3. `laptopbench/PLAN.md` and ledger entries `L1-3` onward for the driver, the gates in their
   Windows form, and the production flags (`SPLIT=1`, four compute threads).

## Phases

Write one ledger entry per phase in `laptopbench/LEDGER.md`, numbered `L3-1`, `L3-2`, ..., with
the exact commands, the numbers, and anything retracted, before starting the next phase.

**Phase 0, preflight.** Same checks as before (power plan, AC, Defender exclusions for `C:\tmoe`
and the WSL disk, free disk). Paste the `Get-MpPreference` exclusion output into the ledger if an
elevated shell is available, since that item is still open from `L1`.

**Phase 1, model and engine.** Convert the model, verify size and sha256. Apply the engine patch
and build `llama-bench-temporal` and `llama-perplexity` natively as in `L1-3`.

**Phase 2, routing check.** With `LLAMA_TEMPORAL_ROUTE=trained`, confirm the engine prints
`swaps_per_layer_token` close to 1.0 (the phone session measured 1.0000) and that evictions are
above zero when only 18 experts are resident. If you can build the F32 file and run
`check_engine_routing.py` in WSL, do it and record how many layer-token positions match the
reference (the phone session got 106,496 of 106,496). If you cannot, say so.

**Phase 3, driver.** Generalize `laptopbench/lapbench.py` for this model: name, sha256 and
geometry (13 MoE layers, 192 experts, 18 active) are currently hard-coded for the 11B model.
Re-scale the memory caps: the old 12 GB, 4 GB and 2.5 GB caps separate nothing for a 0.7 GB
model, so pick caps just above the measured footprints and say how you chose them. Dry-run every
stage first.

**Phase 4, gates.** Byte accounting, perplexity identity between streamed (18 resident) and
all-resident (192) on the same binary, and repacked against plain kernels, in the forms accepted
in `L1`. Run perplexity so every token goes through the routing window (batch size 1). Nothing is
timed until all three pass. A failed gate is debugged, never loosened.

**Phase 5, benchmark.** Context 0, 1k and 2k. Also 4k for speed only, labelled as beyond the
2,048-token training context. Arms: (a) deploy, 18 resident, trained routing, streaming from
NVMe; (b) same routing with all 192 experts resident; (c) same weights fully resident with plain
top-k, stated as a different routing function. Existing protocol: three batches per arm from
rest, clock probe before each, rests between, interleaved arms. Record tokens per second, peak
working set, swaps per layer per token, and the pool counters for every row.

**Phase 6, quality.** Report perplexity of the Q4_0 file under trained routing from the gate run,
next to the all-resident value. If you can rebuild the test tokens (README step 2) also report
loss in nats per token on the first 512 test samples, to compare with the PyTorch reference of
3.19202. If not, say so; the phone session reports that number.

**Phase 7, pack and push.** Rows, logs and the ledger under `laptopbench/` and `comms/laptop/`,
committed and pushed to branch `laptop-trained-model`. Commit messages describe the measurement.
No assistant or vendor names anywhere in commits or files.

## Rules

- Same rules as `laptopbench/SESSION_PROMPT.md`: numbers before opinions, n=3 for anything
  reported, a surprising number is a measurement bug until shown otherwise, no reboots or system
  changes without asking Mohsen, nothing outside `C:\tmoe`, `~/tmoe`, `.wslconfig` and the
  Defender exclusions.
- If a phase cannot be completed, finish everything that does not depend on it, then say exactly
  what is blocked and why, with the raw output.

## Final report

Under 400 words: gate results; a table of arm, context, tokens per second, ratio to arm (b) and
to arm (c), peak working set, swaps per token; the quality numbers; anything retracted; what is
blocked; the commit hash on `laptop-trained-model`.
