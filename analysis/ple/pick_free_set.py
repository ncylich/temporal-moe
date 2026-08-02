#!/usr/bin/env python3
"""Choose which free-set cell to extend, from the cells on disk.

Prints one line: `tag free_set bpb rationale`, for a shell `read` to split.

Two rules, both stated before any of the cells they rank had run:

**Only data-seed-0 cells compete.** A replicate trains on a different draw of the corpus. Ranking it
alongside the others would let the luckier draw win and then be reported as a better configuration,
which is the error the replicate exists to detect.

**A tie inside the noise bar goes to the cheaper cell.** Sorting on BPB alone would hand the win to
+175% resident memory over +131% for a difference the program's own pre-registration calls noise.
Resident expert slots are the currency the residency thesis is denominated in, so they are the
tie-break, not an afterthought to mention in the write-up.

    pick_free_set.py <data-dir> <bar-in-bpb>
"""
import glob
import json
import os
import sys


def main(data, bar):
    cells = []
    for p in glob.glob(os.path.join(data, "ple_*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if "final_bpb" not in r or str(r.get("rank")) != "off":
            continue
        # Two spellings of the same thing: --free-set names the layers, --free-layers N means the
        # first N. ce_free2, the first cell of this line, used the second, so a filter that reads
        # only free_set drops it -- and it drops it silently, leaving a one-candidate field that
        # still looks like a comparison.
        layers = [x.strip() for x in (r.get("free_set") or "").split(",") if x.strip()] \
            or [str(i) for i in range(r.get("free_layers", 0))]
        if not layers or r.get("data_seed", 0) != 0:
            continue
        n = len(layers)
        cells.append({"bpb": r["final_bpb"], "slots": (16 - n) * 8 + n * 64,
                      "tag": r["tag"], "free_set": ",".join(layers), "tokens": r["train_tokens"]})
    if not cells:
        print("NONE - - no-free-set-cells-found")
        return 1

    cells.sort(key=lambda c: c["bpb"])
    best = cells[0]
    tied = [c for c in cells if c["bpb"] - best["bpb"] <= bar]
    pick = min(tied, key=lambda c: (c["slots"], c["bpb"]))
    why = (f"{len(cells)}cells,best={best['tag']}@{best['bpb']:.6f},"
           f"tied={len(tied)},on=" + ("memory" if pick is not best else "bpb"))
    for c in cells:
        mark = "<-" if c is pick else "  "
        print(f"# {mark} {c['tag']:24s} bpb={c['bpb']:.6f} slots={c['slots']:3d} "
              f"(+{c['slots'] / 128 * 100 - 100:5.1f}%) tokens={c['tokens'] / 1e6:.0f}M",
              file=sys.stderr)
    print(f"{pick['tag']} {pick['free_set']} {pick['bpb']:.6f} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], float(sys.argv[2])))
