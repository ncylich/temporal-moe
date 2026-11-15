#!/usr/bin/env python3
"""Prompt token-layers where the engine's top-k differs from torch.topk on the same logits: how many are exact
ties at the cut (the K-th and (K+1)-th router logits are equal), i.e. two equally valid selections.

    python3 pixel_skliar_ties.py route.bin [--K 18 --E 192]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpubench", "released"))
from route_check import read_dump

ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--K", type=int, default=18); ap.add_argument("--E", type=int, default=192)
a = ap.parse_args()
hdr, ids, lg = read_dump(a.dump, a.K, a.E)
p = (hdr[:, 0] >= 0) & (hdr[:, 1] > 1)
srt = -np.sort(-lg[p], axis=1)
tie = srt[:, a.K - 1] == srt[:, a.K]
ref = np.argsort(-lg[p], axis=1, kind="stable")[:, :a.K]
diff = np.array([set(r) != set(e) for r, e in zip(ref.tolist(), ids[p].tolist())])
# a differing set is valid if every selected logit is >= the K-th largest and every unselected one is <= it
kth = srt[:, a.K - 1]
sel_min = np.take_along_axis(lg[p], ids[p], axis=1).min(axis=1)
valid = sel_min >= kth
d = (hdr[:, 0] >= 0) & (hdr[:, 1] == 1)
sd = -np.sort(-lg[d], axis=1)
print(json.dumps({"prompt_token_layers": int(p.sum()), "tie_at_cut": int(tie.sum()), "differ_from_stable_topk": int(diff.sum()),
                  "differ_and_tie": int((diff & tie).sum()), "engine_selection_not_a_valid_topk": int((~valid).sum()),
                  "decode_token_layers": int(d.sum()), "decode_exact_logit_ties_at_cut": int((sd[:, a.K - 1] == sd[:, a.K]).sum())}))
