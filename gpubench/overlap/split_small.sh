#!/usr/bin/env bash
# The MoE split on the 1.26 B models, real text (P1-9 protocol through fast_small.sh), streamed with 18 slots:
#   full     the arm as served
#   nb       the same kernels, no bytes moved (TEMPORAL_ROUTE_NOCOPY=1)
#   skipAB   no bytes and the fused op's two launches skipped (TEMPORAL_FUSED_SKIP=3)
#   stub     routing kernel replaced by an empty launch (TEMPORAL_ROUTE_STUB=1; it selects no load)
# op = nb - skipAB, routing kernel = nb - stub, exposed load = full - nb. Passes run the four in order, then
# reversed, so drift does not decide a difference.
#   split_small.sh <tag> "<switches>" [passes]      e.g. split_small.sh all "TEMPORAL_QKV=2 ... TEMPORAL_ATTN1=1" 2
set -uo pipefail
TAG=$1; SW=$2; NP=${3:-2}; O=/workspace/rel/ov/small
S="bash /workspace/rel/repo/gpubench/overlap/fast_small.sh"
for p in $(seq 1 $NP); do
  L="full nb skipAB stub"; R=0; [ $((p % 2)) = 0 ] && { L="stub skipAB nb full"; R=1; }
  for c in $L; do
    case $c in full) E="";; nb) E="TEMPORAL_ROUTE_NOCOPY=1";; skipAB) E="TEMPORAL_ROUTE_NOCOPY=1 TEMPORAL_FUSED_SKIP=3";; stub) E="TEMPORAL_ROUTE_STUB=1";; esac
    REV=$R X="$SW $E" $S split-$TAG-$c-p$p
  done
done
python3 - $O "$TAG" $NP <<'PY'
import json, sys
o, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ms = {}
for c in ("full", "nb", "skipAB", "stub"):
    for p in range(1, n + 1):
        for l in open(f"{o}/split-{tag}-{c}-p{p}.jsonl"):
            r = json.loads(l)
            if "tok_s" in r: ms.setdefault((r["tag"], c), []).append(1000.0 / r["tok_s"])
m = {k: sum(v) / len(v) for k, v in ms.items()}
print(f"split {tag}: ms per token, mean of {n} passes x rows")
tot = {}
for a in ("ours", "skliar", "trained"):
    op, rt, ld = m[a, "nb"] - m[a, "skipAB"], m[a, "nb"] - m[a, "stub"], m[a, "full"] - m[a, "nb"]
    tot[a] = op + rt + ld
    print(f"  {a:8s} token {m[a,'full']:.4f} ({1000/m[a,'full']:.1f} tok/s) | op {op:.4f} | routing kernel {rt:.4f} | exposed load {ld:.4f} | MoE {tot[a]:.4f} = {100*tot[a]/m[a,'full']:.1f} % of the token"
          f" | no bytes {m[a,'nb']:.4f} skipAB {m[a,'skipAB']:.4f} stub {m[a,'stub']:.4f}")
o_, s_ = "ours", "skliar"
d_rt = (m[s_, "nb"] - m[s_, "stub"]) - (m[o_, "nb"] - m[o_, "stub"]); d_ld = (m[s_, "full"] - m[s_, "nb"]) - (m[o_, "full"] - m[o_, "nb"])
print(f"  Skliar behind ours: {100*(m[s_,'full']/m[o_,'full']-1):.1f} % of the token; at the MoE layers {100*(tot[s_]/tot[o_]-1):.1f} % "
      f"({100*d_rt/tot[o_]:.1f} points routing kernel, {100*d_ld/tot[o_]:.1f} points load pattern, {100*((m[s_,'nb']-m[s_,'skipAB'])-(m[o_,'nb']-m[o_,'skipAB']))/tot[o_]:.1f} op)")
PY
