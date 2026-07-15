# Ablation programs — plans + verdicts

Each doc is one mechanism-search program run on the temporal router: the plan, then the appended
verdicts. Numbers live in `../../../results/ablations/` (see its README for per-CSV provenance).

| Program | Verdict | Backing data |
|---|---|---|
| [alignment-program.md](alignment-program.md) | Track A (aux-free → Karen momentum): promotion disqualified (residency degeneracy); Track B (anticipatory): Goodharts its target | `alignment_cells.csv`, `karen_promotion_s2_1e17.csv`, `seed_replicates.csv` |
| [local-global-program.md](local-global-program.md) | LG1 logratio momentum: diversity-safe but A3-inert; LG2 bursty loss: Goodhart collapse; LG3 declined; R-knob: monotone BPB-vs-R frontier | `temporal_router_*_sweep.csv`, `rsweep.csv` |
| [coherence-loss.md](coherence-loss.md) | Temporal-coherence BCE loss: no quality gain | `alignment_cells.csv` |
| [decision-time-alignment.md](decision-time-alignment.md) | Program guardrails (what this work is NOT) + decision-time plan | — |
