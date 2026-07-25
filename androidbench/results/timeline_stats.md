| metric | temporal (two-pass, repacked) | baseline (resident, repacked) |
|---|---|---|
| decode tok/s | 20.25 | 30.46 |
| % of ceiling | 66% | 100% |
| per-layer wall (us) | 901 | 468 |
| per-GEMV (us) | 7.71 | 7.62 |
| fetches/token | 266 | 0 |
| layers per 7 ms | 7.8 | 15 |

Two-pass layer split (median): resident pass 453 us + stall 422 us + new-expert pass 28 us = 903 us.
