## rows tagged `big1`

| arm | texts | tok/s, mean over texts [min, max] | sd over reps, mean | ms per token | loads per decoded token-layer | 0 / 1 / 2 / 3 / 4+ loads, % of token-layers | peak RSS MB | void rows |
|---|---|---|---|---|---|---|---|---|
| big_one_load | 1 | 14.42 [14.42, 14.42] | 0.44 | 69.34 | 1.0000 | 0.0 / 100.0 / 0.0 / 0.0 / 0.0 | 4409 | 0 |
| big_skliar_loads | 1 | 13.40 [13.40, 13.40] | 0.40 | 74.63 | 1.0277 | 20.6 / 63.4 / 11.0 / 3.4 / 1.6 | 4430 | 0 |

Per text (tok/s, the three repetitions):

| arm | text 0 |
|---|---|
| big_one_load | 14.42 (14.0, 14.9, 14.3) |
| big_skliar_loads | 13.40 (12.9, 13.6, 13.6) |

Per decoded token (engine token log; ms):

**big_one_load**: 768 tokens, mean 69.38; first token after the prompt 142.8 (n 6); later tokens 68.81, median 65.92, p95 83.14, max 129.8

| largest per-layer load count on the token | tokens | share | ms per token | loads on the token (all MoE layers) | ms blocked on fetches |
|---|---|---|---|---|---|
| 1 | 762 | 100.0% | 68.81 | 45.0 | 24.36 |

**big_skliar_loads**: 768 tokens, mean 74.67; first token after the prompt 269.0 (n 6); later tokens 73.14, median 71.07, p95 91.61, max 137.9

| largest per-layer load count on the token | tokens | share | ms per token | loads on the token (all MoE layers) | ms blocked on fetches |
|---|---|---|---|---|---|
| 2 | 74 | 9.7% | 68.99 | 39.9 | 23.24 |
| 3 | 292 | 38.3% | 71.94 | 44.7 | 24.64 |
| 4 | 258 | 33.9% | 74.51 | 47.9 | 25.65 |
| 5+ | 138 | 18.1% | 75.36 | 49.7 | 26.66 |

Least squares, tokens after the first of each repetition: ms = 43.75 + 0.636 x loads on the token (r^2 0.13).

