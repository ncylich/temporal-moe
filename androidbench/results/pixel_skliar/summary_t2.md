## rows tagged `t2`

| arm | texts | tok/s, mean over texts [min, max] | sd over reps, mean | ms per token | loads per decoded token-layer | 0 / 1 / 2 / 3 / 4+ loads, % of token-layers | peak RSS MB | void rows |
|---|---|---|---|---|---|---|---|---|
| trained_deploy | 1 | 32.60 [32.60, 32.60] | 0.17 | 30.68 | -- | -- | 762 | 0 |
| skliar_stream | 5 | 32.20 [30.03, 33.23] | 0.78 | 31.05 | 1.0287 | 20.5 / 63.2 / 11.3 / 3.4 / 1.5 | 889 | 0 |

Per text (tok/s, the three repetitions):

| arm | text 0 | text 1 | text 2 | text 3 | text 4 | text 5 |
|---|---|---|---|---|---|---|
| trained_deploy | 32.60 (32.4, 32.7, 32.6) | -- | -- | -- | -- | -- |
| skliar_stream | -- | 32.76 (32.0, 33.6, 32.7) | 32.98 (32.1, 33.7, 33.1) | 30.03 (29.2, 30.5, 30.5) | 32.01 (30.9, 32.7, 32.4) | 33.23 (32.6, 33.6, 33.5) |

Per decoded token (engine token log; ms):

**trained_deploy**: 384 tokens, mean 30.67; first token after the prompt 68.8 (n 3); later tokens 30.37, median 29.57, p95 35.60, max 51.4

**skliar_stream**: 1920 tokens, mean 31.10; first token after the prompt 90.1 (n 15); later tokens 30.64, median 29.90, p95 38.54, max 93.3

| largest per-layer load count on the token | tokens | share | ms per token | loads on the token (13 layers) | ms blocked on fetches |
|---|---|---|---|---|---|
| 1 | 552 | 29.0% | 28.45 | 9.3 | 4.49 |
| 2 | 708 | 37.2% | 30.24 | 12.3 | 5.47 |
| 3 | 373 | 19.6% | 31.79 | 15.6 | 6.49 |
| 4 | 171 | 9.0% | 33.55 | 19.8 | 7.91 |
| 5+ | 101 | 5.3% | 36.21 | 23.2 | 9.46 |

Least squares, tokens after the first of each repetition: ms = 24.29 + 0.477 x loads on the token (r^2 0.32).

