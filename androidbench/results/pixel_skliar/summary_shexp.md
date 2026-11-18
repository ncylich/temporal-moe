| arm | switch | texts | tok/s, mean over texts [min, max] | ms per token | loads per decoded token-layer | later tokens: mean ms | p95 ms | slowest ms | ms blocked on fetches per token | fit: ms = a + b x loads | peak RSS MB | void rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trainedtext_stream | off | 5 | 32.36 [31.64, 32.75] | 30.90 | 1.0000 | 30.68 | 36.77 | 56.8 | 6.64 | -- | 848 | 0 |
| trainedtext_stream | on | 5 | 33.67 [33.56, 33.90] | 29.70 | 1.0000 | 29.45 | 35.28 | 50.4 | 5.65 | -- | 849 | 0 |
| skliar_stream | off | 5 | 32.54 [32.04, 32.89] | 30.73 | 1.0022 | 30.43 | 37.64 | 61.1 | 6.15 | 24.06 + 0.491 | 888 | 0 |
| skliar_stream | on | 5 | 33.30 [32.59, 34.12] | 30.03 | 1.0022 | 29.70 | 36.66 | 61.3 | 5.64 | 23.41 + 0.485 | 890 | 0 |
| stock_resident | off | 5 | 48.09 [48.02, 48.18] | 20.79 | -- | 20.79 | 20.92 | 22.5 | 0.00 | -- | 886 | 0 |
| stock_resident | on | 5 | 48.10 [48.01, 48.17] | 20.79 | -- | 20.78 | 20.90 | 23.4 | 0.00 | -- | 888 | 0 |

Per text, tok/s (three repetitions), off then on, and on against off in percent:

| arm | text 1 | text 2 | text 3 | text 4 | text 5 | mean of the paired differences [min, max] |
|---|---|---|---|---|---|---|
| trainedtext_stream | 32.62 / 33.90 (+3.9%) | 32.32 / 33.60 (+3.9%) | 32.75 / 33.69 (+2.9%) | 31.64 / 33.59 (+6.2%) | 32.48 / 33.56 (+3.3%) | +4.05% [+2.9, +6.2] |
| skliar_stream | 32.89 / 32.94 (+0.1%) | 32.73 / 33.21 (+1.5%) | 32.43 / 33.64 (+3.8%) | 32.04 / 32.59 (+1.7%) | 32.63 / 34.12 (+4.6%) | +2.32% [+0.1, +4.6] |
| stock_resident | 48.18 / 48.16 (-0.0%) | 48.02 / 48.17 (+0.3%) | 48.10 / 48.06 (-0.1%) | 48.10 / 48.01 (-0.2%) | 48.05 / 48.10 (+0.1%) | +0.02% [-0.2, +0.3] |

Trained model against Skliar on the same text, tok/s, positive = trained faster:

| switch | text 1 | text 2 | text 3 | text 4 | text 5 | mean of the paired differences [min, max] | ratio of the means |
|---|---|---|---|---|---|---|---|
| off | -0.8% | -1.2% | +1.0% | -1.3% | -0.5% | -0.56% [-1.3, +1.0] | -0.56% |
| on | +2.9% | +1.2% | +0.1% | +3.1% | -1.6% | +1.14% [-1.6, +3.1] | +1.11% |

Skliar, later tokens grouped by the largest number of experts one layer loaded on the token (ms per token, off / on):

| largest per-layer load count | tokens off / on | ms per token off / on | ms blocked on fetches off / on |
|---|---|---|---|
| 1 | 623 / 623 | 28.41 / 27.91 | 4.88 / 4.35 |
| 2 | 696 / 696 | 30.28 / 29.15 | 5.99 / 5.27 |
| 3 | 335 / 335 | 31.72 / 31.14 | 6.90 / 6.75 |
| 4 | 169 / 169 | 33.52 / 33.03 | 8.33 / 8.01 |
| 5+ | 82 / 82 | 35.35 / 35.19 | 9.65 / 9.22 |
