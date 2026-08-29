# From-scratch on-policy sweep (one knob per cell; GSM8K n=1319; paired vs the running best)

| cell | env | KL@50 -> @100 | free | R8 | R16 | R8 vs best (fixed/broken, z) | verdict |
|---|---|---|---|---|---|---|---|
| baseline | {'TMOE_ANCHOR_W': '0', 'TMOE_BUDGET_ON': 'sampled'} | 0.556 -> 0.526 | 86.6 | 81.7 | 86.1 | -0.8 (73/84, z=-0.9) | no gain |
