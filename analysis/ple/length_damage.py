#!/usr/bin/env python3
"""Response length against benchmark damage. Lengths are reconstructed as final logged
vLLM output throughput x elapsed / items (estimate, ~10%); damages from the audited
genbench CSVs. Finding: pooled Spearman +0.72 (p=0.01) with damage NEGATIVE, i.e.
SHORT-response tasks take the damage; no evidence of error accumulation over length.
Data table inline (log-derived lengths pinned at analysis time; logs are ephemeral)."""
DATA = {  # (model, task): (mean_free_len_tokens, dmg_at_k, dmg_at_12p5)
 ("olmoe", "gsm8k"): (91, -28.5, None), ("olmoe", "humaneval"): (101, -8.5, None),
 ("olmoe", "mmlu"): (125, -15.8, None), ("olmoe", "ifeval"): (370, -9.0, None),
 ("gemma4", "gsm8k"): (229, -9.5, -1.0), ("gemma4", "mmlu"): (334, -5.7, -2.6),
 ("gemma4", "ifeval"): (336, -1.0, -2.5),
 ("qwen35", "humaneval"): (145, -4.9, -3.7), ("qwen35", "gsm8k"): (877, -7.5, -4.0),
 ("qwen35", "mmlu"): (965, 3.9, 2.2), ("qwen35", "ifeval"): (1264, 0.5, 0.5),
}
if __name__ == "__main__":
    from scipy.stats import spearmanr
    pts = [(v[0], v[1]) for v in DATA.values()]
    rho, p = spearmanr(*zip(*pts))
    print(f"pooled spearman(length, damage@R=k) = {rho:+.2f} (n={len(pts)}, p={p:.3f})")
