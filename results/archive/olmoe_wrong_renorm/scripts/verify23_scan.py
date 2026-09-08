import sys, torch
sys.path.insert(0, "/workspace/FLAME-MoE")
from temporal.temporal_router import compute_resident_mask, compute_resident_mask_accel
torch.manual_seed(0)
S, B, E, k = 4096, 2, 64, 8
lg = torch.randn(S, B, E)
m_torch = compute_resident_mask(lg, k, evict="min_logit")
# VERIFY 3: swaps per token <= 1 (experts newly resident vs previous token), and exactly k resident
resident_count = m_torch.sum(-1)                    # [S,B]
added = (m_torch[1:] & ~m_torch[:-1]).sum(-1)       # [S-1,B] experts added at each t>=1
print(f"VERIFY3 resident/token: min={int(resident_count.min())} max={int(resident_count.max())} (want =={k})")
print(f"VERIFY3 swaps/token: max={int(added.max())} (want <=1)  -> {'PASS' if added.max()<=1 and resident_count.min()==k and resident_count.max()==k else 'FAIL'}")
# VERIFY 2: triton accel matches torch reference
try:
    m_accel = compute_resident_mask_accel(lg.cuda(), k, evict="min_logit").cpu()
    match = torch.equal(m_torch, m_accel)
    disagree = (m_torch != m_accel).float().mean().item()
    print(f"VERIFY2 triton-vs-torch parity: exact_match={match}  disagree_frac={disagree:.2e}  -> {'PASS' if disagree < 1e-6 else 'FAIL'}")
except Exception as e:
    print(f"VERIFY2 accel path error: {type(e).__name__}: {str(e)[:120]}")
