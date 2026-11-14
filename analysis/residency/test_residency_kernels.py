#!/usr/bin/env python3
"""Bit-exactness of residency_kernels against the plain-torch references, on the GPU.

  decode_step  vs reference_step : random logits, live rows/slots subset, rho, swaps, dtypes,
                                   E in {64,128,256}; masked logits, bank rows and swap
                                   counts must all match exactly.
  chunk_scan   vs reference_scan : seeded and unseeded chunks, n in {1,2,7,300}.
Then the continuous-batching walker test (test_vllm_walker.py) under TEMPORAL_WALKER=fast.
"""
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency_kernels as RK                                       # noqa: E402

dev = torch.device("cuda")
g = torch.Generator(device="cpu").manual_seed(7)
cases = 0
for E in (64, 128, 256):
    for dtype in (torch.float32, torch.bfloat16):
        for rho in (0.0, 0.5, 1.25):
            for swaps in (1, 2):
                N, cap, D, R = 50, 64, 37, 8
                logits = torch.randn(N, E, generator=g).to(dev, dtype)
                # a few exact ties to exercise first-index tie-breaking
                logits[3, 5] = logits[3, 9]; logits[10, :4] = logits[10, 0]
                bank = torch.zeros(cap, E, dtype=torch.int8, device=dev)
                for s in range(cap):
                    bank[s].scatter_(0, torch.randperm(E, generator=g)[:R].to(dev), 1)
                rows = torch.randperm(N, generator=g)[:D].to(dev, torch.int32)
                slots = torch.randperm(cap, generator=g)[:D].to(dev, torch.int32)
                rows_b = torch.zeros(cap, dtype=torch.int32, device=dev); rows_b[:D] = rows
                slots_b = torch.zeros(cap, dtype=torch.int32, device=dev); slots_b[:D] = slots
                ndec = torch.tensor([D], dtype=torch.int32, device=dev)
                count = torch.zeros(2, dtype=torch.int64, device=dev)
                out = logits.clone(); bank2 = bank.clone()
                RK.decode_step(out, logits, bank2, rows_b, slots_b, ndec, count, rho, swaps, cap)
                ref, done = RK.reference_step(logits.index_select(0, rows.long()),
                                              bank.index_select(0, slots.long()).bool(), rho, swaps)
                assert torch.equal(bank2.index_select(0, slots.long()).bool(), ref), (E, dtype, rho, swaps, "bank")
                untouched = torch.ones(cap, dtype=torch.bool, device=dev); untouched[slots.long()] = False
                assert torch.equal(bank2[untouched], bank[untouched]), "untouched slots changed"
                want = logits.index_select(0, rows.long()).masked_fill(~ref, float("-inf"))
                assert torch.equal(out.index_select(0, rows.long()), want), (E, dtype, rho, swaps, "out")
                keep = torch.ones(N, dtype=torch.bool, device=dev); keep[rows.long()] = False
                assert torch.equal(out[keep], logits[keep]), "non-decode rows changed"
                assert int(count[0]) == int(done.sum()) and int(count[1]) == D, (int(count[0]), int(done.sum()))
                cases += 1
                for n in (1, 2, 7, 300):
                    lg = torch.randn(n, E, generator=g).to(dev, dtype)
                    for seeded in (False, True):
                        res0 = bank[3].clone() if seeded else None
                        c = torch.zeros(2, dtype=torch.int64, device=dev)
                        m = RK.chunk_scan(lg, res0, R, rho, swaps, c)
                        refm, total = RK.reference_scan(lg, res0, R, rho, swaps)
                        assert torch.equal(m.bool(), refm), (E, dtype, rho, swaps, n, seeded, "scan")
                        assert int(c[0]) == total, (int(c[0]), total)
                        cases += 1
print(f"KERNEL PARITY PASS ({cases} cases: decode_step + chunk_scan, E/dtype/rho/swaps/seeded)")

env = dict(os.environ, TEMPORAL_WALKER="fast")
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "test_vllm_walker.py")],
                   env=env, capture_output=True, text=True)
print(r.stdout.strip()[-300:]); print(r.stderr.strip()[-600:] if r.returncode else "", end="")
assert r.returncode == 0, "walker test failed under the fast walker"
