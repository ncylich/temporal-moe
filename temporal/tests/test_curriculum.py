#!/usr/bin/env python3
"""Curriculum knobs (temporal -> free schedules, heterogeneous batches): pure-function tests for
schedule_step, schedule_interp, free_rows and current_iteration in temporal/temporal_router.py.
Run: $PY -m pytest temporal/tests/test_curriculum.py
"""
import os, sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from temporal.temporal_router import (schedule_step, schedule_interp, free_rows,  # noqa: E402
                                      current_iteration, banner_knobs)


def test_schedule_step_picks_last_start_at_or_below():
    spec = "0:18,1931:E"
    assert schedule_step(spec, 0) == "18"
    assert schedule_step(spec, 1930) == "18"
    assert schedule_step(spec, 1931) == "E"
    assert schedule_step(spec, 99999) == "E"


def test_schedule_step_unsorted_and_before_first():
    spec = "2000:72, 500:36, 1000:E"
    assert schedule_step(spec, 100) == "36"      # before the first point: its value
    assert schedule_step(spec, 700) == "36"
    assert schedule_step(spec, 1500) == "E"
    assert schedule_step(spec, 2000) == "72"


def test_schedule_interp_linear_and_clamped():
    spec = "0:0,1544:0,3089:1"
    assert schedule_interp(spec, 0) == 0.0
    assert schedule_interp(spec, 1544) == 0.0
    assert abs(schedule_interp(spec, 1544 + (3089 - 1544) // 2) - 0.5) < 1e-3
    assert schedule_interp(spec, 3089) == 1.0
    assert schedule_interp(spec, 5000) == 1.0
    assert schedule_interp("100:0.3", 0) == 0.3


def test_free_rows_sets_fraction_of_batch_all_true():
    S, B, E = 7, 10, 16
    mask = torch.zeros(S, B, E, dtype=torch.bool)
    mask[:, :, :3] = True
    out = free_rows(mask, 0.3, it=5, layer=2)
    freed = [b for b in range(B) if bool(out[:, b, :].all())]
    assert len(freed) == 3
    for b in range(B):
        if b not in freed:
            assert torch.equal(out[:, b, :], mask[:, b, :])
    assert not mask[:, :, 3:].any()                     # input untouched


def test_free_rows_edges_and_determinism():
    mask = torch.zeros(4, 6, 8, dtype=torch.bool)
    assert free_rows(mask, 0.0, 1, 1) is mask
    assert free_rows(mask, 1.0, 1, 1).all()
    a = free_rows(mask, 0.5, it=3, layer=1); b = free_rows(mask, 0.5, it=3, layer=1)
    assert torch.equal(a, b)
    c = free_rows(mask, 0.5, it=4, layer=1); d = free_rows(mask, 0.5, it=3, layer=2)
    assert not torch.equal(a, c) or not torch.equal(a, d)   # keyed on iteration and layer


def test_current_iteration_is_zero_without_megatron_args():
    assert current_iteration() == 0


def test_banner_names_curriculum_knobs(monkeypatch):
    monkeypatch.setenv("TEMPORAL_ITER_SCHEDULE", "0:18,1931:E")
    monkeypatch.setenv("TEMPORAL_SHADOW", "1")
    b = banner_knobs()
    assert "iter_schedule=0:18,1931:E" in b and "shadow=1" in b
