#!/usr/bin/env python3
"""Backward-compat shim — the temporal router moved to the `temporal/` package (Slice A refactor).

  core       -> temporal.temporal_router      (rolling-residency routing, Triton scan, install)
  ablations  -> temporal.ablation_mechanisms  (default-off negative-result experimental knobs)

This module re-exports the combined public surface so existing importers keep working unmodified
(`import temporal_router`; pretrain_temporal.py -> temporal_router.install()). New code should
import from the `temporal` package directly.
"""
import sys

from temporal import temporal_router as _core, ablation_mechanisms as _ab

# Backward-compat: expose the experimental ablation knobs as attributes of the core module so both
# `temporal_router.<name>` and `from temporal_router import <name>` resolve for core AND ablation
# symbols. The core module itself references the ablation functions module-qualified, so this
# injection is purely for the legacy import surface and does not change core behavior.
for _n in _ab.__all__:
    setattr(_core, _n, getattr(_ab, _n))

# Make `import temporal_router` return the real core module object, so module-level globals that
# callers reach into (e.g. _scan_path, _graph_cache — used by the GPU fast-path tests) stay live
# rather than being shadowed by a stale copy on this shim.
sys.modules[__name__] = _core
