"""Repo-root-relative paths for the analysis and probe scripts.

Resolution order:
  1. $TMOE_ROOT, exported by scripts/env.sh
  2. `git rev-parse --show-toplevel` from this file's directory
  3. this file's parent directory (analysis/../), so a plain copy still works

Usage:
    from analysis.paths import ROOT, RUNS, CACHE, FIGDATA

or, when the probe is run as a script from inside analysis/:
    from paths import ROOT, RUNS, CACHE, FIGDATA
"""

import os
import subprocess

__all__ = ["ROOT", "RUNS", "CACHE", "FIGDATA"]


def _root() -> str:
    env = os.environ.get("TMOE_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            ["git", "-C", here, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            return out
    except (OSError, subprocess.CalledProcessError):
        pass
    return os.path.dirname(here)


ROOT = _root()
RUNS = os.path.join(ROOT, "results/phase0/runs")
CACHE = os.path.join(ROOT, "results/phase0/probe_batch_cache")
FIGDATA = os.path.join(ROOT, "results/phase0/figure_data")  # small CSVs behind every figure
