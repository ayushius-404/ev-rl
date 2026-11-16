"""Small helpers for exploratory analysis."""
from __future__ import annotations

import numpy as np

from .dataset import DaySessions


def asap_load_profile(s: DaySessions, steps: int, dt: float) -> np.ndarray:
    """Station load (kW) if every EV charged at its own max rate from arrival (no station cap)."""
    load = np.zeros(steps)
    for a, dpt, e, r in zip(s.arr, s.dep, s.energy, s.rate):
        remaining = e
        for t in range(a, dpt):
            if remaining <= 1e-9:
                break
            p = min(r, remaining / dt)
            load[t] += p
            remaining -= p * dt
    return load


def plugged_in_count(s: DaySessions, steps: int) -> np.ndarray:
    """Number of EVs present (plugged in) at each step."""
    n = np.zeros(steps, dtype=int)
    for a, dpt in zip(s.arr, s.dep):
        n[a:dpt] += 1
    return n
