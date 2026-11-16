"""Inner dispatch rule: split the station's power budget among the plugged-in EVs.

The RL agent decides HOW MUCH power the station draws this step; this classical rule decides WHO gets it.
Rule: Least Laxity First (LLF). Laxity = (time left) - (time needed at the EV's max rate).
The EV that is closest to missing its deadline is served first, at its full feasible rate.
"""
from __future__ import annotations

import numpy as np


def laxity_hours(dep: np.ndarray, remaining: np.ndarray, rate: np.ndarray, t: int, dt: float) -> np.ndarray:
    return (dep - t) * dt - remaining / rate


def llf_dispatch(budget_kw: float, remaining: np.ndarray, rate: np.ndarray, dep: np.ndarray,
                 t: int, dt: float) -> np.ndarray:
    """Return the power (kW) given to each EV in the arrays passed in (all arrays = active EVs only)."""
    n = len(remaining)
    p = np.zeros(n)
    if n == 0 or budget_kw <= 0:
        return p
    max_p = np.minimum(rate, remaining / dt)            # cannot exceed rate, nor deliver more than needed
    order = np.argsort(laxity_hours(dep, remaining, rate, t, dt), kind="stable")
    budget = budget_kw
    for k in order:
        x = min(max_p[k], budget)
        p[k] = x
        budget -= x
        if budget <= 1e-9:
            break
    return p
