"""Offline optimum with perfect foresight (continuous per-EV power) solved as a linear program.

This is an UPPER BOUND on how good any online policy can be on a given day: it knows every future arrival,
departure and price, and is not restricted to discrete station power levels.

    minimise   sum_t price_t * load_t * dt  +  w_unmet * sum_i unmet_i
    subject to 0 <= p[i,t] <= rate_i (only while EV i is plugged in), sum_i p[i,t] <= cap,
               sum_t p[i,t]*dt + unmet_i = energy_i,  unmet_i >= 0
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from ..data.dataset import DaySessions


def solve_day(s: DaySessions, prices: np.ndarray, cap_kw: float, dt: float, w_unmet: float) -> dict:
    T, n = len(prices), s.n
    if n == 0:
        return {"cost": 0.0, "load": np.zeros(T), "remaining": np.zeros(0)}
    mask = np.zeros((n, T))
    for i in range(n):
        mask[i, s.arr[i]:s.dep[i]] = 1.0
    P = cp.Variable((n, T), nonneg=True)
    U = cp.Variable(n, nonneg=True)
    load = cp.sum(P, axis=0)
    cons = [
        P <= cp.multiply(mask, s.rate[:, None]),
        load <= cap_kw,
        cp.sum(P, axis=1) * dt + U == s.energy,
    ]
    obj = cp.Minimize(dt * prices @ load + w_unmet * cp.sum(U))
    prob = cp.Problem(obj, cons)
    prob.solve()
    if P.value is None:
        raise RuntimeError(f"LP failed: {prob.status}")
    p_val = np.maximum(P.value, 0.0)
    load_v = p_val.sum(axis=0)
    return {
        "cost": float(dt * prices @ load_v),
        "load": load_v,
        "remaining": np.maximum(U.value, 0.0),
    }
