"""Run any policy (baseline or trained agent) on a list of days and collect metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..baselines.lp_optimal import solve_day
from ..baselines.policies import Policy


class AgentPolicy(Policy):
    """Adapts a trained BaseAgent to the baseline Policy interface."""

    def __init__(self, agent, name: str):
        self.agent, self.name = agent, name

    def act(self, obs, env) -> int:
        return self.agent.predict(obs)


def run_episode(policy: Policy, env, day: str, record: bool = False):
    obs, _ = env.reset(options={"day": day})
    policy.reset(env)
    rec, done = [], False
    while not done:
        obs, _, term, trunc, info = env.step(policy.act(obs, env))
        done = term or trunc
        if record:
            rec.append(info)
    summary = env.summary()
    if record:
        ts = pd.DataFrame(rec)
        ts["cum_cost"] = ts["cost"].cumsum()
        return summary, ts
    return summary


def evaluate_policy(policy: Policy, env, days: list[str], label: str | None = None, seed: int = 0) -> pd.DataFrame:
    rows = []
    for d in days:
        s = run_episode(policy, env, d)
        s.update(policy=label or policy.name, seed=seed)
        rows.append(s)
    return pd.DataFrame(rows)


def lp_day_summary(env, day: str, w_unmet: float | None = None, record: bool = False):
    """Same metrics as run_episode, but from the perfect-foresight LP."""
    s = env.ds.days[day]
    prices = env.tariff.price_vector(day)
    w = env.cfg.w_unmet if w_unmet is None else w_unmet
    res = solve_day(s, prices, env.cap, env.dt, w)
    demand = float(s.energy.sum())
    unmet = float(res["remaining"].sum())
    summary = {
        "day": day, "n_evs": s.n, "demand_kwh": demand,
        "delivered_kwh": demand - unmet, "unmet_kwh": unmet, "cost": res["cost"],
        "peak_kw": float(res["load"].max()) if len(res["load"]) else 0.0,
        "avg_wait_min": float("nan"),  # not defined for the continuous LP plan
        "energy_served_pct": 100.0 * (1 - unmet / demand) if demand > 0 else 100.0,
        "evs_fully_served_pct": 100.0 * float(np.mean(res["remaining"] <= 0.05)) if s.n else 100.0,
    }
    if record:
        ts = pd.DataFrame({"price": prices, "load": res["load"]})
        ts["cost"] = ts["price"] * ts["load"] * env.dt
        ts["cum_cost"] = ts["cost"].cumsum()
        return summary, ts
    return summary


def evaluate_lp(env, days: list[str], label: str = "lp_optimal") -> pd.DataFrame:
    rows = []
    for d in days:
        s = lp_day_summary(env, d)
        s.update(policy=label, seed=0)
        rows.append(s)
    return pd.DataFrame(rows)
