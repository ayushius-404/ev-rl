"""Gymnasium environment: one charging station, one operating day per episode, real sessions replayed.

MDP summary
-----------
Step      : `step_minutes` (default 15 min) -> 96 steps per episode.
Action    : Discrete(n_levels). Station power level = level/(n_levels-1) * station_cap_kw.
            Power is split among plugged-in EVs by Least Laxity First (see dispatch.py).
State     : see `_obs` (14 numbers, all scaled to [-1, 1]).
Reward    : -(price*energy + w_unmet*unmet_kwh_at_departure + w_peak*peak_excess) * reward_scale
Episode   : ends after the last step of the day (terminated=True; time-of-day is in the state).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import EnvConfig
from ..data.dataset import Dataset
from ..data.tariff import Tariff
from .dispatch import laxity_hours, llf_dispatch

OBS_NAMES_BASE = [
    "tod_sin", "tod_cos", "is_weekend", "price_now",
]  # followed by `forecast_steps` future prices, then the station-state features below
OBS_NAMES_STATE = [
    "n_active", "total_remaining", "min_laxity", "mean_laxity", "required_power", "last_load",
]


class EVStationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset: Dataset, tariff: Tariff, days: list[str], cfg: EnvConfig,
                 order: str = "random"):
        """
        days  : list of 'YYYY-MM-DD' operating days this env may replay.
        order : 'random' (training) or 'sequential' (evaluation, cycles through `days` in order).
        """
        super().__init__()
        assert order in ("random", "sequential")
        if not days:
            raise ValueError("EVStationEnv needs at least one day")
        self.ds, self.tariff, self.days, self.cfg, self.order = dataset, tariff, list(days), cfg, order
        self.T = dataset.steps_per_day
        self.dt = dataset.step_minutes / 60.0
        self.cap = float(cfg.station_cap_kw)
        self.levels = np.linspace(0.0, 1.0, cfg.n_levels)
        self.level_kw = self.levels * self.cap
        self._ptr = 0

        self.action_space = spaces.Discrete(cfg.n_levels)
        self.obs_dim = len(OBS_NAMES_BASE) + cfg.forecast_steps + len(OBS_NAMES_STATE)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.obs_names = OBS_NAMES_BASE + [f"price_+{i}" for i in range(1, cfg.forecast_steps + 1)] + OBS_NAMES_STATE

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options and "day" in options:
            day = options["day"]
        elif self.order == "sequential":
            day = self.days[self._ptr % len(self.days)]
            self._ptr += 1
        else:
            day = self.days[int(self.np_random.integers(len(self.days)))]
        s = self.ds.days[day]
        self.day = day
        self.arr, self.dep, self.rate = s.arr, s.dep, s.rate
        self.demand_per_ev = s.energy.copy()
        self.remaining = s.energy.copy()
        self.prices = self.tariff.price_vector(day)
        self.weekend = float(datetime.strptime(day, "%Y-%m-%d").weekday() >= 5)
        self.t = 0
        self.last_load = 0.0
        self._cost = self._delivered = self._unmet = 0.0
        self._peak = 0.0
        self._wait_steps = 0
        return self._obs(), {"day": day}

    def step(self, action: int):
        t = self.t
        idx = self._active(t)
        p = llf_dispatch(self.level_kw[int(action)], self.remaining[idx], self.rate[idx], self.dep[idx], t, self.dt)
        e = p * self.dt
        self.remaining[idx] = np.maximum(self.remaining[idx] - e, 0.0)

        energy, load, price = float(e.sum()), float(p.sum()), float(self.prices[t])
        cost = price * energy
        unmet = float(self.remaining[self.dep == t + 1].sum())  # EVs leaving after this step
        peak_excess = 0.0
        if self.cfg.peak_threshold_kw is not None:
            peak_excess = max(0.0, load - self.cfg.peak_threshold_kw)

        # --- REWARD ---------------------------------------------------------------------------
        # TODO(YOU): this is the heart of the project. Read it, understand every term, and decide
        # whether to extend it. Ideas to try (each is a nice ablation for the report):
        #   * a small "delay" penalty per step an EV waits with zero power,
        #   * a fairness term (penalise the most-starved EV),
        #   * a small bonus for energy delivered (reward shaping).
        # Remember: w_unmet must stay >> max price, otherwise "never charge" becomes optimal.
        n_wait = int(np.sum(p <= 1e-9)) if len(idx) else 0   # EVs present, still hungry, but given no power
        raw = -(cost + self.cfg.w_unmet * unmet + self.cfg.w_peak * peak_excess + self.cfg.w_wait * n_wait)
        reward = raw * self.cfg.reward_scale
        # --------------------------------------------------------------------------------------

        self._cost += cost
        self._delivered += energy
        self._unmet += unmet
        self._peak = max(self._peak, load)
        self._wait_steps += n_wait
        self.last_load = load
        self.t += 1
        terminated = self.t >= self.T
        info = {"price": price, "energy": energy, "load": load, "cost": cost, "unmet": unmet,
                "n_active": int(len(idx)), "n_wait": n_wait}
        return self._obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ helpers used by baselines/eval
    def _active(self, t: int) -> np.ndarray:
        return np.where((self.arr <= t) & (t < self.dep) & (self.remaining > 1e-9))[0]

    def active_laxity(self):
        """(indices, laxity in hours) of EVs that can charge right now."""
        idx = self._active(min(self.t, self.T - 1))
        return idx, laxity_hours(self.dep[idx], self.remaining[idx], self.rate[idx], self.t, self.dt)

    def urgent_power_kw(self, margin_h: float = 0.0) -> float:
        """Power needed THIS step to keep every EV with laxity <= margin on schedule."""
        idx, lax = self.active_laxity()
        if len(idx) == 0:
            return 0.0
        urgent = idx[lax <= margin_h]
        return float(np.minimum(self.rate[urgent], self.remaining[urgent] / self.dt).sum())

    def level_for_power(self, power_kw: float) -> int:
        """Smallest action whose power >= power_kw (capped at the max level)."""
        a = int(np.searchsorted(self.level_kw, power_kw - 1e-9))
        return min(a, len(self.level_kw) - 1)

    def summary(self) -> dict:
        """Episode-level metrics (call after the episode ended)."""
        demand = float(self.demand_per_ev.sum())
        served_evs = float(np.mean(self.remaining <= 0.05)) if len(self.remaining) else 1.0
        return {
            "day": self.day, "n_evs": int(len(self.remaining)), "demand_kwh": demand,
            "delivered_kwh": self._delivered, "unmet_kwh": float(self.remaining.sum()),
            "cost": self._cost, "peak_kw": self._peak,
            "avg_wait_min": (self._wait_steps * self.ds.step_minutes / len(self.remaining)) if len(self.remaining) else 0.0,
            "energy_served_pct": 100.0 * (1 - self.remaining.sum() / demand) if demand > 0 else 100.0,
            "evs_fully_served_pct": 100.0 * served_evs,
        }

    # ------------------------------------------------------------------ observation
    def _obs(self) -> np.ndarray:
        t = min(self.t, self.T - 1)
        minutes = (self.ds.day_start_hour * 60 + t * self.ds.step_minutes) % (24 * 60)
        ang = 2 * np.pi * minutes / (24 * 60)
        pmax = self.tariff.max_price

        fut = [self.prices[min(t + i, self.T - 1)] / pmax for i in range(1, self.cfg.forecast_steps + 1)]

        idx, lax = self.active_laxity() if self.t < self.T else (np.array([], dtype=int), np.array([]))
        if len(idx) == 0:
            n_act, tot_rem, min_lax, mean_lax, req, = 0.0, 0.0, 1.0, 1.0, 0.0
        else:
            n_act = len(idx) / self.cfg.max_evs_norm
            tot_rem = self.remaining[idx].sum() / (self.cap * 2.0)       # "two hours of full station power"
            min_lax = np.clip(lax.min() / 12.0, -1, 1)
            mean_lax = np.clip(lax.mean() / 12.0, -1, 1)
            time_left = np.maximum((self.dep[idx] - t) * self.dt, self.dt)
            req = (self.remaining[idx] / time_left).sum() / self.cap     # avg power needed to just finish
        v = [np.sin(ang), np.cos(ang), self.weekend, self.prices[t] / pmax, *fut,
             n_act, tot_rem, min_lax, mean_lax, req, self.last_load / self.cap]
        return np.clip(np.asarray(v, dtype=np.float32), -1.0, 1.0)
