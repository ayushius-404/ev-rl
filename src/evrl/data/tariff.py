"""Time-of-use tariff -> price vector for one operating day."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


class Tariff:
    """Builds the per-step price ($/kWh) for a given operating day from the YAML periods."""

    def __init__(self, tariff_cfg: dict, step_minutes: int, day_start_hour: int):
        self.cfg = tariff_cfg
        self.step_minutes = step_minutes
        self.day_start_hour = day_start_hour
        self.steps = 24 * 60 // step_minutes
        self._cache: dict[str, np.ndarray] = {}
        prices = [p["price"] for key in ("weekday", "weekend") for p in tariff_cfg[key]]
        self.max_price = float(max(prices))
        self.min_price = float(min(prices))

    @staticmethod
    def _lookup(periods: list[dict], hour: float) -> float:
        for p in periods:
            if p["start"] <= hour < p["end"]:
                return float(p["price"])
        raise ValueError(f"Tariff has no period covering hour {hour}")

    def price_vector(self, day: str) -> np.ndarray:
        """Prices for every step of the operating day that starts on `day` (YYYY-MM-DD)."""
        if day in self._cache:
            return self._cache[day]
        t0 = datetime.strptime(day, "%Y-%m-%d") + timedelta(hours=self.day_start_hour)
        out = np.empty(self.steps)
        for s in range(self.steps):
            t = t0 + timedelta(minutes=s * self.step_minutes)
            periods = self.cfg["weekday"] if t.weekday() < 5 else self.cfg["weekend"]
            out[s] = self._lookup(periods, t.hour + t.minute / 60.0)
        self._cache[day] = out
        return out
