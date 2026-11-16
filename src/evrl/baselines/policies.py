"""Rule-based baselines. All share the same interface as a trained agent: act(obs, env) -> action."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    name: str = "policy"

    def reset(self, env) -> None:  # called at the start of every episode
        pass

    @abstractmethod
    def act(self, obs, env) -> int: ...


class ASAP(Policy):
    """Charge at full station power whenever possible (what an uncontrolled station does)."""
    name = "asap"

    def act(self, obs, env) -> int:
        return len(env.level_kw) - 1


class JustInTime(Policy):
    """Delay charging as long as possible; draw only the power needed to keep EVs feasible (ALAP)."""
    name = "just_in_time"

    def __init__(self, margin_h: float = 0.25):
        self.margin_h = margin_h

    def act(self, obs, env) -> int:
        return env.level_for_power(env.urgent_power_kw(self.margin_h))


class GreedyCheap(Policy):
    """Charge at full power when the price is in the cheapest part of the REMAINING day, else only urgent EVs.

    TODO(YOU): this is a simple heuristic. Try improving it (e.g. use a price percentile per EV's own
    remaining window instead of the whole remaining day) and check whether it closes the gap to the LP optimum.
    """
    name = "greedy_cheap"

    def __init__(self, quantile: float = 40.0, margin_h: float = 0.25):
        self.q, self.margin_h = quantile, margin_h

    def act(self, obs, env) -> int:
        t = min(env.t, env.T - 1)
        thr = np.percentile(env.prices[t:], self.q)
        if env.prices[t] <= thr + 1e-12:
            return len(env.level_kw) - 1
        return env.level_for_power(env.urgent_power_kw(self.margin_h))


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env) -> int:
        return int(self.rng.integers(len(env.level_kw)))


def make_baselines() -> list[Policy]:
    return [ASAP(), JustInTime(), GreedyCheap(), RandomPolicy()]
