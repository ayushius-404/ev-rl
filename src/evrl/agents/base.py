"""Common interface for every learning algorithm in this repo.

To add a new algorithm, subclass BaseAgent (or SB3Agent if it comes from Stable-Baselines3) and register it
in agents/__init__.py. Training, evaluation and the demo only talk to this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def train(self, total_timesteps: int, val_env=None, eval_freq: int = 10_000,
              n_eval_episodes: int = 1, save_dir: str | Path | None = None) -> None:
        """Train. If val_env is given, evaluate on it every eval_freq steps and keep the best model in save_dir."""

    @abstractmethod
    def predict(self, obs) -> int:
        """Greedy (deterministic) action for one observation."""

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseAgent": ...
