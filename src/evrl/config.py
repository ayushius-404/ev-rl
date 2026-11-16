"""Config loading helpers."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Read the YAML config into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class EnvConfig:
    """Parameters of the charging-station environment (the `env:` block of the YAML)."""

    station_cap_kw: float = 100.0
    n_levels: int = 5
    forecast_steps: int = 4
    w_unmet: float = 5.0
    w_peak: float = 0.0
    w_wait: float = 0.0          # $ penalty per EV per step that is plugged in, still needs energy, but gets 0 power
    peak_threshold_kw: Optional[float] = None
    reward_scale: float = 0.1
    max_evs_norm: int = 40

    @classmethod
    def from_dict(cls, d: dict, **overrides) -> "EnvConfig":
        known = {f.name for f in fields(cls)}
        merged = {**d, **overrides}
        return cls(**{k: v for k, v in merged.items() if k in known})
