"""Loading the processed sessions into per-day arrays that the environment replays."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DaySessions:
    """All charging sessions of one operating day, as parallel arrays (one entry per EV)."""

    arr: np.ndarray      # arrival step (int), EV can charge from this step
    dep: np.ndarray      # departure step (int), last chargeable step is dep - 1
    energy: np.ndarray   # energy the EV needs (kWh)
    rate: np.ndarray     # max charging power of this EV (kW)

    @property
    def n(self) -> int:
        return len(self.arr)


@dataclass
class Dataset:
    days: dict[str, DaySessions]
    splits: dict[str, list[str]]   # {"train": [...], "val": [...], "test": [...]} (chronological)
    step_minutes: int
    day_start_hour: int

    @property
    def steps_per_day(self) -> int:
        return 24 * 60 // self.step_minutes


def load_dataset(processed_dir: str | Path = "data/processed") -> Dataset:
    """Load data/processed/sessions.csv + meta.json (created by scripts/prepare_data.py)."""
    processed_dir = Path(processed_dir)
    sess_path, meta_path = processed_dir / "sessions.csv", processed_dir / "meta.json"
    if not sess_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No processed data in {processed_dir}. Run `python scripts/prepare_data.py` first."
        )
    df = pd.read_csv(sess_path)
    with open(meta_path) as f:
        meta = json.load(f)

    days: dict[str, DaySessions] = {}
    for day, g in df.groupby("day"):
        days[str(day)] = DaySessions(
            arr=g["arr_step"].to_numpy(dtype=int),
            dep=g["dep_step"].to_numpy(dtype=int),
            energy=g["energy_kwh"].to_numpy(dtype=float),
            rate=g["rate_kw"].to_numpy(dtype=float),
        )
    splits = {k: [d for d in v if d in days] for k, v in meta["splits"].items()}
    return Dataset(days, splits, int(meta["step_minutes"]), int(meta["day_start_hour"]))
