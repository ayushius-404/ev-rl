"""Shared setup for the scripts: load config, dataset and tariff in one call."""
from __future__ import annotations

from evrl.config import load_config
from evrl.data import Tariff, load_dataset


def setup(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)
    ds = load_dataset(cfg["data"]["processed_dir"])
    tariff = Tariff(cfg["tariff"], ds.step_minutes, ds.day_start_hour)
    return cfg, ds, tariff
