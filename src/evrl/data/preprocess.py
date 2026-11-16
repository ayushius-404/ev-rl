"""Raw ACN-style sessions -> per-day, step-indexed sessions + chronological train/val/test split.

Output (in data.processed_dir):
    sessions.csv  columns: day, arr_step, dep_step, energy_kwh, rate_kw
    meta.json     step_minutes, day_start_hour, splits, summary counts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Download data (evrl/data/download.py) or generate synthetic data.")
    if p.suffix == ".json":
        with open(p) as f:
            obj = json.load(f)
        items = obj["_items"] if isinstance(obj, dict) and "_items" in obj else obj
        return pd.DataFrame(items)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError("raw data must be .json or .csv")


def preprocess(cfg: dict) -> dict:
    d = cfg["data"]
    step_min = int(d["step_minutes"])
    T = 24 * 60 // step_min
    dt = step_min / 60.0
    energy_col = d["energy_column"]
    log: dict[str, int] = {}

    raw = load_raw(d["raw_path"])
    need = ["connectionTime", "disconnectTime", energy_col]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw data is missing columns {missing}. Found: {list(raw.columns)}")
    log["raw_sessions"] = len(raw)

    tz = d["timezone"]

    def to_local(col: str) -> pd.Series:
        s = pd.to_datetime(raw[col], utc=True, errors="coerce")
        return s.dt.tz_convert(tz).dt.tz_localize(None)  # wall-clock local time (DST ignored on purpose)

    df = pd.DataFrame({
        "arr_time": to_local("connectionTime"),
        "dep_time": to_local("disconnectTime"),
        "energy": pd.to_numeric(raw[energy_col], errors="coerce"),
    }).dropna()
    df = df[df["dep_time"] > df["arr_time"]]
    log["after_cleaning"] = len(df)

    # operating day: [day_start_hour, day_start_hour + 24h)
    off = pd.Timedelta(hours=int(d["day_start_hour"]))
    step = pd.Timedelta(minutes=step_min)
    df["day"] = (df["arr_time"] - off).dt.normalize()
    win_start = df["day"] + off
    win_end = win_start + pd.Timedelta(days=1)
    dep_clipped = df["dep_time"].where(df["dep_time"] < win_end, win_end)  # EVs staying past the window end
    df["arr_step"] = ((df["arr_time"] - win_start) // step).astype(int).clip(0, T)
    df["dep_step"] = ((dep_clipped - win_start) // step).astype(int).clip(0, T)

    min_steps = max(1, int(np.ceil(d["min_duration_min"] / step_min)))
    df = df[(df["dep_step"] - df["arr_step"]) >= min_steps]
    log["after_min_duration"] = len(df)

    # make each EV's demand physically feasible on its own (rate x time available)
    df["rate_kw"] = float(d["max_ev_rate_kw"])
    feasible = df["rate_kw"] * (df["dep_step"] - df["arr_step"]) * dt
    df["energy_kwh"] = np.minimum(df["energy"], feasible)
    df = df[df["energy_kwh"] >= d["min_energy_kwh"]]
    log["after_min_energy"] = len(df)

    counts = df.groupby("day").size()
    good_days = counts[counts >= d["min_sessions_per_day"]].index
    df = df[df["day"].isin(good_days)].copy()
    log["after_min_sessions_per_day"] = len(df)
    df["day"] = df["day"].dt.strftime("%Y-%m-%d")

    days = sorted(df["day"].unique())
    fr = d["split_fractions"]
    n = len(days)
    n_tr, n_va = int(round(fr[0] * n)), int(round(fr[1] * n))
    splits = {"train": days[:n_tr], "val": days[n_tr:n_tr + n_va], "test": days[n_tr + n_va:]}

    out_dir = Path(d["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df[["day", "arr_step", "dep_step", "energy_kwh", "rate_kw"]].to_csv(out_dir / "sessions.csv", index=False)
    meta = {
        "step_minutes": step_min, "day_start_hour": int(d["day_start_hour"]),
        "energy_column": energy_col, "splits": splits,
        "n_days": n, "n_sessions": len(df), "filter_log": log,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta
