"""Synthetic sessions with the same raw schema as ACN-Data.

Use this to develop and test the whole pipeline BEFORE you have the real data.
    python -m evrl.data.synthetic          # writes data/raw/synthetic_sessions.csv
Then set `data.raw_path: data/raw/synthetic_sessions.csv` in configs/default.yaml.
The final results in your report must come from the REAL data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_raw(n_days: int = 240, seed: int = 0, tz: str = "America/Los_Angeles") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2019-01-01") + pd.Timedelta(days=d)
        weekend = day.weekday() >= 5
        n = rng.poisson(14 if weekend else 60)
        for _ in range(n):
            # bimodal morning-heavy arrivals for a workplace site
            if rng.random() < 0.8:
                arr_h = rng.normal(8.8, 1.3)
            else:
                arr_h = rng.normal(13.0, 2.0)
            arr_h = float(np.clip(arr_h, 0.0, 23.5))
            dur_h = float(np.clip(rng.normal(6.5 if not weekend else 4.0, 2.0), 0.4, 14.0))
            arr = day + pd.Timedelta(hours=arr_h)
            dep = arr + pd.Timedelta(hours=dur_h)
            energy = float(np.clip(rng.lognormal(np.log(9.0), 0.6), 1.0, 40.0))
            energy = min(energy, 6.6 * dur_h)
            rows.append((arr, dep, energy))
    df = pd.DataFrame(rows, columns=["connectionTime", "disconnectTime", "kWhDelivered"])
    for c in ("connectionTime", "disconnectTime"):
        df[c] = df[c].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    return df.dropna().sort_values("connectionTime").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/synthetic_sessions.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_raw(a.days, a.seed).to_csv(a.out, index=False)
    print(f"Wrote {a.out}")
