"""Exploratory analysis. Run this BEFORE choosing the tariff and the station cap.   python scripts/eda.py

Produces results/eda/eda.png and prints the ASAP peak-load percentiles (use them to set env.station_cap_kw).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import setup
from evrl.data.stats import asap_load_profile, plugged_in_count

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg, ds, tariff = setup(a.config)
    T, dt, sm = ds.steps_per_day, ds.step_minutes / 60, ds.step_minutes

    rows = []
    for d, s in ds.days.items():
        for i in range(s.n):
            rows.append((d, s.arr[i], s.dep[i], s.energy[i]))
    df = pd.DataFrame(rows, columns=["day", "arr", "dep", "energy"])
    df["arr_h"] = (ds.day_start_hour + df["arr"] * sm / 60) % 24
    df["dep_h"] = (ds.day_start_hour + df["dep"] * sm / 60) % 24
    df["dur_h"] = (df["dep"] - df["arr"]) * dt

    peaks = np.array([asap_load_profile(s, T, dt).max() for s in ds.days.values()])
    mean_plugged = np.mean([plugged_in_count(s, T) for s in ds.days.values()], axis=0)
    mean_asap = np.mean([asap_load_profile(s, T, dt) for s in ds.days.values()], axis=0)
    hours = ds.day_start_hour + np.arange(T) * sm / 60

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ax[0, 0].hist(df["arr_h"], bins=24, range=(0, 24)); ax[0, 0].set_title("Arrival hour")
    ax[0, 1].hist(df["dep_h"], bins=24, range=(0, 24)); ax[0, 1].set_title("Departure hour")
    ax[0, 2].hist(df["dur_h"], bins=40); ax[0, 2].set_title("Parking duration (h)")
    ax[1, 0].hist(df["energy"], bins=40); ax[1, 0].set_title("Energy demand per session (kWh)")
    ax[1, 1].plot(hours, mean_plugged, label="EVs plugged in")
    ax2 = ax[1, 1].twinx(); ax2.step(hours, tariff.price_vector(list(ds.days)[0]), "r", alpha=0.6, where="post")
    ax[1, 1].set_title("Mean occupancy (blue) vs price on first day (red)")
    ax[1, 2].plot(hours, mean_asap); ax[1, 2].set_title("Mean ASAP station load (kW)")
    for x in ax.flat:
        x.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("results/eda"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "eda.png", dpi=500)

    print(f"days: {len(ds.days)}   sessions: {len(df)}   sessions/day (mean): {len(df)/len(ds.days):.1f}")
    print(f"energy/session  mean {df.energy.mean():.1f} kWh, median {df.energy.median():.1f} kWh")
    print(f"duration        mean {df.dur_h.mean():.1f} h, median {df.dur_h.median():.1f} h")
    print("ASAP daily peak load percentiles (kW):",
          {p: round(float(np.percentile(peaks, p)), 1) for p in (10, 25, 50, 75, 90, 100)})
    print(f"-> suggestion: env.station_cap_kw about {0.65 * np.median(peaks):.0f} kW (65% of the median peak)")
    print(f"Saved {out / 'eda.png'}")
