"""Plots for EDA, day replays and training curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAVE_DPI = 500  # report figures must be 400-600 dpi


def plot_day_comparison(day: str, prices: np.ndarray, cap_kw: float, plugged: np.ndarray,
                        series: dict[str, pd.DataFrame], day_start_hour: int = 0, step_minutes: int = 15):
    """4 stacked panels: price, EVs plugged in, station load per policy, cumulative cost per policy."""
    T = len(prices)
    hours = day_start_hour + np.arange(T) * step_minutes / 60.0
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    ax[0].step(hours, prices, where="post", color="tab:red")
    ax[0].set_ylabel("price ($/kWh)")
    ax[0].set_title(f"Operating day starting {day}")
    ax[1].fill_between(hours, plugged, step="post", alpha=0.5)
    ax[1].set_ylabel("EVs plugged in")
    for name, ts in series.items():
        ax[2].step(hours, ts["load"].to_numpy(), where="post", label=name)
        ax[3].plot(hours, ts["cum_cost"].to_numpy(), label=name)
    ax[2].axhline(cap_kw, color="k", ls="--", lw=0.8, label="station cap")
    ax[2].set_ylabel("station load (kW)")
    ax[2].legend(fontsize=8)
    ax[3].set_ylabel("cumulative cost ($)")
    ax[3].set_xlabel("hour of day (wall clock, may exceed 24 after midnight)")
    ax[3].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_training_curves(run_dirs: dict[str, list[Path]], window: int = 50, out: str | Path | None = None):
    """run_dirs: {label: [run_dir_seed0, run_dir_seed1, ...]}. Reads SB3 Monitor files (monitor.monitor.csv)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, dirs in run_dirs.items():
        curves = []
        for d in dirs:
            f = Path(d) / "monitor.monitor.csv"
            if not f.exists():
                continue
            r = pd.read_csv(f, skiprows=1)["r"].rolling(window, min_periods=1).mean().to_numpy()
            curves.append(r)
        if not curves:
            continue
        L = min(len(c) for c in curves)
        arr = np.stack([c[:L] for c in curves])
        m, s = arr.mean(0), arr.std(0)
        ax.plot(m, label=f"{label} ({len(curves)} seeds)")
        ax.fill_between(range(L), m - s, m + s, alpha=0.2)
    ax.set_xlabel("training episode")
    ax.set_ylabel(f"episode reward (rolling mean, {window})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=SAVE_DPI)
    return fig


def bar_comparison(summary: pd.DataFrame, out: str | Path | None = None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    summary["cost"].plot.bar(ax=ax[0], yerr=summary.get("cost_std_over_seeds"), color="tab:blue")
    ax[0].set_ylabel("mean daily cost ($)")
    ax[0].set_title("Cost (lower is better)")
    summary["energy_served_pct"].plot.bar(ax=ax[1], color="tab:green")
    ax[1].set_ylabel("energy served (%)")
    ax[1].set_ylim(max(0, summary["energy_served_pct"].min() - 5), 100.5)
    ax[1].set_title("Service (higher is better)")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=SAVE_DPI)
    return fig
