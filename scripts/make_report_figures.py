"""Create every figure of the report (500 dpi PNG) from the real project outputs.

    python scripts/make_report_figures.py          # the 3 essential figures: workflow, convergence, test comparison
    python scripts/make_report_figures.py --all    # plus data overview, training curves, reward variants, example day

Needs (all created by the earlier steps):
    data/processed/                      -> fig_data
    runs/{dqn,ppo,a2c}/seed*/evaluations.npz -> fig_convergence (validation score vs steps)
    runs/{dqn,ppo,a2c}/seed*/monitor.monitor.csv -> fig_training_curves (training score vs episode)
    results/test_summary.csv             -> fig_test_score, fig_variants
    results/val_summary.csv  (optional)  -> reference lines in fig_convergence
    runs/dqn/seed<k>/best_model.zip      -> fig_replay (needs PyTorch; skipped if it cannot be loaded)
Figures are written to report/figs/. A figure whose input is missing is skipped with a message.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import setup

warnings.filterwarnings("ignore", category=RuntimeWarning)
DPI = 500
OUT = Path("report/figs")
COL = {"dqn": "#1f77b4", "ppo": "#ff7f0e", "a2c": "#2ca02c", "asap": "#7f7f7f",
       "greedy_cheap": "#9467bd", "lp_optimal": "#d62728"}
NAMES = {"lp_optimal": "LP optimum (offline)", "dqn": "DQN", "ppo": "PPO", "a2c": "A2C",
         "dqn_wait": "DQN + wait penalty", "dqn_peak": "DQN + peak penalty",
         "greedy_cheap": "Cheapest-Greedy", "asap": "ASAP", "just_in_time": "Just-in-Time", "random": "Random"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT / name}")


# ---------------------------------------------------------------- Fig: agent-environment workflow
def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    w, gap = 0.18, 0.025
    boxes = [
        ("1 State $s_t$", "#e8f0fe", ["time of day", "price, next 6 h", "EVs connected", "energy still needed", "laxity (urgency)", "station load"]),
        ("2 RL agent", "#efe6fb", ["DQN / PPO / A2C", "", "neural network", "2 x 128 hidden units", "", "picks power level"]),
        ("3 Action $a_t$", "#e5f4ea", ["station power", "level, one of", "0, 25, 50, 75,", "100 % of 50 kW", "", ""]),
        ("4 Dispatch", "#fdf0dd", ["Least Laxity First", "", "most urgent EV", "gets power first", "", ""]),
        ("5 Station", "#eceff3", ["replays real ACN", "sessions of a day", "applies 50 kW cap", "updates energy left,", "cost, unmet energy", "advances 15 min"]),
    ]
    y0, h = 0.40, 0.50
    centers = []
    for i, (title, colr, lines) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(plt.Rectangle((x, y0), w, h, facecolor=colr, edgecolor="#555", linewidth=0.8, zorder=1))
        ax.text(x + w / 2, y0 + h - 0.07, title, ha="center", va="center", fontsize=8.2, fontweight="bold")
        for j, t in enumerate(lines):
            ax.text(x + w / 2, y0 + h - 0.17 - j * 0.058, t, ha="center", va="center", fontsize=6.5)
        centers.append(x + w / 2)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap, y0 + h / 2), xytext=(x + w, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2, mutation_scale=9))
    # feedback loop
    ylo = 0.17
    ax.plot([centers[-1], centers[-1]], [y0, ylo], color="#1a56db", lw=1.3)
    ax.plot([centers[-1], centers[0]], [ylo, ylo], color="#1a56db", lw=1.3)
    ax.annotate("", xy=(centers[0], y0), xytext=(centers[0], ylo),
                arrowprops=dict(arrowstyle="-|>", color="#1a56db", lw=1.3, mutation_scale=9))
    ax.text(0.5, ylo - 0.065, r"reward $r_t=-0.1\,(\mathrm{cost}_t + 5\times\mathrm{unmet\ energy}_t)$   and next state $s_{t+1}$",
            ha="center", va="center", fontsize=8, color="#1a56db")
    ax.text(0.5, 0.02, "One episode = one day = 96 decisions of 15 minutes", ha="center", fontsize=7.5, color="#444")
    save(fig, "fig_workflow.png")


# ---------------------------------------------------------------- Fig: data overview
def fig_data(ds, tariff):
    sm, T = ds.step_minutes, ds.steps_per_day
    arr_h, dur_h, energy = [], [], []
    occ = np.zeros(T)
    for s in ds.days.values():
        arr_h += list((ds.day_start_hour + s.arr * sm / 60) % 24)
        dur_h += list((s.dep - s.arr) * sm / 60)
        energy += list(s.energy)
        for a, d in zip(s.arr, s.dep):
            occ[a:d] += 1
    occ /= len(ds.days)
    day_wd = next(d for d in ds.days if pd.Timestamp(d).weekday() < 5)
    hours = ds.day_start_hour + np.arange(T) * sm / 60
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9))
    ax[0].hist(arr_h, bins=24, range=(0, 24), color="#4c78a8"); ax[0].set_title("(a) Arrival hour"); ax[0].set_xlabel("hour of day")
    ax[1].hist(dur_h, bins=30, range=(0, 16), color="#4c78a8"); ax[1].set_title("(b) Parking time"); ax[1].set_xlabel("hours")
    ax[2].hist(energy, bins=30, range=(0, 40), color="#4c78a8"); ax[2].set_title("(c) Energy per session"); ax[2].set_xlabel("kWh")
    ax[3].plot(hours, occ, color="#4c78a8"); ax[3].set_title("(d) Occupancy and price"); ax[3].set_xlabel("hour of day")
    ax[3].set_ylabel("EVs plugged in (mean)")
    a2 = ax[3].twinx(); a2.step(hours, tariff.price_vector(day_wd), where="post", color="#d62728", lw=1)
    a2.set_ylabel("price ($/kWh)", color="#d62728"); a2.grid(False)
    ticks = [4, 10, 16, 22, 28]
    ax[3].set_xticks(ticks); ax[3].set_xticklabels([str(t % 24) for t in ticks])
    ax[0].set_ylabel("sessions")
    fig.tight_layout()
    save(fig, "fig_data.png")


# ---------------------------------------------------------------- Fig: convergence
def fig_convergence(cfg):
    """One panel per algorithm: thin lines = single seeds, thick line = mean of the seeds, y = validation score."""
    from matplotlib.lines import Line2D

    scale = cfg["env"]["reward_scale"]
    data = {}
    for algo in ("dqn", "ppo", "a2c"):
        curves, steps = [], None
        for f in sorted(Path("runs", algo).glob("seed*/evaluations.npz")):
            d = np.load(f)
            curves.append(-d["results"].mean(axis=1) / scale)      # validation score J5 (lower is better)
            steps = d["timesteps"]
        if curves:
            L = min(len(c) for c in curves)
            data[algo] = (steps[:L] / 1000, np.stack([c[:L] for c in curves]))
    if not data:
        raise FileNotFoundError("no runs/<algo>/seed*/evaluations.npz found")

    ref = {}
    vs = Path("results/val_summary.csv")
    if vs.exists():
        v = pd.read_csv(vs).set_index("policy")["objective_w5"]
        ref = {p: float(v[p]) for p in ("asap", "greedy_cheap", "lp_optimal") if p in v.index}
    lo = (ref.get("lp_optimal") or min(c.min() for _, c in data.values())) - 3
    hi = 1.6 * ref["asap"] if "asap" in ref else 1.5 * max(c.mean(0).max() for _, c in data.values())

    fig, axes = plt.subplots(1, len(data), figsize=(7.2, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (algo, (x, arr)) in zip(axes, data.items()):
        for c in arr:
            ax.plot(x, c, color=COL[algo], alpha=0.3, lw=0.8)
        ax.plot(x, arr.mean(0), color=COL[algo], lw=2)
        for p, ls in (("asap", ":"), ("greedy_cheap", "--"), ("lp_optimal", "-.")):
            if p in ref:
                ax.axhline(ref[p], color=COL[p], ls=ls, lw=1.1)
        n_out = int((arr > hi).sum())
        if n_out:
            ax.text(0.97, 0.95, f"{n_out} of {arr.size} values\nabove {hi:.0f} not shown", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, color="#a00")
        ax.set_ylim(lo, hi)
        ax.set_title(f"{algo.upper()} ({arr.shape[0]} seeds)")
        ax.set_xlabel("training steps (thousands)")
    axes[0].set_ylabel("validation score\n(cost + 5 x unmet, lower is better)")
    handles = [Line2D([], [], color="#444", lw=2), Line2D([], [], color="#444", lw=0.8, alpha=0.4)]
    labels = ["mean of seeds", "single seed"]
    for p, ls, lab in (("asap", ":", "ASAP"), ("greedy_cheap", "--", "Cheapest-Greedy"), ("lp_optimal", "-.", "LP optimum")):
        if p in ref:
            handles.append(Line2D([], [], color=COL[p], ls=ls, lw=1.2)); labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout()
    save(fig, "fig_convergence.png")


# ---------------------------------------------------------------- Fig: training-episode curves
def fig_training_curves(cfg, window=50):
    """Score of the training episodes (random training days, with exploration): noisy, but shows the learning phase."""
    scale = cfg["env"]["reward_scale"]
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    found = 0
    for algo in ("dqn", "ppo", "a2c"):
        curves = []
        for f in sorted(Path("runs", algo).glob("seed*/monitor.monitor.csv")):
            r = pd.read_csv(f, skiprows=1)["r"]
            curves.append((-r / scale).rolling(window, min_periods=window).mean().to_numpy())
        if not curves:
            continue
        L = min(len(c) for c in curves)
        arr = np.stack([c[:L] for c in curves]); m, s = np.nanmean(arr, 0), np.nanstd(arr, 0)
        x = np.arange(L)
        ax.plot(x, m, color=COL[algo], label=f"{algo.upper()} ({len(curves)} seeds)", lw=1.2)
        ax.fill_between(x, m - s, m + s, color=COL[algo], alpha=0.18)
        found += 1
    if not found:
        raise FileNotFoundError("no runs/<algo>/seed*/monitor.monitor.csv found")
    ax.set_xlabel("training episode (one episode = one day)")
    ax.set_ylabel(f"training score, rolling mean of {window}\nlower is better")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_training_curves.png")


# ---------------------------------------------------------------- Fig: test comparison
def fig_test_score():
    t = pd.read_csv("results/test_summary.csv").set_index("policy")
    show = [p for p in t.sort_values("objective_w5").index if p not in ("random", "just_in_time")]
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    y = np.arange(len(show))[::-1]
    cost = t.loc[show, "cost"].to_numpy(); pen = 5 * t.loc[show, "unmet_kwh"].to_numpy()
    ax.barh(y, cost, color="#4c78a8", label="cost")
    ax.barh(y, pen, left=cost, color="#f58518", label="5 x unmet energy")
    for yi, c, p in zip(y, cost, pen):
        ax.text(c + p + 0.4, yi, f"{c + p:.1f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([NAMES.get(p, p) for p in show])
    ax.set_xlabel("score J5 = cost + 5 x unmet energy (lower is better)")
    ax.set_xlim(0, max(cost + pen) * 1.15)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    save(fig, "fig_test_score.png")


# ---------------------------------------------------------------- Fig: reward variants
def fig_variants():
    t = pd.read_csv("results/test_summary.csv").set_index("policy")
    names = [p for p in ("dqn", "dqn_wait", "dqn_peak") if p in t.index]
    labels = {"dqn": "base", "dqn_wait": "+ wait", "dqn_peak": "+ peak"}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0))
    x = np.arange(len(names))
    spec = [("cost", "mean daily cost ($)", "(a) Cost"), ("avg_wait_min", "average waiting (min)", "(b) Waiting time"),
            ("peak_kw", "mean daily peak load (kW)", "(c) Peak load")]
    for a, (col, ylab, title) in zip(ax, spec):
        err = t.loc[names, "cost_std_over_seeds"] if col == "cost" else None
        a.errorbar(x, t.loc[names, col], yerr=err, fmt="o", color="#1f77b4", capsize=3, ms=5)
        for xi, v in zip(x, t.loc[names, col]):
            a.text(xi + 0.08, v, f"{v:.1f}", fontsize=7, va="bottom")
        a.set_xticks(x); a.set_xticklabels([labels[n] for n in names]); a.set_xlim(-0.5, len(names) - 0.5)
        a.set_ylabel(ylab); a.set_title(title)
        pad = 0.15 * (t.loc[names, col].max() - t.loc[names, col].min() + 1e-9)
        if col == "peak_kw":
            a.set_ylim(0, 50)          # full range up to the 50 kW cap, so tiny differences are not exaggerated
            a.axhline(50, color="k", ls="--", lw=0.8)
        else:
            a.set_ylim(t.loc[names, col].min() - 3 * pad - 0.5, t.loc[names, col].max() + 3 * pad + 0.5)
    fig.tight_layout()
    save(fig, "fig_variants.png")


# ---------------------------------------------------------------- Fig: one-day replay
def fig_replay(cfg, ds, tariff, day, seed):
    from evrl.baselines import ASAP, GreedyCheap
    from evrl.data.stats import plugged_in_count
    from evrl.envs import make_env
    from evrl.evaluation import AgentPolicy, lp_day_summary, run_episode

    env = make_env(ds, tariff, cfg, "test", "sequential")
    if day not in ds.splits["test"]:
        day = ds.splits["test"][0]
        print(f"  (example day not in test split, using {day})")
    series = {"asap": run_episode(ASAP(), env, day, record=True)[1],
              "greedy_cheap": run_episode(GreedyCheap(), env, day, record=True)[1]}
    try:
        from evrl.agents import load_agent
        agent = load_agent("dqn", Path("runs/dqn") / f"seed{seed}" / "best_model.zip")
        series["dqn"] = run_episode(AgentPolicy(agent, "dqn"), env, day, record=True)[1]
    except Exception as e:
        print(f"  (DQN not loaded: {e}) - figure will not show the DQN line")
    series["lp_optimal"] = lp_day_summary(env, day, record=True)[1]
    T = env.T
    hours = ds.day_start_hour + np.arange(T) * ds.step_minutes / 60
    fig, ax = plt.subplots(3, 1, figsize=(3.45, 5.0), sharex=True)
    ax[0].step(hours, env.prices, where="post", color="#d62728"); ax[0].set_ylabel("price ($/kWh)")
    ax0b = ax[0].twinx(); ax0b.fill_between(hours, plugged_in_count(ds.days[day], T), step="post", alpha=0.25, color="#4c78a8")
    ax0b.set_ylabel("EVs plugged in"); ax0b.grid(False)
    for name, ts in series.items():
        ax[1].step(hours, ts["load"].to_numpy(), where="post", color=COL[name], label=NAMES[name], lw=1.2)
        ax[2].plot(hours, ts["cum_cost"].to_numpy(), color=COL[name], label=f"{NAMES[name]}: ${ts['cum_cost'].iloc[-1]:.1f}", lw=1.4)
    ax[1].axhline(env.cap, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("station load (kW)"); ax[1].legend(loc="upper right", fontsize=6.5)
    ax[2].set_ylabel("cumulative cost ($)"); ax[2].legend(loc="upper left", fontsize=6.5)
    ticks = [4, 8, 12, 16, 20, 24, 28]
    ax[2].set_xticks(ticks); ax[2].set_xticklabels([str(t % 24) for t in ticks]); ax[2].set_xlabel("hour of day")
    ax[0].set_title(f"Test day {day}: {ds.days[day].n} EVs, {ds.days[day].energy.sum():.0f} kWh requested")
    fig.tight_layout()
    save(fig, "fig_replay.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--day", default="2019-11-01")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="also make the optional figures (data, training curves, variants, replay)")
    a = ap.parse_args()
    cfg, ds, tariff = setup(a.config)
    jobs = [("workflow diagram", fig_workflow),
            ("convergence (validation)", lambda: fig_convergence(cfg)),
            ("test comparison", fig_test_score)]
    if a.all:
        jobs += [("data overview", lambda: fig_data(ds, tariff)),
                 ("training curves", lambda: fig_training_curves(cfg)),
                 ("reward variants", fig_variants),
                 ("example day", lambda: fig_replay(cfg, ds, tariff, a.day, a.seed))]
    for label, fn in jobs:
        print(f"[{label}]")
        try:
            fn()
        except Exception as e:  # keep going; report what failed
            print(f"  SKIPPED: {type(e).__name__}: {e}")
    print(f"\nDone. Figures are in {OUT}/")
