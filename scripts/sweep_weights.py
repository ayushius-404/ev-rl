"""Experiment: trade-off between cost and unmet energy by sweeping w_unmet.   (the Pareto-style plot)

    python scripts/sweep_weights.py --values 0.5 1 2 5 10 --timesteps 100000

Trains one DQN per value (a short run each), evaluates on the chosen split and plots cost vs. energy served.
TODO(YOU): extend it - e.g. sweep w_peak, or run each value with several seeds and plot error bars.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from _common import setup
from evrl.agents import make_agent
from evrl.envs import make_env
from evrl.evaluation import AgentPolicy, evaluate_policy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--values", type=float, nargs="+", default=[0.5, 1, 2, 5, 10])
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg, ds, tariff = setup(a.config)
    rows = []
    for w in a.values:
        tr = Monitor(make_env(ds, tariff, cfg, "train", "random", w_unmet=w))
        agent = make_agent(cfg["train"]["algo"], tr, cfg, seed=a.seed)
        agent.train(a.timesteps)
        env = make_env(ds, tariff, cfg, a.split, "sequential", w_unmet=w)
        df = evaluate_policy(AgentPolicy(agent, f"w={w}"), env, ds.splits[a.split])
        rows.append({"w_unmet": w, "cost": df["cost"].mean(), "energy_served_pct": df["energy_served_pct"].mean()})
        print(rows[-1])
    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/sweep_w_unmet.csv", index=False)
    plt.figure(figsize=(6, 4.5))
    plt.plot(res["energy_served_pct"], res["cost"], "o-")
    for _, r in res.iterrows():
        plt.annotate(f"{r.w_unmet:g}", (r.energy_served_pct, r.cost))
    plt.xlabel("energy served (%)"); plt.ylabel("mean daily cost ($)"); plt.grid(alpha=0.3)
    plt.title("Cost vs service trade-off (labels = w_unmet)")
    plt.tight_layout(); plt.savefig("results/sweep_w_unmet.png", dpi=500)
    print("Saved results/sweep_w_unmet.png")
