"""Evaluate baselines, the LP optimum and every trained agent in runs/ on held-out days.

    python scripts/evaluate.py                       # test split
    python scripts/evaluate.py --cap-scale 0.8       # robustness: 20% tighter station cap, no retraining
    python scripts/evaluate.py --split val

Writes results/<tag>_per_day.csv, results/<tag>_summary.csv and results/<tag>_comparison.png
"""
import argparse
from pathlib import Path

import pandas as pd

from _common import setup
from evrl.agents import REGISTRY, load_agent
from evrl.baselines import make_baselines
from evrl.envs import make_env
from evrl.evaluation import AgentPolicy, evaluate_lp, evaluate_policy, summarize
from evrl.evaluation.plots import bar_comparison

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--model", default="best_model", help="best_model | final_model")
    ap.add_argument("--cap-scale", type=float, default=1.0, help="multiply station cap (robustness test)")
    ap.add_argument("--no-lp", action="store_true", help="skip the LP optimum (it is the slow part)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    cfg, ds, tariff = setup(a.config)
    over = {"station_cap_kw": cfg["env"]["station_cap_kw"] * a.cap_scale}
    env = make_env(ds, tariff, cfg, a.split, "sequential", **over)
    days = ds.splits[a.split]
    tag = a.tag or f"{a.split}" + (f"_cap{a.cap_scale}" if a.cap_scale != 1.0 else "")

    frames = [evaluate_policy(p, env, days) for p in make_baselines()]
    if not a.no_lp:
        print("Solving LP optimum for each day ...")
        frames.append(evaluate_lp(env, days))

    for algo in REGISTRY:  # every registered algorithm that has trained models
        for seed_dir in sorted(Path(a.runs).glob(f"{algo}*/seed*")):
            f = seed_dir / f"{a.model}.zip"
            if not f.exists():
                continue
            seed = int(seed_dir.name.replace("seed", ""))
            label = seed_dir.parent.name
            agent = load_agent(algo, f)
            frames.append(evaluate_policy(AgentPolicy(agent, label), env, days, label=label, seed=seed))

    per_day = pd.concat(frames, ignore_index=True)
    summary = summarize(per_day)
    out = Path("results"); out.mkdir(exist_ok=True)
    per_day.to_csv(out / f"{tag}_per_day.csv", index=False)
    summary.to_csv(out / f"{tag}_summary.csv")
    bar_comparison(summary, out / f"{tag}_comparison.png")
    pd.set_option("display.width", 200)
    print(f"\n=== {a.split} split, {len(days)} days, cap x{a.cap_scale} ===")
    print(summary)
