"""Train an agent.   python scripts/train.py --algo dqn --seeds 0 1 2

Outputs go to runs/<algo><tag>/seed<k>/ : best_model.zip (best on validation days), final_model.zip,
monitor.monitor.csv (training episode rewards + wall-clock time), evaluations.npz.

Reward variants (for the "different reward formulations" experiment) without touching the config:
    python scripts/train.py --algo dqn --seeds 0 1 2 3 4 --tag _peak --set w_peak=0.05 peak_threshold_kw=30
    python scripts/train.py --algo dqn --seeds 0 1 2 3 4 --tag _wait --set w_wait=0.02
"""
import argparse
from pathlib import Path

import yaml

from stable_baselines3.common.monitor import Monitor

from _common import setup
from evrl.agents import make_agent
from evrl.envs import make_env

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--algo", default=None, help="defaults to train.algo in the config")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="", help="optional suffix for the run folder (used by sweeps)")
    ap.add_argument("--w-unmet", type=float, default=None, help="override env.w_unmet for this run")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any env setting for this run, e.g. w_peak=0.05 peak_threshold_kw=30")
    a = ap.parse_args()

    cfg, ds, tariff = setup(a.config)
    algo = a.algo or cfg["train"]["algo"]
    seeds = a.seeds if a.seeds is not None else cfg["train"]["seeds"]
    steps = a.timesteps or cfg["train"]["total_timesteps"]
    over = {"w_unmet": a.w_unmet} if a.w_unmet is not None else {}
    for item in a.set:
        k, v = item.split("=", 1)
        over[k] = yaml.safe_load(v)          # numbers stay numbers, "null" becomes None
    if over:
        print("env overrides for this run:", over)

    for seed in seeds:
        run_dir = Path(a.out) / (algo + a.tag) / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        train_env = Monitor(make_env(ds, tariff, cfg, "train", "random", **over), str(run_dir / "monitor"))
        val_env = make_env(ds, tariff, cfg, "val", "sequential", **over)  # SB3 wraps it in its own Monitor
        train_env.reset(seed=seed)
        agent = make_agent(algo, train_env, cfg, seed=seed)
        print(f"[{algo}] seed {seed}: {steps} steps, {len(ds.splits['train'])} train days, "
              f"{len(ds.splits['val'])} val days -> {run_dir}")
        agent.train(steps, val_env=val_env, eval_freq=cfg["train"]["eval_freq"],
                    n_eval_episodes=len(ds.splits["val"]), save_dir=run_dir)
