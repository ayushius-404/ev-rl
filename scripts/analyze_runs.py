"""Compare training runs: steps to converge, wall-clock time and network size.   python scripts/analyze_runs.py

For every folder in runs/ (dqn, ppo, a2c, dqn_peak, ...) it reads the training logs of all seeds and writes
results/algorithm_comparison.csv.

Definitions (state them in the report):
  * validation reward   = mean reward over the 53 fixed validation days, measured every `eval_freq` steps
                          with the greedy policy (saved in evaluations.npz). Much less noisy than training reward,
                          because training episodes use random days and, for DQN, random exploration.
  * final value         = mean validation reward of the last 5 checkpoints
  * steps to converge   = first checkpoint where the validation reward and the next one are both within 5% of
                          the final value (resolution = eval_freq, so 10000 means "already there at the first check")
  * wall-clock time     = total training time from the Monitor log; it includes the periodic validation runs
  * parameters          = trainable weights of the network that acts (needs a trained best_model.zip)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evrl.agents import REGISTRY, load_agent


def analyse_run(seed_dir: Path, tol: float = 0.05, last_k: int = 5) -> dict:
    mon = pd.read_csv(seed_dir / "monitor.monitor.csv", skiprows=1)
    out = {"wall_clock_min": float(mon["t"].iloc[-1]) / 60.0, "total_steps": int(mon["l"].sum())}
    ev = np.load(seed_dir / "evaluations.npz")
    steps, val = ev["timesteps"], ev["results"].mean(axis=1)      # mean validation reward per checkpoint
    final = float(val[-last_k:].mean())
    ok = val >= final - tol * abs(final)
    both = np.where(ok[:-1] & ok[1:])[0]
    if len(both):
        i = int(both[0])
    else:
        i = int(np.where(ok)[0][0]) if ok.any() else len(val) - 1
    out.update(converge_steps=int(steps[i]), final_val_reward=final)
    return out


def mean_std(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.1f} ± {x.std(ddof=1):.1f}" if len(x) > 1 else f"{x.mean():.1f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()

    rows = []
    for run_dir in sorted(p for p in Path(a.runs).iterdir() if p.is_dir() and not p.name.startswith("_")):
        seeds = sorted(run_dir.glob("seed*"))
        per_seed = [analyse_run(s) for s in seeds
                    if (s / "monitor.monitor.csv").exists() and (s / "evaluations.npz").exists()]
        if not per_seed:
            continue
        d = pd.DataFrame(per_seed)
        algo = run_dir.name.split("_")[0]
        n_params = None
        if algo in REGISTRY and (seeds[0] / "best_model.zip").exists():
            try:
                n_params = load_agent(algo, seeds[0] / "best_model.zip").n_parameters()
            except Exception as e:  # keep going, the other columns are still useful
                print(f"  (could not count parameters for {run_dir.name}: {e})")
        rows.append({
            "run": run_dir.name, "seeds": len(d),
            "steps_to_converge": mean_std(d["converge_steps"]),
            "final_val_reward": f"{d['final_val_reward'].mean():.2f} ± {d['final_val_reward'].std(ddof=1):.2f}" if len(d) > 1 else f"{d['final_val_reward'].mean():.2f}",
            "wall_clock_min": mean_std(d["wall_clock_min"]),
            "sec_per_1k_steps": f"{(d['wall_clock_min'] * 60 / (d['total_steps'] / 1000)).mean():.2f}",
            "parameters": n_params,
            "_converge_mean": d["converge_steps"].mean(),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No training logs found in runs/.")
    Path("results").mkdir(exist_ok=True)
    out.drop(columns="_converge_mean").to_csv("results/algorithm_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(out.drop(columns="_converge_mean").to_string(index=False))
    print("\nSaved results/algorithm_comparison.csv")