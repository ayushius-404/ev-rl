"""Training curves (mean +- std over seeds) from runs/.   python scripts/plot_training.py"""
import argparse
from pathlib import Path

from evrl.evaluation.plots import plot_training_curves

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--window", type=int, default=50)
    a = ap.parse_args()
    groups = {p.name: sorted(p.glob("seed*")) for p in sorted(Path(a.runs).iterdir()) if p.is_dir()}
    Path("results").mkdir(exist_ok=True)
    plot_training_curves(groups, a.window, "results/training_curves.png")
    print("Saved results/training_curves.png")
