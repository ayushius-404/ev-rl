"""Aggregate per-day results into the comparison table used in the report."""
from __future__ import annotations

import pandas as pd

METRICS = ["cost", "energy_served_pct", "evs_fully_served_pct", "unmet_kwh", "peak_kw", "avg_wait_min"]
W_REF = 5.0  # fixed weight used to compare policies that were trained with different rewards


def summarize(per_day: pd.DataFrame, reference: str = "asap", optimal: str = "lp_optimal") -> pd.DataFrame:
    """Mean per policy (averaged over days, then over seeds) plus savings vs. a reference policy.

    savings_vs_asap_pct   : (cost_asap - cost_policy) / cost_asap
    pct_of_optimal_saving : how much of the LP's saving the policy recovers (100 = as good as perfect foresight)
    """
    per_seed = per_day.groupby(["policy", "seed"])[METRICS].mean()
    mean = per_seed.groupby("policy").mean()
    std = per_seed.groupby("policy")["cost"].std().rename("cost_std_over_seeds")
    out = mean.join(std)
    out["objective_w5"] = out["cost"] + W_REF * out["unmet_kwh"]   # cost + 5 x unmet energy, same for every policy
    if reference in out.index:
        ref = out.loc[reference, "cost"]
        out["savings_vs_asap_pct"] = 100 * (ref - out["cost"]) / ref
        if optimal in out.index:
            best = ref - out.loc[optimal, "cost"]
            out["pct_of_optimal_saving"] = 100 * (ref - out["cost"]) / best if best > 1e-9 else float("nan")
    return out.sort_values("objective_w5").round(3)
