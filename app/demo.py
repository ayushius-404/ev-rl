"""Interactive replay of a test day:  streamlit run app/demo.py

Pick a day and some policies and compare station load, price and cumulative cost side by side.
TODO(YOU): make it yours - add a slider for the station cap, a w_unmet selector, or a live "step through the day" mode.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import setup  # noqa: E402
from evrl.agents import REGISTRY, load_agent  # noqa: E402
from evrl.baselines import make_baselines  # noqa: E402
from evrl.data.stats import plugged_in_count  # noqa: E402
from evrl.envs import make_env  # noqa: E402
from evrl.evaluation import AgentPolicy, lp_day_summary, run_episode  # noqa: E402
from evrl.evaluation.plots import plot_day_comparison  # noqa: E402

st.set_page_config(page_title="EV charging RL demo", layout="wide")
st.title("EV charging scheduling: RL vs baselines")

cfg, ds, tariff = setup()
split = st.sidebar.selectbox("Split", ["test", "val", "train"])
day = st.sidebar.selectbox("Day", ds.splits[split])
env = make_env(ds, tariff, cfg, split, "sequential")

policies = {p.name: p for p in make_baselines()}
for algo in REGISTRY:
    for seed_dir in sorted(Path("runs").glob(f"{algo}*/seed*")):
        f = seed_dir / "best_model.zip"
        if f.exists():
            label = f"{seed_dir.parent.name}_{seed_dir.name}"
            policies[label] = AgentPolicy(load_agent(algo, f), label)
chosen = st.sidebar.multiselect("Policies", list(policies), default=[n for n in ("asap", "just_in_time") if n in policies]
                                + [n for n in policies if n.startswith("dqn")][:1])
use_lp = st.sidebar.checkbox("Include LP optimum (perfect foresight)", value=True)

series, rows = {}, []
for name in chosen:
    summ, ts = run_episode(policies[name], env, day, record=True)
    series[name] = ts
    rows.append({"policy": name, **{k: summ[k] for k in ("cost", "energy_served_pct", "evs_fully_served_pct", "peak_kw")}})
if use_lp:
    summ, ts = lp_day_summary(env, day, record=True)
    series["lp_optimal"] = ts
    rows.append({"policy": "lp_optimal", **{k: summ[k] for k in ("cost", "energy_served_pct", "evs_fully_served_pct", "peak_kw")}})

if series:
    fig = plot_day_comparison(day, tariff.price_vector(day), env.cap, plugged_in_count(ds.days[day], env.T),
                              series, ds.day_start_hour, ds.step_minutes)
    c1, c2 = st.columns([2, 1])
    c1.pyplot(fig)
    c2.dataframe(rows, hide_index=True)
    c2.caption(f"{ds.days[day].n} EVs this day, demand {ds.days[day].energy.sum():.0f} kWh.")
else:
    st.info("Pick at least one policy in the sidebar.")
