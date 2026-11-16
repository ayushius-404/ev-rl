"""Live simulation: EVs arrive through the day and the trained agent decides the charging power step by step.

    streamlit run app/live_sim.py

- Pick a day and a policy (your trained DQN or a baseline) in the sidebar.
- "Play" runs the day in (accelerated) real time, "Step" advances one 15-minute step, "Run to end" jumps to the end.
- "Add an EV now" injects an extra car at the current time, to see how the agent reacts to a surprise.
TODO(YOU): make it yours. Ideas: show the agent's Q-values per action, compare two policies side by side,
add a "block 50% of the station power for 2 hours" button to simulate a grid event.
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import setup  # noqa: E402
from evrl.agents import REGISTRY, load_agent  # noqa: E402
from evrl.baselines import ASAP, make_baselines  # noqa: E402
from evrl.envs import make_env  # noqa: E402
from evrl.envs.dispatch import llf_dispatch  # noqa: E402
from evrl.evaluation import AgentPolicy, run_episode  # noqa: E402

st.set_page_config(page_title="EV charging live simulation", layout="wide")
st.title("Live simulation: the agent allots charging as EVs arrive")

GLOSSARY = """
**Time**: the simulated clock. One step = 15 minutes. The operating day runs 04:00 to 04:00.

**Price**: electricity price right now ($ per kWh). It follows a time-of-use tariff that is known in advance.

**Station load / cap**: power the station is drawing now (kW) versus the station's maximum power (the *cap*). The agent can never exceed the cap.

**Agent power level**: the agent's decision for this step, as a share of the cap (0, 25, 50, 75 or 100%).
It only decides *how much* power to use. A fixed rule, **Least Laxity First**, then decides *who* gets it.

**Laxity**: time an EV has left minus the time it needs to finish charging at full speed. Small laxity means urgent. The most urgent EV is served first.

**Cost so far**: the electricity bill of the day up to now (price x energy, summed over steps). Lower is better.

**Unmet energy**: kWh that EVs asked for but did not receive before leaving. Counted when an EV departs. Should stay near 0, because the agent is penalised heavily for it.

**Charged %**: how much of the EV's requested energy it has received so far.

**ASAP**: the "no control" policy that charges every EV at full power as soon as it arrives. It is the reference we want to beat.
"""
if hasattr(st, "popover"):
    with st.popover("? What do these mean"):
        st.markdown(GLOSSARY)
else:
    with st.expander("? What do these mean"):
        st.markdown(GLOSSARY)


@st.cache_resource
def get_setup():
    return setup()


@st.cache_resource
def get_agent(algo: str, path: str):
    return load_agent(algo, path)


cfg, ds, tariff = get_setup()

# ---------------------------------------------------------------- sidebar
split = st.sidebar.selectbox("Split", ["test", "val", "train"])
day = st.sidebar.selectbox("Day", ds.splits[split])

policy_names = {p.name: p for p in make_baselines()}
for algo in REGISTRY:
    for seed_dir in sorted(Path("runs").glob(f"{algo}*/seed*")):
        f = seed_dir / "best_model.zip"
        if f.exists():
            label = f"{seed_dir.parent.name}_{seed_dir.name}"
            policy_names[label] = (algo, str(f), label)
options = list(policy_names)
default = next((i for i, n in enumerate(options) if n.startswith("dqn")), options.index("greedy_cheap"))
policy_name = st.sidebar.selectbox("Policy", options, index=default)

st.sidebar.markdown("---")
st.sidebar.subheader("Add an EV now")
add_energy = st.sidebar.slider("Energy needed (kWh)", 2, 40, 20)
add_stay = st.sidebar.slider("Will stay (hours)", 1, 10, 4)


def build_policy(name):
    p = policy_names[name]
    if isinstance(p, tuple):
        algo, path, label = p
        return AgentPolicy(get_agent(algo, path), label)
    return p


# ---------------------------------------------------------------- session state
key = (split, day, policy_name)
if st.session_state.get("key") != key or st.sidebar.button("Reset"):
    env = make_env(ds, tariff, cfg, split, "sequential")
    obs, _ = env.reset(options={"day": day})
    st.session_state.update(key=key, env=env, obs=obs, policy=build_policy(policy_name), hist=[],
                            last_power={}, last_action=0, done=False, injected=0)
ss = st.session_state
env = ss.env


def clock(step: int) -> str:
    minutes = (ds.day_start_hour * 60 + step * ds.step_minutes) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def do_step():
    if ss.done:
        return
    t = env.t
    a = ss.policy.act(ss.obs, env)
    idx = env._active(t)  # same dispatch the env will use, so we can show who got how much power
    p = llf_dispatch(env.level_kw[a], env.remaining[idx], env.rate[idx], env.dep[idx], t, env.dt)
    ss.last_power = {int(i): float(x) for i, x in zip(idx, p)}
    ss.last_action = int(a)
    ss.obs, _, term, trunc, info = env.step(a)
    ss.hist.append({"step": t, **info})
    ss.done = bool(term or trunc)


def add_ev():
    t = env.t
    if ss.done or t >= env.T - 1:
        return
    dep = min(env.T, t + max(1, int(round(add_stay / env.dt))))
    rate = float(env.rate[0]) if len(env.rate) else 6.6
    energy = float(min(add_energy, rate * (dep - t) * env.dt))
    env.arr, env.dep = np.append(env.arr, t), np.append(env.dep, dep)
    env.rate, env.remaining = np.append(env.rate, rate), np.append(env.remaining, energy)
    env.demand_per_ev = np.append(env.demand_per_ev, energy)
    ss.obs = env._obs()
    ss.injected += 1


def render(container):
    t = env.t
    h = pd.DataFrame(ss.hist)
    with container.container():
        c = st.columns(6)
        c[0].metric("Time", clock(min(t, env.T - 1)) if not ss.done else "end of day",
                    help="Simulated clock. One step is 15 minutes. The operating day runs from 04:00 to 04:00.")
        price = env.prices[min(t, env.T - 1)]
        c[1].metric("Price ($/kWh)", f"{price:.2f}",
                    help="Electricity price right now, from the time-of-use tariff (known in advance). Cheap hours are when the agent prefers to buy energy.")
        load = h["load"].iloc[-1] if len(h) else 0.0
        c[2].metric("Station load", f"{load:.0f} / {env.cap:.0f} kW",
                    help="Power the station drew in the last step, out of its maximum (the station cap). The agent can never exceed the cap.")
        c[3].metric("Agent power level", f"{env.levels[ss.last_action] * 100:.0f}%",
                    help="The agent's decision: what share of the station's maximum power to use this step (0, 25, 50, 75 or 100%). A fixed rule (Least Laxity First) then splits that power among the plugged-in EVs, most urgent first.")
        c[4].metric("Cost so far", f"${env._cost:.2f}",
                    help="Electricity bill of the day up to now: price x energy delivered, summed over all steps. Lower is better.")
        c[5].metric("Unmet energy", f"{env._unmet:.1f} kWh",
                    help="Energy that EVs asked for but did not receive before leaving (counted when an EV departs). It should stay near 0: the agent is penalised heavily for it.")

        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("EVs plugged in")
            present = np.where((env.arr <= t) & (t < env.dep))[0] if not ss.done else []
            rows = []
            for i in present:
                demand = env.demand_per_ev[i]
                rows.append({
                    "EV": f"#{i}" + (" (added)" if i >= len(ds.days[day].arr) else ""),
                    "arrived": clock(int(env.arr[i])), "leaves": clock(int(env.dep[i]) % env.T if env.dep[i] < env.T else env.T),
                    "needs (kWh)": round(float(demand), 1),
                    "left (kWh)": round(float(env.remaining[i]), 1),
                    "charged %": 100.0 * (1 - env.remaining[i] / demand) if demand > 0 else 100.0,
                    "power now (kW)": round(ss.last_power.get(int(i), 0.0), 1),
                })
            if rows:
                df = pd.DataFrame(rows).sort_values("power now (kW)", ascending=False)
                st.dataframe(df, hide_index=True, height=360,
                             column_config={
                                 "EV": st.column_config.TextColumn("EV", help="Car id. '(added)' = injected by you."),
                                 "arrived": st.column_config.TextColumn("arrived", help="Time the EV plugged in."),
                                 "leaves": st.column_config.TextColumn("leaves", help="Time the EV will unplug. It must be charged before this."),
                                 "needs (kWh)": st.column_config.NumberColumn("needs (kWh)", help="Total energy this EV requested."),
                                 "left (kWh)": st.column_config.NumberColumn("left (kWh)", help="Energy still to be delivered."),
                                 "charged %": st.column_config.ProgressColumn(
                                     "charged %", min_value=0, max_value=100, format="%d%%",
                                     help="Share of the requested energy delivered so far."),
                                 "power now (kW)": st.column_config.NumberColumn(
                                     "power now (kW)", help="Power this EV received in the last step. Urgent EVs are served first."),
                             })
            else:
                st.info("No EVs plugged in right now.")
        with right:
            st.subheader("Station load vs price")
            fig, ax = plt.subplots(figsize=(6, 3.4))
            hours = ds.day_start_hour + np.arange(env.T) * ds.step_minutes / 60.0
            if len(h):
                ax.step(hours[h["step"]], h["load"], where="post", color="tab:blue")
            ax.axhline(env.cap, color="k", ls="--", lw=0.8)
            ax.set_xlim(hours[0], hours[-1]); ax.set_ylim(0, env.cap * 1.1)
            ax.set_ylabel("load (kW)"); ax.set_xlabel("hour of day")
            ax2 = ax.twinx()
            ax2.step(hours, env.prices, where="post", color="tab:red", alpha=0.5)
            ax2.set_ylabel("price ($/kWh)", color="tab:red")
            if not ss.done:
                ax.axvline(hours[min(t, env.T - 1)], color="gray", lw=1)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if ss.done:
            s = env.summary()
            msg = (f"Day finished. Cost ${s['cost']:.2f}, energy served {s['energy_served_pct']:.2f}%, "
                   f"EVs fully served {s['evs_fully_served_pct']:.1f}%, peak {s['peak_kw']:.0f} kW.")
            if ss.injected == 0:
                asap = run_episode(ASAP(), make_env(ds, tariff, cfg, split, "sequential"), day)
                msg += f" ASAP (no control) would have cost ${asap['cost']:.2f}."
            st.success(msg)


# ---------------------------------------------------------------- controls
b = st.columns([1, 1, 1.2, 1.6, 3])
step_clicked = b[0].button("Step", help="Advance one 15-minute step.")
play_clicked = b[1].button("Play", type="primary", help="Run the day step by step at the chosen speed.")
end_clicked = b[2].button("Run to end", help="Jump straight to the end of the day.")
add_clicked = b[3].button("Add an EV now", help="Inject an extra car right now (size and stay time are set in the sidebar).")
delay = b[4].slider("Speed: seconds per step", 0.0, 3.0, 0.5, 0.1,
                    help="Pause between steps while Play is running. 0 = as fast as possible, 3 = very slow. "
                         "If you change it during Play, Play stops: press Play again.")
st.caption(f"{ds.days[day].n} EVs in the original day. Added so far: {ss.injected}.")

main = st.empty()
if add_clicked:
    add_ev()
if step_clicked:
    do_step()
if end_clicked:
    while not ss.done:
        do_step()
if play_clicked:
    while not ss.done:
        do_step()
        render(main)
        time.sleep(delay)
render(main)
