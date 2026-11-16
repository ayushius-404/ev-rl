# EV Charging Scheduling with Reinforcement Learning

A modular, reproducible project for the RL course: a DQN agent learns **when to buy energy** at an EV charging
station, using **real charging sessions** (ACN-Data, Caltech) replayed in a Gymnasium environment, and is compared
against rule-based baselines and a **perfect-foresight LP optimum**.

> **One-sentence pitch (for the viva):** A station has a power limit and time-varying electricity prices; EVs arrive
> and leave at real, unpredictable times; every 15 minutes the agent chooses how much power to draw so that each EV
> gets its energy before it leaves, at minimum cost.

---

## 1. Problem formulation (MDP)

| Element | Definition |
|---|---|
| **Episode** | One operating day (04:00 to 04:00), 96 steps of 15 min. A random training day is replayed each episode. |
| **Arrivals** | Not modelled: the real sessions (arrival, departure, energy) are replayed. |
| **Action** | `Discrete(5)`: station power level in {0, 25, 50, 75, 100}% of `station_cap_kw`. |
| **Dispatch** | The chosen power is split among plugged-in EVs by **Least Laxity First** (most urgent EV first, at its full rate). The agent decides *how much*; the rule decides *who*. |
| **State** (14 numbers in [-1, 1]) | time of day (sin, cos), weekend flag, current price, next 4 prices, #EVs plugged in, total energy still needed, min and mean laxity, required average power to just finish, last step's load. |
| **Reward** | `-(price*energy + w_unmet*unmet_kWh_at_departure + w_peak*peak_excess) * reward_scale` |
| **Goal** | Minimise cost while delivering the requested energy to (almost) every EV, within the station cap. |
| **Termination** | After the last step of the day. Time of day is in the state, so a finite horizon is a proper MDP. |

**Laxity** = (time until departure) - (time needed to finish at the EV's max rate). Small or negative laxity means
the EV is urgent.

**Why not "never charge"?** Not charging is the cheapest action. `w_unmet` (default 5 $/kWh) is much bigger than any
price (max 0.40 $/kWh), so leaving energy undelivered is always worse than buying it.

### Design decisions you should be able to defend
- **Discrete power level + LLF dispatch** instead of one action per EV: keeps the action space at 5 and DQN-friendly,
  and keeps the learned part focused on timing (price vs urgency).
- **Replay of real sessions** instead of a simulated arrival process: no distribution assumptions about arrivals.
- **Chronological train/val/test split** (70/15/15 by day): no leakage from the future into training.
- **Per-EV demand is clipped to what is physically possible** (`rate x time available`), so no EV is infeasible by
  construction. The *station cap* can still make days infeasible, which is why unmet energy is a metric.
- **Assumed tariff:** ACN has sessions, not prices. The tariff is an explicit assumption (see config).
  Say so in the report.

---

## 2. Repository layout

```
ev-rl/
├── README.md
├── pyproject.toml / requirements.txt
├── configs/default.yaml        # ALL tunable settings (data, tariff, env, reward weights, DQN hyper-params)
├── data/{raw,processed}/       # raw ACN export in, processed per-day sessions out (git-ignored)
├── src/evrl/
│   ├── config.py               # YAML loading, EnvConfig dataclass
│   ├── data/
│   │   ├── download.py         # ACN-Data API download           <- needs YOUR token (TODO)
│   │   ├── preprocess.py       # raw -> per-day step-indexed sessions + chronological split
│   │   ├── dataset.py          # loads processed data into per-day arrays
│   │   ├── tariff.py           # time-of-use price vector for a day
│   │   ├── synthetic.py        # fake sessions to develop the pipeline before the real data arrives
│   │   └── stats.py            # ASAP load profile, occupancy (for EDA)
│   ├── envs/
│   │   ├── ev_station_env.py   # the Gymnasium environment (state, action, reward live here)
│   │   └── dispatch.py         # Least-Laxity-First power split
│   ├── baselines/
│   │   ├── policies.py         # ASAP, JustInTime, GreedyCheap, Random
│   │   └── lp_optimal.py       # perfect-foresight LP (upper bound), cvxpy
│   ├── agents/
│   │   ├── base.py             # BaseAgent interface (train / predict / save / load)
│   │   ├── sb3_agent.py        # generic Stable-Baselines3 wrapper
│   │   ├── dqn_agent.py        # DQN (main algorithm)
│   │   ├── ppo_agent.py        # placeholder for a 2nd algorithm (disabled)
│   │   └── __init__.py         # REGISTRY of algorithms
│   └── evaluation/
│       ├── runner.py           # run any policy/agent/LP on days -> metrics
│       ├── metrics.py          # summary table, savings vs ASAP, % of optimal saving
│       └── plots.py            # day replay, training curves, comparison bars
├── scripts/                    # thin command-line entry points
│   ├── prepare_data.py   eda.py   train.py   evaluate.py
│   └── plot_training.py  sweep_weights.py
├── app/demo.py                 # Streamlit replay demo
├── tests/test_env.py           # hand-checked unit tests
├── notebooks/  runs/  results/
```

Everything algorithm-specific sits in `agents/`; everything problem-specific sits in `envs/` and `data/`. Training,
evaluation and the demo only use the `BaseAgent` / `Policy` interfaces.

---

## 3. Setup (venv)

```bash
# from the repo root
python -m venv .venv

# activate:
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows (cmd / PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                 # makes `import evrl` work everywhere (needed once)
pytest -q                        # sanity check: all tests should pass
```

PyTorch tip: if you have no NVIDIA GPU, install the small CPU build first
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then `pip install -r requirements.txt`.
This network is tiny; a CPU is fine.

---

## 4. Workflow (follow in order)

### Step 0: develop on synthetic data (optional, 2 minutes)
```bash
python -m evrl.data.synthetic --days 200          # writes data/raw/synthetic_sessions.csv
# in configs/default.yaml set  data.raw_path: data/raw/synthetic_sessions.csv
```
Good for checking the code runs end to end. **Never report results from synthetic data.**

### Step 1: get the real data
Read `src/evrl/data/download.py` (it has a TODO checklist: token, verify the API, terms of use), then
```bash
export ACN_API_TOKEN=your_token             # Windows: set ACN_API_TOKEN=your_token
python -m evrl.data.download --site caltech --start 2019-01-01 --end 2019-12-31
```
Any export with columns `connectionTime`, `disconnectTime` and `kWhDelivered` (or `kWhRequested`) works, as `.json`
(API format) or `.csv`. Put it in `data/raw/` and point `data.raw_path` at it.

### Step 2: preprocess, then explore
```bash
python scripts/prepare_data.py     # prints how many sessions each filter removed, and the split
python scripts/eda.py              # results/eda/eda.png + ASAP peak-load percentiles
```
Look at the plots **before** choosing the tariff and the cap:
- **Tariff:** savings only exist if a cheap window lies inside the hours EVs are actually parked. A workplace
  site has most EVs parked roughly 08:00 to 17:00, so a tariff whose only cheap hours are at night gives the agent
  nothing to optimise. The default is an illustrative "solar dip" tariff (cheap 10:00 to 15:00).
- **Station cap:** `eda.py` suggests ~65% of the median ASAP daily peak so the cap really binds.

### Step 3: sanity-check baselines first (no training needed)
```bash
python scripts/evaluate.py --split val
```
This runs ASAP, JustInTime, GreedyCheap, Random and the LP optimum. Check that the ordering makes sense
(LP best; Random worst on service). Note that `JustInTime` can be **worse** than ASAP: if you delay charging into
an expensive evening peak you pay more. That is tariff-dependent, and a good talking point.

### Step 4: train
```bash
python scripts/train.py --algo dqn --seeds 0            # quick first run
python scripts/train.py --algo dqn --seeds 0 1 2 3 4    # final runs (do at least 3 seeds)
python scripts/plot_training.py                         # results/training_curves.png
```
Each run saves `runs/dqn/seed<k>/best_model.zip` (best on the validation days) and `final_model.zip`.

### Step 5: final evaluation on the test days
```bash
python scripts/evaluate.py --split test
```
Outputs `results/test_summary.csv`, `results/test_per_day.csv`, `results/test_comparison.png`.
Only touch the test split for final numbers. Tune on **val**.

### Step 6: experiments for the report
| # | Experiment | How |
|---|---|---|
| 1 | All policies vs optimum | `python scripts/evaluate.py --split test` |
| 2 | Training curves, mean +- std over seeds | `python scripts/plot_training.py` |
| 3 | Cost vs service trade-off | `python scripts/sweep_weights.py --values 0.5 1 2 5 10` |
| 4 | Robustness: tighter cap, no retraining | `python scripts/evaluate.py --cap-scale 0.8` |
| 5 | Ablation: no price forecast | set `env.forecast_steps: 0`, retrain with `--tag _nofc`, compare |

For the ablation, copy the config, change one value, train with `--tag` so runs don't overwrite each other, and
evaluate. `evaluate.py` automatically picks up every folder in `runs/` that starts with a registered algorithm name
(`dqn`, `dqn_nofc`, ...). Note: an ablation run must be evaluated with the same env settings it was trained with.

### Step 7: demo
```bash
streamlit run app/demo.py
```
Pick a test day and policies; see price, occupancy, station load and cumulative cost side by side (with the LP
optimum for reference).

---

## 5. Metrics

| Metric | Meaning |
|---|---|
| `cost` | Mean daily electricity cost ($) |
| `savings_vs_asap_pct` | Cost reduction vs the uncontrolled station |
| `pct_of_optimal_saving` | Share of the LP's saving that the policy recovers (100 = perfect foresight) |
| `energy_served_pct` | Delivered / requested energy |
| `evs_fully_served_pct` | EVs whose demand was met (within 0.05 kWh) |
| `unmet_kwh` | Total undelivered energy per day |
| `peak_kw` | Highest station load of the day |

Always read cost **together with** service: a policy that saves money by not delivering energy is not a good policy.

**About the LP optimum:** it uses continuous per-EV power and perfect foresight, so it is an *upper bound*, not
something an online agent can fully reach. "Our agent recovers X% of the optimal saving" is the honest framing.

---

## 6. Configuration quick reference (`configs/default.yaml`)

| Key | What it does |
|---|---|
| `data.energy_column` | `kWhDelivered` (default) or `kWhRequested` |
| `data.day_start_hour` | Where the operating day starts (04:00 keeps most sessions inside one day) |
| `tariff.*` | Time-of-use periods in $/kWh for weekdays/weekends |
| `env.station_cap_kw` | Station power limit |
| `env.n_levels` | Number of discrete power levels |
| `env.forecast_steps` | Future prices visible to the agent (0 = ablation) |
| `env.w_unmet`, `env.w_peak`, `env.peak_threshold_kw` | Reward weights |
| `train.total_timesteps`, `train.seeds` | Training budget and seeds |
| `dqn.*` | DQN hyper-parameters |

---

## 7. Adding a second algorithm (e.g. PPO) later

Nothing else changes. See `src/evrl/agents/ppo_agent.py` for the checklist:
1. implement `PPOAgent` (mirror `dqn_agent.py`, ~10 lines),
2. add a `ppo:` block to the config,
3. register it in `agents/__init__.py` (`REGISTRY`),
4. `python scripts/train.py --algo ppo --seeds 0 1 2`, then `python scripts/evaluate.py`.

It will appear in the comparison table, plots and demo automatically.

---

## 8. Your TODO list (search for `TODO(YOU)` in the code)

| Where | What you need to do |
|---|---|
| `data/download.py` | Get the ACN token, verify the API details, check the terms of use and citation |
| `configs/default.yaml` -> `energy_column` | Choose delivered vs requested energy and justify it |
| `configs/default.yaml` -> `tariff` | Choose a tariff you can defend (after EDA) |
| `configs/default.yaml` -> `station_cap_kw` | Set it from the EDA output |
| `configs/default.yaml` -> `seeds` | Use 3-5 seeds for final results |
| `envs/ev_station_env.py` -> reward | Understand it, decide whether to extend it (delay, fairness, shaping) |
| `baselines/policies.py` -> `GreedyCheap` | Improve the heuristic if you want a stronger baseline |
| `agents/dqn_agent.py` | Optional: write your own DQN and compare with the library |
| `agents/ppo_agent.py`, `agents/__init__.py` | Optional: second algorithm |
| `scripts/sweep_weights.py` | Extend to multiple seeds / `w_peak` |
| `app/demo.py` | Make the demo your own (sliders, live stepping) |

---

## 9. Tests

```bash
pytest -q
```
`tests/test_env.py` checks, by hand-computed examples: ASAP cost and energy, the "never charge" penalty, the station
cap, Least-Laxity-First ordering, just-in-time behaviour, that the LP shifts load into cheap hours, and that
observations stay inside the declared bounds. If you change the environment, run the tests.

---

## 10. Limitations (state them yourself in the report)

- Tariff is assumed, not observed. Savings depend on how it lines up with real parking hours.
- Single site (Caltech, a workplace), so results may not transfer to home or highway charging.
- Charging is modelled as constant power up to the EV's limit: no battery charge curve, no charging efficiency.
- No V2G (discharging), no PV generation, no demand charges.
- Wall-clock local time is used, so daylight-saving shifts are ignored.
- The agent chooses total station power; *who* gets it is a fixed rule (LLF), not learned.

## 11. Likely viva questions

- Why is this an MDP and what is in the state? (see Section 1)
- Why a discrete action with an inner dispatch rule? (small action space; separates timing from allocation)
- Why does `w_unmet` have to be large? (not charging is the cheapest action)
- How do you know it is not memorising training days? (chronological split; test only used once)
- How close is it to optimal? (`pct_of_optimal_saving` against the LP)
- Why can JustInTime lose to ASAP? (delaying into expensive hours)
- What would change with a different tariff? (savings and policy behaviour; try one)

## Data citation
Add the ACN-Data citation required by the dataset's terms of use to your report.
