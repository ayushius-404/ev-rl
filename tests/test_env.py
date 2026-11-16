"""Hand-checkable tests of the environment, dispatch rule, baselines and LP.

Toy world: 15-min steps, flat price $0.10/kWh, station cap 10 kW, one EV that needs 5 kWh, rate 10 kW,
present for steps 0..7 (8 steps = 2 hours).
At full power 10 kW the EV gets 2.5 kWh per step, so it is full after 2 steps -> cost = 0.10 * 5 = $0.50.
"""
import numpy as np
import pytest

from evrl.baselines import ASAP, JustInTime, solve_day
from evrl.config import EnvConfig
from evrl.data.dataset import Dataset, DaySessions
from evrl.data.tariff import Tariff
from evrl.envs import EVStationEnv
from evrl.envs.dispatch import llf_dispatch
from evrl.evaluation.runner import run_episode

DAY = "2019-03-04"  # a Monday
TARIFF_CFG = {"weekday": [{"start": 0, "end": 24, "price": 0.10}], "weekend": [{"start": 0, "end": 24, "price": 0.10}]}


def make_toy(arr, dep, energy, rate, cap=10.0, **cfg_kw):
    s = DaySessions(np.array(arr), np.array(dep), np.array(energy, float), np.array(rate, float))
    ds = Dataset({DAY: s}, {"train": [DAY], "val": [DAY], "test": [DAY]}, 15, 0)
    tariff = Tariff(TARIFF_CFG, 15, 0)
    cfg = EnvConfig(station_cap_kw=cap, n_levels=5, forecast_steps=2, w_unmet=5.0, reward_scale=1.0, **cfg_kw)
    return EVStationEnv(ds, tariff, [DAY], cfg, order="sequential")


def test_asap_cost_and_energy():
    env = make_toy([0], [8], [5.0], [10.0])
    s = run_episode(ASAP(), env, DAY)
    assert s["delivered_kwh"] == pytest.approx(5.0)
    assert s["cost"] == pytest.approx(0.50)
    assert s["unmet_kwh"] == pytest.approx(0.0)
    assert s["peak_kw"] == pytest.approx(10.0)


def test_never_charging_gives_unmet_penalty():
    env = make_toy([0], [8], [5.0], [10.0])
    obs, _ = env.reset(options={"day": DAY})
    total = 0.0
    for _ in range(env.T):
        obs, r, term, _, _ = env.step(0)
        total += r
    assert env.summary()["unmet_kwh"] == pytest.approx(5.0)
    assert total == pytest.approx(-5.0 * 5.0)  # w_unmet * unmet, no cost


def test_station_cap_respected():
    env = make_toy([0, 0], [8, 8], [5.0, 5.0], [10.0, 10.0], cap=10.0)
    obs, _ = env.reset(options={"day": DAY})
    _, _, _, _, info = env.step(env.action_space.n - 1)
    assert info["load"] <= 10.0 + 1e-9


def test_llf_serves_most_urgent_first():
    # EV0 has lots of slack, EV1 is tight. Budget only covers one EV.
    remaining = np.array([2.0, 2.0]); rate = np.array([8.0, 8.0]); dep = np.array([40, 3])
    p = llf_dispatch(8.0, remaining, rate, dep, t=0, dt=0.25)
    assert p[1] == pytest.approx(8.0) and p[0] == pytest.approx(0.0)


def test_just_in_time_delays_but_still_serves():
    env = make_toy([0], [8], [5.0], [10.0])
    s = run_episode(JustInTime(margin_h=0.0), env, DAY)
    assert s["unmet_kwh"] == pytest.approx(0.0, abs=0.05)


def test_lp_not_worse_than_asap_and_cheap_hours_used():
    # two prices: expensive first half, cheap second half of the stay -> LP should shift charging to the cheap half
    prices = np.array([0.4] * 4 + [0.1] * 4 + [0.1] * 88)
    s = DaySessions(np.array([0]), np.array([8]), np.array([5.0]), np.array([10.0]))
    res = solve_day(s, prices, cap_kw=10.0, dt=0.25, w_unmet=5.0)
    assert res["cost"] == pytest.approx(0.5, abs=1e-3)           # 5 kWh at $0.10
    assert res["load"][:4].sum() == pytest.approx(0.0, abs=1e-4)  # nothing in the expensive hours
    assert res["remaining"].sum() == pytest.approx(0.0, abs=1e-4)


def test_observation_in_bounds_and_right_size():
    env = make_toy([0], [8], [5.0], [10.0])
    obs, _ = env.reset(options={"day": DAY})
    assert obs.shape == env.observation_space.shape
    for _ in range(env.T):
        obs, _, term, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
    assert term


def test_wait_penalty_and_metric():
    # two EVs, only enough power for one of them -> exactly one EV waits in every charging step
    env = make_toy([0, 0], [8, 8], [5.0, 5.0], [10.0, 10.0], cap=10.0)
    env.cfg.w_wait = 0.5
    env.reset(options={"day": DAY})
    _, r, _, _, info = env.step(env.action_space.n - 1)   # full power: one EV gets 10 kW, the other 0
    assert info["n_wait"] == 1
    expected_cost = 0.10 * info["energy"]
    assert r == pytest.approx(-(expected_cost + 0.5 * 1))
    for _ in range(env.T - 1):
        env.step(env.action_space.n - 1)
    assert env.summary()["avg_wait_min"] > 0
