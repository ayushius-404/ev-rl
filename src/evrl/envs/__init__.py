from ..config import EnvConfig
from ..data.dataset import Dataset
from ..data.tariff import Tariff
from .ev_station_env import EVStationEnv


def make_env(dataset: Dataset, tariff: Tariff, cfg: dict, split: str = "train",
             order: str = "random", **env_overrides) -> EVStationEnv:
    """Build an environment on one split ('train' | 'val' | 'test').

    env_overrides let you change env settings without editing the YAML, e.g. station_cap_kw=80,
    forecast_steps=0, w_unmet=10.
    """
    env_cfg = EnvConfig.from_dict(cfg["env"], **env_overrides)
    return EVStationEnv(dataset, tariff, dataset.splits[split], env_cfg, order=order)


__all__ = ["EVStationEnv", "EnvConfig", "make_env"]
