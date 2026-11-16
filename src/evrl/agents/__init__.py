"""Agent registry. Imports are lazy so that torch/SB3 are only needed when an agent is actually used."""
from __future__ import annotations

from .base import BaseAgent


def _dqn():
    from .dqn_agent import DQNAgent
    return DQNAgent


def _ppo():
    from .ppo_agent import PPOAgent
    return PPOAgent


def _a2c():
    from .a2c_agent import A2CAgent
    return A2CAgent


REGISTRY = {
    "dqn": _dqn,
    "ppo": _ppo,
    "a2c": _a2c,
}


def get_agent_class(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Registered: {list(REGISTRY)}")
    return REGISTRY[name]()


def make_agent(name: str, env, cfg: dict, seed: int = 0) -> BaseAgent:
    cls = get_agent_class(name)
    return cls(env=env, hparams=cfg.get(name, {}), seed=seed)


def load_agent(name: str, path) -> BaseAgent:
    return get_agent_class(name).load(path)


__all__ = ["BaseAgent", "REGISTRY", "make_agent", "load_agent", "get_agent_class"]
