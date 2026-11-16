"""A2C agent (advantage actor-critic, on-policy). Thin wrapper over Stable-Baselines3 A2C."""
from __future__ import annotations

from .sb3_agent import SB3Agent


def _a2c_cls():
    from stable_baselines3 import A2C
    return A2C


class A2CAgent(SB3Agent):
    name = "a2c"

    def __init__(self, env=None, hparams=None, seed=0):
        type(self).algo_cls = _a2c_cls()
        super().__init__(env, hparams, seed)

    @classmethod
    def load(cls, path):
        cls.algo_cls = _a2c_cls()
        return super().load(path)
