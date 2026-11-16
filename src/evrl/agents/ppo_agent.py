"""PPO agent (policy-gradient, on-policy). Thin wrapper over Stable-Baselines3 PPO."""
from __future__ import annotations

from .sb3_agent import SB3Agent


def _ppo_cls():
    from stable_baselines3 import PPO
    return PPO


class PPOAgent(SB3Agent):
    name = "ppo"

    def __init__(self, env=None, hparams=None, seed=0):
        type(self).algo_cls = _ppo_cls()
        super().__init__(env, hparams, seed)

    @classmethod
    def load(cls, path):
        cls.algo_cls = _ppo_cls()
        return super().load(path)
