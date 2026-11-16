"""DQN agent (main algorithm). Thin wrapper over Stable-Baselines3 DQN."""
from __future__ import annotations

from .sb3_agent import SB3Agent


def _dqn_cls():
    from stable_baselines3 import DQN
    return DQN


class DQNAgent(SB3Agent):
    name = "dqn"

    def __init__(self, env=None, hparams=None, seed=0):
        type(self).algo_cls = _dqn_cls()
        super().__init__(env, hparams, seed)

    @classmethod
    def load(cls, path):
        cls.algo_cls = _dqn_cls()
        return super().load(path)

    # TODO(YOU) (optional, great for learning): re-implement DQN yourself in PyTorch (replay buffer, target
    # network, epsilon-greedy, Double-DQN target) as `agents/my_dqn.py`, subclassing BaseAgent, and register it as
    # "my_dqn". Then you can show in the viva that your own implementation matches the library's results.
