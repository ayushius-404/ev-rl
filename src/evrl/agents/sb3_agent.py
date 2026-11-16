"""Generic wrapper around a Stable-Baselines3 algorithm (DQN, PPO, ...)."""
from __future__ import annotations

from pathlib import Path

from .base import BaseAgent


class SB3Agent(BaseAgent):
    algo_cls = None  # set by subclasses, e.g. stable_baselines3.DQN

    def __init__(self, env=None, hparams: dict | None = None, seed: int = 0):
        self.model = None
        if env is not None:
            hp = dict(hparams or {})
            net_arch = hp.pop("net_arch", [128, 128])
            self.model = self.algo_cls(
                "MlpPolicy", env, seed=seed, verbose=0,
                policy_kwargs=dict(net_arch=list(net_arch)), **hp,
            )

    def train(self, total_timesteps, val_env=None, eval_freq=10_000, n_eval_episodes=1, save_dir=None):
        from stable_baselines3.common.callbacks import EvalCallback

        cb = None
        if val_env is not None and save_dir is not None:
            cb = EvalCallback(val_env, best_model_save_path=str(save_dir), log_path=str(save_dir),
                              eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=True)
        self.model.learn(total_timesteps=total_timesteps, callback=cb)
        if save_dir is not None:
            self.save(Path(save_dir) / "final_model")

    def n_parameters(self) -> int:
        """Number of trainable weights of the network that is actually used to act (for the complexity comparison)."""
        net = getattr(self.model, "q_net", None)            # DQN: only the online Q-network (not the target copy)
        params = net.parameters() if net is not None else self.model.policy.parameters()
        return int(sum(p.numel() for p in params))

    def predict(self, obs) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def save(self, path):
        self.model.save(str(path))

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = cls.algo_cls.load(str(path))
        return obj
