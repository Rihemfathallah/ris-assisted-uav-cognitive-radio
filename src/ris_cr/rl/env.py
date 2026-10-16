"""Gymnasium environment: one flight segment of ``episode_len`` steps.

Reward
  power_mode == "project":  r = R          (analytic power => feasible by construction)
  power_mode == "learn":    r = R - lam * penalty(I_PU)   (spec, Sec. 5.6)

The interference violation is logged separately in ``info`` because a good
average reward can hide frequent violations.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from ..channel import TRAIN_EPISODE_SEED_OFFSET, ChannelGenerator
from ..config import Config
from ..scenarios import Bank
from .. import system as S
from .core import ActionMap, ObsBuilder, penalty


class RisCrEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, power_mode: str | None = None, penalty_kind: str | None = None,
                 lam: float | None = None, bank: Bank | None = None, action_param: str | None = None):
        super().__init__()
        td3 = cfg.td3
        self.cfg = cfg
        self.N = int(cfg.N)
        self.T = int(cfg.episode_len)
        self.amap = ActionMap(cfg, self.N, power_mode or td3["power_mode"], action_param or td3.get("action_param", "linear"))
        self.obs_builder = ObsBuilder(cfg, self.N, self.amap.dim)
        self.penalty_kind = penalty_kind or td3["penalty"]
        self.lam = float(td3["lam"] if lam is None else lam)
        self.I_th = cfg.I_th_norm
        self.bank = bank
        self._bank_ep = 0
        self.gen = ChannelGenerator(cfg)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.amap.dim,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_builder.dim,), np.float32)

    # ------------------------------------------------------------------ helpers
    def _state(self):
        if self.bank is not None:
            return self.bank.state(self._ep, self.t)
        return self.gen.state()

    def _obs(self):
        return self.obs_builder.build(self._ch, self._prev_u, self._last_R, self._last_I)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.t = 0
        if self.bank is not None:
            self._ep = int(options.get("episode", self._bank_ep)) % self.bank.shape[0]
            self._bank_ep = self._ep + 1
        else:
            ep_seed = TRAIN_EPISODE_SEED_OFFSET + int(self.np_random.integers(0, 2**31 - 1 - TRAIN_EPISODE_SEED_OFFSET))
            self.gen.new_episode(ep_seed, 0)
        self._ch = self._state()
        self._prev_u = np.zeros(self.amap.dim, np.float32)
        self._last_R, self._last_I = 0.0, self.I_th
        return self._obs(), {}

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)   # circular pairs are already inside
        d = self.amap.decision(self._ch, u)
        R, I = S.rate(self._ch, d), S.interference(self._ch, d)
        viol = S.is_violation(I, self.I_th)
        r = R if self.amap.power_mode == "project" else R - self.lam * penalty(I, self.I_th, self.penalty_kind)
        info = {"rate": R, "interference": I, "violation": viol, "P": d.P}
        self._prev_u, self._last_R, self._last_I = u, R, I
        self.t += 1
        if self.bank is None:
            self.gen.step()
        truncated = self.t >= self.T
        # Time-limit truncation is not a terminal state: TD3 bootstraps through it, so the
        # returned observation must be the true next channel (available from the generator).
        if self.bank is None or not truncated:
            self._ch = self._state()
        return self._obs(), float(r), False, truncated, info
