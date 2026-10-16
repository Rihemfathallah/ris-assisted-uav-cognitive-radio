"""TD3 controller: a trained Stable-Baselines3 actor behind the common Controller interface.

It reuses the *same* ObsBuilder and ActionMap objects as the training
environment and keeps the same recurrent features (previous action, last rate,
last interference), reset at every episode, so train/serve skew is impossible
by construction.

Inference runs the SB3 actor directly under ``torch.no_grad()`` with a single
torch thread: that is the cost a deployed controller would pay, and it is what
the runtime benchmark reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from .. import system as S
from ..rl.core import ActionMap, ObsBuilder
from ..rl.sb3 import actor_forward, load_actor, output_layer
from ..types import ChannelState, Decision

torch.set_num_threads(1)


class TD3Controller:
    def __init__(self, cfg: Config, actor, power_mode: str, name: str = "td3",
                 action_param: str = "circular", model=None):
        self.name = name
        self.cfg = cfg
        self.actor = actor
        self.model = model                       # kept for model.predict() timing only
        self.N = int(cfg.N)
        self.amap = ActionMap(cfg, self.N, power_mode, action_param)
        self.obs = ObsBuilder(cfg, self.N, self.amap.dim)
        assert actor.mu[0].in_features == self.obs.dim, "checkpoint / observation size mismatch"
        assert output_layer(actor).out_features == self.amap.dim, "checkpoint / action size mismatch"
        self.reset()

    @classmethod
    def from_run(cls, cfg: Config, run_dir: str | Path, which: str = "best", name: str = "td3"):
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text())
        model, actor = load_actor(run_dir / f"model_{which}")
        return cls(cfg.with_(N=meta["N"]), actor, meta["power_mode"], name,
                   meta.get("action_param", "linear"), model=model)

    @classmethod
    def from_model(cls, cfg: Config, model, power_mode: str, name: str = "td3",
                   action_param: str = "circular"):
        """Wrap a live SB3 model (used by the in-training validation callback)."""
        model.policy.set_training_mode(False)
        return cls(cfg, model.policy.actor, power_mode, name, action_param, model=model)

    def reset(self) -> None:
        self._prev_u = np.zeros(self.amap.dim, np.float32)
        self._last_R, self._last_I = 0.0, self.cfg.I_th_norm

    def action(self, ch: ChannelState) -> np.ndarray:
        """Deterministic action for the current channel state (no state update)."""
        o = self.obs.build(ch, self._prev_u, self._last_R, self._last_I)
        return actor_forward(self.actor, o)

    def decide(self, ch: ChannelState) -> Decision:
        u = self.action(ch)
        d = self.amap.decision(ch, u)
        # feedback features, exactly as the environment computes them (current-CSI convention)
        self._prev_u = u.astype(np.float32)
        self._last_R, self._last_I = S.rate(ch, d), S.interference(ch, d)
        return d
