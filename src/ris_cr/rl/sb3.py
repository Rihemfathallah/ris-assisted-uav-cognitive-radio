"""Stable-Baselines3 (PyTorch) backend for TD3.

Everything framework-specific lives here: model construction from the YAML
hyper-parameters, the residual-head bias trick, the validation callback used for
model selection, and single-threaded actor inference for the deployed controller.
The environment, the observation builder and the action decoding stay in
``core.py`` / ``env.py`` and know nothing about torch.

Action heads (see ``core.ActionMap``)
  * ``linear``   : u in [-1, 1]^N, theta = pi (u + 1).  The tanh actor saturates
    against the wrap-around at 0 = 2 pi; ``saturated_fraction`` measures it.
  * ``circular`` / ``residual`` : one (c_n, s_n) pair per element, decoded with
    ``atan2(s_n, c_n)``.  atan2 is scale-invariant, so a tanh-bounded pair needs
    no projection onto the unit circle: every direction in the plane is still
    reachable, and the critic is periodic in theta by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

RESIDUAL_COS_BIAS = 2.0   # pre-tanh bias on every cos component: atan2(tanh(z), tanh(2+z)) ~ 0


def output_layer(actor):
    """The actor's final Linear (SB3 ends ``actor.mu`` with a Tanh, so ``mu[-1]`` is not it)."""
    for layer in reversed(actor.mu):
        if isinstance(layer, torch.nn.Linear):
            return layer
    raise TypeError("actor.mu has no Linear layer")


def build_model(env, hp: dict, seed: int, buffer_size: int | None = None, gamma: float | None = None) -> TD3:
    """``stable_baselines3.TD3`` configured from ``configs/default.yaml``."""
    act_dim = int(env.action_space.shape[0])
    return TD3(
        "MlpPolicy", env,
        learning_rate=float(hp["learning_rate"]),
        buffer_size=int(hp["buffer_size"] if buffer_size is None else buffer_size),
        batch_size=int(hp["batch_size"]),
        gamma=float(hp["gamma"] if gamma is None else gamma),
        tau=float(hp["tau"]),
        policy_delay=int(hp["policy_delay"]),
        target_policy_noise=float(hp["target_noise"]),
        target_noise_clip=float(hp["target_noise_clip"]),
        learning_starts=int(hp["learning_starts"]),
        action_noise=NormalActionNoise(np.zeros(act_dim), float(hp["action_noise"]) * np.ones(act_dim)),
        policy_kwargs=dict(net_arch=list(hp["net_arch"])),
        train_freq=1, gradient_steps=1,
        seed=int(seed), device="cpu", verbose=0,
    )


def apply_residual_bias(model: TD3, n_pairs: int, value: float = RESIDUAL_COS_BIAS) -> None:
    """Anchor an untrained residual policy on co-phasing.

    The residual head decodes ``atan2(s_n, c_n)`` as a *correction* to the
    analytic co-phasing phase, so a zero correction needs s_n ~ 0 and c_n > 0.
    Adding a positive constant to the pre-tanh bias of every cos component
    (the even entries of the action vector) pushes c_n towards +1 while leaving
    s_n centred on 0, so the decoded correction starts near zero.
    Applied to the actor *and* the target actor, so the TD3 target is anchored too.
    """
    if n_pairs <= 0:
        raise ValueError("the residual bias only applies to a (cos, sin) head")
    with torch.no_grad():
        for net in (model.policy.actor, model.policy.actor_target):
            bias = output_layer(net).bias    # last Linear of the actor MLP (pre-tanh)
            bias[0: 2 * n_pairs: 2] += value


def actor_forward(actor, obs: np.ndarray) -> np.ndarray:
    """Raw deterministic actor forward pass, single-threaded, no autograd.

    This is what a deployed controller would run; ``model.predict`` adds
    observation reshaping/normalisation bookkeeping on top and is timed separately.
    """
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        return actor(x).cpu().numpy().reshape(-1)


def saturated_fraction(u: np.ndarray, thr: float = 0.999) -> float:
    """Fraction of action components pinned against the tanh bound."""
    u = np.asarray(u, dtype=float)
    return float(np.mean(np.abs(u) > thr)) if u.size else float("nan")


def load_actor(path: str | Path):
    """Load a saved SB3 TD3 model and return (model, actor) ready for inference."""
    model = TD3.load(str(path), device="cpu")
    model.policy.set_training_mode(False)
    return model, model.policy.actor


class ValidationCallback(BaseCallback):
    """Model selection on the frozen validation flights.

    Every ``eval_every`` steps the current actor is evaluated on the validation
    bank with the *serving* code path (``TD3Controller``), one row is appended to
    ``learning_curve.csv``, and the best model is saved.  With learned power a
    model is only eligible if its interference violation rate is <= 5 %.
    """

    MAX_VIOLATION = 0.05

    def __init__(self, evaluate_fn, eval_every: int, out_dir: Path, power_mode: str,
                 writer, flush, verbose_prefix: str | None = None):
        super().__init__(verbose=0)
        self.evaluate_fn, self.eval_every, self.out = evaluate_fn, int(eval_every), Path(out_dir)
        self.power_mode, self.writer, self.flush = power_mode, writer, flush
        self.prefix = verbose_prefix
        self.best = -np.inf
        self.best_step = -1

    def _eligible(self, m: dict) -> bool:
        return self.power_mode == "project" or m["violation_rate"] <= self.MAX_VIOLATION

    def _evaluate_and_record(self) -> None:
        m = self.evaluate_fn(self.model)
        self.writer(self.num_timesteps, m)
        self.flush()
        if m["rate"] > self.best and self._eligible(m):
            self.best, self.best_step = m["rate"], self.num_timesteps
            self.model.save(self.out / "model_best")
        if self.prefix:
            print(f"{self.prefix} step {self.num_timesteps:>7} val_rate={m['rate']:.4f} "
                  f"viol={m['violation_rate']:.3f} sat={m['saturated_frac']:.3f}", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0:
            self._evaluate_and_record()
        return True

    def on_training_end(self) -> None:
        if self.num_timesteps % self.eval_every != 0:
            self._evaluate_and_record()
