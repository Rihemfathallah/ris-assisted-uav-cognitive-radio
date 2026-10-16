"""Core data types shared by every module.

For a single-antenna UAV the cascaded channels ``a = h_r * g`` and
``b = q_r * g`` together with the direct links are sufficient statistics for
the rate and the interference, so that is all a controller ever sees.
All channels are divided by sigma (noise-normalised units).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ChannelState:
    t: int                   # step index inside the episode
    uav_pos: np.ndarray      # (3,) metres
    h_d: complex             # UAV -> SU direct link
    q_d: complex             # UAV -> PU direct link
    a: np.ndarray            # (N,) cascaded UAV -> RIS -> SU
    b: np.ndarray            # (N,) cascaded UAV -> RIS -> PU

    @property
    def N(self) -> int:
        return int(self.a.shape[0])


@dataclass(frozen=True)
class Decision:
    theta: np.ndarray        # (N,) RIS phases in [0, 2*pi)
    P: float                 # transmit power, watts
    ris_active: bool = True  # False only for the "no RIS" reference


class Controller(Protocol):
    """Every method (baselines, optimisers, TD3) implements this interface."""

    name: str

    def reset(self) -> None:
        """Called at the start of every episode (stateful controllers clear memory)."""

    def decide(self, ch: ChannelState) -> Decision:
        """Map the current channel state to an RIS configuration and a power."""
