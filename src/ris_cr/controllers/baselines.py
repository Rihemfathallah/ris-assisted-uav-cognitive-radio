"""Reference controllers.

All controllers that choose phases also choose power with the analytic
projection P = min(P_max, I_th / |q_eff|^2), so every method is evaluated on
the same (feasible) footing and differences come only from the phases.
"""
from __future__ import annotations

import numpy as np

from ..config import Config, rng
from ..system import closed_form_phases, projected_decision
from ..types import ChannelState, Decision


class _Base:
    name = "base"

    def __init__(self, cfg: Config, constrained: bool = True):
        self.cfg = cfg
        self.P_max = cfg.P_max_W
        # "unconstrained" mode (Fig. 1) simply sets an infinite threshold
        self.I_th = cfg.I_th_norm if constrained else np.inf

    def reset(self) -> None:
        pass


class NoRIS(_Base):
    """RIS physically absent: only the direct links exist."""
    name = "no_ris"

    def decide(self, ch: ChannelState) -> Decision:
        return projected_decision(ch, np.zeros(ch.N), self.P_max, self.I_th, ris_active=False)


class RandomPhases(_Base):
    """I.i.d. uniform phases (seeded, reproducible)."""
    name = "random"

    def __init__(self, cfg: Config, constrained: bool = True, seed: int = 0):
        super().__init__(cfg, constrained)
        self._seed = seed
        self._g = rng(seed, "baseline_random")

    def reset(self) -> None:
        pass  # keep the stream running across episodes; re-seeded only at construction

    def decide(self, ch: ChannelState) -> Decision:
        return projected_decision(ch, self._g.uniform(0, 2 * np.pi, ch.N), self.P_max, self.I_th)


class ClosedForm(_Base):
    """SU-optimal phases (co-phasing with h_d), PU-agnostic; power projected.

    Without the PU constraint this is the optimum of (P1) (Fig. 1 bound).  With
    the constraint it is a natural "greedy" heuristic that ignores where the
    reflected energy lands at the PU.
    """
    name = "closed_form"

    def decide(self, ch: ChannelState) -> Decision:
        return projected_decision(ch, closed_form_phases(ch), self.P_max, self.I_th)
