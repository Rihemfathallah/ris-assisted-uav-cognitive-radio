"""Element-wise (coordinate-ascent) refinement over discrete phase levels.

Objective per candidate: P*(theta) |h_eff|^2 with P* the projected power, which
is monotone in the SU rate of (P1).  One element is updated at a time with the
others fixed; the 16 candidate values are evaluated in closed form (O(L) per
element thanks to incremental h_eff / q_eff updates).  A move is accepted only
if it strictly improves the objective, so the procedure is monotone and
terminates; it starts from the (continuous) co-phasing solution, therefore it
can never be worse than the ClosedForm controller.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import ChannelState, Decision
from ..system import closed_form_phases, project_power
from .baselines import _Base


class ElementWise(_Base):
    name = "elementwise"

    def __init__(self, cfg: Config, constrained: bool = True, levels: int | None = None,
                 max_sweeps: int | None = None):
        super().__init__(cfg, constrained)
        self.levels = int(levels or cfg.elementwise["levels"])
        self.max_sweeps = int(max_sweeps or cfg.elementwise["max_sweeps"])
        self._cand_theta = 2 * np.pi * np.arange(self.levels) / self.levels
        self._cand = np.exp(1j * self._cand_theta)
        self.last_sweeps = 0

    def _value(self, h, q):
        qg = np.abs(q) ** 2
        with np.errstate(divide="ignore"):
            P = np.minimum(self.P_max, self.I_th / qg)
        return P * np.abs(h) ** 2

    def decide(self, ch: ChannelState) -> Decision:
        a, b = ch.a, ch.b
        theta = closed_form_phases(ch)
        e = np.exp(1j * theta)
        sweeps = 0
        for sweeps in range(1, self.max_sweeps + 1):
            h = ch.h_d + a @ e           # recompute to avoid drift of incremental updates
            q = ch.q_d + b @ e
            cur = self._value(h, q)
            changed = False
            for n in range(ch.N):
                h_rest, q_rest = h - a[n] * e[n], q - b[n] * e[n]
                vals = self._value(h_rest + a[n] * self._cand, q_rest + b[n] * self._cand)
                k = int(np.argmax(vals))
                if vals[k] > cur * (1.0 + 1e-12):
                    e[n], theta[n], cur = self._cand[k], self._cand_theta[k], vals[k]
                    h, q = h_rest + a[n] * e[n], q_rest + b[n] * e[n]
                    changed = True
            if not changed:
                break
        self.last_sweeps = sweeps
        qg = abs(ch.q_d + b @ np.exp(1j * theta)) ** 2
        return Decision(theta % (2 * np.pi), project_power(qg, self.P_max, self.I_th))
