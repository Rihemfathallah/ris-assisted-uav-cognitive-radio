"""The single source of truth for featurisation and action decoding.

Used by the Gymnasium environment during training *and* by the TD3 controller
at evaluation, so train/serve skew is impossible by construction.

Observation (spec, Sec. 5.6) with two documented refinements:
  [Re a, Im a, Re b, Im b, Re h_d, Im h_d, Re q_d, Im q_d, prev_action, last_rate, last_interference]
  * size 5N+7 with learned power, 5N+6 with projected power (action has N entries)
  * phase canonicalisation: every channel is multiplied by exp(-j arg h_d).
    A common phase rotation of all UAV-side channels changes neither the rates
    nor the optimal RIS phases (theta_n* = arg h_d - arg a_n is invariant), so
    this removes a nuisance degree of freedom losslessly (E04, test_rl.py).
  * fixed scaling: each channel group is divided by its RMS value at the circle
    centre (from the path-loss model), so features are O(1) without running
    statistics that would complicate reproducibility.
"""
from __future__ import annotations

import numpy as np

from ..channel import reference_gains
from ..config import Config
from ..system import project_power
from ..types import ChannelState, Decision


class ObsBuilder:
    def __init__(self, cfg: Config, N: int, action_dim: int, canonicalize: bool = True):
        g = reference_gains(cfg)
        s2 = cfg.sigma2_W
        self.s_a = np.sqrt(g["uav_ris"] * g["ris_su"] / s2)
        self.s_b = np.sqrt(g["uav_ris"] * g["ris_pu"] / s2)
        self.s_hd = np.sqrt(g["uav_su"] / s2)
        self.s_qd = np.sqrt(g["uav_pu"] / s2)
        self.I_th = cfg.I_th_norm
        self.N, self.action_dim, self.canonicalize = N, action_dim, canonicalize
        self.dim = 4 * N + 4 + action_dim + 2

    def build(self, ch: ChannelState, prev_u: np.ndarray, last_rate: float, last_I: float) -> np.ndarray:
        rot = np.exp(-1j * np.angle(ch.h_d)) if self.canonicalize else 1.0
        a, b = ch.a * rot / self.s_a, ch.b * rot / self.s_b
        hd, qd = ch.h_d * rot / self.s_hd, ch.q_d * rot / self.s_qd
        i_db = 10.0 * np.log10(max(last_I, 1e-12) / self.I_th) / 10.0
        return np.concatenate([
            a.real, a.imag, b.real, b.imag,
            [hd.real, hd.imag, qd.real, qd.imag],
            prev_u,
            [last_rate / 10.0, np.clip(i_db, -3.0, 3.0)],
        ]).astype(np.float32)


class ActionMap:
    """Action vector -> Decision.

    param == "linear"   (spec):  u in [-1, 1]^N,   theta_n = pi (u_n + 1)
    param == "circular" (E05) :  u = [c_1, s_1, ..., c_N, s_N], theta_n = atan2(s_n, c_n).
                                 atan2 is scale-invariant, so the tanh-bounded pair needs no
                                 projection onto the unit circle: every direction in the plane
                                 is still reachable.  There is no boundary, so the actor cannot
                                 saturate against the wrap-around at 0 = 2*pi, and the critic is
                                 periodic in theta by construction.
    param == "residual" (E09):  same (cos, sin) head, but the decoded angle is a correction
                                 on top of the analytic co-phasing solution,
                                 theta_n = theta_n^cf + atan2(s_n, c_n): the agent only has to learn
                                 the PU-aware deviation (physics-informed residual policy).
    With power_mode == "learn" one extra entry p in [-1, 1] gives P = P_max (p + 1) / 2.
    """

    def __init__(self, cfg: Config, N: int, power_mode: str, param: str = "linear"):
        if power_mode not in ("project", "learn"):
            raise ValueError(power_mode)
        if param not in ("linear", "circular", "residual"):
            raise ValueError(param)
        self.N, self.power_mode, self.param = N, power_mode, param
        self.P_max, self.I_th = cfg.P_max_W, cfg.I_th_norm
        self.n_phase = N if param == "linear" else 2 * N           # circular and residual: (cos, sin) pairs
        self.n_pairs = 0 if param == "linear" else N
        self.dim = self.n_phase + (power_mode == "learn")

    def phases(self, u: np.ndarray, ch: ChannelState | None = None) -> np.ndarray:
        if self.param == "linear":
            return (np.pi * (np.clip(u[: self.N], -1.0, 1.0) + 1.0)) % (2 * np.pi)
        ang = np.arctan2(u[1: self.n_phase: 2], u[0: self.n_phase: 2])
        if self.param == "residual":
            ang = ang + (np.angle(ch.h_d) - np.angle(ch.a))
        return ang % (2 * np.pi)

    def decision(self, ch: ChannelState, u: np.ndarray) -> Decision:
        u = np.asarray(u, dtype=float)
        theta = self.phases(u, ch)
        if self.power_mode == "project":
            qg = abs(ch.q_d + ch.b @ np.exp(1j * theta)) ** 2
            P = project_power(qg, self.P_max, self.I_th)
        else:
            P = self.P_max * (np.clip(u[self.n_phase], -1.0, 1.0) + 1.0) / 2.0
        return Decision(theta, float(P))


def penalty(I: float, I_th: float, kind: str) -> float:
    if kind == "linear":                                   # spec: max(0, (I - I_th) / I_th)
        return max(0.0, (I - I_th) / I_th)
    if kind == "db":                                       # bounded alternative: dB excess / 10
        return max(0.0, 10.0 * np.log10(max(I, 1e-300) / I_th)) / 10.0
    raise ValueError(kind)
