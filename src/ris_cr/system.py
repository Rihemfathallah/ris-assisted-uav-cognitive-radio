"""System model (single-antenna UAV). Pure functions, noise-normalised units.

    h_eff = h_d + a^T e,     SNR = P |h_eff|^2,     R = log2(1 + SNR)
    q_eff = q_d + b^T e,     I   = P |q_eff|^2       (interference-to-noise ratio)

Key identity used everywhere (power projection): for fixed phases the
optimal feasible power of problem (P1) is

    P*(theta) = min(P_max, I_th / |q_eff(theta)|^2)

because R is increasing in P and the only other constraint is linear in P.
"""
from __future__ import annotations

import numpy as np

from .types import ChannelState, Decision

VIOLATION_RTOL = 1e-6  # relative slack before an interference sample counts as a violation


def reflection(theta: np.ndarray) -> np.ndarray:
    return np.exp(1j * np.asarray(theta))


def h_eff(ch: ChannelState, d: Decision) -> complex:
    return ch.h_d + (ch.a @ reflection(d.theta) if d.ris_active else 0.0)


def q_eff(ch: ChannelState, d: Decision) -> complex:
    return ch.q_d + (ch.b @ reflection(d.theta) if d.ris_active else 0.0)


def snr(ch: ChannelState, d: Decision) -> float:
    return float(d.P * abs(h_eff(ch, d)) ** 2)


def rate(ch: ChannelState, d: Decision) -> float:
    return float(np.log2(1.0 + snr(ch, d)))


def interference(ch: ChannelState, d: Decision) -> float:
    return float(d.P * abs(q_eff(ch, d)) ** 2)


def is_violation(I: float, I_th: float) -> bool:
    return bool(I > I_th * (1.0 + VIOLATION_RTOL))


def project_power(q_gain: float, P_max: float, I_th: float) -> float:
    """P* = min(P_max, I_th / |q_eff|^2) given the PU gain |q_eff|^2."""
    if q_gain <= 0.0:
        return float(P_max)
    return float(min(P_max, I_th / q_gain))


def projected_decision(ch: ChannelState, theta: np.ndarray, P_max: float, I_th: float,
                       ris_active: bool = True) -> Decision:
    d0 = Decision(np.asarray(theta, float) % (2 * np.pi), 0.0, ris_active)
    qg = abs(q_eff(ch, d0)) ** 2
    return Decision(d0.theta, project_power(qg, P_max, I_th), ris_active)


def quad_forms(ch: ChannelState) -> tuple[np.ndarray, np.ndarray]:
    """R_SU, R_PU such that |h_d + a^T e|^2 = v^H R_SU v with v = [e; 1] (spec, Sec. 5.5)."""
    def build(c: np.ndarray, d: complex) -> np.ndarray:
        N = c.shape[0]
        R = np.empty((N + 1, N + 1), complex)
        R[:N, :N] = np.outer(np.conj(c), c)
        R[:N, N] = np.conj(c) * d
        R[N, :N] = np.conj(d) * c
        R[N, N] = abs(d) ** 2
        return R
    return build(ch.a, ch.h_d), build(ch.b, ch.q_d)


def closed_form_phases(ch: ChannelState) -> np.ndarray:
    """SU-optimal phases without the PU constraint: align every path with h_d."""
    return (np.angle(ch.h_d) - np.angle(ch.a)) % (2 * np.pi)


def closed_form_snr_bound(ch: ChannelState, P: float) -> float:
    return float(P * (abs(ch.h_d) + np.sum(np.abs(ch.a))) ** 2)


# ------------------------------------------------------------------ vectorised helpers (evaluation)
def batch_rate(h_d, a, theta, P) -> np.ndarray:
    """Rates for a batch: h_d (B,), a (B,N), theta (B,N) or (N,), P (B,) or scalar."""
    h = h_d + np.sum(a * np.exp(1j * theta), axis=-1)
    return np.log2(1.0 + P * np.abs(h) ** 2)
