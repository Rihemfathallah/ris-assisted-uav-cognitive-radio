"""Geometry, path loss, array responses, Rician fading and UAV trajectory.

Conventions (documented once, used everywhere)
----------------------------------------------
* Array responses use *direction cosines*: for a half-wavelength ULA along the
  unit axis ``e``, ``a(u)[n] = exp(j*pi*n*(u.e))`` with ``u`` the unit vector
  pointing from the array towards the other node.  This is the spec's
  ``exp(j*pi*n*sin(phi))`` with ``sin(phi) = u.e`` (phi measured from broadside).
* RIS: uniform planar array in the y-z plane (wall facing the x direction).
  Horizontal axis = y (``nh`` elements), vertical axis = z (``nv`` elements).
  Element index ``n = iv * nh + ih``, i.e. ``a_RIS = kron(a_z, a_y)``.
  Elements are isotropic: no element pattern and no half-space blocking is
  modelled (same simplification as Wu & Zhang, 2019).  See LOG.md, E01.
* UAV array (Phase 4, M > 1): ULA along the global x-axis, fixed heading.
* LoS terms include the propagation phase ``exp(-j*2*pi*d/lambda)``, so the
  LoS component rotates as the UAV moves (sub-wavelength motion per step).
* Fading: links with a moving end (UAV-RIS, UAV-SU, UAV-PU) evolve with a
  Gauss-Markov model of coefficient rho; links between static nodes (RIS-SU,
  RIS-PU) are block-faded: drawn once per episode and then held constant.
* Everything returned by :class:`ChannelGenerator` is divided by sigma.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, db2lin, rng
from .types import ChannelState

TRAIN_EPISODE_SEED_OFFSET = 1_000_000  # training episode seeds live far away from test seeds


# ----------------------------------------------------------------------------- helpers
def crandn(gen: np.random.Generator, *shape) -> np.ndarray:
    """Circularly-symmetric complex Gaussian CN(0, 1)."""
    return (gen.standard_normal(shape) + 1j * gen.standard_normal(shape)) / np.sqrt(2.0)


def ula(n: int, u_axis: float) -> np.ndarray:
    return np.exp(1j * np.pi * np.arange(n) * u_axis)


def upa_yz(nh: int, nv: int, u: np.ndarray) -> np.ndarray:
    return np.kron(ula(nv, u[2]), ula(nh, u[1]))


def unit(v: np.ndarray) -> tuple[np.ndarray, float]:
    d = float(np.linalg.norm(v))
    return v / d, d


def path_gain(cfg: Config, d: float, link: str) -> float:
    """Average power gain PL(d) = C0 * d^-alpha (linear), plus blockage on direct ground links."""
    alpha = cfg.alpha[link]
    g = db2lin(cfg.C0_dB) * d ** (-alpha)
    if link == "uav_ground":
        g *= db2lin(-cfg.blockage_dB)
    return float(g)


def rician(pl: float, K_dB: float, los, nlos):
    K = db2lin(K_dB)
    return np.sqrt(pl) * (np.sqrt(K / (K + 1.0)) * los + np.sqrt(1.0 / (K + 1.0)) * nlos)


def uav_position(cfg: Config, psi: float) -> np.ndarray:
    R = cfg.uav["radius_m"]
    return np.array([R * np.cos(psi), R * np.sin(psi), cfg.uav["height_m"]])


# ----------------------------------------------------------------------------- generator
@dataclass
class Links:
    """Raw (normalised) links for a UAV with M antennas and an N-element RIS."""
    G: np.ndarray    # (N, M) UAV -> RIS
    h_d: np.ndarray  # (M,)   UAV -> SU
    q_d: np.ndarray  # (M,)   UAV -> PU
    h_r: np.ndarray  # (N,)   RIS -> SU
    q_r: np.ndarray  # (N,)   RIS -> PU


class ChannelGenerator:
    def __init__(self, cfg: Config, M: int | None = None):
        self.cfg = cfg
        self.N = int(cfg.N)
        self.nh, self.nv = cfg.ris_shape
        self.M = int(M if M is not None else cfg.M)
        self.lam = cfg.wavelength_m
        self.sigma = cfg.sigma
        self.rho = float(cfg.rho_fading)
        self.dpsi = cfg.uav["speed_mps"] * cfg.dt / cfg.uav["radius_m"]  # rad per step (CCW)
        self.p_ris, self.p_su, self.p_pu = cfg.pos("ris"), cfg.pos("su"), cfg.pos("pu")
        # static LoS parts of the RIS-ground links (never change)
        self._hr_los = self._los_ris_to(self.p_su)
        self._qr_los = self._los_ris_to(self.p_pu)
        self._d_rs = np.linalg.norm(self.p_su - self.p_ris)
        self._d_rp = np.linalg.norm(self.p_pu - self.p_ris)

    # -- LoS building blocks --------------------------------------------------------
    def _los_ris_to(self, p: np.ndarray) -> np.ndarray:
        u, d = unit(p - self.p_ris)
        return np.exp(-2j * np.pi * d / self.lam) * upa_yz(self.nh, self.nv, u)

    def _uav_array(self, u: np.ndarray) -> np.ndarray:
        return ula(self.M, u[0])

    # -- episode control ----------------------------------------------------------------
    def new_episode(self, seed: int, episode: int = 0) -> None:
        g_traj = rng(seed, "trajectory", episode)
        self._fad_uav = rng(seed, "fading_uav", episode)
        g_static = rng(seed, "fading_static", episode)
        self.psi = float(g_traj.uniform(0.0, 2.0 * np.pi))
        self.t = 0
        N, M = self.N, self.M
        self._G_nl = crandn(self._fad_uav, N, M)
        self._hd_nl = crandn(self._fad_uav, M)
        self._qd_nl = crandn(self._fad_uav, M)
        self._hr_nl = crandn(g_static, N)
        self._qr_nl = crandn(g_static, N)

    def step(self) -> None:
        self.psi += self.dpsi
        self.t += 1
        r, s = self.rho, np.sqrt(1.0 - self.rho**2)
        f = self._fad_uav
        self._G_nl = r * self._G_nl + s * crandn(f, self.N, self.M)
        self._hd_nl = r * self._hd_nl + s * crandn(f, self.M)
        self._qd_nl = r * self._qd_nl + s * crandn(f, self.M)

    @property
    def uav_pos(self) -> np.ndarray:
        return uav_position(self.cfg, self.psi)

    # -- channels -------------------------------------------------------------------------
    def links(self) -> Links:
        cfg, p_u = self.cfg, self.uav_pos
        u_ur, d_ur = unit(p_u - self.p_ris)              # RIS -> UAV direction
        G_los = np.exp(-2j * np.pi * d_ur / self.lam) * np.outer(
            upa_yz(self.nh, self.nv, u_ur), self._uav_array(-u_ur)
        )
        G = rician(path_gain(cfg, d_ur, "uav_ris"), cfg.K_dB["uav_ris"], G_los, self._G_nl)

        def direct(p_ground, nl):
            u, d = unit(p_ground - p_u)
            los = np.exp(-2j * np.pi * d / self.lam) * self._uav_array(u)
            return rician(path_gain(cfg, d, "uav_ground"), cfg.K_dB["uav_ground"], los, nl)

        h_d = direct(self.p_su, self._hd_nl)
        q_d = direct(self.p_pu, self._qd_nl)
        h_r = rician(path_gain(cfg, self._d_rs, "ris_ground"), cfg.K_dB["ris_ground"], self._hr_los, self._hr_nl)
        q_r = rician(path_gain(cfg, self._d_rp, "ris_ground"), cfg.K_dB["ris_ground"], self._qr_los, self._qr_nl)
        s = self.sigma
        return Links(G / s, h_d / s, q_d / s, h_r / s, q_r / s)

    def state(self) -> ChannelState:
        """Sufficient statistics for the single-antenna problem (uses antenna 0 if M > 1)."""
        L = self.links()
        g = L.G[:, 0]
        # NB: each of G and h_r is divided by sigma, so the cascade carries 1/sigma^2.
        # Multiply one sigma back so that |a|^2 is in noise units like |h_d|^2.
        a = L.h_r * g * self.sigma
        b = L.q_r * g * self.sigma
        return ChannelState(self.t, self.uav_pos.copy(), complex(L.h_d[0]), complex(L.q_d[0]), a, b)


# ----------------------------------------------------------------------------- episodes
def generate_episode(cfg: Config, seed: int, episode: int = 0, T: int | None = None) -> dict[str, np.ndarray]:
    """One flight segment as arrays (used for the frozen evaluation bank)."""
    T = int(T or cfg.episode_len)
    gen = ChannelGenerator(cfg)
    gen.new_episode(seed, episode)
    N = gen.N
    out = {
        "h_d": np.empty(T, complex), "q_d": np.empty(T, complex),
        "a": np.empty((T, N), complex), "b": np.empty((T, N), complex),
        "uav_pos": np.empty((T, 3)),
    }
    for t in range(T):
        st = gen.state()
        out["h_d"][t], out["q_d"][t], out["a"][t], out["b"][t], out["uav_pos"][t] = st.h_d, st.q_d, st.a, st.b, st.uav_pos
        gen.step()
    return out


def reference_gains(cfg: Config, uav_pos: np.ndarray | None = None) -> dict[str, float]:
    """Average path gains (linear, before normalisation) at a UAV position; default: circle centre."""
    p_u = np.array([0.0, 0.0, cfg.uav["height_m"]]) if uav_pos is None else np.asarray(uav_pos, float)
    p_r, p_s, p_p = cfg.pos("ris"), cfg.pos("su"), cfg.pos("pu")
    d = lambda x, y: float(np.linalg.norm(x - y))
    return {
        "uav_ris": path_gain(cfg, d(p_u, p_r), "uav_ris"),
        "ris_su": path_gain(cfg, d(p_r, p_s), "ris_ground"),
        "ris_pu": path_gain(cfg, d(p_r, p_p), "ris_ground"),
        "uav_su": path_gain(cfg, d(p_u, p_s), "uav_ground"),
        "uav_pu": path_gain(cfg, d(p_u, p_p), "uav_ground"),
    }
