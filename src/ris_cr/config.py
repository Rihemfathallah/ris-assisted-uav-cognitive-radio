"""Configuration: YAML -> frozen dataclass with derived physical quantities.

Design notes
------------
* Everything downstream works in *noise-normalised* units: every channel is
  divided by sigma (so SNR = P * |h|^2 with P in watts), and the interference
  threshold becomes an interference-to-noise ratio ``I_th_norm``.
* Random streams are keyed by *name*, not by creation order, so adding a new
  stream never perturbs the existing ones (a classic source of "the results
  changed and I don't know why").
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import brentq
from scipy.special import j0

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

# Stable integer ids for named random streams (never renumber, only append).
_STREAM_IDS = {
    "trajectory": 1,
    "fading_uav": 2,       # links with a moving end (UAV-RIS, UAV-SU, UAV-PU)
    "fading_static": 3,    # RIS-SU, RIS-PU (both ends static -> block fading)
    "baseline_random": 4,
    "policy": 5,
    "exploration": 6,
    "sensing": 7,
    "sdr_randomization": 8,
    "octave_export": 9,
}


def db2lin(x_db):
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def lin2db(x):
    return 10.0 * np.log10(np.asarray(x, dtype=float))


def dbm2w(x_dbm):
    return db2lin(np.asarray(x_dbm, dtype=float) - 30.0)


def w2dbm(x_w):
    return lin2db(x_w) + 30.0


def rng(seed: int, stream: str, *extra: int) -> np.random.Generator:
    """Independent generator for (seed, stream[, extra...])."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), _STREAM_IDS[stream], *map(int, extra)]))


@dataclass(frozen=True)
class Config:
    raw: dict = field(repr=False)

    # --- convenient accessors -------------------------------------------------
    def __getattr__(self, item: str) -> Any:  # only called when normal lookup fails
        raw = object.__getattribute__(self, "raw")
        if item in raw:
            return raw[item]
        raise AttributeError(item)

    # --- noise / power -----------------------------------------------------------
    @property
    def sigma2_dBm(self) -> float:
        return -174.0 + 10.0 * np.log10(float(self.raw["B_Hz"])) + self.raw["NF_dB"]

    @property
    def sigma2_W(self) -> float:
        return float(dbm2w(self.sigma2_dBm))

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self.sigma2_W))

    @property
    def P_max_W(self) -> float:
        return float(dbm2w(self.raw["P_max_dBm"]))

    @property
    def I_th_norm(self) -> float:
        """Interference threshold in units of noise power (an INR)."""
        return float(dbm2w(self.raw["I_th_dBm"]) / self.sigma2_W)

    # --- propagation -----------------------------------------------------------------
    @property
    def wavelength_m(self) -> float:
        """C0 = (lambda / 4 pi)^2 (free-space reference at 1 m)."""
        return float(4.0 * np.pi * np.sqrt(db2lin(self.raw["C0_dB"])))

    @property
    def carrier_Hz(self) -> float:
        return 299_792_458.0 / self.wavelength_m

    @property
    def doppler_Hz(self) -> float:
        return self.raw["uav"]["speed_mps"] / self.wavelength_m

    @property
    def dt(self) -> float:
        """Control period. 'auto' makes the Gauss-Markov rho consistent with Jakes' model."""
        v = self.raw.get("dt_s", "auto")
        if v != "auto":
            return float(v)
        rho = self.raw["rho_fading"]
        x = brentq(lambda z: j0(z) - rho, 1e-9, 2.4048)  # first zero of J0 is 2.4048
        return float(x / (2.0 * np.pi * self.doppler_Hz))

    @property
    def ris_shape(self) -> tuple[int, int]:
        shapes = {int(k): tuple(v) for k, v in self.raw["ris_shapes"].items()}
        N = int(self.raw["N"])
        if N not in shapes:
            raise ValueError(f"No UPA layout configured for N={N}")
        nh, nv = shapes[N]
        assert nh * nv == N
        return nh, nv

    # --- sensing ---------------------------------------------------------------------
    @property
    def C_r(self) -> float:
        """Monostatic radar constant lambda^2 * RCS / (4 pi)^3 (isotropic elements)."""
        lam = self.wavelength_m
        return float(lam**2 * self.raw["sensing"]["rcs_m2"] / (4.0 * np.pi) ** 3)

    def pos(self, name: str) -> np.ndarray:
        return np.asarray(self.raw["positions_m"][name], dtype=float)

    def with_(self, **overrides) -> "Config":
        """Return a copy with (possibly dotted) keys overridden, e.g. with_(N=32, **{'td3.gamma': 0.99})."""
        raw = copy.deepcopy(self.raw)
        for key, val in overrides.items():
            node = raw
            parts = key.split(".")
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = val
        return Config(raw)

    def digest(self) -> str:
        return hashlib.sha256(yaml.safe_dump(self.raw, sort_keys=True).encode()).hexdigest()[:12]

    def summary(self) -> str:
        return (
            f"N={self.raw['N']} M={self.raw['M']} Pmax={self.raw['P_max_dBm']} dBm "
            f"Ith={self.raw['I_th_dBm']} dBm sigma2={self.sigma2_dBm:.1f} dBm "
            f"lambda={self.wavelength_m:.3f} m (fc={self.carrier_Hz/1e6:.0f} MHz) "
            f"dt={self.dt*1e3:.2f} ms blockage={self.raw['blockage_dB']} dB"
        )


def load_config(path: str | Path | None = None, **overrides) -> Config:
    with open(path or DEFAULT_CONFIG) as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw)
    return cfg.with_(**overrides) if overrides else cfg


def source_digest() -> str:
    """Hash of the source tree, stored with every run for provenance."""
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]
