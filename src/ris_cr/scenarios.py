"""Frozen scenario bank.

Evaluation trajectories are generated once per (split, N) and stored as .npz.
Every controller (and the Octave cross-check) consumes exactly these arrays, so
"all methods on the same channel realisations" is a property of the code.

Trajectories are independent of N (separate RNG stream), so the sweep over N
sees the same flights with different RIS sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .channel import generate_episode
from .config import ROOT, Config
from .types import ChannelState

BANK_DIR = ROOT / "results" / "bank"
SPLITS = {"test": "test_seeds", "val": "val_seeds"}


@dataclass
class Bank:
    h_d: np.ndarray       # (E, T)
    q_d: np.ndarray       # (E, T)
    a: np.ndarray         # (E, T, N)
    b: np.ndarray         # (E, T, N)
    uav_pos: np.ndarray   # (E, T, 3)
    seeds: np.ndarray     # (E,) seed of each episode
    episode: np.ndarray   # (E,) episode index within its seed

    @property
    def shape(self) -> tuple[int, int, int]:
        E, T, N = self.a.shape
        return E, T, N

    def state(self, e: int, t: int) -> ChannelState:
        return ChannelState(t, self.uav_pos[e, t], complex(self.h_d[e, t]), complex(self.q_d[e, t]),
                            self.a[e, t], self.b[e, t])

    def episode_states(self, e: int, stride: int = 1):
        T = self.a.shape[1]
        return [self.state(e, t) for t in range(0, T, stride)]


def bank_path(split: str, N: int) -> Path:
    return BANK_DIR / f"{split}_N{N}.npz"


def build_bank(cfg: Config, split: str = "test", episodes_per_seed: int | None = None) -> Bank:
    seeds = cfg.raw[SPLITS[split]]
    eps = int(episodes_per_seed or (cfg.episodes_per_seed if split == "test" else cfg.td3["val_episodes"] // len(seeds)))
    parts, sd, ei = [], [], []
    for s in seeds:
        for e in range(eps):
            parts.append(generate_episode(cfg, s, e))
            sd.append(s)
            ei.append(e)
    stack = {k: np.stack([p[k] for p in parts]) for k in parts[0]}
    return Bank(**stack, seeds=np.array(sd), episode=np.array(ei))


def save_bank(bank: Bank, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bank.__dict__)


def load_bank(split: str, N: int, cfg: Config | None = None, rebuild: bool = False) -> Bank:
    p = bank_path(split, N)
    if rebuild or not p.exists():
        if cfg is None:
            raise FileNotFoundError(f"{p} missing and no config given to build it")
        bank = build_bank(cfg.with_(N=N), split)
        save_bank(bank, p)
        return bank
    with np.load(p) as z:
        return Bank(**{k: z[k] for k in z.files})
