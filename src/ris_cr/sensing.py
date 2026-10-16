"""Phase 4 — integrated sensing and communication with an M-antenna UAV (time sharing).

Communication (per frame, fixed during the frame)
  1. MRT on the direct link, reduce the MISO problem to the single-antenna form
     (h_d -> h_d^T w, a -> h_r * (G w)), run the element-wise RIS optimiser on it;
  2. re-optimise the beam for the resulting effective channels h_eff, q_eff with a
     1-D search over w(alpha) ~ (1 - alpha) w_MRT + alpha w_ZF, where w_ZF nulls the
     PU; power by projection.  With M = 8 the UAV can null the PU spatially, which
     turns the interference-limited single-antenna problem into a power-limited one.
Sensing slots
  * probing x[l] ~ CN(0, P_s/M I) ("isotropic") or confined to the PU null space
    ("null"); P_s obeys the same interference constraint as communication;
  * monostatic LoS echo y[l] = beta a(phi) a(phi)^H x[l] + n[l], |beta|^2 = C_r d^-4,
    C_r = lambda^2 RCS / (4 pi)^3 (radar equation, isotropic antennas);
  * MUSIC on a 0.1 deg grid, noise subspace of dimension M - 1;
  * CRB: deterministic (unknown-waveform) bound for one source,
        var(phi) >= 6 / (pi^2 cos^2 phi  M (M^2 - 1)  sum_l |s_l|^2),  s_l = beta a^H x[l]
    (noise-normalised).  MUSIC ignores the known probing signal, so this is the
    matching bound; an estimator exploiting x[l] could do better.
Localisation: the ULA (along x) measures u_x = sin(phi).  With the UAV position and
the true range known (spec assumption), and the target on the ground plane, the
position follows up to the sign of the y-offset, taken from the known target sector.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .channel import ChannelGenerator, crandn, ula, unit
from .config import ROOT, Config, load_config, rng
from .controllers.elementwise import ElementWise
from .types import ChannelState

CSV = ROOT / "results" / "csv"


# ------------------------------------------------------------------------ communication
def comm_design(cfg: Config, gen: ChannelGenerator, n_alpha: int = 51):
    """Returns (rate, q_eff, e) for the current MISO channel of ``gen``."""
    L = gen.links()
    s = gen.sigma
    P_max, I_th = cfg.P_max_W, cfg.I_th_norm
    w = np.conj(L.h_d) / np.linalg.norm(L.h_d)
    Gw = L.G @ w
    red = ChannelState(0, gen.uav_pos, complex(L.h_d @ w), complex(L.q_d @ w), L.h_r * Gw * s, L.q_r * Gw * s)
    theta = ElementWise(cfg).decide(red).theta
    e = np.exp(1j * theta)
    h_eff = L.h_d + (L.G.T @ (L.h_r * e)) * s       # (M,)
    q_eff = L.q_d + (L.G.T @ (L.q_r * e)) * s
    w_mrt = np.conj(h_eff) / np.linalg.norm(h_eff)
    qn = q_eff / np.linalg.norm(q_eff)
    w_zf = w_mrt - np.conj(qn) * (qn @ w_mrt)       # q_eff^T w_zf = 0
    w_zf /= np.linalg.norm(w_zf)
    al = np.linspace(0.0, 1.0, n_alpha)[:, None]
    W = (1 - al) * w_mrt + al * w_zf
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    gh, gq = np.abs(W @ h_eff) ** 2, np.abs(W @ q_eff) ** 2
    with np.errstate(divide="ignore"):
        P = np.minimum(P_max, I_th / gq)
    k = int(np.argmax(P * gh))
    return float(np.log2(1 + P[k] * gh[k])), q_eff, e


def sensing_power(cfg: Config, q_eff: np.ndarray, M: int, probing: str) -> float:
    if probing == "pu_null":
        return cfg.P_max_W
    return float(min(cfg.P_max_W, cfg.I_th_norm * M / np.sum(np.abs(q_eff) ** 2)))


def probing_signals(g, M, Ls, P_s, q_eff, probing):
    z = crandn(g, M, Ls)
    if probing == "isotropic":
        return np.sqrt(P_s / M) * z
    qn = q_eff / np.linalg.norm(q_eff)
    Pperp = np.eye(M) - np.outer(np.conj(qn), qn)   # q^T Pperp = 0
    return np.sqrt(P_s / (M - 1)) * (Pperp @ z)


# ------------------------------------------------------------------------ estimation
class Music:
    def __init__(self, M: int, grid_deg: float = 0.1):
        self.M = M
        self.grid = np.deg2rad(np.arange(-90.0, 90.0 + 1e-9, grid_deg))
        self.A = np.exp(1j * np.pi * np.arange(M)[:, None] * np.sin(self.grid)[None, :])

    def estimate(self, Y: np.ndarray) -> float:
        R = Y @ Y.conj().T / Y.shape[1]
        _, V = np.linalg.eigh(R)
        En = V[:, : self.M - 1]                     # one source -> M - 1 noise eigenvectors
        spec = 1.0 / np.sum(np.abs(En.conj().T @ self.A) ** 2, axis=0)
        return float(self.grid[int(np.argmax(spec))])


def crb_phi(phi: float, M: int, sum_s2: float) -> float:
    return 6.0 / (np.pi**2 * np.cos(phi) ** 2 * M * (M**2 - 1) * sum_s2)


def echo(g, beta, phi, X, noise=None):
    a = ula(X.shape[0], np.sin(phi))
    s = beta * (a.conj() @ X)                       # (Ls,)
    return np.outer(a, s) + (crandn(g, *X.shape) if noise is None else noise), s


def localize(phi_hat: float, p_uav: np.ndarray, p_tgt: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(p_tgt - p_uav)
    dx = d * np.sin(phi_hat)
    dy2 = d**2 - p_uav[2] ** 2 - dx**2
    dy = np.sign(p_tgt[1] - p_uav[1]) * np.sqrt(max(dy2, 0.0))
    return np.array([p_uav[0] + dx, p_uav[1] + dy, 0.0])


def target_geometry(cfg: Config, p_uav: np.ndarray):
    u, d = unit(cfg.pos("target") - p_uav)
    return float(np.arcsin(u[0])), d                 # array along x: sin(phi) = u_x


# ------------------------------------------------------------------------ experiments
def exp_rmse_vs_snr(cfg: Config, snr_db=np.arange(-20, 21, 2.5), Ls_list=(5, 8, 10, 20, 50), trials=None):
    """Fig. 7: angle RMSE of MUSIC vs per-antenna echo SNR (fixed geometry, UAV at psi = 0)."""
    M = int(cfg.sensing["M"])
    trials = int(trials or cfg.sensing["mc_trials"])
    mu = Music(M, cfg.sensing["grid_deg"])
    p_u = np.array([cfg.uav["radius_m"], 0.0, cfg.uav["height_m"]])
    phi, _ = target_geometry(cfg, p_u)
    rows = []
    for Ls in Ls_list:
        for snr in snr_db:
            g = rng(cfg.seed, "sensing", 7, Ls, int(snr * 10) + 1000)
            beta_abs = np.sqrt(10 ** (snr / 10))    # P_s = 1 normalisation: SNR = |beta|^2 E|a^H x|^2
            err, crb = [], []
            for _ in range(trials):
                X = crandn(g, M, Ls) / np.sqrt(M)   # E|a^H x|^2 = 1
                Y, s = echo(g, beta_abs * np.exp(2j * np.pi * g.uniform()), phi, X)
                err.append(mu.estimate(Y) - phi)
                crb.append(crb_phi(phi, M, np.sum(np.abs(s) ** 2)))
            rows.append({"Ls": Ls, "snr_db": snr, "rmse_deg": np.rad2deg(np.sqrt(np.mean(np.square(err)))),
                         "crb_deg": np.rad2deg(np.sqrt(np.mean(crb))), "trials": trials})
    return pd.DataFrame(rows)


def exp_tradeoff(cfg: Config, L: int, probing: str, trials=None):
    """Fig. 8: (1 - rho) R vs localisation RMSE over rho; common random numbers across rho."""
    c8 = cfg.with_(M=int(cfg.sensing["M"]))
    M = int(c8.M)
    trials = int(trials or cfg.sensing["mc_trials"])
    mu = Music(M, cfg.sensing["grid_deg"])
    gen = ChannelGenerator(c8, M=M)
    rho_grid = list(cfg.sensing["rho_grid"])
    R, pos_err = np.zeros(trials), {r: [] for r in rho_grid}
    ang_err, snr_echo = {r: [] for r in rho_grid}, []
    for k in range(trials):
        gen.new_episode(300_000 + k)                 # random position on the circle + fresh fading
        R[k], q_eff, _ = comm_design(c8, gen)
        p_u, p_t = gen.uav_pos, cfg.pos("target")
        phi, d = target_geometry(cfg, p_u)
        beta = np.sqrt(cfg.C_r * d**-4 / cfg.sigma2_W) * np.exp(2j * np.pi * rng(k, "sensing", 1).uniform())
        P_s = sensing_power(c8, q_eff, M, probing)
        snr_echo.append(abs(beta) ** 2 * P_s)
        # common random numbers: one probing/noise sequence per trial, nested across rho
        # (the rho = 0.10 frame uses the first 10 snapshots of the rho = 0.50 frame, ...)
        Lmax = int(round(max(rho_grid) * L))
        g = rng(k, "sensing", 2, L)
        X_full = probing_signals(g, M, Lmax, P_s, q_eff, probing)
        N_full = crandn(g, M, Lmax)
        for r in rho_grid:
            Ls = int(round(r * L))
            Y, _ = echo(g, beta, phi, X_full[:, :Ls], N_full[:, :Ls])
            ph = mu.estimate(Y)
            ang_err[r].append(ph - phi)
            pos_err[r].append(np.linalg.norm(localize(ph, p_u, p_t) - p_t))
    rows = [{"L": L, "probing": probing, "rho": r, "Ls": int(round(r * L)),
             "rate_eff": (1 - r) * R.mean(), "rate_full": R.mean(),
             "angle_rmse_deg": np.rad2deg(np.sqrt(np.mean(np.square(ang_err[r])))),
             "loc_rmse_m": np.sqrt(np.mean(np.square(pos_err[r]))),
             "loc_median_m": np.median(pos_err[r]), "loc_p90_m": np.percentile(pos_err[r], 90),
             "loc_outage_5m": np.mean(np.array(pos_err[r]) > 5.0),
             "echo_snr_db_median": 10 * np.log10(np.median(snr_echo)), "trials": trials} for r in rho_grid]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["rmse", "tradeoff"], choices=["rmse", "tradeoff"])
    ap.add_argument("--trials", type=int)
    a = ap.parse_args()
    cfg = load_config()
    CSV.mkdir(parents=True, exist_ok=True)
    if "rmse" in a.part:
        t0 = time.perf_counter()
        exp_rmse_vs_snr(cfg, trials=a.trials).to_csv(CSV / "fig7_music_rmse_vs_snr.csv", index=False)
        print(f"fig7 done in {time.perf_counter() - t0:.0f}s", flush=True)
    if "tradeoff" in a.part:
        frames = []
        for L, probing in ((100, "isotropic"), (200, "isotropic"), (100, "pu_null")):
            t0 = time.perf_counter()
            frames.append(exp_tradeoff(cfg, L, probing, a.trials))
            print(f"fig8 L={L} {probing} done in {time.perf_counter() - t0:.0f}s", flush=True)
        pd.concat(frames).to_csv(CSV / "fig8_isac_tradeoff.csv", index=False)


if __name__ == "__main__":
    main()
