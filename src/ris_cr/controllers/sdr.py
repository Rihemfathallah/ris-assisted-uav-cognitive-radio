"""Semidefinite relaxation of (P1).

Joint formulation (used for all reported results)
-------------------------------------------------
With v = [e; 1] and W = P v v^H every diagonal entry of W equals P because
|v_n| = 1.  Problem (P1) is therefore *exactly*

    max_W  tr(R_SU W)   s.t.  tr(R_PU W) <= I_th,  W_nn = W_{N+1,N+1} <= P_max,
                              W >= 0,  rank(W) = 1,

and dropping the rank constraint gives a single convex SDP whose optimum is an
upper bound on (P1).  No alternation over P is needed.  We solve it in units of
P_max (so diag(W~) <= 1) with the R matrices scaled to O(1) for conditioning,
and compile it once with CVXPY parameters (DPP) so each decision only pays for
the numerical solve.

Rounding: principal eigenvector + K Gaussian randomisations, each mapped to
unit-modulus phases e_n = exp(j arg(v_n / v_{N+1})); power by projection (so
every candidate is feasible); the best rate wins.

``SDRAlternating`` implements the specification's procedure (fixed-P SDR,
power update, repeat) and exists to cross-validate the joint formulation.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

from ..config import Config, rng
from ..system import closed_form_phases, project_power, quad_forms
from ..types import ChannelState, Decision
from .baselines import _Base


def _solve(prob: cp.Problem, solver: str) -> bool:
    for s in (solver, "SCS"):
        try:
            prob.solve(solver=s, **({"eps": 1e-7, "max_iters": 20000} if s == "SCS" else {}))
            if prob.status in ("optimal", "optimal_inaccurate"):
                return True
        except cp.error.SolverError:
            pass
    return False


def gaussian_rounding(W: np.ndarray, ch: ChannelState, P_max: float, I_th: float,
                      K: int, gen: np.random.Generator) -> tuple[np.ndarray, float]:
    """Return (theta, P) of the best feasible candidate among eigvec + K randomisations."""
    N = ch.N
    lam, U = np.linalg.eigh((W + W.conj().T) / 2)
    lam = np.clip(lam, 0.0, None)
    principal = (U[:, -1] * np.sqrt(lam[-1]))[:, None]              # deterministic candidate
    if K > 0:
        r = (gen.standard_normal((N + 1, K)) + 1j * gen.standard_normal((N + 1, K))) / np.sqrt(2)
        V = np.hstack([principal, (U * np.sqrt(lam)) @ r])          # (N+1, K+1)
    else:
        V = principal
    theta = np.angle(V[:N, :] / V[N:N + 1, :]).T % (2 * np.pi)   # (K+1, N)
    E = np.exp(1j * theta)
    h = ch.h_d + E @ ch.a
    q = ch.q_d + E @ ch.b
    with np.errstate(divide="ignore"):
        P = np.minimum(P_max, I_th / np.abs(q) ** 2)
    k = int(np.argmax(P * np.abs(h) ** 2))
    return theta[k], float(P[k])


class SDR(_Base):
    name = "sdr"

    def __init__(self, cfg: Config, constrained: bool = True, randomizations: int | None = None,
                 solver: str | None = None, seed: int = 0):
        super().__init__(cfg, constrained)
        self.K = int(cfg.sdr["randomizations"] if randomizations is None else randomizations)
        self.solver = solver or cfg.sdr["solver"]
        self._gen = rng(seed, "sdr_randomization")
        self._N = None
        self.failures = 0
        self.last_upper_bound = np.nan   # rate upper bound log2(1 + tr(R_SU W*))
        self.last_rank_ratio = np.nan    # lambda_max / trace  (1.0 = tight relaxation)

    def _build(self, N: int) -> None:
        n = N + 1
        self._W = cp.Variable((n, n), hermitian=True)
        self._Rsu = cp.Parameter((n, n), hermitian=True)
        self._Rpu = cp.Parameter((n, n), hermitian=True)
        self._thr = cp.Parameter(nonneg=True)
        d = cp.real(cp.diag(self._W))
        cons = [self._W >> 0, d[:N] == d[N], d[N] <= 1.0]
        if np.isfinite(self.I_th):
            cons.append(cp.real(cp.trace(self._Rpu @ self._W)) <= self._thr)
        self._prob = cp.Problem(cp.Maximize(cp.real(cp.trace(self._Rsu @ self._W))), cons)
        assert self._prob.is_dcp(dpp=True)
        self._N = N

    def decide(self, ch: ChannelState) -> Decision:
        if self._N != ch.N:
            self._build(ch.N)
        R_su, R_pu = quad_forms(ch)
        c_su = float(np.max(np.abs(R_su)))
        c_pu = float(np.max(np.abs(R_pu)))
        self._Rsu.value = R_su / c_su
        self._Rpu.value = R_pu / c_pu
        if np.isfinite(self.I_th):
            self._thr.value = self.I_th / (self.P_max * c_pu)
        if not _solve(self._prob, self.solver):
            self.failures += 1
            th = closed_form_phases(ch)
            qg = abs(ch.q_d + ch.b @ np.exp(1j * th)) ** 2
            self.last_upper_bound = self.last_rank_ratio = np.nan
            return Decision(th, project_power(qg, self.P_max, self.I_th))
        W = self._W.value * self.P_max                      # back to watts
        self.last_upper_bound = float(np.log2(1.0 + np.real(np.trace(R_su @ W))))
        ev = np.linalg.eigvalsh((W + W.conj().T) / 2)
        self.last_rank_ratio = float(ev[-1] / max(np.sum(np.clip(ev, 0, None)), 1e-300))
        theta, P = gaussian_rounding(W, ch, self.P_max, self.I_th, self.K, self._gen)
        return Decision(theta, P)


class SDRAlternating(_Base):
    """The specification's procedure (Sec. 5.5): fixed-P SDR, then P update, repeated."""
    name = "sdr_alternating"

    def __init__(self, cfg: Config, iters: int | None = None, randomizations: int | None = None, seed: int = 0):
        super().__init__(cfg, True)
        self.iters = int(iters or cfg.sdr["alternating_iters"])
        self.K = int(cfg.sdr["randomizations"] if randomizations is None else randomizations)
        self._gen = rng(seed, "sdr_randomization")

    def decide(self, ch: ChannelState) -> Decision:
        N = ch.N
        R_su, R_pu = quad_forms(ch)
        c_su, c_pu = float(np.max(np.abs(R_su))), float(np.max(np.abs(R_pu)))
        # Starting at P = P_max (as in the spec) can make the first fixed-P SDP infeasible when
        # the RIS cannot cancel enough of the direct PU path; start from the power that makes the
        # co-phasing solution feasible instead, so iteration 1 always has a feasible point (LOG E03).
        th0 = closed_form_phases(ch)
        P = project_power(abs(ch.q_d + ch.b @ np.exp(1j * th0)) ** 2, self.P_max, self.I_th)
        V_best, obj_best = None, -np.inf
        for _ in range(self.iters):
            V = cp.Variable((N + 1, N + 1), hermitian=True)
            cons = [V >> 0, cp.real(cp.diag(V)) == 1,
                    P * c_pu * cp.real(cp.trace((R_pu / c_pu) @ V)) <= self.I_th]
            prob = cp.Problem(cp.Maximize(cp.real(cp.trace((R_su / c_su) @ V))), cons)
            if not _solve(prob, "CLARABEL"):
                break
            Vv = V.value
            obj = P * np.real(np.trace(R_su @ Vv))
            if obj > obj_best:
                V_best, obj_best = Vv, obj
            P_new = project_power(float(np.real(np.trace(R_pu @ Vv))), self.P_max, self.I_th)
            if abs(P_new - P) <= 1e-9 * self.P_max:
                break
            P = P_new
        if V_best is None:
            raise RuntimeError("alternating SDR failed at the first iteration")
        theta, Pr = gaussian_rounding(V_best, ch, self.P_max, self.I_th, self.K, self._gen)
        return Decision(theta, Pr)
