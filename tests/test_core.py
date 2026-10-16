"""Correctness tests for the physics and the optimisation baselines.

Run:  pytest -q
"""
import warnings

import numpy as np
import pytest
from scipy.special import j0

from ris_cr import system as S
from ris_cr.channel import TRAIN_EPISODE_SEED_OFFSET, ChannelGenerator, generate_episode, path_gain
from ris_cr.config import load_config, rng
from ris_cr.controllers.baselines import ClosedForm, NoRIS
from ris_cr.controllers.elementwise import ElementWise
from ris_cr.controllers.sdr import SDR, SDRAlternating
from ris_cr.types import Decision

warnings.filterwarnings("ignore", message="Solution may be inaccurate")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _states(cfg, n=5, seed=123, N=None):
    c = cfg.with_(N=N) if N else cfg
    gen = ChannelGenerator(c)
    out = []
    for e in range(n):
        gen.new_episode(seed, e)
        for _ in range(7):
            gen.step()
        out.append(gen.state())
    return out


# ------------------------------------------------------------------ Phase 1: closed form
@pytest.mark.parametrize("N", [8, 16, 32, 64])
def test_closed_form_matches_analytic_bound_and_beats_random(cfg, N):
    ch = _states(cfg, 1, N=N)[0]
    P = cfg.P_max_W
    th = S.closed_form_phases(ch)
    snr_cf = S.snr(ch, Decision(th, P))
    assert snr_cf == pytest.approx(S.closed_form_snr_bound(ch, P), rel=1e-12)
    g = rng(0, "baseline_random")
    rand = g.uniform(0, 2 * np.pi, (10_000, N))
    snr_rand = P * np.abs(ch.h_d + np.exp(1j * rand) @ ch.a) ** 2
    assert snr_cf >= snr_rand.max()
    # and the bound is essentially unattainable by chance (sanity on the test itself)
    assert snr_rand.max() < snr_cf


def test_quadratic_form_identity(cfg):
    """v^H R v = |h_d + a^T e|^2 — catches the classic misplaced-conjugate bug in the SDR."""
    for ch in _states(cfg, 5):
        R_su, R_pu = S.quad_forms(ch)
        for th in rng(1, "baseline_random").uniform(0, 2 * np.pi, (20, ch.N)):
            v = np.append(np.exp(1j * th), 1.0)
            assert np.real(v.conj() @ R_su @ v) == pytest.approx(abs(ch.h_d + ch.a @ np.exp(1j * th)) ** 2, rel=1e-10)
            assert np.real(v.conj() @ R_pu @ v) == pytest.approx(abs(ch.q_d + ch.b @ np.exp(1j * th)) ** 2, rel=1e-10)
        assert np.allclose(R_su, R_su.conj().T) and np.allclose(R_pu, R_pu.conj().T)


def test_power_projection_never_violates(cfg):
    ch = _states(cfg, 1)[0]
    for th in rng(2, "baseline_random").uniform(0, 2 * np.pi, (10_000, ch.N))[::10]:
        d = S.projected_decision(ch, th, cfg.P_max_W, cfg.I_th_norm)
        assert not S.is_violation(S.interference(ch, d), cfg.I_th_norm)
        assert 0.0 <= d.P <= cfg.P_max_W


def test_projection_is_optimal_power_for_fixed_phases(cfg):
    ch = _states(cfg, 1)[0]
    th = S.closed_form_phases(ch)
    d = S.projected_decision(ch, th, cfg.P_max_W, cfg.I_th_norm)
    for P in np.linspace(0, cfg.P_max_W, 201):
        dd = Decision(th, P)
        if not S.is_violation(S.interference(ch, dd), cfg.I_th_norm):
            assert S.rate(ch, dd) <= S.rate(ch, d) + 1e-12


# ------------------------------------------------------------------ Phase 2: optimisation
def test_sdr_unconstrained_recovers_closed_form(cfg):
    sdr, cf = SDR(cfg, constrained=False), ClosedForm(cfg, constrained=False)
    for ch in _states(cfg, 3):
        r_sdr, r_cf = S.rate(ch, sdr.decide(ch)), S.rate(ch, cf.decide(ch))
        assert r_sdr == pytest.approx(r_cf, abs=1e-4)
        assert sdr.last_upper_bound >= r_sdr - 1e-6


def test_ordering_and_feasibility_constrained(cfg):
    ctrls = [NoRIS(cfg), ClosedForm(cfg), ElementWise(cfg), SDR(cfg)]
    for ch in _states(cfg, 5, seed=321):
        r = {}
        for c in ctrls:
            d = c.decide(ch)
            assert not S.is_violation(S.interference(ch, d), cfg.I_th_norm), c.name
            r[c.name] = S.rate(ch, d)
        assert r["elementwise"] >= r["closed_form"] - 1e-12      # monotone refinement
        assert r["sdr"] >= r["closed_form"] - 1e-3
        assert ctrls[-1].last_upper_bound >= r["sdr"] - 1e-4     # relaxation is a bound
        assert ctrls[-1].last_upper_bound >= r["elementwise"] - 1e-4


def test_joint_sdr_dominates_spec_alternating(cfg):
    joint, alt = SDR(cfg), SDRAlternating(cfg)
    for ch in _states(cfg, 3, seed=99):
        assert S.rate(ch, joint.decide(ch)) >= S.rate(ch, alt.decide(ch)) - 1e-3


# ------------------------------------------------------------------ channel model
def test_gauss_markov_keeps_unit_variance(cfg):
    gen = ChannelGenerator(cfg)
    gen.new_episode(5)
    x = []
    for _ in range(10_000):
        gen.step()
        x.append(gen._hd_nl[0])
    x = np.array(x)
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=0.1)
    # lag-1 autocorrelation equals rho
    ac = np.real(np.vdot(x[:-1], x[1:])) / np.real(np.vdot(x, x))
    assert ac == pytest.approx(cfg.rho_fading, abs=0.02)


def test_average_gain_matches_path_loss(cfg):
    """E|h_d|^2 = PL / sigma^2 (Rician normalisation has unit average power)."""
    gen = ChannelGenerator(cfg)
    vals, pls = [], []
    for e in range(400):
        gen.new_episode(77, e)
        ch = gen.state()
        d = np.linalg.norm(cfg.pos("su") - gen.uav_pos)
        vals.append(abs(ch.h_d) ** 2)
        pls.append(path_gain(cfg, d, "uav_ground") / cfg.sigma2_W)
    assert np.mean(vals) / np.mean(pls) == pytest.approx(1.0, rel=0.12)


def test_reproducible_and_trajectory_independent_of_N(cfg):
    e1 = generate_episode(cfg, 100, 0)
    e2 = generate_episode(cfg, 100, 0)
    for k in e1:
        assert np.array_equal(e1[k], e2[k])
    e3 = generate_episode(cfg.with_(N=64), 100, 0)
    assert np.array_equal(e1["uav_pos"], e3["uav_pos"])       # same flight, different RIS


def test_static_links_are_block_faded(cfg):
    gen = ChannelGenerator(cfg)
    gen.new_episode(1)
    hr0 = gen.links().h_r
    for _ in range(50):
        gen.step()
    assert np.allclose(gen.links().h_r, hr0)


def test_dt_is_jakes_consistent(cfg):
    assert j0(2 * np.pi * cfg.doppler_Hz * cfg.dt) == pytest.approx(cfg.rho_fading, abs=1e-9)


def test_train_and_test_seeds_disjoint(cfg):
    assert TRAIN_EPISODE_SEED_OFFSET > max(cfg.test_seeds + cfg.val_seeds)
