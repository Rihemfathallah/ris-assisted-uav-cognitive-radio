import numpy as np
import pytest

from ris_cr.channel import ChannelGenerator, crandn, ula
from ris_cr.config import load_config
from ris_cr.sensing import Music, comm_design, crb_phi, localize, probing_signals, sensing_power, target_geometry


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_music_noiseless_recovers_angle_to_grid_resolution():
    M, mu = 8, Music(8, 0.1)
    g = np.random.default_rng(0)
    for phi in np.deg2rad([-60.0, -12.34, 0.0, 17.77, 45.0]):
        a = ula(M, np.sin(phi))
        Y = np.outer(a, crandn(g, 20)) + 1e-6 * crandn(g, M, 20)
        assert abs(np.rad2deg(mu.estimate(Y) - phi)) <= 0.05 + 1e-9


def test_crb_decreases_with_energy_and_array_size():
    assert crb_phi(0.3, 8, 100.0) < crb_phi(0.3, 8, 10.0)
    assert crb_phi(0.3, 16, 10.0) < crb_phi(0.3, 8, 10.0)


def test_localisation_is_exact_with_exact_angle(cfg):
    p_u = np.array([30.0, 0.0, 100.0])
    phi, _ = target_geometry(cfg, p_u)
    assert np.allclose(localize(phi, p_u, cfg.pos("target")), cfg.pos("target"), atol=1e-9)


def test_comm_design_feasible_and_null_probing_silent_at_pu(cfg):
    c8 = cfg.with_(M=8)
    gen = ChannelGenerator(c8, M=8)
    gen.new_episode(3)
    R, q_eff, _ = comm_design(c8, gen)
    assert np.isfinite(R) and R > 0
    X = probing_signals(np.random.default_rng(1), 8, 50, 1.0, q_eff, "pu_null")
    assert np.max(np.abs(q_eff @ X)) < 1e-9 * np.linalg.norm(q_eff)
    # isotropic probing respects the average interference constraint by construction
    P_s = sensing_power(c8, q_eff, 8, "isotropic")
    assert P_s / 8 * np.sum(np.abs(q_eff) ** 2) <= c8.I_th_norm * (1 + 1e-9)
