import dataclasses

import numpy as np
import pytest
import torch
from stable_baselines3.common.env_checker import check_env

from ris_cr import system as S
from ris_cr.config import load_config
from ris_cr.controllers.td3 import TD3Controller
from ris_cr.rl.core import ActionMap, ObsBuilder, penalty
from ris_cr.rl.env import RisCrEnv
from ris_cr.rl.sb3 import actor_forward, apply_residual_bias, build_model, saturated_fraction
from ris_cr.scenarios import build_bank

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def bank(cfg):
    return build_bank(cfg, "val", episodes_per_seed=1)


def _model(cfg, env, seed=0):
    """Small-buffer SB3 model: the tests only need the networks, never the replay buffer."""
    return build_model(env, cfg.td3, seed=seed, buffer_size=1)


# ----------------------------------------------------------------- environment contract
def test_env_passes_sb3_checker(cfg):
    check_env(RisCrEnv(cfg), skip_render_check=True)
    check_env(RisCrEnv(cfg, power_mode="learn"), skip_render_check=True)


def test_observation_size_matches_spec(cfg):
    lin = dict(action_param="linear")
    assert RisCrEnv(cfg, power_mode="learn", **lin).observation_space.shape[0] == 5 * 16 + 7   # 87, as in the spec
    assert RisCrEnv(cfg, power_mode="project", **lin).observation_space.shape[0] == 5 * 16 + 6
    assert RisCrEnv(cfg, power_mode="project", action_param="circular").observation_space.shape[0] == 6 * 16 + 6


def test_episode_length_and_projection_feasibility(cfg):
    env = RisCrEnv(cfg)
    env.reset(seed=3)
    n, viol = 0, 0
    while True:
        _, _, term, trunc, info = env.step(env.action_space.sample())
        n += 1
        viol += info["violation"]
        if term or trunc:
            break
    assert n == cfg.episode_len and viol == 0


def test_canonicalisation_is_lossless(cfg, bank):
    """Rotating every UAV-side channel by a common phase changes neither obs nor rates."""
    ob = ObsBuilder(cfg, 16, 16)
    ch = bank.state(0, 5)
    rot = np.exp(1j * 1.234)
    ch2 = dataclasses.replace(ch, h_d=ch.h_d * rot, q_d=ch.q_d * rot, a=ch.a * rot, b=ch.b * rot)
    u = np.zeros(16, np.float32)
    assert np.allclose(ob.build(ch, u, 1.0, 1.0), ob.build(ch2, u, 1.0, 1.0), atol=1e-6)
    th = np.linspace(0, 6, 16)
    d = S.projected_decision(ch, th, cfg.P_max_W, cfg.I_th_norm)
    d2 = S.projected_decision(ch2, th, cfg.P_max_W, cfg.I_th_norm)
    assert S.rate(ch, d) == pytest.approx(S.rate(ch2, d2), rel=1e-12)


def test_penalties(cfg):
    I_th = cfg.I_th_norm
    assert penalty(0.5 * I_th, I_th, "linear") == 0.0 == penalty(0.5 * I_th, I_th, "db")
    assert penalty(100 * I_th, I_th, "linear") == pytest.approx(99.0)
    assert penalty(100 * I_th, I_th, "db") == pytest.approx(2.0)


# ----------------------------------------------------------------- action decoding
def test_circular_head_decodes_angles_and_is_scale_invariant(cfg):
    """atan2 recovers the angle, and scaling a (cos, sin) pair does not change it.

    This is why the tanh-bounded SB3 actor needs no projection onto the unit circle.
    """
    am = ActionMap(cfg, 16, "project", "circular")
    th = np.linspace(0.1, 6.2, 16)
    uu = np.stack([np.cos(th), np.sin(th)], -1).reshape(-1)
    assert np.allclose(am.phases(uu), th)
    scaled = (np.stack([np.cos(th), np.sin(th)], -1) * np.linspace(0.05, 0.9, 16)[:, None]).reshape(-1)
    assert np.allclose(am.phases(scaled), th)


def test_residual_head_with_zero_correction_is_co_phasing(cfg, bank):
    ch = bank.state(0, 3)
    am = ActionMap(cfg, 16, "project", "residual")
    zero = np.tile([1.0, 0.0], 16)                                # atan2(0, 1) = 0 for every element
    assert np.allclose(am.phases(zero, ch), S.closed_form_phases(ch))


def test_untrained_residual_policy_is_anchored_on_co_phasing(cfg, bank):
    """With the cos-component bias, an untrained residual actor starts at co-phasing.

    Without the bias the decoded correction is an arbitrary angle, so the two cases are
    compared against each other rather than against a hand-picked tolerance.
    """
    env = RisCrEnv(cfg, action_param="residual")
    ch = bank.state(0, 7)

    def mean_dev(biased):
        model = _model(cfg, env, seed=1)
        if biased:
            apply_residual_bias(model, env.amap.n_pairs)
        model.policy.set_training_mode(False)
        ctrl = TD3Controller(cfg, model.policy.actor, "project", action_param="residual", model=model)
        d = ctrl.decide(ch)
        err = (d.theta - S.closed_form_phases(ch) + np.pi) % (2 * np.pi) - np.pi
        return np.degrees(np.abs(err).mean())

    biased, plain = mean_dev(True), mean_dev(False)
    assert biased < 0.25 * plain, f"bias did not anchor the policy ({biased:.2f} vs {plain:.2f} deg)"
    assert biased < 15.0, f"untrained residual policy is {biased:.2f} deg away from co-phasing"


def test_saturated_fraction_flags_the_tanh_bound():
    assert saturated_fraction(np.array([0.0, 0.5, 0.9999, -1.0])) == pytest.approx(0.5)
    assert saturated_fraction(np.zeros(8)) == 0.0


# ----------------------------------------------------------------- train / serve
def test_train_serve_parity(cfg, bank):
    """The controller must see exactly the observations the environment produces."""
    env_lin = RisCrEnv(cfg, action_param="linear")
    model = _model(cfg, env_lin, seed=1)
    model.policy.set_training_mode(False)
    ctrl = TD3Controller(cfg, model.policy.actor, "project", action_param="linear", model=model)
    env = RisCrEnv(cfg, bank=bank, action_param="linear")
    obs, _ = env.reset(options={"episode": 0})
    for t in range(30):
        ch = bank.state(0, t)
        o_ctrl = ctrl.obs.build(ch, ctrl._prev_u, ctrl._last_R, ctrl._last_I)
        assert np.allclose(obs, o_ctrl, atol=1e-6), f"skew at t={t}"
        d = ctrl.decide(ch)
        obs, r, *_ = env.step(ctrl._prev_u)
        assert r == pytest.approx(S.rate(ch, d), rel=1e-5)


def test_model_save_load_round_trip_gives_identical_actions(cfg, bank, tmp_path):
    from ris_cr.rl.sb3 import load_actor
    env = RisCrEnv(cfg, action_param="circular")
    model = _model(cfg, env, seed=2)
    model.policy.set_training_mode(False)
    obs = [ObsBuilder(cfg, 16, 32).build(bank.state(0, t), np.zeros(32, np.float32), 0.0, cfg.I_th_norm)
           for t in range(10)]
    before = np.array([actor_forward(model.policy.actor, o) for o in obs])
    model.save(tmp_path / "m")
    _, actor2 = load_actor(tmp_path / "m")
    after = np.array([actor_forward(actor2, o) for o in obs])
    assert np.array_equal(before, after), "save/load changed the deterministic action"


def test_controller_rejects_mismatched_architecture(cfg):
    env = RisCrEnv(cfg, action_param="circular")
    model = _model(cfg, env, seed=0)
    with pytest.raises(AssertionError):
        TD3Controller(cfg, model.policy.actor, "project", action_param="linear", model=model)
