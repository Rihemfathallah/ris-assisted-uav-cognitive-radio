"""Train TD3 (Stable-Baselines3, PyTorch) on the RIS-UAV cognitive-radio environment.

    python -m ris_cr.rl.train_td3 --N 16 --seed 0
    python -m ris_cr.rl.train_td3 --N 16 --seed 0 --gamma 0.99 --tag gamma099
    python -m ris_cr.rl.train_td3 --N 16 --seed 0 --power-mode learn --penalty linear --tag learnP

Outputs (results/runs/<tag>_N<N>_seed<k>/):
  meta.json            full provenance (config digest, source digest, args, wall time)
  config.yaml          exact configuration used
  learning_curve.csv   validation metrics every eval_every steps (validation seeds only)
  train_episodes.csv   per-episode training return / violation rate
  model_best.zip       SB3 model with the best *validation* rate (model selection)
  model_final.zip      SB3 model at the end of training
The test flights are never touched here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import yaml
from stable_baselines3.common.env_checker import check_env

from ..config import ROOT, load_config, source_digest
from ..controllers.td3 import TD3Controller
from ..scenarios import load_bank
from .. import system as S
from .env import RisCrEnv
from .sb3 import ValidationCallback, apply_residual_bias, build_model, saturated_fraction

torch.set_num_threads(1)


def evaluate_on_bank(cfg, model, power_mode, bank, action_param="circular") -> dict:
    """Validation metrics through the *serving* code path (TD3Controller)."""
    ctrl = TD3Controller.from_model(cfg, model, power_mode, action_param=action_param)
    E, T, _ = bank.shape
    R, V, P, U = [], [], [], []
    for e in range(E):
        ctrl.reset()
        for t in range(T):
            ch = bank.state(e, t)
            d = ctrl.decide(ch)
            R.append(S.rate(ch, d))
            V.append(S.is_violation(S.interference(ch, d), cfg.I_th_norm))
            P.append(d.P)
            U.append(ctrl._prev_u)
    model.policy.set_training_mode(True)          # the callback resumes training after this
    return {"rate": float(np.mean(R)), "violation_rate": float(np.mean(V)),
            "P_mean_W": float(np.mean(P)), "saturated_frac": saturated_fraction(np.concatenate(U))}


def train(N: int, seed: int, tag: str = "td3", gamma: float | None = None, power_mode: str | None = None,
          penalty: str | None = None, lam: float | None = None, timesteps: int | None = None,
          action_param: str | None = None, lr: float | None = None,
          out_root: Path | None = None, quiet: bool = False) -> Path:
    cfg = load_config(N=N)
    over = {}
    if gamma is not None: over["td3.gamma"] = gamma
    if power_mode is not None: over["td3.power_mode"] = power_mode
    if penalty is not None: over["td3.penalty"] = penalty
    if lam is not None: over["td3.lam"] = lam
    if timesteps is not None: over["td3.total_timesteps"] = timesteps
    if action_param is not None: over["td3.action_param"] = action_param
    if lr is not None: over["td3.learning_rate"] = lr
    cfg = cfg.with_(**over)
    hp = cfg.td3

    out = (out_root or ROOT / "results" / "runs") / f"{tag}_N{N}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False))

    env = RisCrEnv(cfg)
    check_env(env, skip_render_check=True)                      # SB3's own checker
    val_bank = load_bank("val", N, cfg)
    model = build_model(env, hp, seed)
    if env.amap.param == "residual":
        # anchor the untrained policy on the analytic co-phasing solution (see sb3.apply_residual_bias)
        apply_residual_bias(model, env.amap.n_pairs)

    lc_f = open(out / "learning_curve.csv", "w", newline="")
    lc = csv.writer(lc_f)
    lc.writerow(["step", "val_rate", "val_violation_rate", "val_P_mean_W", "val_saturated_frac",
                 "wall_s", "steps_per_s"])
    t0 = time.perf_counter()

    def write_row(step, m):
        wall = time.perf_counter() - t0
        lc.writerow([step, round(m["rate"], 5), round(m["violation_rate"], 5), m["P_mean_W"],
                     round(m["saturated_frac"], 5), round(wall, 1), round(step / max(wall, 1e-9), 1)])

    cb = ValidationCallback(
        evaluate_fn=lambda mdl: evaluate_on_bank(cfg, mdl, hp["power_mode"], val_bank, env.amap.param),
        eval_every=int(hp["eval_every"]), out_dir=out, power_mode=hp["power_mode"],
        writer=write_row, flush=lc_f.flush,
        verbose_prefix=None if quiet else f"[{tag} N={N} s={seed}]")

    total = int(hp["total_timesteps"])
    model.learn(total_timesteps=total, callback=cb, log_interval=None)
    model.save(out / "model_final")
    if not (out / "model_best.zip").exists():                   # never met the violation budget
        model.save(out / "model_best")
    lc_f.close()

    import stable_baselines3 as sb3
    meta = {"N": N, "seed": seed, "tag": tag, "power_mode": hp["power_mode"], "penalty": hp["penalty"],
            "action_param": env.amap.param, "learning_rate": hp["learning_rate"],
            "lam": hp["lam"], "gamma": hp["gamma"], "total_timesteps": total,
            "obs_dim": int(env.observation_space.shape[0]), "act_dim": int(env.action_space.shape[0]),
            "best_val_rate": None if not np.isfinite(cb.best) else cb.best, "best_step": cb.best_step,
            "wall_s": round(time.perf_counter() - t0, 1),
            "config_digest": cfg.digest(), "source_digest": source_digest(),
            "backend": "sb3", "sb3_version": sb3.__version__, "torch_version": torch.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="td3")
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--power-mode", choices=["project", "learn"])
    ap.add_argument("--penalty", choices=["linear", "db"])
    ap.add_argument("--lam", type=float)
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--action-param", choices=["linear", "circular", "residual"])
    ap.add_argument("--lr", type=float)
    ap.add_argument("--skip-existing", action="store_true", help="resume a sweep: skip runs with meta.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    for s in a.seed:
        run = ROOT / "results" / "runs" / f"{a.tag}_N{a.N}_seed{s}"
        if a.skip_existing and (run / "meta.json").exists():
            print(f"skip {run.name}", flush=True)
            continue
        train(a.N, s, a.tag, a.gamma, a.power_mode, a.penalty, a.lam, a.timesteps, a.action_param, a.lr,
              quiet=a.quiet)


if __name__ == "__main__":
    main()
