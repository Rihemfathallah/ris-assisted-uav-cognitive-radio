"""Do the untrained action heads start where we think they do?

The residual head decodes theta_n = theta_n^cf + atan2(s_n, c_n), so an untrained
residual policy is only equivalent to analytic co-phasing if its decoded correction
starts at zero.  ``sb3.apply_residual_bias`` adds a constant to the pre-tanh bias of
every cos component to push it there.  This script measures, on the frozen validation
bank, what the untrained policies actually score against co-phasing, and writes
results/csv/untrained_policies.csv.

    python scripts/untrained_policies.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ris_cr import system as S                                        # noqa: E402
from ris_cr.config import load_config                                 # noqa: E402
from ris_cr.controllers.baselines import ClosedForm                   # noqa: E402
from ris_cr.controllers.td3 import TD3Controller                      # noqa: E402
from ris_cr.rl.env import RisCrEnv                                    # noqa: E402
from ris_cr.rl.sb3 import apply_residual_bias, build_model            # noqa: E402
from ris_cr.scenarios import load_bank                                # noqa: E402

torch.set_num_threads(1)
N, SEEDS = 16, (0, 1, 2)


def score(ctrl, bank, cfg, collect_angle_err=False):
    """Mean rate / violation rate of a controller over the whole bank."""
    E, T, _ = bank.shape
    R, V, D = [], [], []
    for e in range(E):
        ctrl.reset()
        for t in range(T):
            ch = bank.state(e, t)
            d = ctrl.decide(ch)
            R.append(S.rate(ch, d))
            V.append(S.is_violation(S.interference(ch, d), cfg.I_th_norm))
            if collect_angle_err:
                # signed angular distance between the chosen phases and co-phasing
                err = (d.theta - S.closed_form_phases(ch) + np.pi) % (2 * np.pi) - np.pi
                D.append(np.abs(err))
    out = {"rate": float(np.mean(R)), "violation_rate": float(np.mean(V))}
    if collect_angle_err:
        D = np.concatenate(D)
        out["mean_abs_dev_from_cophasing_deg"] = float(np.degrees(D.mean()))
        out["max_abs_dev_from_cophasing_deg"] = float(np.degrees(D.max()))
    return out


def main() -> None:
    cfg = load_config(N=N)
    bank = load_bank("val", N, cfg)
    rows = []

    cf = score(ClosedForm(cfg), bank, cfg)
    rows.append({"policy": "closed_form_cophasing", "seed": np.nan, **cf,
                 "mean_abs_dev_from_cophasing_deg": 0.0, "max_abs_dev_from_cophasing_deg": 0.0,
                 "rate_minus_cophasing": 0.0})
    print(f"co-phasing reference: rate={cf['rate']:.6f}")

    for param, biased in (("residual", True), ("residual", False), ("circular", False)):
        for seed in SEEDS:
            env = RisCrEnv(cfg, action_param=param)
            model = build_model(env, cfg.td3, seed=seed, buffer_size=1)
            if biased:
                apply_residual_bias(model, env.amap.n_pairs)
            model.policy.set_training_mode(False)
            ctrl = TD3Controller(cfg, model.policy.actor, "project", action_param=param, model=model)
            m = score(ctrl, bank, cfg, collect_angle_err=True)
            name = f"untrained_{param}" + ("_cos_bias" if biased else "_no_bias")
            rows.append({"policy": name, "seed": seed, **m,
                         "rate_minus_cophasing": m["rate"] - cf["rate"]})
            print(f"{name} seed {seed}: rate={m['rate']:.6f} "
                  f"(co-phasing {m['rate'] - cf['rate']:+.6f})  "
                  f"mean|dtheta|={m['mean_abs_dev_from_cophasing_deg']:.3f} deg")

    out = ROOT / "results" / "csv" / "untrained_policies.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
