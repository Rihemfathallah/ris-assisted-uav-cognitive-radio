"""E02 — operating-regime check across direct-link blockage values.

For each blockage value, on the same trajectories: fraction of states where the
PU constraint binds under PU-agnostic co-phasing, and the mean rate of every
method. Decides whether the default operating point is interesting at all.

    python scripts/regime_check.py --blockage 0 5 10 --states 200
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from ris_cr.channel import ChannelGenerator
from ris_cr.config import ROOT, load_config
from ris_cr.controllers.baselines import ClosedForm, NoRIS, RandomPhases
from ris_cr.controllers.elementwise import ElementWise
from ris_cr.controllers.sdr import SDR
from ris_cr import system as S

warnings.filterwarnings("ignore", message="Solution may be inaccurate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blockage", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    ap.add_argument("--states", type=int, default=200)
    args = ap.parse_args()
    rows = []
    for blk in args.blockage:
        cfg = load_config(blockage_dB=blk)
        ctrls = [NoRIS(cfg), RandomPhases(cfg), ClosedForm(cfg), ElementWise(cfg), SDR(cfg)]
        gen = ChannelGenerator(cfg)
        acc = {c.name: [] for c in ctrls}
        binding, ub = [], []
        per_ep = 10
        for ep in range(args.states // per_ep):
            gen.new_episode(20_000, ep)
            for k in range(per_ep):
                ch = gen.state()
                for c in ctrls:
                    d = c.decide(ch)
                    acc[c.name].append(S.rate(ch, d))
                    if c.name == "closed_form":
                        binding.append(d.P < cfg.P_max_W * (1 - 1e-9))
                    if c.name == "sdr":
                        ub.append(c.last_upper_bound)
                for _ in range(20):
                    gen.step()
        row = {"blockage_dB": blk, "binding_frac": np.mean(binding)}
        row.update({k: np.mean(v) for k, v in acc.items()})
        row["sdp_bound"] = np.mean(ub)
        row["gain_ris_vs_noris"] = row["sdr"] - row["no_ris"]
        row["gain_pu_aware"] = row["sdr"] - row["closed_form"]
        rows.append(row)
        print(pd.DataFrame([row]).round(3).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "csv" / "e02_regime_check.csv"
    df.to_csv(out, index=False)
    print("\n", df.round(3).to_string(index=False), f"\n-> {out}")


if __name__ == "__main__":
    main()
