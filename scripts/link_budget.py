"""Pre-flight link budget: is the RIS path relevant compared with the direct link?

Prints (i) deterministic average path gains around the UAV circle and (ii) the
empirical |h_d|^2 vs best-case RIS gain (sum |a_n|)^2 from the actual channel
generator, for each N and each blockage value given on the command line.

    python scripts/link_budget.py --blockage 0 10
"""
from __future__ import annotations

import argparse

import numpy as np

from ris_cr.channel import ChannelGenerator, reference_gains, uav_position
from ris_cr.config import lin2db, load_config, w2dbm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blockage", type=float, nargs="+", default=[0.0, 10.0])
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args()
    base = load_config()
    print(base.summary(), "\n")

    for blk in args.blockage:
        cfg = base.with_(blockage_dB=blk)
        print(f"=== blockage_dB = {blk:g} ===")
        print(f"{'UAV angle':>9} | {'UAV-RIS':>8} {'RIS-SU':>8} {'RIS-PU':>8} {'UAV-SU':>8} {'UAV-PU':>8}  [dB]")
        for deg in (0, 90, 180, 270):
            g = reference_gains(cfg, uav_position(cfg, np.deg2rad(deg)))
            print(f"{deg:>8}° | " + " ".join(f"{lin2db(g[k]):8.1f}" for k in ("uav_ris", "ris_su", "ris_pu", "uav_su", "uav_pu")))

        g = reference_gains(cfg, uav_position(cfg, 0.0))
        I_noris = w2dbm(cfg.P_max_W * g["uav_pu"])
        print(f"PU interference, no RIS, P_max={cfg.P_max_dBm} dBm, UAV at 0°: {I_noris:.1f} dBm "
              f"(I_th = {cfg.I_th_dBm} dBm -> constraint {'ACTIVE' if I_noris > cfg.I_th_dBm else 'inactive'})")

        print(f"{'N':>4} | {'E|h_d|^2':>9} {'E(Σ|a_n|)^2':>12} {'RIS/direct':>10} {'max SNR gain':>12}   (noise units, dB)")
        for N in (8, 16, 32, 64):
            c = cfg.with_(N=N)
            gen = ChannelGenerator(c)
            hd2, ris2, gain = [], [], []
            for s in range(args.samples // 200):
                gen.new_episode(10_000 + s)
                for _ in range(200):
                    st = gen.state()
                    hd2.append(abs(st.h_d) ** 2)
                    ris2.append(np.sum(np.abs(st.a)) ** 2)
                    gain.append((abs(st.h_d) + np.sum(np.abs(st.a))) ** 2 / abs(st.h_d) ** 2)
                    gen.step()
            hd2, ris2 = np.mean(hd2), np.mean(ris2)
            print(f"{N:>4} | {lin2db(hd2):9.1f} {lin2db(ris2):12.1f} {lin2db(ris2 / hd2):+10.1f} "
                  f"{lin2db(np.median(gain)):+12.1f} (median)")
        print()


if __name__ == "__main__":
    main()
