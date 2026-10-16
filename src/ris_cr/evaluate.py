"""Evaluation driver.  Every method runs on the frozen test bank; results go to CSV only
(figures are produced separately by scripts/make_figures.py and never recompute anything).

    python -m ris_cr.evaluate --part fig1      # rate vs N, no PU constraint (+ Octave export)
    python -m ris_cr.evaluate --part sweeps    # Fig. 2 (P_max) and Fig. 3 (I_th), N = 16
    python -m ris_cr.evaluate --part runtime   # Table 1 / Fig. 6: time per decision vs N
    python -m ris_cr.evaluate --part main      # per-step results of all baselines, N = 16
    python -m ris_cr.evaluate --part val_refs  # baseline levels on the validation bank (Fig. 4)
    python -m ris_cr.evaluate --part td3       # every trained run in results/runs/

Runtime protocol: time.perf_counter() around decide() only, single thread, first
call of each method excluded (warm-up / compilation), mean over the timed calls.
CSI-delay protocol: the decision computed from the state at t is applied to the
state at t+1 of the same flight (one control period, 8.1 ms, of outdated CSI).
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

import numpy as np
import pandas as pd
import scipy.io

from .config import ROOT, load_config, rng
from .controllers.baselines import ClosedForm, NoRIS, RandomPhases
from .controllers.elementwise import ElementWise
from .controllers.sdr import SDR
from .controllers.td3 import TD3Controller
from .rl.core import ActionMap, ObsBuilder
from .rl.sb3 import actor_forward, build_model
from .rl.env import RisCrEnv
from .scenarios import load_bank
from . import system as S

CSV = ROOT / "results" / "csv"
OCT = ROOT / "results" / "octave"
BASELINES = ("no_ris", "random", "closed_form", "elementwise", "sdr")


def make(name: str, cfg, constrained: bool = True):
    return {"no_ris": NoRIS, "random": RandomPhases, "closed_form": ClosedForm,
            "elementwise": ElementWise, "sdr": SDR}[name](cfg, constrained)


def run_controller(ctrl, bank, cfg, stride: int = 1, episodes=None, timing: bool = True) -> pd.DataFrame:
    """Per-step records for one controller on a bank (episode-wise, reset per episode)."""
    E, T, _ = bank.shape
    rows, first = [], True
    for e in (range(E) if episodes is None else episodes):
        ctrl.reset()
        for t in range(0, T, stride):
            ch = bank.state(e, t)
            t0 = time.perf_counter()
            d = ctrl.decide(ch)
            dt = time.perf_counter() - t0
            I = S.interference(ch, d)
            rec = {"method": ctrl.name, "episode": e, "t": t, "rate": S.rate(ch, d), "I_norm": I,
                   "violation": S.is_violation(I, cfg.I_th_norm), "P_W": d.P,
                   "runtime_s": np.nan if (first or not timing) else dt}
            if t + stride < T:                                   # one-step-outdated CSI
                ch2 = bank.state(e, t + 1)
                I2 = S.interference(ch2, d)
                rec.update(rate_delayed=S.rate(ch2, d), violation_delayed=S.is_violation(I2, cfg.I_th_norm))
            if ctrl.name == "sdr":
                rec.update(sdp_bound=ctrl.last_upper_bound, rank_ratio=ctrl.last_rank_ratio)
            rows.append(rec)
            first = False
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by=("method",)) -> pd.DataFrame:
    g = df.groupby(list(by))
    out = g.agg(rate=("rate", "mean"), rate_std=("rate", "std"), n=("rate", "size"),
                violation_rate=("violation", "mean"), runtime_ms=("runtime_s", lambda x: 1e3 * np.nanmean(x)))
    out["rate_sem"] = out["rate_std"] / np.sqrt(out["n"])
    return out.reset_index()


# ------------------------------------------------------------------------------------ parts
def part_fig1(cfg) -> None:
    """Rate vs N without the PU constraint, P = P_max; exports the exact inputs for Octave."""
    OCT.mkdir(parents=True, exist_ok=True)
    rows = []
    for N in (8, 16, 32, 64):
        c = cfg.with_(N=N)
        bank = load_bank("test", N, c)
        for name in ("no_ris", "random", "closed_form"):
            df = run_controller(make(name, c, constrained=False), bank, c, stride=4, timing=False)
            rows.append({"N": N, "method": name, "rate": df.rate.mean(), "rate_sem": df.rate.std() / np.sqrt(len(df))})
        # exact Octave cross-check inputs: the same states plus the random phases actually used
        E, T, _ = bank.shape
        idx = [(e, t) for e in range(E) for t in range(0, T, 4)]
        th = rng(0, "octave_export", N).uniform(0, 2 * np.pi, (len(idx), N))
        h_d = np.array([bank.h_d[e, t] for e, t in idx])
        a = np.array([bank.a[e, t] for e, t in idx])
        py_rand = S.batch_rate(h_d, a, th, c.P_max_W)
        py_cf = np.log2(1 + c.P_max_W * (np.abs(h_d) + np.abs(a).sum(1)) ** 2)
        py_no = np.log2(1 + c.P_max_W * np.abs(h_d) ** 2)
        scipy.io.savemat(OCT / f"fig1_input_N{N}.mat", {"h_d": h_d[:, None], "a": a, "theta": th,
                         "P": c.P_max_W, "py_rand": py_rand[:, None], "py_cf": py_cf[:, None], "py_no": py_no[:, None]})
        # independent-MC reference for Octave's own generator: i.i.d. positions and fading
        from .channel import ChannelGenerator
        gen, vals = ChannelGenerator(c), []
        for k in range(2000):
            gen.new_episode(500_000 + k)
            st = gen.state()
            vals.append(np.log2(1 + c.P_max_W * (abs(st.h_d) + np.abs(st.a).sum()) ** 2))
        rows.append({"N": N, "method": "closed_form_iid_python", "rate": np.mean(vals), "rate_sem": np.std(vals) / np.sqrt(len(vals))})
        print(f"fig1 N={N} done", flush=True)
    pd.DataFrame(rows).to_csv(CSV / "fig1_rate_vs_N.csv", index=False)


def part_sweeps(cfg) -> None:
    bank = load_bank("test", 16, cfg)
    for key, grid, fname in (("P_max_dBm", [0, 5, 10, 15, 20, 25, 30], "fig2_rate_vs_Pmax.csv"),
                             ("I_th_dBm", [-100, -95, -90, -85, -80], "fig3_rate_vs_Ith.csv")):
        rows = []
        for val in grid:
            c = cfg.with_(**{key: val})
            for name in BASELINES:
                ctrl = make(name, c)
                df = run_controller(ctrl, bank, c, stride=20, timing=False)
                rows.append({key: val, "method": name, "rate": df.rate.mean(), "rate_sem": df.rate.std() / np.sqrt(len(df)),
                             "violation_rate": df.violation.mean(),
                             "P_dBm": 10 * np.log10(df.P_W.clip(lower=1e-30)).mean() + 30})
                if name == "sdr":
                    rows.append({key: val, "method": "sdp_bound", "rate": df.sdp_bound.mean(),
                                 "rate_sem": df.sdp_bound.std() / np.sqrt(len(df)), "violation_rate": np.nan})
            print(f"{key}={val} done", flush=True)
        pd.DataFrame(rows).to_csv(CSV / fname, index=False)


def part_runtime(cfg, n_real: int = 100) -> None:
    rows = []
    for N in (8, 16, 32, 64):
        c = cfg.with_(N=N)
        bank = load_bank("test", N, c)
        E, T, _ = bank.shape
        states = [bank.state(e, t) for e in range(E) for t in range(0, T, T // (n_real // E + 1))][: n_real + 1]
        ctrls = [make(n, c) for n in BASELINES]
        # TD3 inference cost depends only on the architecture, not on the weight values:
        # an untrained SB3 actor of the deployed shape is built for every N (the trained one for
        # N=16 is timed inside part_td3).  buffer_size=1 keeps construction cheap.
        env = RisCrEnv(c, action_param="circular")
        model = build_model(env, c.td3, seed=0, buffer_size=1)
        model.policy.set_training_mode(False)
        td3 = TD3Controller(c, model.policy.actor, "project", "td3_arch", "circular", model=model)
        for ctrl in [*ctrls, td3]:
            ctrl.reset()
            ctrl.decide(states[0])                                  # warm-up / DPP compilation
            ts = []
            for st in states[1:]:
                t0 = time.perf_counter(); ctrl.decide(st); ts.append(time.perf_counter() - t0)
            rows.append({"N": N, "method": "td3" if ctrl.name == "td3_arch" else ctrl.name,
                         "runtime_ms": 1e3 * np.mean(ts), "runtime_ms_std": 1e3 * np.std(ts), "n": len(ts)})
        # SB3 exposes two inference paths: the raw actor forward pass (what a deployed controller
        # runs) and model.predict(), which adds observation bookkeeping.  Both are timed.
        obs = [td3.obs.build(st, np.zeros(td3.amap.dim, np.float32), 0.0, c.I_th_norm) for st in states]
        for label, fn in (("td3_actor_forward", lambda o: actor_forward(td3.actor, o)),
                          ("td3_model_predict", lambda o: model.predict(o, deterministic=True))):
            fn(obs[0])                                              # warm-up
            ts = []
            for o in obs[1:]:
                t0 = time.perf_counter(); fn(o); ts.append(time.perf_counter() - t0)
            rows.append({"N": N, "method": label, "runtime_ms": 1e3 * np.mean(ts),
                         "runtime_ms_std": 1e3 * np.std(ts), "n": len(ts)})
        pd.DataFrame(rows).to_csv(CSV / "table1_runtime_vs_N.csv", index=False)   # incremental: survives interruption
        print(f"runtime N={N} done", flush=True)


def part_main(cfg) -> None:
    bank = load_bank("test", 16, cfg)
    frames = []
    for name in BASELINES:
        t0 = time.perf_counter()
        frames.append(run_controller(make(name, cfg), bank, cfg))
        print(f"main {name}: {time.perf_counter() - t0:.0f}s", flush=True)
        pd.concat(frames).to_csv(CSV / "main_steps_baselines.csv", index=False)


def part_val_refs(cfg) -> None:
    bank = load_bank("val", 16, cfg)
    df = pd.concat([run_controller(make(n, cfg), bank, cfg, timing=False) for n in BASELINES])
    summarize(df).to_csv(CSV / "val_references.csv", index=False)


def part_td3(cfg) -> None:
    bank = load_bank("test", 16, cfg)
    frames = []
    for run in sorted((ROOT / "results" / "runs").glob("*_N16_seed*")):
        if not (run / "meta.json").exists():
            continue
        ctrl = TD3Controller.from_run(cfg, run, "best", name=run.name)
        df = run_controller(ctrl, bank, cfg)
        df["tag"] = run.name.split("_N16")[0]
        df["seed"] = int(run.name.split("seed")[-1])
        frames.append(df)
        print(f"td3 eval {run.name}: rate={df.rate.mean():.4f} viol={df.violation.mean():.4f}", flush=True)
    pd.concat(frames).to_csv(CSV / "main_steps_td3.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", required=True,
                    choices=["fig1", "sweeps", "runtime", "main", "val_refs", "td3"])
    args = ap.parse_args()
    CSV.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    for p in args.part:
        t0 = time.perf_counter()
        globals()[f"part_{p}"](cfg)
        print(f"== part {p} finished in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
