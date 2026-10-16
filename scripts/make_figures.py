"""All figures and tables, generated from results/csv only (nothing is recomputed here).

    python scripts/make_figures.py            # everything whose inputs exist
Outputs: results/figures/*.pdf|png, results/tables/*.md|tex|csv
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV, FIG, TAB = ROOT / "results" / "csv", ROOT / "results" / "figures", ROOT / "results" / "tables"
RUNS, OCT = ROOT / "results" / "runs", ROOT / "results" / "octave"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False, "axes.spines.right": False,
})
W1, W2 = 3.5, 7.16          # IEEE single / double column widths (inches)
STYLE = {  # consistent identity for every method across all figures
    "no_ris": dict(label="No RIS", color="0.55", marker="v", ls=":"),
    "random": dict(label="Random phases", color="0.35", marker="x", ls="--"),
    "closed_form": dict(label="Co-phasing (PU-agnostic)", color="#1f77b4", marker="o", ls="-"),
    "elementwise": dict(label="Element-wise (16 levels)", color="#2ca02c", marker="s", ls="-"),
    "sdr": dict(label="Joint SDR", color="#d62728", marker="D", ls="-"),
    "sdp_bound": dict(label="SDP upper bound", color="k", marker="", ls=(0, (1, 1))),
    "td3": dict(label="TD3 (pure)", color="#9467bd", marker="^", ls="-"),
    "td3res": dict(label="TD3-residual", color="#ff7f0e", marker="P", ls="-"),
    "td3_actor_forward": dict(label="TD3 actor forward pass", color="#9467bd", marker="", ls="--"),
    "td3_model_predict": dict(label="TD3 model.predict()", color="#9467bd", marker="", ls=":"),
}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}")


def have(*files):
    missing = [f for f in files if not (CSV / f).exists() and not Path(f).exists()]
    if missing:
        print(f"  skip (missing {missing})")
    return not missing


def md_table(df: pd.DataFrame, path: Path, floatfmt=".3f"):
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(df.to_markdown(index=False, floatfmt=floatfmt))
    path.with_suffix(".tex").write_text(df.to_latex(index=False, float_format=lambda x: f"{x:{floatfmt}}"))


# ------------------------------------------------------------------------------------ Fig. 1
def fig1():
    print("fig1")
    if not have("fig1_rate_vs_N.csv"):
        return
    d = pd.read_csv(CSV / "fig1_rate_vs_N.csv")
    o = pd.read_csv(OCT / "rate_vs_N_octave.csv") if (OCT / "rate_vs_N_octave.csv").exists() else None
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for m in ("no_ris", "random", "closed_form"):
        s = d[d.method == m]
        st = dict(STYLE[m]); st["label"] = st["label"].replace(" (PU-agnostic)", " (bound)")
        ax.errorbar(s.N, s.rate, yerr=1.96 * s.rate_sem, capsize=2, **st)
        if o is not None:
            so = o[o.method == m]
            ax.plot(so.N, so.rate, ls="none", marker="+", ms=8, mew=0.9, color=st["color"])
    s = d[d.method == "closed_form_iid_python"]
    ax.plot(s.N, s.rate, color="#1f77b4", ls=(0, (4, 2)), lw=0.8, label="Bound, i.i.d. positions (Python)")
    if o is not None:
        so = o[o.method == "closed_form_iid_octave"]
        ax.errorbar(so.N, so.rate, yerr=1.96 * so.rate_sem, color="#ff7f0e", ls="none", marker="*", ms=5,
                    capsize=2, label="Bound, i.i.d. (independent Octave)")
        ax.plot([], [], ls="none", marker="+", ms=8, color="k", label="Octave on identical realisations")
    ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32, 64]); ax.set_xticklabels(["8", "16", "32", "64"])
    ax.set_xlabel("Number of RIS elements $N$"); ax.set_ylabel("SU rate (bit/s/Hz)")
    ax.set_ylim(5.2, 10.6)
    ax.legend(loc="upper left", ncol=2, fontsize=5.5, columnspacing=0.8)
    save(fig, "fig1_rate_vs_N")


# ------------------------------------------------------------------------------------ Figs. 2-3
def sweep(fname, key, xlabel, out):
    print(out)
    if not have(fname):
        return
    d = pd.read_csv(CSV / fname)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for m in ("no_ris", "random", "closed_form", "elementwise", "sdr", "sdp_bound"):
        s = d[d.method == m].sort_values(key)
        ax.errorbar(s[key], s.rate, yerr=1.96 * s.rate_sem, capsize=1.5, **STYLE[m])
    ax.set_xlabel(xlabel); ax.set_ylabel("SU rate (bit/s/Hz)")
    ax.legend(loc="upper left")
    save(fig, out)


# ------------------------------------------------------------------------------------ runtime
def runtime():
    print("table1 / fig6")
    if not have("table1_runtime_vs_N.csv"):
        return
    d = pd.read_csv(CSV / "table1_runtime_vs_N.csv")
    t = d.pivot(index="method", columns="N", values="runtime_ms")
    table_order = [m for m in ("no_ris", "random", "closed_form", "elementwise", "sdr", "td3",
                               "td3_actor_forward", "td3_model_predict") if m in t.index]
    order = [m for m in ("no_ris", "random", "closed_form", "elementwise", "sdr", "td3") if m in t.index]
    t = t.loc[table_order].reset_index()
    t["method"] = t["method"].map(lambda m: STYLE[m]["label"])
    t.columns = ["Method"] + [f"N={c} (ms)" for c in t.columns[1:]]
    md_table(t, TAB / "table1_runtime_vs_N", ".4f")
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for m in order:
        s = d[d.method == m]
        ax.plot(s.N, s.runtime_ms, **STYLE[m])
    Ns = np.array([8, 16, 32, 64])
    s = d[d.method == "sdr"].set_index("N").runtime_ms
    ax.plot(Ns, s[16] * ((Ns + 1) / 17.0) ** 4.5, color="#d62728", lw=0.6, ls=(0, (5, 3)),
            label=r"$\mathcal{O}((N{+}1)^{4.5})$ worst case (anchored at N=16)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(Ns); ax.set_xticklabels(map(str, Ns))
    ax.set_xlabel("Number of RIS elements $N$"); ax.set_ylabel("Time per decision (ms)")
    ax.legend(loc="upper left", fontsize=6)
    save(fig, "fig6_runtime_vs_N")


# ------------------------------------------------------------------------------------ learning curves
def _curves(tag):
    runs = sorted(RUNS.glob(f"{tag}_N16_seed*"))
    frames = []
    for r in runs:
        if (r / "meta.json").exists():
            c = pd.read_csv(r / "learning_curve.csv"); c["seed"] = r.name.split("seed")[-1]; frames.append(c)
    return pd.concat(frames) if frames else None


def fig4():
    print("fig4")
    main = _curves("td3")
    if main is None:
        print("  skip (no finished td3 runs)"); return
    refs = pd.read_csv(CSV / "val_references.csv").set_index("method") if (CSV / "val_references.csv").exists() else None
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3))
    g = main.groupby("step").val_rate
    m, s, n = g.mean(), g.std().fillna(0), g.size()
    for ax in axs:
        ax.plot(m.index / 1e3, m, **{**STYLE["td3"], "marker": "", "label": f"TD3 pure, circular head ({n.max()} seeds)"})
        ax.fill_between(m.index / 1e3, m - s, m + s, color=STYLE["td3"]["color"], alpha=0.2, lw=0)
        res = _curves("td3res")
        if res is not None:
            gr = res.groupby("step").val_rate
            ax.plot(gr.mean().index / 1e3, gr.mean(), **{**STYLE["td3res"], "marker": "",
                    "label": f"TD3-residual ({res.seed.nunique()} seeds)"})
            ax.fill_between(gr.mean().index / 1e3, gr.mean() - gr.std().fillna(0), gr.mean() + gr.std().fillna(0),
                            color=STYLE["td3res"]["color"], alpha=0.2, lw=0)
        if refs is not None:
            for k in ("sdr", "elementwise", "closed_form", "no_ris"):
                ax.axhline(refs.loc[k, "rate"], color=STYLE[k]["color"], ls=STYLE[k]["ls"], lw=0.8, label=STYLE[k]["label"])
        ax.set_xlabel("Training steps ($\\times 10^3$)"); ax.set_ylabel("Validation SU rate (bit/s/Hz)")
    axs[0].set_title("(a) main configuration", fontsize=8); axs[0].legend(loc="lower right")
    abl = {"abl_linearhead": ("spec tanh head $\\theta=\\pi(u+1)$", "#8c564b"),
           "abl_gamma099": ("$\\gamma$ = 0.99", "#e377c2"),
           "abl_learnP_linpen": ("learned $P$, linear penalty (spec)", "#bcbd22"),
           "abl_learnP_dbpen": ("learned $P$, dB penalty", "#17becf")}
    for tag, (lab, col) in abl.items():
        c = _curves(tag)
        if c is None:
            continue
        g = c.groupby("step").val_rate
        axs[1].plot(g.mean().index / 1e3, g.mean(), color=col, label=f"{lab} ({c.seed.nunique()} seeds)")
        axs[1].fill_between(g.mean().index / 1e3, g.min(), g.max(), color=col, alpha=0.15, lw=0)
    axs[1].set_title("(b) ablations (band = min/max over seeds)", fontsize=8)
    axs[1].legend(loc="lower right", fontsize=5.5)
    save(fig, "fig4_learning_curves")


# ------------------------------------------------------------------------------------ Table 2 / Fig. 5
def table2_fig5():
    print("table2 / fig5 / ablations / csi-delay")
    if not have("main_steps_baselines.csv"):
        return
    b = pd.read_csv(CSV / "main_steps_baselines.csv")
    t = pd.read_csv(CSV / "main_steps_td3.csv") if (CSV / "main_steps_td3.csv").exists() else None
    sdr_rate = b[b.method == "sdr"].rate.mean()
    bound = b[b.method == "sdr"].sdp_bound.mean()
    rows = []
    for m in ("no_ris", "random", "closed_form", "elementwise", "sdr"):
        s = b[b.method == m]
        rows.append({"Method": STYLE[m]["label"], "Rate (bit/s/Hz)": s.rate.mean(), "± (seeds)": np.nan,
                     "% of SDR": 100 * s.rate.mean() / sdr_rate, "% of SDP bound": 100 * s.rate.mean() / bound,
                     "Violation rate": s.violation.mean(), "Time/decision (ms)": 1e3 * s.runtime_s.mean(),
                     "Rate, 1-step-old CSI": s.rate_delayed.mean(), "Violations, 1-step-old CSI": s.violation_delayed.mean()})
    if t is not None:
        for tag, lab in (("td3", "TD3 (pure, circular head)"), ("td3res", "TD3-residual (physics-informed)"),
                         ("abl_linearhead", "TD3, spec tanh head"),
                         ("abl_gamma099", "TD3, gamma=0.99"), ("abl_learnP_linpen", "TD3, learned P + linear penalty"),
                         ("abl_learnP_dbpen", "TD3, learned P + dB penalty")):
            s = t[t.tag == tag]
            if s.empty:
                continue
            per_seed = s.groupby("seed").agg(r=("rate", "mean"), v=("violation", "mean"), rt=("runtime_s", "mean"),
                                             rd=("rate_delayed", "mean"), vd=("violation_delayed", "mean"))
            rows.append({"Method": f"{lab} [{len(per_seed)} seeds]", "Rate (bit/s/Hz)": per_seed.r.mean(),
                         "± (seeds)": per_seed.r.std(), "% of SDR": 100 * per_seed.r.mean() / sdr_rate,
                         "% of SDP bound": 100 * per_seed.r.mean() / bound, "Violation rate": per_seed.v.mean(),
                         "Time/decision (ms)": 1e3 * per_seed.rt.mean(), "Rate, 1-step-old CSI": per_seed.rd.mean(),
                         "Violations, 1-step-old CSI": per_seed.vd.mean()})
    tab = pd.DataFrame(rows)
    md_table(tab, TAB / "table2_main_comparison")
    # compact LaTeX version for the report
    def f(x, k=2):
        return "--" if pd.isna(x) else f"{x:.{k}f}"
    lines = [r"\begin{table*}[t]\centering\caption{Test flights ($N=16$, " + f"{int((b.method == 'sdr').sum())}"
             + r" decisions per method). TD3 rows: mean over seeds ($\pm$ std). Right: same decisions applied one control period (8.1\,ms) late.}\label{tab:main}",
             r"\begin{tabular}{lrrrrrr}\toprule",
             r"Method & Rate (bit/s/Hz) & \% of SDR & Violations (\%) & Time/decision (ms) & Rate, old CSI & Violations, old CSI (\%)\\\midrule"]
    for _, r in tab.iterrows():
        rate = f(r["Rate (bit/s/Hz)"], 3) + ("" if pd.isna(r["± (seeds)"]) else f" $\\pm$ {r['± (seeds)']:.3f}")
        lines.append(f"{r.Method.replace('_', ' ')} & {rate} & {f(r['% of SDR'], 1)} & {f(100 * r['Violation rate'], 1)} & "
                     f"{r['Time/decision (ms)']:.3g} & {f(r['Rate, 1-step-old CSI'], 3)} & {f(100 * r['Violations, 1-step-old CSI'], 1)}\\\\")
    lines += [r"\midrule SDP upper bound & " + f"{bound:.3f}" + r" & & & & & \\", r"\bottomrule\end{tabular}\end{table*}"]
    (ROOT / "report" / "table2.tex").write_text("\n".join(lines) + "\n")
    (TAB / "table2_meta.json").write_text(json.dumps({
        "sdr_rate": sdr_rate, "sdp_bound": bound, "states": int((b.method == "sdr").sum()),
        "sdr_rank_ratio_min": float(b[b.method == "sdr"].rank_ratio.min()),
        "sdr_rank_ratio_median": float(b[b.method == "sdr"].rank_ratio.median())}, indent=2))
    print(tab.round(4).to_string(index=False))

    # Fig. 5: rate along one test flight
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    e = 0
    for m in ("no_ris", "closed_form", "elementwise", "sdr"):
        s = b[(b.method == m) & (b.episode == e)]
        ax.plot(s.t * 8.1e-3, s.rate, **{**STYLE[m], "marker": ""})
    if t is not None and not t[t.tag == "td3"].empty:
        s = t[(t.tag == "td3") & (t.seed == 0) & (t.episode == e)]
        ax.plot(s.t * 8.1e-3, s.rate, **{**STYLE["td3"], "marker": "", "label": "TD3 pure (seed 0)"})
    if t is not None and not t[t.tag == "td3res"].empty:
        s = t[(t.tag == "td3res") & (t.seed == 0) & (t.episode == e)]
        ax.plot(s.t * 8.1e-3, s.rate, **{**STYLE["td3res"], "marker": "", "label": "TD3-residual (seed 0)"})
    ax.set_xlabel("Flight time (s)"); ax.set_ylabel("SU rate (bit/s/Hz)")
    ax.legend(loc="lower left", ncol=2, fontsize=6)
    save(fig, "fig5_rate_along_trajectory")


# ------------------------------------------------------------------------------------ Figs. 7-8
def fig7():
    print("fig7")
    if not have("fig7_music_rmse_vs_snr.csv"):
        return
    d = pd.read_csv(CSV / "fig7_music_rmse_vs_snr.csv")
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    cols = plt.cm.viridis(np.linspace(0, 0.85, d.Ls.nunique()))
    for c, (Ls, s) in zip(cols, d.groupby("Ls")):
        lab = f"$L_s$={Ls}" + (" (< M)" if Ls < 8 else "")
        ax.semilogy(s.snr_db, s.rmse_deg, marker="o", color=c, label=f"MUSIC, {lab}")
        ax.semilogy(s.snr_db, s.crb_deg, ls="--", lw=0.7, color=c)
    ax.axhline(0.1 / np.sqrt(12), color="k", lw=0.6, ls=":", label="0.1° grid floor ($\\Delta/\\sqrt{12}$)")
    ax.plot([], [], "k--", lw=0.7, label="$\\sqrt{\\mathrm{CRB}}$ (dashed)")
    ax.set_xlabel("Per-antenna echo SNR (dB)"); ax.set_ylabel("Angle RMSE (deg)")
    ax.legend(loc="lower left", fontsize=5.5, ncol=2)
    save(fig, "fig7_music_rmse_vs_snr")


def fig8():
    print("fig8")
    if not have("fig8_isac_tradeoff.csv"):
        return
    d = pd.read_csv(CSV / "fig8_isac_tradeoff.csv", keep_default_na=False)
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    spec = {(100, "isotropic"): ("L=100, isotropic probing", "#1f77b4", "o"),
            (200, "isotropic"): ("L=200, isotropic probing", "#ff7f0e", "s"),
            (100, "pu_null"): ("L=100, PU-null-space probing", "#2ca02c", "^")}
    for (L, pr), s in d.groupby(["L", "probing"]):
        lab, col, mk = spec[(L, pr)]
        ax.errorbar(s.loc_median_m, s.rate_eff, xerr=[np.zeros(len(s)), s.loc_p90_m - s.loc_median_m],
                    color=col, marker=mk, capsize=1.5, elinewidth=0.5, label=lab)
        if (L, pr) == (100, "isotropic"):
            for _, r in s.iloc[[0, 1, 3, 5, 9]].iterrows():
                ax.annotate(f"$\\rho$={r.rho:.2f}", (r.loc_median_m, r.rate_eff), xytext=(4, 2),
                            textcoords="offset points", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel("Localisation error, median (bar: to 90th pct.) (m)")
    ax.set_ylabel("Effective rate $(1-\\rho)R$ (bit/s/Hz)")
    ax.legend(loc="center right", fontsize=6)
    save(fig, "fig8_isac_tradeoff")
    keep = ["L", "probing", "rho", "Ls", "rate_eff", "angle_rmse_deg", "loc_median_m", "loc_p90_m", "loc_rmse_m",
            "loc_outage_5m", "echo_snr_db_median"]
    md_table(d[keep], TAB / "table3_isac_tradeoff")


def saturation():
    """How hard does each action head push against the tanh bound?

    E05's claim (the spec's theta = pi(u+1) head saturates) is re-checked against the SB3
    agent: the validation callback logs the fraction of action components with |u| > 0.999.
    """
    print("saturation (E05 / E15)")
    rows = []
    for tag, lab in (("abl_linearhead", "Linear head, theta = pi(u+1) (spec)"),
                     ("td3", "Circular head (cos, sin)"),
                     ("td3res", "Residual head (cos, sin) on co-phasing")):
        c = _curves(tag)
        if c is None or "val_saturated_frac" not in c:
            continue
        last = c.sort_values("step").groupby("seed").tail(1)
        rows.append({"Head": lab, "seeds": int(c.seed.nunique()),
                     "Saturated actions, final (%)": 100 * last.val_saturated_frac.mean(),
                     "Saturated actions, mean over training (%)": 100 * c.val_saturated_frac.mean(),
                     "Final validation rate": last.val_rate.mean()})
    if not rows:
        print("  skip (no learning curves)"); return
    tab = pd.DataFrame(rows)
    md_table(tab, TAB / "table4_action_saturation")
    print(tab.round(3).to_string(index=False))


def regime():
    print("regime (E02)")
    if have("e02_regime_check.csv"):
        d = pd.read_csv(CSV / "e02_regime_check.csv")
        md_table(d, TAB / "tableS1_regime_check")


# ------------------------------------------------------------------------------------ report macros
def numbers():
    """Write report/numbers.tex: every number quoted in the report comes from here."""
    print("numbers.tex")
    M = {}

    def put(name, val, fmt="{:.2f}"):
        M[name] = "??" if val is None or (isinstance(val, float) and not np.isfinite(val)) else fmt.format(val)

    def csv(f, **kw):
        return pd.read_csv(CSV / f, **kw) if (CSV / f).exists() else None

    t2 = pd.read_csv(TAB / "table2_main_comparison.csv") if (TAB / "table2_main_comparison.csv").exists() else None
    if t2 is not None:
        def row(prefix):
            r = t2[t2.Method.str.startswith(prefix)]
            return None if r.empty else r.iloc[0]
        for key, pre in (("NoRIS", "No RIS"), ("Rand", "Random"), ("CF", "Co-phasing"), ("EW", "Element-wise"),
                         ("SDR", "Joint SDR"), ("TD", "TD3 (pure"), ("TDres", "TD3-residual"),
                         ("TDlin", "TD3, spec tanh"), ("TDgam", "TD3, gamma"), ("TDlp", "TD3, learned P + linear"),
                         ("TDdb", "TD3, learned P + dB")):
            r = row(pre)
            if r is None:
                continue
            put(f"Rate{key}", r["Rate (bit/s/Hz)"]); put(f"Std{key}", r["± (seeds)"], "{:.3f}")
            put(f"Pct{key}", r["% of SDR"], "{:.1f}"); put(f"PctB{key}", r["% of SDP bound"], "{:.1f}")
            put(f"Viol{key}", 100 * r["Violation rate"], "{:.1f}"); put(f"Time{key}", r["Time/decision (ms)"], "{:.3g}")
            put(f"DRate{key}", r["Rate, 1-step-old CSI"]); put(f"DViol{key}", 100 * r["Violations, 1-step-old CSI"], "{:.1f}")
        meta = json.loads((TAB / "table2_meta.json").read_text())
        put("RateBound", meta["sdp_bound"]); put("RankMin", meta["sdr_rank_ratio_min"], "{:.4f}")
        put("NStates", meta["states"], "{:d}")
    rt = csv("table1_runtime_vs_N.csv")
    if rt is not None:
        g = rt.set_index(["method", "N"]).runtime_ms
        for m, k in (("sdr", "SDR"), ("elementwise", "EW"), ("td3", "TD"), ("closed_form", "CF")):
            for N in (8, 16, 32, 64):
                put(f"RT{k}{'ABCD'[[8,16,32,64].index(N)]}", g.get((m, N)), "{:.3g}")
        put("SpeedupSDRTD", g.get(("sdr", 64)) / g.get(("td3", 64)), "{:.0f}")
        for m, k in (("td3_actor_forward", "Fwd"), ("td3_model_predict", "Pred")):
            for N in (8, 16, 32, 64):
                put(f"RT{k}{'ABCD'[[8, 16, 32, 64].index(N)]}", g.get((m, N)), "{:.3g}")
    sat = (TAB / "table4_action_saturation.csv")
    if sat.exists():
        d = pd.read_csv(sat).set_index("Head")
        for key, head in (("Lin", "Linear head, theta = pi(u+1) (spec)"),
                          ("Circ", "Circular head (cos, sin)"),
                          ("Res", "Residual head (cos, sin) on co-phasing")):
            if head in d.index:
                put(f"Sat{key}", d.loc[head, "Saturated actions, final (%)"], "{:.1f}")

    e2 = csv("e02_regime_check.csv")
    if e2 is not None:
        e2 = e2.set_index("blockage_dB")
        put("BindZero", 100 * e2.loc[0, "binding_frac"], "{:.1f}"); put("BindTen", 100 * e2.loc[10, "binding_frac"], "{:.1f}")
        put("GainRISZero", e2.loc[0, "gain_ris_vs_noris"]); put("GainPUZero", e2.loc[0, "gain_pu_aware"])
        put("GainPUTen", e2.loc[10, "gain_pu_aware"])
    f1 = csv("fig1_rate_vs_N.csv")
    if f1 is not None:
        g = f1.set_index(["method", "N"]).rate
        put("CFeight", g[("closed_form", 8)]); put("CFsixtyfour", g[("closed_form", 64)])
        put("NoRISsixtyfour", g[("no_ris", 64)]); put("Randsixtyfour", g[("random", 64)])
    f8 = csv("fig8_isac_tradeoff.csv", keep_default_na=False)
    if f8 is not None:
        g = f8.set_index(["L", "probing", "rho"])
        put("RateISAC", f8.rate_full.iloc[0])
        put("LocIsoTen", g.loc[(100, "isotropic", 0.10), "loc_median_m"]); put("LocNullTen", g.loc[(100, "pu_null", 0.10), "loc_median_m"])
        put("LocIsoFifty", g.loc[(100, "isotropic", 0.50), "loc_median_m"]); put("LocNullFifty", g.loc[(100, "pu_null", 0.50), "loc_median_m"])
        put("OutIsoTen", 100 * g.loc[(100, "isotropic", 0.10), "loc_outage_5m"], "{:.0f}")
        put("OutNullTen", 100 * g.loc[(100, "pu_null", 0.10), "loc_outage_5m"], "{:.0f}")
        put("EchoIso", g.loc[(100, "isotropic", 0.10), "echo_snr_db_median"], "{:.1f}")
        put("EchoNull", g.loc[(100, "pu_null", 0.10), "echo_snr_db_median"], "{:.1f}")
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (ROOT / "report" / "numbers.tex").write_text("% AUTO-GENERATED by scripts/make_figures.py -- do not edit\n" + "\n".join(lines) + "\n")
    print(f"  {len(M)} macros")


if __name__ == "__main__":
    fig1(); sweep("fig2_rate_vs_Pmax.csv", "P_max_dBm", "Maximum UAV power $P_{\\max}$ (dBm)", "fig2_rate_vs_Pmax")
    sweep("fig3_rate_vs_Ith.csv", "I_th_dBm", "PU interference threshold $I_{\\mathrm{th}}$ (dBm)", "fig3_rate_vs_Ith")
    runtime(); fig4(); table2_fig5(); fig7(); fig8(); regime(); saturation(); numbers()
