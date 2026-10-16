# RIS-Assisted UAV Cognitive Radio with Integrated Sensing (RIS-CR-DRL)

**Author: Rihem Fathallah **

A UAV relays data to a secondary user (SU) with the help of a Reconfigurable Intelligent
Surface (RIS), while sharing spectrum with a licensed primary user (PU) it must not interfere
with above a fixed threshold. The project compares closed-form, convex-optimisation, and
reinforcement-learning (TD3) controllers for the joint power / RIS-phase decision, and adds an
integrated sensing and communication (ISAC) study on top (MUSIC angle estimation of the PU from
the RIS echo).

Everything here — including the wrong turns — is written up as it happened in [LOG.md](LOG.md).
That log, not this README, is the real account of how the project was built.

## Tech stack

| Tool | Used for |
|---|---|
| **Python** | everything below |
| **NumPy** | channel model, system model, vectorised evaluation |
| **SciPy** | Jakes/Bessel control-period solve, MUSIC eigendecomposition, `.mat` export |
| **CVXPY** | the convex optimisation baseline (semidefinite relaxation), compiled once with DPP |
| **PyTorch** | neural networks and inference behind TD3 |
| **Stable-Baselines3** | the TD3 agent itself (`TD3("MlpPolicy", ...)`) |
| **Gymnasium** | the training environment API (`RisCrEnv`) |
| **GNU Octave** | an independent re-implementation of the channel model, used as a cross-check |
| pandas | results tables and CSV handling |
| matplotlib | all figures |
| pytest | the test suite |

Pinned versions are in [`requirements.txt`](requirements.txt).

## Why this exists

I wanted to check a specific claim from the project spec: that a UAV-RIS link is only useful to
a secondary user once blockage kills the direct path. The regime check in `LOG.md` (E02) says
that's true for the *unconstrained* rate but wrong for the *constrained* problem — at 0 dB
blockage the RIS is doing real work even with a weak SU-only gain, because it also steers energy
away from the PU and buys the UAV more transmit power. That's the operating point kept throughout
the rest of the project.

## What's in here

- `src/ris_cr/` — channel model, the convex optimisation baseline (semidefinite relaxation),
  the element-wise heuristic, TD3 (Stable-Baselines3 / PyTorch), sensing (MUSIC).
- `scripts/` — preflight checks, training queue, untrained-policy check, figure generation.
- `octave/` — an independent re-implementation of the channel model used to cross-check the
  Python one (E07).
- `tests/` — regression tests tied to specific bugs found along the way
  (e.g. `test_joint_sdr_dominates_spec_alternating`,
  `test_residual_head_with_zero_correction_is_co_phasing`).
- `results/` — generated CSVs, figures, and trained policy checkpoints.
- `report/main.tex` — the write-up. Compile with `make report` (needs a LaTeX install with
  `latexmk`) or read [report/main.pdf](report/main.pdf) directly.

## The controllers

| Controller | What it does |
|---|---|
| No RIS | direct links only — the floor |
| Random phases | i.i.d. uniform phases |
| Co-phasing (closed form) | SU-optimal phases, PU-agnostic; the analytic heuristic |
| Element-wise | coordinate ascent over 16 phase levels, initialised at co-phasing |
| **Joint SDR** | **the convex optimisation baseline (semidefinite relaxation)**: (P1) is rewritten exactly as a rank-constrained SDP in `W = P vv^H`; dropping the rank constraint gives one convex SDP whose value upper-bounds (P1), so it also certifies how far the others are from optimal |
| TD3 (pure) | Stable-Baselines3 TD3 with a circular `(cos, sin)` action head |
| TD3-residual | the same, but the action is a *correction* on top of co-phasing |

## Reproducing

```bash
pip install -e . && pip install -r requirements.txt
make all      # test, preflight, bank, train, eval, sensing, octave, figures, report
```

Training is the long pole. The queue (`scripts/run_training_queue.sh`) runs the specification's
budget — 5 seeds x 200k steps for each of the two main agents, 3 seeds x 100k for each of the
four ablations, 3.2M environment steps in total — as parallel single-threaded processes, at
most `JOBS` at a time (default: the number of cores). It is resumable: finished runs are skipped,
so re-running the script after an interruption picks up where it stopped.

```bash
JOBS=6 ./scripts/run_training_queue.sh                              # limit concurrency
STEPS_MAIN=50000 STEPS_ABL=25000 ./scripts/run_training_queue.sh    # a shorter budget
```

The runtime benchmark (`make eval` / `--part runtime`) must be run on an otherwise idle machine:
it measures time per decision, so competing load inflates every number in Table 1.

## Deviations from the specification

| Spec says | What this repo does | Why |
|---|---|---|
| Alternate SDR over phases and power | one *joint* SDP in `W = P vv^H` | the alternation provably stalls at its starting power whenever the interference constraint is active, and can start infeasible (E03) |
| `theta = pi (u + 1)` from a tanh actor | circular `(cos, sin)` head, `theta = atan2(s, c)` | the spec head saturates: `u = ±1` are the same phase but the critic extrapolates past the boundary (E05). Kept as ablation `abl_linearhead` |
| Learn `P` with an interference penalty | analytic power projection `P* = min(P_max, I_th/\|q_eff\|^2)` | optimal for fixed phases, so every action is feasible by construction and the penalty weight disappears. Kept as ablations `abl_learnP_*` (E05) |
| Evolve all links with Gauss-Markov | block fading on links whose both ends are static | RIS-SU and RIS-PU have no moving end (E01) |
| — | isotropic RIS elements | no single wall orientation puts the UAV circle, the SU and the PU all in front of the RIS (E01). Stated as a limitation, not fixed |
| — | observations phase-canonicalised by `e^{-j arg h_d}` | lossless, removes a nuisance rotation (E04, `test_canonicalisation_is_lossless`) |

## Known limitations

- RIS elements are treated as isotropic (E01).
- Static (ground-to-ground) links use block fading rather than continuously evolving, a
  deliberate deviation from a literal reading of the spec (E01).
- Perfect current CSI of the PU cascade is assumed; the cost of one-step-outdated CSI is
  reported alongside every method in Table 2.
- Target position error assumes the range is known.

## Report

The report is in [`report/main.tex`](report/main.tex), built with `make report`. The compiled
[`report/main.pdf`](report/main.pdf) is committed alongside it so it can be read without a LaTeX
toolchain.

## License

[MIT](LICENSE) © Rihem Fathallah
