# Experiment log

Conventions: every entry states *what was run*, *what came out* (numbers copied from the
script output, never retyped from memory), and *what was decided*. Commands are relative to
the repository root. Machine for every number below: 1 CPU core, 3 GB RAM, no GPU.

---

## E00 — Link-budget pre-flight (2026-09-18)

`python scripts/link_budget.py --blockage 0 10 15`

Deterministic path gains, UAV at 0° on its circle (30, 0, 100):

| link | UAV-RIS | RIS-SU | RIS-PU | UAV-SU | UAV-PU |
|---|---|---|---|---|---|
| gain (dB) | -73.5 | -60.5 | -62.9 | -102.9 | -102.5 |

Empirical, from the generator (noise units, dB), blockage 0:

| N | E\|h_d\|² | E(Σ\|a_n\|)² | RIS / direct | median max SNR gain |
|---|---|---|---|---|
| 8 | 29.3 | 15.8 | -13.5 | +2.0 |
| 16 | 28.6 | 21.6 | -7.0 | +3.6 |
| 32 | 28.6 | 27.6 | -1.0 | +6.2 |
| 64 | 28.7 | 33.6 | +4.9 | +9.8 |

No-RIS interference at P_max = 20 dBm: **-82.5 dBm**, i.e. 7.5 dB above I_th = -90 dBm, so the
PU constraint is active at the default operating point. The spec's warning box is right: the
direct link dominates the *SU-only* gain for N ≤ 32. Whether that makes the study meaningless
depends on the constraint — see E02.

## E01 — Channel-model decisions (2026-09-18)

* **Carrier.** C0 = -30 dB at 1 m is the free-space value for λ = 4π·10^(C0/20) = 0.397 m
  (fc ≈ 754 MHz). λ is derived from C0 rather than chosen separately, so the LoS phases and
  the radar constant are consistent with the path loss.
* **Control period.** The spec fixes ρ = 0.9 but not the step duration. Jakes' model gives
  ρ = J0(2π f_D Δt), with f_D = v/λ = 12.6 Hz, so Δt = **8.10 ms**
  (`dt_s: auto`, `test_dt_is_jakes_consistent`). One 200-step episode is therefore 1.6 s,
  or 8.1 m of flight. The episode start angle is random, so the test bank still covers the
  whole circle.
* **Block fading on static links.** RIS-SU and RIS-PU have both ends static. Their NLoS part
  is drawn once per episode and held fixed, and only links touching the UAV evolve
  (Gauss-Markov). A literal reading of the spec would also evolve the static links.
* **Isotropic RIS elements.** I checked whether a single wall orientation can put the UAV
  circle, the SU and the PU all in front of the RIS: it cannot, because the SU (x = 65) and
  the PU (x = 55) sit on opposite sides of x = 60 and the UAV is above. Element patterns are
  therefore ignored, as in Wu & Zhang (2019). This is a stated limitation.
* **Gotcha.** PyYAML parses `1.0e6` as a *string* (YAML 1.1 requires a signed exponent), and
  the noise computation crashed on it. Fixed in the YAML (`1.0e+6`), with a `float()` cast
  added in code.

## E02 — Is the operating point interesting? (2026-09-18)

`python scripts/regime_check.py --blockage 0 5 10` (200 test-like states, same flights)

| blockage (dB) | PU constraint binding | no RIS | co-phasing | element-wise | SDR | SDR − no RIS | SDR − co-phasing |
|---|---|---|---|---|---|---|---|
| 0 | 81.5 % | 4.301 | 5.519 | 5.772 | 5.787 | +1.485 | +0.268 |
| 5 | 44.0 % | 3.785 | 5.651 | 5.797 | 5.814 | +2.028 | +0.163 |
| 10 | 7.5 % | 2.733 | 5.314 | 5.332 | 5.336 | +2.603 | +0.022 |

**Decision: keep the spec geometry (`blockage_dB = 0`).** Before any code existed, I had
recommended adding 10–15 dB of blockage. The data says that was wrong for the *constrained*
problem: at 10 dB the constraint almost never binds, every method collapses to closed-form
co-phasing (PU-aware gain 0.02 bit/s/Hz), and a DRL comparison would be empty. At 0 dB the
RIS is worth +1.49 bit/s/Hz, even though its SU-only gain is small, because it also steers
energy *away* from the PU and so lets the UAV transmit more power.

## E03 — SDR: joint formulation vs the spec's alternation (2026-09-18)

* **Joint SDP.** Substitute W = P·vvᴴ. Since |v_n| = 1, every diagonal entry of W equals P,
  and (P1) becomes one SDP after dropping rank(W) = 1. It is compiled once with CVXPY
  parameters (DPP) and solved with Clarabel in units of P_max.
* **The relaxation is tight.** On the smoke-test states, λ_max/tr(W) = 1.0000, so the rounded
  solution equals the SDP bound: SDR reaches the *global optimum* of (P1) there. The full-bank
  statistic is in `results/tables/table2_meta.json`.
* **Spec alternation, bug 1.** The first iteration uses P = P_max. When the RIS cannot cancel
  enough of the direct PU path at full power, that fixed-P SDP is **infeasible**
  (RuntimeError on the first try).
  Fix: start from the projected power of the co-phasing solution, which is always feasible.
* **Spec alternation, bug 2 (structural).** When the interference constraint is active, the
  fixed-P SDR uses the whole interference budget, so the power update returns the same P.
  The starting point is therefore a fixed point. Over 20 states at blockage 0: alternating
  **5.364** vs joint **5.656** bit/s/Hz, 702 ms vs 141 ms per decision.
  `test_joint_sdr_dominates_spec_alternating` keeps this true.

## E04 — Tooling: TD3 in JAX instead of Stable-Baselines3 (2026-09-18)

`pip install torch` pulls the CUDA wheel plus about 3 GB of `nvidia-*` libraries, which
overflowed the 10 GB disk. Without them `import torch` fails in `_preload_cuda_lib`.
JAX's PyPI wheel is CPU-only, so `ris_cr/rl/td3_jax.py` reimplements TD3 and mirrors SB3 v2
line by line:

* nn.Linear initialisation;
* Adam with eps = 1e-8;
* summed twin-critic MSE;
* target smoothing noise 0.2, clipped at 0.5;
* policy delay 2 for the actor *and* the targets, τ = 0.005;
* uniform actions during `learning_starts`;
* bootstrapping through time-limit truncation.

`gymnasium.utils.env_checker.check_env` replaces SB3's checker, which imports torch.
Deployed inference is plain numpy and is tested equal to the JAX forward pass.

## E05 — The spec's action parameterisation saturates (2026-09-18)

With θ = π(u+1) and a tanh actor (10k-step smoke run, N = 16):

* **100 %** of actions sat at |u| > 0.999, with a mean pre-tanh |z| of **18.6**;
* the validation rate was frozen at **4.2039** at both 5k and 10k steps, the same as random
  phases.

Cause: u = −1 and u = +1 are the same phase, but the critic does not know it. It extrapolates
Q linearly past the boundary, and the actor follows that gradient into saturation.

**Fix:** a circular head. The actor outputs one (c, s) pair per element, normalised onto the
unit circle, and θ = atan2(s, c). There is no boundary, and the critic is periodic in θ by
construction. A small radial regulariser (1e-3) keeps the pre-normalisation head well scaled.
Smoke run: 4.96 / 5.04 / **5.07** at 5k / 10k / 15k steps. The spec's head is kept as
ablation `abl_linearhead`.

Also: the observation is phase-canonicalised (every channel multiplied by e^{-j arg h_d}).
This is lossless (`test_canonicalisation_is_lossless`) and removes a nuisance rotation.

## E06 — Compute budget and training protocol (2026-09-18)

Throughput is 105–135 env steps/s alone, and 37–41 steps/s while the SDR evaluation shares
the core. The spec budget (5 seeds × 200k) would take about 2.5 h of pure compute.
Protocol used:

* main pure TD3: 2 seeds × 60k;
* physics-informed residual TD3 (E09): 3 seeds × 30k;
* each ablation: 1 seed × 20k, compared with the main runs *at equal step counts*.

Model selection uses the **validation** flights (seeds 50–51) every 5k steps. The **test**
flights (seeds 100–104, 20 × 200 decisions) are only touched by `evaluate.py`.

Lesson (infrastructure): background jobs were killed when the shell that launched them
returned, and one queue silently died with an empty log. Jobs are now started with `setsid`,
and the queue is resumable (`--skip-existing`).

## E07 — Octave cross-check (2026-09-18)

`octave-cli octave/rate_vs_N.m`

* **(A) Identical realisations** exported by Python: max |Δrate| = **1.8e-15** for no-RIS,
  random and co-phasing, at every N.
* **(B) Independent Octave channel generator** (`channel_model.m`, Octave's own RNG),
  2000 i.i.d. snapshots per N, compared with Python's i.i.d. estimate:

| N | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| Octave | 6.406 ± 0.028 | 6.980 ± 0.024 | 7.864 ± 0.018 | 9.096 ± 0.013 |
| Python | 6.436 ± 0.029 | 7.006 ± 0.024 | 7.887 ± 0.018 | 9.082 ± 0.014 |

All differences are under 1σ of the combined standard error. `tests/test_octave_params.py`
prevents the Octave parameter mirror from drifting away from the YAML.

## E08 — ISAC (2026-09-18)

* **MUSIC with fewer snapshots than antennas does not break down for a single target.** Only
  the principal eigenvector is needed. What changes is the threshold SNR: at 0 dB, the angle
  RMSE is 5.27° for L_s = 5 vs 0.84° for L_s = 8. This corrects the spec's pitfall list and
  my own earlier claim.
* **Efficiency and grid floor.** MUSIC reaches √CRB at high SNR, then flattens onto the 0.1°
  grid floor (Δ/√12 = 0.0289°): at 20 dB with L_s = 50 the RMSE is 0.0298° against a
  √CRB of 0.0284°.
* **First trade-off run was non-monotone in ρ.** Angle RMSE 0.93° at ρ = 0.25, then 2.50° at
  ρ = 0.30. Two causes: independent noise per ρ (no common random numbers), and RMSE being
  dominated by rare threshold outliers. Fix: one probing/noise sequence per trial, nested
  across ρ, with the median, 90th percentile and 5 m outage reported next to RMSE.
* **Gotcha.** pandas read the label `"null"` as NaN. The label is now `pu_null`, read with
  `keep_default_na=False`.
* **Physics of the trade-off.** With M = 8 the UAV zero-forces the PU for communication, so
  the constraint is inactive for data (R ≈ 9.5 bit/s/Hz). *Isotropic* probing is still
  interference-limited (median echo SNR −4.0 dB). Confining the probing to the PU null space
  lets sensing use P_max (−0.06 dB). At ρ = 0.10 that cuts the median localisation error from
  2.96 m to 1.45 m, roughly the benefit of doubling the frame length.

## E09 — Pure TD3 plateaus below the analytic heuristic → residual policy (2026-09-18)

Main run, seed 0, on the validation flights:

| step | 10k | 15k | 20k | 25k | 30k |
|---|---|---|---|---|---|
| validation rate (bit/s/Hz) | 5.039 | 5.071 | 5.083 | 5.106 | 5.166 |

References on the same flights: co-phasing 5.537, element-wise 5.805, SDR 5.824. Pure TD3
is still **below the 0.02 ms closed-form co-phasing**. The problem is myopic, and co-phasing
already captures most of the achievable rate, so the agent spends most of its budget
rediscovering it.

**Physics-informed residual head:** θ_n = θ_n^cf + atan2(s_n, c_n). The agent only has to
learn the PU-aware deviation. With a zero correction it reproduces co-phasing exactly
(`test_residual_head_with_zero_correction_is_co_phasing`).

## E13 — Port from JAX to Stable-Baselines3 (2026-09-20)

The project description promises PyTorch + Stable-Baselines3; E04 explains why the original
build machine could not install torch. On the machine used for this re-run
(Intel i7-13620H, 10 physical cores, 32 GB RAM, Windows 11) `pip install torch` and
`pip install stable-baselines3` both succeed, so the JAX agent was removed and training now
runs on `stable_baselines3.TD3("MlpPolicy", ...)` with the hyper-parameters already in
`configs/default.yaml`. Versions: torch 2.10.0+cpu, stable-baselines3 2.9.0, Python 3.13.7.

What moved where:

* `src/ris_cr/rl/sb3.py` (new) holds everything torch-specific: model construction from the
  YAML, the residual-head bias, the validation callback, single-threaded actor inference.
* `src/ris_cr/rl/core.py` (ObsBuilder, ActionMap) and `env.py` are unchanged in behaviour and
  still know nothing about the learning framework. All three action encodings survive.
* `src/ris_cr/rl/td3_jax.py` is deleted; `jax`/`optax` are out of `requirements.txt`.
* `gymnasium.utils.env_checker.check_env` is replaced by
  `stable_baselines3.common.env_checker.check_env`, in training and in the tests.
* Checkpoints are now SB3 `.zip` archives (`model_best.zip` / `model_final.zip`) instead of
  `actor_*.npz`, so `TD3Controller.from_run` loads a model rather than a list of weight arrays.

**Two JAX-only devices were dropped, deliberately.** The circular and residual heads used to
project each (cos, sin) pair onto the unit circle and carried a radial regulariser
(`radial_reg`, E05) to keep that projection well-conditioned. Neither is needed with SB3's
tanh actor: the decoding is `atan2(s_n, c_n)`, which is *scale-invariant*, so a tanh-bounded
pair of any radius still reaches every direction in the plane. `radial_reg` is removed from the
config and `test_circular_head_decodes_angles_and_is_scale_invariant` pins the property.

**Gotcha.** SB3's actor MLP is `actor.mu = Sequential(Linear, ReLU, Linear, ReLU, Linear, Tanh)`,
so `mu[-1]` is the *Tanh*, not the output Linear. The first version of the residual bias and of
the controller's shape assertion both indexed `mu[-1]` and failed with
`'Tanh' object has no attribute 'out_features'`. `sb3.output_layer()` now walks back to the last
`Linear`.

**Saturation is still measured.** The unit-circle projection used to hide how the linear (spec)
head saturates; the validation callback now logs the fraction of action components with
|u| > 0.999 in `learning_curve.csv` (`val_saturated_frac`), so the E05 finding stays checkable
against the SB3 agent rather than being inherited from the JAX one.

## E14 — Anchoring the untrained residual policy on co-phasing (2026-09-20)

`python scripts/untrained_policies.py` → `results/csv/untrained_policies.csv`

The residual head decodes theta_n = theta_n^cf + atan2(s_n, c_n), so it only reproduces
co-phasing if the decoded correction starts at zero. An untrained SB3 actor does not: its
output layer has the default `nn.Linear` initialisation, and the decoded angle is essentially
uniform. Adding **+2.0** to the pre-tanh bias of every cos component (even entries of the
action vector), in the actor *and* the actor target, pushes each pair towards (c, s) = (+1, 0).

Validation bank (seeds 50-51, 4 flights x 200 decisions), co-phasing reference **5.537099**:

| policy | seed 0 | seed 1 | seed 2 | mean abs. deviation from co-phasing |
|---|---|---|---|---|
| untrained residual, +2.0 cos bias | 5.533744 | 5.533319 | 5.533527 | 2.7-3.3 deg |
| untrained residual, no bias | 4.496447 | 4.308858 | 4.314318 | 91-94 deg |
| untrained circular, no bias | 4.350022 | 4.392699 | 4.331907 | 90-92 deg |

So the bias works, but **"equal to co-phasing" is an approximation, not an identity**: the
untrained residual policy sits **0.0034-0.0038 bit/s/Hz below** co-phasing (0.06 %), because
tanh of a small random pre-activation is small but not zero, leaving a ~3 deg residual
correction per element. The exact identity holds for the *decoding*, not for an untrained
network, and that is what `test_residual_head_with_zero_correction_is_co_phasing` asserts;
`test_untrained_residual_policy_is_anchored_on_co_phasing` asserts the weaker, true statement
(biased deviation < 1/4 of unbiased, and < 15 deg).
