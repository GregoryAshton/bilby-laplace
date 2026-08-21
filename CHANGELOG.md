# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions correspond to git tags; version numbers follow
[Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

- `fisher_method='waveform'` now supports phase/time/distance-marginalised likelihoods: the
  marginalised parameters are reinstated in the Fisher (evaluated at the injection, or reconstructed
  from the likelihood at the MAP) and marginalised out via the Schur complement of that block —
  equivalent to inverting the full precision and keeping the sampled-parameter sub-block. New
  `marginalized_reference` argument on `LaplacePosteriorEstimator` supplies the reference point.
  Calibration marginalisation (a discrete index) remains unsupported. The waveform precision is
  floored at the prior precision (generalising the unit-cube prior bound to parameter space via
  rescaling by the prior standard deviation), so unconstrained or phase-degenerate directions —
  e.g. the polarisation angle under phase marginalisation — fall back to prior width instead of
  producing a runaway variance. Detector-based sky frames (`reference_frame`, i.e. zenith/azimuth
  sampling) are supported: the Fisher applies the likelihood's zenith/azimuth-to-ra/dec conversion
  at each finite-difference point so those parameters are constrained rather than appearing null.
- SMC resampling via aspire (`resample='smc'`), including multi-mode discovery and Gaussian mixture proposals.
- `inprior` resampling mode — filters proposal samples to prior support without likelihood evaluation.
- Aligned initial samples for SMC — uses `_draw_inprior_samples()` helper to match rejection/importance sampling.
- `jacobian_cap_scale` parameter — caps the Jacobian at user-specified fraction of uniform-prior density, addressing prior-dominated parameters.
- `hessian_kwargs` parameter — forwards custom kwargs to `scipy.differentiate.hessian` for finer control.
- `sampling_cov` parameter — pass a precomputed covariance (parameter-named DataFrame or `(names, cov)` tuple) to use in place of the Laplace estimate.
- `max_iterations` parameter (default 1e6) — aborts rejection/importance sampling if acceptance rate drops below 1%.
- Unit-cube Hessian computation (`use_unit_cube=True`) — avoids boundary issues when MAP sits near prior edges.
- Validation of Hessian-derived covariance along principal axes — inflates eigendirections where posterior is wider than Gaussian predicts.
- `.github/workflows/pre-commit.yml` — automated code quality checks on push and PRs.
- `.pre-commit-config.yaml` — black, isort, flake8, mypy, trailing-whitespace, end-of-file-fixer hooks.
- Shared `examples/comparison.py` module — eliminates code duplication across example scripts.
- Multi-plot support in HLV example — intrinsic vs extrinsic parameter comparisons.
- Removed GW150914 example (real data download not practical for CI).

### Changed

- `fail_on_error` default changed to `True` (was `False`) — fails fast on sampling issues.
- Examples restructured: gaussian, rosenbrock, hlv (lorentzian kept from stub).
- Jacobian cap now scaled by `jacobian_cap_scale` parameter rather than hardcoded at 1.0.
- SMC pre-scan for rejection sampling uses in-prior-filtered calibration samples.
- Unit-cube Hessian finite-difference settings: `initial_step=0.001`, `step_factor=2`, `maxiter=10`
  (all overridable via `hessian_kwargs`).
- Renamed for terminological precision: the estimated matrix is the negative Hessian of the
  log-posterior (the posterior precision), not the Fisher information matrix. `matrix.py` →
  `laplace.py`; `FisherMatrixPosteriorEstimator` → `LaplacePosteriorEstimator`; `calculate_FIM` →
  `calculate_posterior_precision`; `calculate_iFIM` → `calculate_posterior_covariance`. No
  backwards-compatible aliases (alpha).

### Fixed

- **`Constraint` priors were ignored by the whole Laplace setup stage.** A `Constraint` bounds a
  *derived* quantity (`mass_1`/`mass_2` from a sampled `chirp_mass` and `mass_ratio`), so bilby
  applies it through the `PriorDict`'s `conversion_function` — and `LaplacePosteriorEstimator` never
  stored the `PriorDict`, only a plain per-parameter `priors_dict` from which the conversion
  function, `constraint_keys` and `evaluate_constraints` are all absent. `log_prior` was therefore a
  hand-rolled product of marginals, and everything built on it was blind to constraints: the MAP
  search (differential evolution over the raw prior *box*, plus `sample_subset` starting points),
  the parameter-space Hessian, the mode search and `log_evidence_laplace`. On the BNS_3G prior 39%
  of the box violates the mass constraint, so the MAP could be found — and the proposal centred and
  shaped — in a region of zero prior density. `log_prior` now delegates to `PriorDict.ln_prob`,
  which applies per-parameter support, constraints and the constraint normalisation in one call;
  MAP starting points come from `sample_subset_constrained`.
  Note the *sampling* stages were already correct: `rejection`, `importance`, `inprior` and `smc`
  all filter through `PriorDict.ln_prob` (the last via `aspire_bilby`'s log-prior), so their output
  samples always satisfied constraints. Only the Laplace stage that seeds them did not.
- `log_likelihood_from_array` now screens constraints as well as the per-parameter box, once per
  batch, before dispatching to the pool. Constraint-violating points return `-inf` and are never
  handed to the likelihood. `clip_to_bounds` still clips into the box but cannot rescue a
  constraint violation — there is no per-parameter projection onto a non-rectangular region.
- `log_evidence_laplace` summed marginals directly and so omitted bilby's
  `normalize_constraint_factor`, leaving it on a different normalisation from the
  rejection/importance/SMC evidences, which get the factor via `ln_prob`. On the BNS_3G prior the
  two differed by 0.50 nats. It now goes through `log_prior`.
- `_draw_initial_smc_samples`/`_draw_inprior_samples` applied the `prior_parameters` substitution
  *after* filtering to the prior support, so a fresh draw for a substituted parameter could push an
  already-accepted sample back outside the support — silently reintroducing exactly what the filter
  removes. The substitution now happens before the test, matching `_run_inprior`.
- Mode search: Latin hypercube starts that violate a constraint are dropped (after any
  `mode_search_subspace` pinning, so the configuration tested is the one actually used) rather than
  spent on a polish from a flat `-inf`.
- `resample=None` returned raw Gaussian draws with no support test at all; it now reports what
  fraction falls outside the prior support. The draws are still unfiltered — that is the documented
  meaning of this mode, and `resample='inprior'` is the filtered variant.

- Laplace covariance could collapse to the prior (posterior widths inflated ~250x). The unit-cube
  Hessian's `maxiter` had been raised to 20, driving `scipy.differentiate.hessian` into the
  objective's numerical-noise floor and producing a spurious indefinite curvature. Reverted to
  `maxiter=10`, and — more robustly — the unit-cube precision is now floored at the prior precision
  before inversion, so the covariance is bounded by the prior: a noisy, indefinite, or non-finite
  Hessian degrades to prior width instead of blowing up. A too-large `initial_step` no longer
  crashes the eigendecomposition.

### Removed

- The duplicated prior-bounds test inside `_pool_log_likelihood`, and the `bounds_min`, `bounds_max`
  and `clip_to_bounds` arguments it needed. Support is now decided once per batch in
  `log_likelihood_from_array`, so pool workers receive only in-support columns — one vectorised test
  per batch instead of one scalar test per task, and two fewer arrays bound into every task.
- The redundant bounds test in `log_posterior_from_array`: `log_prior` → `PriorDict.ln_prob` already
  returns `-inf` outside a parameter's range, and `log_posterior` short-circuits before the
  likelihood. The test it replaced was both duplicated and weaker (box only).
- Manual finite-difference Hessian fallback and the `fd_eps` parameter — `scipy.differentiate.hessian`
  (scipy ≥ 1.15) is now required and used unconditionally. Dropped the `packaging` dependency.

---

## [0.1.0] — 2026-03-02

Initial release.

### Added

- `Fisher` sampler class — a Bilby plugin that registers as `bilby.laplace` /
  `laplace` via the `bilby.samplers` entry-point group.
- `FisherMatrixPosteriorEstimator` — computes the MAP estimate and inverse Fisher
  Information Matrix (iFIM) covariance for a given likelihood and prior.
- FIM calculation using `scipy.differentiate.hessian` (scipy ≥ 1.15) with automatic
  finite-difference fallback for older scipy.
- Rejection sampling and importance sampling resampling modes.
- Batched sampling loop with a `tqdm` progress bar reporting per-batch efficiency.
- Optional corner diagnostic plot (`plot_diagnostic=True`) comparing the Gaussian
  proposal with the resampled posterior.
- `result.meta_data["run_statistics"]` populated with sampling efficiency, likelihood
  evaluation count, and wall-clock time.
- `examples/gaussian_example.py` — minimal runnable demo on a 2-D Gaussian likelihood.
- Version managed by `setuptools-scm` from git tags.

[Unreleased]: https://github.com/GregoryAshton/bilby-laplace/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GregoryAshton/bilby-laplace/releases/tag/v0.1.0
