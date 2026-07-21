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
  producing a runaway variance.
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

- Laplace covariance could collapse to the prior (posterior widths inflated ~250x). The unit-cube
  Hessian's `maxiter` had been raised to 20, driving `scipy.differentiate.hessian` into the
  objective's numerical-noise floor and producing a spurious indefinite curvature. Reverted to
  `maxiter=10`, and — more robustly — the unit-cube precision is now floored at the prior precision
  before inversion, so the covariance is bounded by the prior: a noisy, indefinite, or non-finite
  Hessian degrades to prior width instead of blowing up. A too-large `initial_step` no longer
  crashes the eigendecomposition.

### Removed

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
