# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions correspond to git tags; version numbers follow
[Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

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
- Finite-difference step for unit-cube Hessian reduced from 0.01 to 0.1.
- Renamed for terminological precision: the estimated matrix is the negative Hessian of the
  log-posterior (the posterior precision), not the Fisher information matrix. `matrix.py` →
  `laplace.py`; `FisherMatrixPosteriorEstimator` → `LaplacePosteriorEstimator`; `calculate_FIM` →
  `calculate_posterior_precision`; `calculate_iFIM` → `calculate_posterior_covariance`. No
  backwards-compatible aliases (alpha).

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
