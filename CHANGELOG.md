# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions correspond to git tags; version numbers follow
[Semantic Versioning](https://semver.org/).

---

## [Unreleased]

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

[Unreleased]: https://github.com/your-org/bilby-laplace/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/bilby-laplace/releases/tag/v0.1.0
