# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions correspond to git tags; version numbers follow
[Semantic Versioning](https://semver.org/).

---

## [Unreleased]

## [0.2.0] — 2026-11-11

### Added

- Resampling methods: `resample="smc"` (via [aspire](https://github.com/bilby-dev/aspire)),
  `"inprior"` (draw from the proposal within the prior support, no reweighting) and `"emcee"`.
- Multimodal proposals: `n_modes`, with `mode_searches` (`"hypercube"`, `"symmetric"`,
  `"symmetric-polish"`, `"multistart"`), `mode_search_nsamples`, `mode_search_subspace`,
  `mode_separation_sigma`, `mode_symmetries`, `mode_symmetry_tol`, `mode_symmetry_polish_max`,
  `mode_multistart_nstarts` and `mode_weights`. Candidates from all searches are pooled and ranked
  by log-posterior before selection. The mixture is recorded in `meta_data["mode_mixture"]`,
  including which search proposed each mode.
- `proposal_covariance`: `"diagonal"` (independent truncated normals) or `"full"` (the correlated
  Gaussian restricted to the prior box). Recorded in `meta_data["proposal_covariance"]`.
- `fisher_method="waveform"`: the waveform Fisher matrix for gravitational-wave likelihoods,
  including phase/time/distance-marginalised likelihoods (via the Schur complement, with
  `marginalized_reference` on `LaplacePosteriorEstimator`) and detector-based sky frames.
- `map_restarts` (default 4): independent `differential_evolution` runs in the MAP search, keeping
  the best.
- `map_spike_guard` (default True) and `map_spike_delta`: reject MAP candidates that sit on a
  needle spike of the log-posterior.
- `map_vectorized` (default True): evaluate the `differential_evolution` population as one batch,
  so the MAP search uses `npool`.
- `smc_prior_flow`, `smc_prior_fraction`, `smc_progress` and `smc_plot_every` for SMC runs.
- `sampling_cov`: supply a precomputed covariance.
- `cov_scaling` accepts a per-parameter dict.
- `jacobian_cap_scale`, `hessian_kwargs`, `use_unit_cube` and `max_iterations` options.
- Validation of the Hessian-derived covariance along its principal axes; directions wider than
  the Gaussian predicts are inflated.
- Checkpoint and resume for the batched resampling methods and SMC.
- Parallel likelihood evaluation through `npool` / `pool`.
- `bilby_laplace.comparison`: JSD/EMD comparison of results against a reference, used by the
  examples.
- Examples: gaussian, rosenbrock, BBH_HLV, BNS_3G and AntiGlitch.

### Changed

- `resample="inprior"` uses `proposal_covariance="full"` by default; every other method uses
  `"diagonal"`.
- The multiprocessing pool is created before the MAP search.
- A secondary mode with a higher log-posterior than the primary MAP logs a warning.
- `fail_on_error` defaults to True.
- Unit-cube Hessian defaults: `initial_step=0.001`, `step_factor=2`, `maxiter=10`; the unit-cube
  precision is floored at the prior precision.
- Renamed to reflect that the estimated matrix is the posterior precision, not the Fisher
  information: `matrix.py` → `laplace.py`, `FisherMatrixPosteriorEstimator` →
  `LaplacePosteriorEstimator`, `calculate_FIM` → `calculate_posterior_precision`, `calculate_iFIM`
  → `calculate_posterior_covariance`.
- The `sample_array`/`sample_dataframe` sentinel for the expansion point is `sample="MAP"` (was
  `"maxL"`); other strings raise `ValueError`.

### Fixed

- `Constraint` priors are applied in the MAP search, Hessian, mode search and Laplace evidence
  (`log_prior` delegates to `PriorDict.ln_prob`; MAP starts use `sample_subset_constrained`).
- `log_evidence_laplace` includes bilby's constraint normalisation factor.
- `prior_parameters` substitution happens before the prior-support test in the SMC and in-prior
  draws.
- Latin hypercube starts that violate a constraint are dropped from the mode search.
- The hypercube mode search keeps the best polish of a region, not the first.
- An exact symmetry mirror is not rejected as a duplicate of its own source by the separation gate.
- The waveform Fisher uses the injection as the marginalised-parameter reference only when
  `use_injection_for_map=True`.
- Local MAP polishes wrap periodic parameters instead of treating their bounds as walls.
- `resample=None` reports the fraction of draws outside the prior support.
- The sampling seed reaches every stage, including SMC.

### Removed

- The manual finite-difference Hessian fallback and `fd_eps`; `scipy>=1.15` is required.
- `get_maximum_likelihood_sample()`, an alias of `get_MAP_sample()`.
- The bounds arguments of `_pool_log_likelihood`; support is tested once per batch.
- The GW150914 example (needs a data download).

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

[Unreleased]: https://github.com/GregoryAshton/bilby-laplace/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GregoryAshton/bilby-laplace/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GregoryAshton/bilby-laplace/releases/tag/v0.1.0
