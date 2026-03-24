# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install in editable mode (required before running anything):

```bash
pip install -e .
```

This registers the `bilby.laplace` entry-point so that `bilby.run_sampler(..., sampler="laplace")` works automatically.

## Running the examples

```bash
cd examples
make gaussian-laplace
make gaussian-compare
make help  # full list of targets
```

## Versioning

Versions are derived from annotated git tags via `setuptools-scm`. There is no hardcoded version in `pyproject.toml`.

- `src/bilby_laplace/_version.py` is auto-generated at install time and must not be committed (it is in `.gitignore`).
- To cut a release: `git tag -a vX.Y.Z -m "Release vX.Y.Z"` then push the tag.
- Dev installs between tags automatically get a `X.Y.Z.dev<n>+g<hash>` version string.

## Architecture

The package has two modules:

**`matrix.py` — `FisherMatrixPosteriorEstimator`**
Responsible for all maths: finding the MAP estimate, computing the Hessian of the log-posterior, and drawing Gaussian samples. The Hessian is computed using `scipy.differentiate.hessian` (scipy ≥ 1.15) or a finite-difference fallback. The key public methods are `get_MAP_sample()`, `calculate_iFIM()`, `sample_dataframe()`, and `log_likelihood_from_array()`. The last one accepts a `(N_params, N_samples)` array for vectorised likelihood evaluation, which is used in the sampling loop.

**`sampler.py` — `Laplace(Sampler)`**
A bilby `Sampler` subclass. The entry-point name is `bilby.laplace`. `run_sampler()` orchestrates the full workflow: build a `FisherMatrixPosteriorEstimator`, find the MAP, compute the inverse Hessian covariance, build a truncated Gaussian proposal, then run resampling (rejection, importance, inprior, or SMC) to correct for non-Gaussianity. Results are stored as `self.result.samples` (a numpy array) and `self.result.log_likelihood_evaluations`; bilby's `run_sampler()` then calls `result.samples_to_posterior()` to apply the conversion function and build the final posterior DataFrame.

The sampler registers via the `[project.entry-points."bilby.samplers"]` entry-point in `pyproject.toml`, following bilby's plugin convention. No changes to bilby itself are needed.
