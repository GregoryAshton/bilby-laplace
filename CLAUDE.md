# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install in editable mode (required before running anything):

```bash
pip install -e .
```

This registers the `bilby.laplace` entry-point so that `bilby.run_sampler(..., sampler="laplace")` works automatically.

## Running the example

```bash
python examples/gaussian_example.py
python examples/gaussian_example.py --also-dynesty  # adds a dynesty comparison run
```

## Versioning

Versions are derived from annotated git tags via `setuptools-scm`. There is no hardcoded version in `pyproject.toml`.

- `src/bilby_laplace/_version.py` is auto-generated at install time and must not be committed (it is in `.gitignore`).
- To cut a release: `git tag -a vX.Y.Z -m "Release vX.Y.Z"` then push the tag.
- Dev installs between tags automatically get a `X.Y.Z.dev<n>+g<hash>` version string.

## Architecture

The package has two modules:

**`matrix.py` — `FisherMatrixPosteriorEstimator`**
Responsible for all maths: finding the MAP estimate, computing the FIM, and drawing Gaussian samples. The FIM is computed using `scipy.differentiate.hessian` (scipy ≥ 1.15) or a finite-difference fallback. The key public methods are `get_maximum_likelihood_sample()`, `calculate_iFIM()`, `sample_dataframe()`, and `log_likelihood_from_array()`. The last one accepts a `(N_params, N_samples)` array for vectorised likelihood evaluation, which is used in the sampling loop.

**`sampler.py` — `Fisher(Sampler)`**
A thin bilby `Sampler` subclass. The entry-point name is `bilby.laplace`. `run_sampler()` orchestrates the full workflow: build a `FisherMatrixPosteriorEstimator`, find the MAP, compute the iFIM covariance, then run a batched loop drawing from the Gaussian proposal until `target_nsamples` are accepted. Resampling (rejection or importance) corrects for the mismatch between the Gaussian proposal and the true posterior. Results are written into `self.result.posterior` (a DataFrame) before returning.

The sampler registers via the `[project.entry-points."bilby.samplers"]` entry-point in `pyproject.toml`, following bilby's plugin convention. No changes to bilby itself are needed.
