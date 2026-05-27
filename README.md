# bilby-laplace

A [Bilby](https://bilby-dev.github.io/bilby/) sampler plugin that estimates posteriors
via the **Laplace approximation** — a Gaussian fitted at the maximum a posteriori (MAP)
point using the Hessian of the log-posterior — followed by optional resampling to
correct for non-Gaussianity.

The method is fast, scales well to moderate dimensions, and produces asymptotically
exact posterior samples when the true posterior is close to Gaussian. It is
useful as a cheap cross-check against nested sampling results.

> **Documentation:** full guides, the configuration/API reference, and background on
> the method are at **<https://gregoryashton.github.io/bilby-laplace/>**.

NOTE: This is currently in development and derived from
[bilby PR #933](https://github.com/bilby-dev/bilby/pull/933) (Gregory Ashton).

## Installation

```bash
pip install bilby-laplace
```

Or from source:

```bash
git clone https://github.com/GregoryAshton/bilby-laplace
cd bilby-laplace
pip install -e .
```

Bilby discovers the sampler automatically via its plugin entry-point system — no
further configuration is needed.

## Quick start

```python
import bilby

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="laplace",
    outdir="outdir",
    label="my_run",
)

result.plot_corner()
print(result.posterior)
```

## Features

- Several resampling strategies to correct for non-Gaussianity: `rejection`,
  `importance`, `inprior`, `smc` (via [aspire](https://github.com/bilby-dev/aspire)),
  or none.
- Two covariance routes: the numerical Hessian (`fisher_method="hessian"`, any
  likelihood) or the genuine waveform Fisher matrix (`fisher_method="waveform"`, for
  gravitational-wave likelihoods).
- Supply a precomputed covariance via `sampling_cov` (e.g. from gwfast / GWFish).
- Multimodal posteriors via SMC with `n_modes > 1`.
- Laplace log-evidence always available; rejection and SMC add independent estimates.

See the [documentation](https://gregoryashton.github.io/bilby-laplace/) for the full list
of options and guidance on choosing between them.

## Examples

Runnable examples are in [`examples/`](examples/) (gaussian, rosenbrock, BBH, BNS).
Each has a `Makefile`:

```bash
cd examples/gaussian
make laplace
make compare
```

## Documentation development

```bash
pip install -e ".[docs]"
mkdocs serve
```
