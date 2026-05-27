# bilby-laplace

A [Bilby](https://bilby-dev.github.io/bilby/) sampler plugin that estimates posteriors
via the **Laplace approximation** — a Gaussian fitted at the maximum a posteriori (MAP)
point — followed by optional resampling to correct for non-Gaussianity.

The method is fast, scales well to moderate dimensions, and produces asymptotically
exact posterior samples when the true posterior is close to Gaussian. It is useful
as a cheap cross-check against nested sampling results.

!!! note
    This project is in active development and derived from
    [bilby PR #933](https://github.com/bilby-dev/bilby/pull/933) (Gregory Ashton).

## Where to go next

- **[Installation](installation.md)** — install the package and let Bilby discover the sampler.
- **[Quick start](quickstart.md)** — run your first Laplace analysis.
- **[How it works](guide/how-it-works.md)** — the MAP → covariance → proposal → resampling pipeline.
- **[Choosing a resampling method](guide/resampling.md)** — `rejection`, `importance`, `inprior`, `smc`, or none.
- **[Covariance estimation](guide/covariance.md)** — the Hessian vs. waveform-Fisher routes and the options that shape the proposal.
- **[Background](background/theory.md)** — what is actually computed, and why it is *not* the Fisher information matrix.
- **[Configuration & API](reference/api.md)** — the full, auto-generated reference.

## At a glance

```python
import bilby

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="laplace",
    outdir="outdir",
    label="my_run",
)
```

The sampler registers through Bilby's plugin entry-point system, so no extra
configuration is needed once it is installed.
