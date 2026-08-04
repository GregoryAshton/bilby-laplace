# Quick start

Once installed, `laplace` is available as a Bilby sampler. Pass it to
`bilby.run_sampler` like any other sampler:

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
print(result.meta_data["run_statistics"])
```

## What you get back

- `result.posterior` — a `pandas.DataFrame` of posterior samples.
- `result.log_evidence` / `result.log_evidence_err` — the evidence estimate. The
  Laplace evidence is always computed; `resample="rejection"` and `resample="smc"`
  add independent estimates.
- `result.meta_data["run_statistics"]` — efficiency, number of likelihood
  evaluations, sampling time, and the Laplace log-evidence.

`result.log_evidence` is the full log Z, including the noise term, and
`result.log_bayes_factor` is that less `result.log_noise_evidence` — the same
convention every other bilby sampler uses, so the numbers are directly
comparable with (say) dynesty. This holds under `use_ratio` either way: the
estimator always evaluates the full likelihood, and the sampler converts to
bilby's expected footing on the way out. A constant offset cannot move the MAP
or the curvature, so `use_ratio` affects only the reported evidence, never the
posterior.

## Common adjustments

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="laplace",
    # --- the knobs you will reach for first ---
    resample="rejection",     # rejection | importance | inprior | smc | None
    target_nsamples=10000,    # how many posterior samples to return
    cov_scaling=1.0,          # widen (>1) the Gaussian proposal if acceptance is low
    plot_diagnostic=True,     # save proposal/resampling diagnostic plots
)
```

- If acceptance is low or the posterior is wider than the Gaussian predicts,
  increase `cov_scaling` to widen the proposal. See
  [Covariance estimation](guide/covariance.md).
- If the posterior is strongly non-Gaussian, prefer `resample="smc"`. See
  [Choosing a resampling method](guide/resampling.md).
- For a gravitational-wave likelihood you can switch to the genuine Fisher matrix
  with `fisher_method="waveform"`.

The full list of options is in the [Configuration & API reference](reference/api.md).
