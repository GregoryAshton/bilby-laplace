# Choosing a resampling method

The Laplace step produces a Gaussian proposal. Because the true posterior is only
approximately Gaussian, the proposal samples are corrected by reweighting with
\( w \propto L(\theta)\,\pi(\theta)/g(\theta) \). The `resample` option selects how:

| `resample` | What it does | Evidence | Best when |
|---|---|---|---|
| `"rejection"` (default) | Accept each sample with probability \( w/\max w \). Exact reweighting. | ✅ independent estimate | The proposal reasonably covers the posterior; you want unbiased samples. |
| `"importance"` | Resample indices proportional to \( w \) (effective-sample-size per batch). | ✅ independent estimate | Acceptance is too low for rejection but the proposal still overlaps the posterior. |
| `"inprior"` | Draw from the proposal, keep only in-prior samples, evaluate the likelihood. No reweighting. | ❌ | A fast filter when the proposal is well-matched to the prior; quick look. |
| `"smc"` | Use the Laplace Gaussian as the starting distribution for [aspire](https://github.com/bilby-dev/aspire) SMC, which anneals toward the true posterior. | ✅ SMC estimate | The posterior is strongly non-Gaussian or multimodal. |
| `None` / `"None"` | Return raw Gaussian samples, no correction. | Laplace only | Debugging, or when you genuinely want the Gaussian approximation. |

## Rejection vs. importance

Both produce a corrected posterior. **Rejection** gives independent, unbiased
draws but its efficiency falls as the proposal/posterior mismatch grows. A pre-scan
batch fixes the rejection bound before any sample is accepted, so the bound cannot
shift mid-run. If too many samples exceed the bound, the posterior may be slightly
biased — widen the proposal with `cov_scaling`.

**Importance** resampling tolerates a worse match by drawing proportional to the
weights, at the cost of duplicated samples when the effective sample size is low.

If acceptance/efficiency is very low, the run aborts once `max_iterations` proposal
samples have been drawn with acceptance below 1%.

## SMC

`resample="smc"` hands the Laplace Gaussian to aspire as a `prior_flow` and runs an
SMC sampler that iteratively refines samples toward the true posterior. This is the
most robust choice for non-Gaussian posteriors. Configuration is passed through
`smc_kwargs`:

```python
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler="laplace",
    resample="smc",
    smc_kwargs=dict(
        sampler="importance",      # aspire posterior sampler
        n_initial_samples=1000,    # drawn from the Laplace proposal, passed to fit()
        n_final_samples=5000,      # output samples requested
    ),
)
```

Any additional keys in `smc_kwargs` are forwarded directly to
`aspire.Aspire.sample_posterior()`.

### Multimodal posteriors

With `resample="smc"` and `n_modes > 1`, the optimiser restarts from multiple prior
draws (Latin hypercube), distinct MAP estimates are deduplicated by a 3-sigma
separation, and they are combined into an equal-weight Gaussian mixture used as the
SMC starting distribution. `mode_search_nsamples` controls how many prior draws are
evaluated when searching for secondary modes.
