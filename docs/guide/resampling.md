# Choosing a resampling method

The Laplace step produces a Gaussian proposal. Because the true posterior is only
approximately Gaussian, the proposal samples are corrected by reweighting with
\( w \propto L(\theta)\,\pi(\theta)/g(\theta) \). The `resample` option selects how:

| `resample` | What it does | Evidence | Best when |
|---|---|---|---|
| `"rejection"` (default) | Accept each sample with probability \( w/\max w \). Exact reweighting. | ✅ independent estimate | The proposal reasonably covers the posterior; you want unbiased samples. |
| `"importance"` | Resample indices proportional to \( w \) (effective-sample-size per batch). | ✅ independent estimate | Acceptance is too low for rejection but the proposal still overlaps the posterior. |
| `"inprior"` | Draw from the proposal, keep only in-prior samples, evaluate the likelihood. No reweighting. Draws from the full correlated Gaussian by default (`proposal_covariance`). | ❌ | A fast filter when the proposal is well-matched to the prior; quick look at the Laplace approximation itself. |
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

`resample="smc"` seeds [aspire](https://github.com/bilby-dev/aspire) from the
Laplace proposal: a normalising flow is trained on proposal draws (or, with
`smc_prior_flow="laplace"`, the Laplace mixture is used directly) and aspire's
posterior sampler anneals from it toward the true posterior. Choose the sampler
with `smc_kwargs["sampler"]`; the default is aspire's `"importance"`, and an
SMC-family sampler such as `"smc"` or `"minipcn_smc"` is needed for annealing,
checkpointing and progress reporting. Configuration is passed through
`smc_kwargs`:

```python
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler="laplace",
    resample="smc",
    smc_kwargs=dict(
        sampler="smc",             # aspire posterior sampler (default "importance")
        n_initial_samples=1000,    # drawn from the Laplace proposal, passed to fit()
        n_samples=5000,            # SMC particle count (drives cost; = size of final posterior)
    ),
)
```

Any additional keys in `smc_kwargs` are forwarded directly to
`aspire.Aspire.sample_posterior()`.

### Multimodal posteriors

With `n_modes > 1`, secondary modes are searched for after the MAP and combined
with it into a mixture proposal, which every resampling method then draws from
(`resample=None` returns draws from the primary mode only):

- `mode_searches` chooses the searches: `"hypercube"` (Latin hypercube prior
  draws, `mode_search_nsamples` of them, optionally restricted to
  `mode_search_subspace`, ranked and then polished by a local optimiser),
  `"symmetric"` (the modes implied by `mode_symmetries`, e.g.
  `[("delta_phase", np.pi)]`), `"symmetric-polish"` and `"multistart"`.
  The default is `("hypercube", "symmetric")`.
- Candidates closer than `mode_separation_sigma` (default 3) primary-mode
  sigmas to a known mode in every parameter are treated as duplicates.
- `mode_weights` sets the component weights: `"equal"` (default) or
  `"laplace"`, the Laplace evidence of each mode.

For SMC, the mixture reaches aspire through the initial particle cloud, which
is drawn from each mode in proportion to its weight (every mode gets at least
one particle); with the default `smc_prior_flow="learned"`, aspire's flow is
trained on that cloud. `smc_prior_fraction` reserves a fraction of the cloud
for draws from the prior.
