# bilby-laplace

A [Bilby](https://bilby-dev.github.io/bilby/) sampler plugin that estimates posteriors
via the **Laplace approximation** — a Gaussian fitted at the maximum a posteriori (MAP)
point using the Hessian of the log-posterior — followed by optional resampling to
correct for non-Gaussianity.

The method is fast, scales well to moderate dimensions, and produces asymptotically
exact posterior samples when the true posterior is close to Gaussian. It is
useful as a cheap cross-check against nested sampling results.

NOTE: This is currently in development and derived from
[bilby PR #933](https://github.com/bilby-dev/bilby/pull/933) (Gregory Ashton).

---

## Installation

```bash
pip install bilby-laplace
```

Or, to install from source:

```bash
git clone https://github.com/your-org/bilby-laplace
cd bilby-laplace
pip install -e .
```

Once installed, Bilby discovers the sampler automatically via its plugin entry-point
system — no further configuration is needed.

---

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
print(result.meta_data["run_statistics"])
```

---

## How it works

1. **MAP estimation** — The maximum a posteriori (MAP) point is found by
   maximising `log L(θ) + log π(θ)`. By default uses `differential_evolution`
   (a global optimizer); alternatively multi-start `Nelder-Mead` can be used.

2. **Covariance estimation** — The Hessian of the log-posterior is computed at
   the MAP using `scipy.differentiate.hessian` (scipy ≥ 1.15) or a
   finite-difference fallback. Its inverse serves as the Gaussian proposal
   covariance. When `use_unit_cube=True` (default), the Hessian is computed in
   unit-cube space via the prior CDFs, avoiding boundary issues for parameters
   near prior edges.

3. **Proposal construction** — A per-marginal truncated Gaussian proposal is
   built, clipped to the prior bounds. This ensures all drawn samples fall
   within the prior support, even when the Gaussian approximation is much wider
   than the prior.

4. **Batched sampling** — Samples are drawn in batches from the truncated
   Gaussian proposal until the target number of posterior samples is reached.

5. **Resampling** — Proposal samples are reweighted by
   `w ∝ L(θ) π(θ) / g(θ)` where `g` is the proposal density, then either:
   - **rejection**: accept each sample with probability `w / max(w)`
   - **importance**: resample `ESS` indices proportional to `w`
   - **smc**: use the Laplace Gaussian as a starting distribution for
     [aspire](https://github.com/bilby-dev/aspire) SMC posterior sampling,
     which iteratively refines samples toward the true posterior
   - **inprior**: draw from the proposal and keep only samples within the
     prior support, evaluating the likelihood for retained samples. Useful
     as a fast filter when the proposal is well-matched to the prior.
   - **None**: skip resampling entirely and return raw Gaussian samples

6. **Evidence estimation** — The Laplace log-evidence
   `log Z ≈ log L(θ_MAP) + log π(θ_MAP) + (d/2) log(2π) + (1/2) log det(Σ)`
   is always computed. Rejection sampling and SMC provide independent evidence
   estimates.

---

## Configuration

All keyword arguments are passed through `bilby.run_sampler`:

| Argument | Default | Description |
|---|---|---|
| `resample` | `'rejection'` | Resampling method: `'rejection'`, `'importance'`, `'smc'`, `'inprior'`, or `None` |
| `target_nsamples` | `10000` | Target number of posterior samples |
| `batch_nsamples` | `1000` | Proposal samples drawn per batch |
| `prior_nsamples` | `100` | Prior draws used in the MAP search (multi-start only) |
| `minimization_method` | `'differential_evolution'` | `scipy.optimize` method for MAP finding |
| `cov_scaling` | `1` | Multiplicative scale applied to the Laplace covariance |
| `use_injection_for_map` | `True` | Use `injection_parameters` as MAP starting point if set |
| `use_unit_cube` | `True` | Compute the Hessian in unit-cube space via prior CDFs |
| `jacobian_cap_scale` | `1.0` | Scale the Jacobian cap for prior-dominated parameters (< 1 widens proposal) |
| `hessian_kwargs` | `None` | Dict of kwargs forwarded to `scipy.differentiate.hessian` |
| `plot_diagnostic` | `False` | Save diagnostic plots (proposal vs prior, SMC stages) |
| `fail_on_error` | `True` | Raise an error when sampling fails (vs. log a warning) |
| `n_modes` | `1` | Number of posterior modes to search for (SMC only) |
| `mode_search_nsamples` | `500` | Prior draws for multi-mode search (`n_modes > 1`) |
| `max_iterations` | `1e6` | Maximum number of proposal samples before aborting (rejection/importance only) |
| `smc_kwargs` | `None` | Dict of aspire SMC configuration (see docstring for keys) |
| `save` | — | Result file format: `'hdf5'`, `'json'`, etc. |

---

## Examples

Example scripts are provided in the `examples/` directory, each supporting
multiple samplers and a `--compare` mode that prints a summary table and
corner plot:

```bash
cd examples

# Gaussian (2D, exact match for Laplace)
make gaussian-laplace
make gaussian-dynesty
make gaussian-compare

# Rosenbrock (2D, non-Gaussian — stress test)
make rosenbrock-smc
make rosenbrock-compare

# HLV BBH injection (simulated GW data, no download needed)
make hlv-laplace
make hlv-smc
make hlv-compare
```

Run `make help` in the examples directory for the full list of targets.
