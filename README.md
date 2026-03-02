# bilby-laplace

A [Bilby](https://bilby-dev.github.io/bilby/) sampler plugin that estimates posteriors
via the **Laplace approximation** — a Gaussian fitted at the maximum likelihood point
using the Fisher Information Matrix — followed by rejection or importance resampling to
correct for non-Gaussianity.

The method is fast, scales well to moderate dimensions, and produces asymptotically
exact posterior samples when the true posterior is close to Gaussian. It is particularly
useful as a cheap cross-check against nested sampling results.

The implementation is derived from
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
    sampler="laplace",          # or sampler="bilby.laplace"
    outdir="outdir",
    label="my_run",
)

result.plot_corner()
print(result.posterior)
print(result.meta_data["run_statistics"])
```

---

## How it works

1. **Maximum likelihood estimation** — `scipy.optimize.minimize` (Nelder-Mead by default)
   is run from multiple draws from the prior to locate the MAP estimate, avoiding local
   optima.

2. **Covariance estimation** — The Fisher Information Matrix (FIM) is computed at the MAP
   using `scipy.differentiate.hessian` (scipy ≥ 1.15) or a finite-difference fallback.
   Its inverse, the iFIM, serves as the Gaussian proposal covariance.

3. **Batched sampling** — Samples are drawn in batches from the Gaussian proposal
   `N(μ_MAP, iFIM)` until the target number of posterior samples is reached.

4. **Resampling** — Proposal samples are reweighted by
   `w ∝ L(θ) π(θ) / g(θ)` where `g` is the Gaussian proposal, then either:
   - **rejection** (default): accept each sample with probability `w / max(w)`
   - **importance**: resample `ESS` indices proportional to `w`

---

## Configuration

All keyword arguments are passed through `bilby.run_sampler`:

| Argument | Default | Description |
|---|---|---|
| `resample` | `'rejection'` | Resampling method: `'rejection'` or `'importance'` |
| `target_nsamples` | `10000` | Target number of posterior samples |
| `batch_nsamples` | `1000` | Proposal samples drawn per batch |
| `prior_nsamples` | `100` | Prior draws used in the max-likelihood search |
| `minimization_method` | `'Nelder-Mead'` | `scipy.optimize.minimize` method |
| `fd_eps` | `1e-6` | Finite-difference step size (relative to prior width) |
| `cov_scaling` | `1` | Multiplicative scale applied to the iFIM covariance |
| `use_injection_for_maxL` | `True` | Use `injection_parameters` as starting point if set |
| `plot_diagnostic` | `False` | Save a corner diagnostic plot of proposal vs posterior |
| `fail_on_error` | `False` | Raise an error (vs. log a warning) when sampling fails |

---

## Example

```bash
python examples/gaussian_example.py
python examples/gaussian_example.py --also-dynesty   # comparison run
```

---

## Limitations

- Works best when the posterior is approximately Gaussian and narrow relative to the
  prior. In highly non-Gaussian or multimodal problems, efficiency will be low.
- The Nelder-Mead max-likelihood search can fail in high dimensions (≳ 15 parameters).
  Provide `injection_parameters` (or a good starting point) to help.
- Evidence (`ln_Z`) is not computed; use nested sampling for evidence estimation.
