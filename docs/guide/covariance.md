# Covariance estimation

The proposal covariance is the most important ingredient: too narrow and the
proposal misses posterior mass, too wide and acceptance collapses. There are two
routes for estimating it, plus several options that shape the result.

## The two routes (`fisher_method`)

### `"hessian"` (default)

Finite-differences the scalar log-posterior at the MAP and inverts the result. Works
for **any** Bilby likelihood. This computes the *negative Hessian of the
log-posterior* — see [Background](../background/theory.md) for why this is the
observed information (including the prior), not the Fisher matrix.

The Hessian is computed with `scipy.differentiate.hessian` (adaptive step,
Richardson extrapolation). In unit-cube space (`use_unit_cube=True`) the parameter
scales are already normalised by the prior CDF transform, and the precision is
**floored at the prior precision** before inversion (a uniform prior has variance
1/12 in unit-cube coordinates). Because a bounded posterior can never be broader
than its prior, this bounds the covariance by the prior: a noisy, indefinite, or
non-finite curvature estimate degrades gracefully to prior width instead of blowing
up, while well-constrained directions (precision ≫ the prior) are untouched.

!!! note "`hessian_kwargs` and `maxiter`"
    `scipy.differentiate.hessian` *shrinks* the step each iteration
    (`initial_step / step_factor**k`), so a larger `maxiter` reaches a *smaller*
    final step. On a numerically noisy objective (e.g. a marginalised likelihood)
    an over-small step is dominated by round-off and the curvature estimate
    degrades — more iterations is not automatically better. The prior floor above
    keeps such a degraded estimate bounded, but the accurate regime is a step that
    is a modest fraction of the posterior width, not the smallest possible.

### `"waveform"` (gravitational-wave likelihoods)

Builds the genuine Fisher matrix

\[
F_{ij} = \sum_{\rm detectors} \mathrm{Re}\,(\partial_i h \mid \partial_j h)
\]

from derivatives of the projected detector strain, where \( (a\mid b) \) is the
noise-weighted inner product. This is **positive semi-definite by construction**,
needs only *first* derivatives of the waveform (far better behaved under finite
differencing than a scalar second derivative), and drops the noisy,
realisation-dependent term that can make the Hessian indefinite. The result is the
likelihood Fisher plus a diagonal prior precision, returned in parameter space.

```python
result = bilby.run_sampler(
    likelihood=gw_likelihood, priors=priors, sampler="laplace",
    fisher_method="waveform",
    fisher_kwargs=dict(eps=1e-6, eps_mass=1e-8),  # finite-difference steps
)
```

### Marginalised likelihoods

Phase, time, and distance marginalisation are supported. Those parameters are
*reinstated* in the Fisher — built over the augmented set (your sampled parameters
plus the marginalised ones), evaluated at reference values (the injection where
available, otherwise reconstructed from the likelihood at the MAP) — and then
removed via the **Schur complement** of the marginalised block. This is equivalent
to inverting the full precision and keeping the sampled-parameter sub-block, i.e.
it *marginalises* over those parameters (propagating their degeneracies) rather than
*conditioning* on (fixing) them.

The reduced precision is then floored at the prior precision (the same bound as the
unit-cube Hessian path, generalised to parameter space by rescaling with the prior
standard deviation `width / sqrt(12)`), so no marginal variance can exceed the prior.

Detector-based sky frames are handled: if the likelihood samples `zenith`/`azimuth`
(via `reference_frame`), the Fisher applies the likelihood's own conversion to
`ra`/`dec` at each finite-difference point, so those parameters are constrained
correctly rather than appearing as null directions.

!!! note "Accuracy of the marginalised block"
    The Schur complement reproduces the analytic marginalisation exactly only where
    the joint posterior is Gaussian in the marginalised parameters. Distance and time
    are usually well-behaved; **phase is not** — it is periodic and often poorly
    constrained, so its Gaussian approximation is weak and parameters degenerate with
    it (notably the polarisation angle `psi`) tend to be only prior-constrained. The
    prior floor keeps these bounded at prior width rather than letting them run away,
    but if you need such a parameter *tightly* constrained, prefer
    `fisher_method="hessian"`, which differentiates the actual marginalised
    likelihood.

!!! warning "Requirements and limitations"
    `fisher_method="waveform"` requires a `GravitationalWaveTransient`-like
    likelihood (exposing `interferometers` and `waveform_generator`). It **refuses**:

    - calibration marginalisation — it integrates a *discrete* calibration index, not
      a continuous Fisher direction;
    - reduced-order likelihoods (ROQ / relative-binning / multi-band), whose inner
      product differs from the full-resolution one used here.

    It works directly in parameter space, so `use_unit_cube` and
    `jacobian_cap_scale` are ignored.

## Conditioning

Before inversion the precision is **diagonally preconditioned** (rescaled by
\( \sqrt{\mathrm{diag}} \) to near-unit diagonal). This removes scale-driven
ill-conditioning — most effective in the parameter-space path, where raw parameter
scales differ by orders of magnitude. Small or negative eigenvalues are then floored
at a relative threshold, which keeps poorly-constrained or indefinite directions
*wide* (the right behaviour for a proposal) rather than zeroing them out. The
before/after condition number is logged.

In the unit-cube path this relative floor is a secondary safety net: the precision
has already been floored at the prior precision (see above), so those directions are
bounded at prior width rather than left arbitrarily wide.

## Shaping the proposal

| Option | Effect |
|---|---|
| `cov_scaling` | Multiplies the covariance (each value scales a parameter's *variance*, so `4` widens its sigma by `2x`). Pass a scalar to scale everything uniformly, or a dict like `{'chirp_mass': 4.0}` for per-parameter scaling (unlisted parameters default to `1.0`, or use the reserved `'others'` key to set that default, e.g. `{'chirp_mass': 4.0, 'others': 2.0}`). Off-diagonal terms scale by `sqrt(v_i*v_j)`, preserving correlations. Increase to widen the proposal when acceptance is low or the posterior is wider than the Gaussian predicts. |
| `sampling_cov` | Bypass estimation entirely and supply a precomputed covariance (see below). |
| `jacobian_cap_scale` | (Hessian unit-cube path) Caps the Jacobian for prior-dominated parameters; values `<1` widen the proposal for those parameters. |
| `prior_parameters` | Replace the proposal for listed parameters with independent prior draws — for parameters whose posterior is essentially the prior and which the Hessian constrains poorly. |
| `hessian_kwargs` | Forwarded to `scipy.differentiate.hessian` (e.g. `initial_step`). |

Additionally, the Hessian-derived covariance is validated along its principal axes:
at 1-sigma from the MAP the log-likelihood should drop by 0.5; directions where the
posterior is notably wider are inflated to match (never shrunk).

## Supplying your own covariance (`sampling_cov`)

If you already have a covariance — for example from a [gwfast](https://github.com/CosmoStatGW/gwfast)
or [GWFish](https://github.com/janosch314/GWFish) Fisher analysis — pass it directly
and skip the MAP-based estimate of the matrix (the MAP search still runs for the
mean):

```python
import pandas as pd

# Either a parameter-named DataFrame...
cov_df = pd.DataFrame(C, index=parameter_names, columns=parameter_names)
result = bilby.run_sampler(..., sampler="laplace", sampling_cov=cov_df)

# ...or a (names, matrix) tuple
result = bilby.run_sampler(..., sampler="laplace", sampling_cov=(parameter_names, C))
```

The names are validated against the model parameters (missing/extra/duplicate names,
wrong shape, non-symmetry, and non-positive-semi-definiteness all raise up front).
`cov_scaling` and the covariance validation still apply. Not compatible with
`n_modes > 1`.
