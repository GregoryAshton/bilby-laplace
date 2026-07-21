# How it works

The sampler runs a fixed pipeline: find the MAP, estimate the posterior
covariance there, build a Gaussian proposal, then (optionally) resample to
correct for non-Gaussianity.

## 1. MAP estimation

The maximum a posteriori (MAP) point maximises \( \log L(\theta) + \log \pi(\theta) \).
By default this uses `differential_evolution` (a global optimiser, recommended for
real data); `minimization_method="Nelder-Mead"` selects the legacy multi-start
local optimiser. If `injection_parameters` are set and `use_injection_for_map=True`,
they seed the search.

## 2. Covariance estimation

The default route (`fisher_method="hessian"`) finite-differences the
log-posterior with `scipy.differentiate.hessian` (scipy ≥ 1.15) and inverts the
result to obtain the proposal covariance. With `use_unit_cube=True` (default) the
Hessian is computed in unit-cube space via the prior CDFs, which avoids boundary
clipping when the MAP sits near a prior edge.

Before inversion the precision matrix is **diagonally preconditioned** (rescaled to
near-unit diagonal), which removes scale-driven ill-conditioning, and small or
negative eigenvalues are floored so that poorly-constrained or indefinite directions
become wide rather than singular. In the unit-cube path the precision is additionally
floored at the prior precision, so the covariance is bounded by the prior: a noisy or
indefinite Hessian degrades to prior width instead of producing a runaway proposal.

For gravitational-wave likelihoods, `fisher_method="waveform"` instead builds the
genuine Fisher matrix from waveform derivatives. See
[Covariance estimation](covariance.md) for both routes and the
[Background](../background/theory.md) for why they differ.

## 3. Proposal construction

A per-marginal **truncated Gaussian** proposal is built and clipped to the prior
bounds. Every draw therefore lands inside the prior support, even when the Gaussian
is much wider than the prior — eliminating wasted likelihood evaluations on
out-of-bounds samples. Off-diagonal correlations are recovered through the
likelihood during resampling.

## 4. Batched sampling

Samples are drawn in batches from the proposal until `target_nsamples` posterior
samples have been collected.

## 5. Resampling

Proposal samples are reweighted by \( w \propto L(\theta)\,\pi(\theta) / g(\theta) \),
where \( g \) is the proposal density. The `resample` option selects how the weights
are used — rejection, importance, in-prior filtering, SMC, or none. See
[Choosing a resampling method](resampling.md).

## 6. Evidence estimation

The Laplace log-evidence

\[
\log Z \approx \log L(\theta_{\rm MAP}) + \log \pi(\theta_{\rm MAP})
              + \tfrac{d}{2}\log(2\pi) + \tfrac{1}{2}\log\det\Sigma
\]

is always available. Rejection sampling and SMC additionally provide independent
evidence estimates with uncertainties.
