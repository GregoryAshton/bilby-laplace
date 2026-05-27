# Fisher, Hessian & precision

A common point of confusion: the matrix the default route computes is **not** the
Fisher information matrix, even though such code is often loosely called "the Fisher
matrix". This page makes the distinction precise, because it explains both the
naming in the code and why the `"waveform"` route can give a better-behaved result.

## The decomposition

For a Gaussian likelihood with mean \( \mu(\theta) \), data \( d \), and noise
covariance \( C \), write the residual \( r = d - \mu(\theta) \). The second
derivative of the log-likelihood splits cleanly:

\[
-\partial_i \partial_j \log L
  = \underbrace{(\partial_i \mu)^{\mathsf T} C^{-1} (\partial_j \mu)}_{\textstyle F_{ij}\ \text{(Fisher)}}
  \;-\; \underbrace{r^{\mathsf T} C^{-1} (\partial_i \partial_j \mu)}_{\textstyle \text{residual}\times\text{curvature}}
\]

For gravitational waves the noise-weighted inner product
\( (a\mid b) = 4\,\mathrm{Re}\int a^* b / S_n\, \mathrm{d}f \) **is** \( a^{\mathsf T} C^{-1} b \),
so with \( \mu \to h(\theta) \):

\[
-\partial_i \partial_j \log L = (\partial_i h \mid \partial_j h) - (r \mid \partial_i \partial_j h).
\]

Three distinct objects appear:

| Object | Definition | Properties |
|---|---|---|
| **Fisher information** \( F \) | \( \mathbb{E}[-\partial^2 \log L] = (\partial_i h\mid\partial_j h) \) | expectation; model + noise only; positive semi-definite; data-independent |
| **Observed information** \( J \) | \( -\partial^2 \log L \) at a point | single realisation; \( = F - (r\mid\partial^2 h) \); can be indefinite |
| **Posterior precision** | \( -\partial^2 \log(L\pi) = J + (-\partial^2\log\pi) \) | what the Laplace approximation actually uses |

## Why the routes differ

- **`fisher_method="hessian"`** numerically estimates the *posterior precision* —
  the third row, including the prior. It is the observed information plus the prior
  curvature.
- **`fisher_method="waveform"`** computes \( F \) directly — the first term only —
  plus a diagonal prior precision.

The two agree **in expectation**: at the true parameters the residual is pure noise,
\( \mathbb{E}[(r\mid X)] = 0 \), so \( \mathbb{E}[-\partial^2\log L] = F \). This is
the Fisher information identity, and the basis of the Cramér–Rao bound.

For a **single** dataset they differ by the residual–curvature term, which is
zero-mean but has nonzero variance, is **not sign-definite**, and shrinks roughly
like \( \mathcal{O}(1/\mathrm{SNR}) \). Dropping it is the *linear-signal
approximation*. Evaluating at the MAP does not remove it: the gradient vanishes
there, but the residual still has a component off the signal manifold that projects
onto \( \partial_i\partial_j h \).

## Consequences in practice

1. The Hessian route computes the **noisy** object: even with perfect
   differentiation, \( J \) includes the indefinite residual–curvature term, which is
   why the inversion needs eigenvalue flooring. The waveform route computes the clean
   PSD \( F \).
2. The Hessian route differentiates a **scalar** to get a *second* derivative — the
   accuracy ceiling is roughly \( \sqrt{\varepsilon_f} \). The Fisher needs only
   *first* derivatives of \( h \), then a Gram product — better conditioned and
   positive semi-definite for free.

A subtlety worth stating: for a *specific* dataset, the inverse observed information
\( J^{-1} \) is actually the better local Gaussian fit to the realised posterior (the
Laplace approximation *is* a Taylor expansion of the realised log-posterior). Tools
like [gwfast](https://github.com/CosmoStatGW/gwfast) and
[GWFish](https://github.com/janosch314/GWFish) compute \( F \) because they are doing
*forecasting* (expected errors, Cramér–Rao). Here we only need a **proposal**
covariance — and \( F \) is a clean, PSD, well-conditioned surrogate that equals
\( J \) in expectation, so it is usually the better engineering choice when available.

## Conditioning

GW Fisher matrices are notoriously ill-conditioned — condition numbers of
\( 10^{10} \)–\( 10^{20} \) from strong degeneracies (distance–inclination, the
masses, sky position). Inverting in double precision loses about
\( \log_{10}(\kappa) \) digits, so a poorly-conditioned matrix can be inverted to
*zero* correct digits.

Two independent sources contribute:

- **Scale-driven** conditioning (parameters span many orders of magnitude). Removed
  by **diagonal preconditioning** — rescaling by \( \sqrt{\mathrm{diag}} \) before
  inversion. This is the shared trick used by both GWFish and gwfast.
- **Correlation-driven** conditioning (genuine parameter degeneracies). Preconditioning
  cannot help here; only a better parameterisation or the Fisher form does.

The unit-cube Hessian path already normalises scales via the prior CDF transform, so
its residual conditioning is correlation-driven; the parameter-space and
waveform-Fisher paths benefit much more from preconditioning.
