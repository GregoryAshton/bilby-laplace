import datetime
import hashlib
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
import tqdm
from bilby.core.sampler.base_sampler import Sampler, signal_wrapper
from bilby.core.utils import logger, random
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm, truncnorm

from .laplace import LaplacePosteriorEstimator

try:
    from bilby.core.sampler.base_sampler import SamplerError
except ImportError:
    SamplerError = RuntimeError

try:
    from bilby.core.utils import safe_save_figure
except ImportError:

    def safe_save_figure(fig, filename, **kwargs):
        fig.savefig(filename, **kwargs)


class TruncatedMVNProposal:
    """Per-marginal independent Gaussian proposal, bounded to the prior.

    Each parameter is sampled independently with scale ``sqrt(cov[i, i])``.
    Off-diagonal covariance elements are not used in sampling or the log-pdf;
    inter-parameter correlations are recovered through the likelihood during
    the acceptance step.

    Bounded parameters use a truncated normal, so every draw lands inside the
    prior *box* however wide the Gaussian is.  The box is not the whole story
    when the prior carries ``Constraint`` priors: those bound a derived
    quantity, cannot be imposed per-marginal, and are deliberately left to the
    accept/reject step rather than built in here.  Every consumer of this
    proposal weights or filters by ``PriorDict.ln_prob`` (which returns
    ``-inf`` on a violation), so constraint-violating draws are already
    rejected -- imposing them inside ``sample`` as well would only raise the
    acceptance rate, and would require renormalising ``logpdf`` by the
    proposal's own constrained volume, which is not the prior's
    ``normalize_constraint_factor`` and would need its own Monte-Carlo
    estimate.  Correctness does not depend on it; efficiency does.

    *Periodic* parameters are
    **wrapped** instead: truncation renormalises the density inside the range
    and puts zero mass on the far side of the boundary, but for an angle the
    mass below the lower edge belongs just under the upper edge.  On the
    precessing BBH example the ``phi_jl`` MAP sits at 0.835 with a proposal
    sigma near 1.4, so about a quarter of its Gaussian falls below zero;
    truncating lost that tail entirely, leaving 1.6% of the posterior above
    0.85*2pi where dynesty has 12%.
    """

    # Wrapped-normal images summed either side of the principal range.  The
    # proposal sigma never approaches the period in practice, so three is far
    # more than enough for the sum to converge.
    _N_WRAPS = 3

    def __init__(self, mean, cov, lower, upper, periodic=None):
        self.mean = np.asarray(mean, dtype=float)
        # Retained for diagnostics only; sampling uses the per-marginal
        # ``_sigma`` below, not the full covariance.
        self.cov = cov
        self._sigma = np.sqrt(np.diag(cov))
        self._ndim = len(self.mean)
        self._lower = np.asarray(lower, dtype=float)
        self._upper = np.asarray(upper, dtype=float)
        self._period = self._upper - self._lower
        if periodic is None:
            self._periodic = np.zeros(self._ndim, dtype=bool)
        else:
            self._periodic = np.asarray(periodic, dtype=bool)
            if self._periodic.shape != (self._ndim,):
                raise ValueError(f"periodic must have one entry per parameter; got {self._periodic.shape}")
        # Standardised bounds for truncnorm: a = (lo - mu)/sigma, b = (hi - mu)/sigma
        self._a = (self._lower - self.mean) / self._sigma
        self._b = (self._upper - self.mean) / self._sigma
        self._dists = [
            truncnorm(a=self._a[i], b=self._b[i], loc=self.mean[i], scale=self._sigma[i]) for i in range(self._ndim)
        ]

    def _wrap(self, x, i):
        """Fold *x* into ``[lower_i, upper_i)``."""
        return self._lower[i] + np.mod(x - self._lower[i], self._period[i])

    def _wrapped_logpdf(self, x, i):
        """Log density of the wrapped normal on parameter *i*."""
        folded = self._wrap(np.asarray(x, dtype=float), i)
        shifts = np.arange(-self._N_WRAPS, self._N_WRAPS + 1) * self._period[i]
        images = folded[None, :] + shifts[:, None]
        return logsumexp(norm.logpdf(images, loc=self.mean[i], scale=self._sigma[i]), axis=0)

    def sample(self, n):
        columns = []
        for i in range(self._ndim):
            if self._periodic[i]:
                columns.append(self._wrap(random.rng.normal(self.mean[i], self._sigma[i], n), i))
            else:
                columns.append(self._dists[i].rvs(n, random_state=random.rng))
        return np.column_stack(columns)

    def logpdf(self, x):
        x = np.atleast_2d(x)
        terms = [
            self._wrapped_logpdf(x[:, i], i) if self._periodic[i] else self._dists[i].logpdf(x[:, i])
            for i in range(self._ndim)
        ]
        return np.sum(terms, axis=0)


class TruncatedMVNMixtureProposal:
    """Weighted mixture of :class:`TruncatedMVNProposal` components.

    One component per discovered posterior mode.  This is what every
    resampling mode draws from when ``n_modes > 1``, so ``inprior``,
    ``rejection``, ``importance`` and ``smc`` share the *whole* proposal rather
    than only its primary component -- without it the mode search was
    effectively an SMC-only feature, and an ``inprior`` run was a single
    Gaussian on the primary MAP no matter what ``n_modes`` said.

    ``mean`` and ``cov`` name the *heaviest* component. They are reference
    points (the rejection bound's starting value, the diagnostic plots), not a
    description of the mixture, and the heaviest component is the best
    single-point stand-in available. Note the heaviest need not be the primary
    MAP: modes are weighted by their Laplace evidence, and a secondary found by
    the mode search can outweigh the point the optimiser started from.
    """

    def __init__(self, components, weights=None):
        if not components:
            raise ValueError("a mixture proposal needs at least one component")
        self.components = list(components)
        k = len(self.components)
        if weights is None:
            w = np.full(k, 1.0 / k)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (k,):
                raise ValueError(f"weights must have one entry per component; got {w.shape} for k={k}")
            if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
                raise ValueError("weights must be finite, non-negative and not all zero")
            w = w / w.sum()
        self.weights = w
        with np.errstate(divide="ignore"):  # a zero-weight component is legal
            self._log_w = np.log(w)
        heaviest = int(np.argmax(w))
        self.mean = self.components[heaviest].mean
        self.cov = self.components[heaviest].cov

    def sample(self, n):
        counts = random.rng.multinomial(n, self.weights)
        x = np.vstack([c.sample(int(m)) for c, m in zip(self.components, counts) if m])
        # Shuffle: the blocks above come out ordered by component, and callers
        # truncate a concatenated batch to the number of samples they asked for
        # (``_run_inprior``, ``_draw_inprior_samples``), which would drop the
        # last components preferentially and silently reweight the mixture.
        random.rng.shuffle(x, axis=0)
        return x

    def logpdf(self, x):
        log_p = np.array([c.logpdf(x) for c in self.components])
        return logsumexp(log_p + self._log_w[:, None], axis=0)


class _StandardisedGaussian:
    """A multivariate normal built in standardised coordinates.

    Exposes the small part of the ``scipy.stats.multivariate_normal`` surface
    this module uses while doing the arithmetic on a unit-diagonal matrix, so a
    covariance spanning many orders of magnitude in parameter scale is not
    refused by scipy's relative positive-definiteness check.
    """

    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self._sd = np.sqrt(np.diag(self.cov))
        self._dist = multivariate_normal(mean=np.zeros_like(self.mean), cov=self.cov / np.outer(self._sd, self._sd))
        self._log_jacobian = float(np.sum(np.log(self._sd)))

    def logpdf(self, x):
        z = (np.asarray(x, dtype=float) - self.mean) / self._sd
        return self._dist.logpdf(z) - self._log_jacobian

    def rvs(self):
        z = np.asarray(self._dist.rvs(random_state=random.rng))
        return self.mean + z * self._sd


class GaussianFlow:
    """Minimal aspire-compatible Flow wrapping a multivariate Gaussian.

    Implements the ``log_prob`` and ``sample_and_log_prob`` interface required
    by aspire's SMC sampler as the ``prior_flow`` argument.
    """

    def __init__(self, mean, cov):
        self._mean = np.asarray(mean, dtype=float)
        self._cov = np.asarray(cov, dtype=float)
        # Built in standardised coordinates, z = (x - mean) / sd. scipy's _PSD
        # rejects a covariance whose condition number exceeds ~4.5e9 *in the
        # units it is handed*, and a GW covariance spans ~10 orders of
        # magnitude between chirp_mass and lambda on scale alone -- so a sound
        # matrix is refused as "not positive definite". Dividing by the
        # per-parameter sd leaves a unit-diagonal matrix whose conditioning
        # reflects genuine correlation rather than units.
        self._sd = np.sqrt(np.diag(self._cov))
        outer = np.outer(self._sd, self._sd)
        self._dist = multivariate_normal(mean=np.zeros_like(self._mean), cov=self._cov / outer)
        self._log_jacobian = float(np.sum(np.log(self._sd)))

    def log_prob(self, x):
        z = (np.asarray(x, dtype=float) - self._mean) / self._sd
        return self._dist.logpdf(z) - self._log_jacobian

    def sample_and_log_prob(self, n_samples):
        z = np.atleast_2d(self._dist.rvs(size=n_samples, random_state=random.rng))
        x = self._mean + z * self._sd
        return x, self._dist.logpdf(z) - self._log_jacobian

    def sample(self, n_samples):
        return self.sample_and_log_prob(n_samples)[0]

    # See GaussianMixtureFlow for why these exist.
    xp = np

    def fit(self, samples, **kwargs):
        """No-op: this Gaussian is constructed from the Hessian, not learned."""
        logger.info("Using the analytic Laplace Gaussian as aspire's prior flow; no training.")
        return None


class GaussianMixtureFlow:
    """Aspire-compatible Flow wrapping a Gaussian mixture.

    Used as ``prior_flow`` when multiple MAP estimates are available, so the
    SMC annealing path starts from a mixture that covers all discovered modes.
    Components are equally weighted unless ``log_weights`` is given (see
    ``Laplace._laplace_mode_log_weights``); the weights are normalised here, so
    unnormalised log-evidences can be passed straight in.
    """

    def __init__(self, means, covs, log_weights=None):
        # Each component standardised; see GaussianFlow for why.
        self._dists = [_StandardisedGaussian(m, c) for m, c in zip(means, covs)]
        self._k = len(self._dists)
        if log_weights is None:
            self._log_w = np.full(self._k, -np.log(self._k))
        else:
            log_w = np.asarray(log_weights, dtype=float)
            if log_w.shape != (self._k,):
                raise ValueError(f"log_weights must have one entry per component; got {log_w.shape} for k={self._k}")
            self._log_w = log_w - logsumexp(log_w)
        self._w = np.exp(self._log_w)

    @property
    def weights(self):
        """Normalised component weights, ordered as the components were given."""
        return self._w.copy()

    def log_prob(self, x):
        x = np.asarray(x)
        # log_probs: shape (K, N) or (K,) for a single point
        log_probs = np.array([d.logpdf(x) for d in self._dists])
        log_w = self._log_w if log_probs.ndim == 1 else self._log_w[:, None]
        return logsumexp(log_probs + log_w, axis=0)

    def sample_and_log_prob(self, n_samples):
        idx = random.rng.choice(self._k, size=n_samples, p=self._w)
        x = np.array([self._dists[i].rvs() for i in idx])
        return x, self.log_prob(x)

    def sample(self, n_samples):
        return self.sample_and_log_prob(n_samples)[0]

    # Aspire calls ``prior_flow.xp.isfinite`` when screening its initial draw.
    xp = np

    def fit(self, samples, **kwargs):
        """No-op: this mixture is constructed, not learned.

        ``Aspire.fit`` calls this unconditionally, so it has to exist; there is
        nothing to train. The samples it is handed were themselves drawn from
        this mixture.
        """
        logger.info(f"Using the analytic Laplace mixture as aspire's prior flow ({self._k} component(s)); no training.")
        return None


def kish_log_effective_sample_size(ln_weights):
    """Kish effective sample size from log unnormalised weights.

    Returns log(ESS) where ESS = (sum w)^2 / sum(w^2).
    """
    ln_weights = np.asarray(ln_weights, dtype=float)
    # Remove -inf entries for numerical stability
    finite = np.isfinite(ln_weights)
    if not np.any(finite):
        return -np.inf
    ln_w = ln_weights[finite]
    log_ess = 2.0 * logsumexp(ln_w) - logsumexp(2.0 * ln_w)
    return log_ess


class Laplace(Sampler):
    """Bilby sampler implementing the Laplace approximation.

    Finds the MAP (maximum a posteriori) with scipy optimisation, computes the
    inverse Hessian of the log-posterior as a Gaussian proposal covariance,
    then draws posterior samples via a resample method (if requested).

    Parameters
    ----------
    likelihood : bilby.core.likelihood.Likelihood
    priors : bilby.core.prior.PriorDict or dict
    outdir : str
    label : str
    resample : str or None
        Resampling method: ``'rejection'`` (default), ``'importance'``, ``'inprior'``,
        ``'smc'``, ``'emcee'``, or ``None`` / ``'None'`` to skip resampling and return
        raw Laplace-approximation samples.
    npool : int
        Number of processes for parallel likelihood evaluation (bilby standard
        argument, default 1; the aliases ``n_pool``, ``cores``, ``threads`` and
        ``queue_size`` are accepted too). When ``> 1``, every batch of
        likelihood evaluations is spread across a ``multiprocessing.Pool``: the
        proposal batches in the rejection / importance / inprior loops, and the
        per-iteration batches aspire requests under ``resample='smc'``. Drawing
        and accept/reject decisions stay in the main process, so a pooled run is
        numerically identical to a serial one. Only worthwhile when a single
        likelihood evaluation is expensive relative to inter-process overhead.
        Alternatively pass a pre-built ``pool`` object.
    target_nsamples : int
        Target number of posterior samples.
    batch_nsamples : int
        For methods that draw samples in batches, this specifies the number of
        samples drawn per batch from the proposal distribution.
    prior_nsamples : int
        Number of prior draws used in the MAP search (multi-start mode).
    minimization_method : str
        Optimization method. Default is ``'differential_evolution'`` (global
        optimizer; recommended for real data). Set to ``'Nelder-Mead'`` to use
        a multi-start local optimizer instead.
    plot_diagnostic : bool
        If True, produce a corner diagnostic plot after resampling.
    cov_scaling : float or dict
        Multiplicative scale applied to the Laplace covariance. Each value
        multiplies the corresponding parameter's *variance* (so a value of 4
        widens that parameter's marginal sigma by 2x).

        - A scalar (default 1) scales the whole covariance uniformly, i.e.
          ``cov -> cov_scaling * cov``.
        - A dict mapping parameter names to scales applies a per-parameter
          variance scale, e.g. ``{'chirp_mass': 4.0, 'luminosity_distance': 9.0}``.
          The reserved key ``'others'`` sets the scale for every parameter not
          listed explicitly (default 1.0), e.g. ``{'chirp_mass': 4.0, 'others': 2.0}``
          scales chirp_mass by 4 and all remaining parameters by 2. Unknown
          names raise an error.

        Off-diagonal entries scale by the geometric mean ``sqrt(v_i * v_j)`` so
        correlations and positive-definiteness are preserved; a uniform dict
        reproduces the scalar behaviour exactly.

        Applied *after* the eigenvalue-based covariance validation, so the
        requested scale is authoritative: validation repairs the estimated
        covariance, then this scaling is the final word on the proposal width.
    sampling_cov : pd.DataFrame, tuple, or None
        Pre-computed covariance to use in place of the Laplace-estimated
        covariance. Must be one of:

        - A pandas DataFrame whose row index and column index are both the
          parameter names. Any ordering is accepted; the matrix is reordered
          internally.
        - A two-element tuple ``(parameter_names, cov)`` where
          ``parameter_names`` is a sequence of parameter-name strings and
          ``cov`` is an ``(N, N)`` array-like covariance in that order.
        - ``None`` (default) to compute the posterior covariance from the
          Hessian of the log-posterior at the MAP.

        The MAP search still runs; only the covariance step is replaced.
        ``cov_scaling`` and the eigenvalue-based covariance validation are
        still applied.  Not compatible with ``n_modes > 1``.

        Examples
        --------
        Given an ``(N, N)`` array ``C`` and a list ``parameter_names``::

            import pandas as pd
            cov_df = pd.DataFrame(C, index=parameter_names, columns=parameter_names)
            result = bilby.run_sampler(..., sampler="laplace", sampling_cov=cov_df)

        Or pass the tuple form directly::

            result = bilby.run_sampler(
                ..., sampler="laplace", sampling_cov=(parameter_names, C)
            )
    fisher_method : str
        How to estimate the posterior precision. ``'hessian'`` (default)
        finite-differences the scalar log-posterior. ``'waveform'`` builds the
        genuine Fisher matrix from gravitational-wave waveform derivatives
        (positive semi-definite by construction). Requires a
        ``GravitationalWaveTransient``-like likelihood without marginalisation
        and not a reduced-order (ROQ / relative-binning / multi-band) variant;
        it works directly in parameter space, so ``use_unit_cube`` and
        ``jacobian_cap_scale`` are ignored.
    fisher_kwargs : dict or None
        Keyword arguments forwarded to the waveform-Fisher computation when
        ``fisher_method='waveform'`` (recognised keys: ``eps``, ``eps_mass``).
    use_injection_for_map : bool
        If True and injection_parameters are set, use them as the starting
        point for the MAP search.
    fail_on_error : bool
        If True, raise SamplerError when sampling fails; otherwise just log.
    n_modes : int
        Number of distinct posterior modes to search for.  When ``n_modes > 1``
        the optimiser is restarted from multiple prior draws and the distinct
        MAP estimates are combined into a Gaussian mixture proposal.  Modes are
        deduplicated by requiring a normalised separation of at least
        ``mode_separation_sigma`` in some parameter.  Default is 1 (single
        Gaussian, original behaviour).

        The mixture is the proposal for **every** value of ``resample``:
        ``inprior`` draws from it, ``rejection`` and ``importance`` weight
        against its density, and ``smc`` seeds its cloud from it.  Until
        2026-08 the search ran inside the SMC branch only, so an ``inprior``
        run silently used a single Gaussian on the primary MAP however many
        modes were asked for -- which made the cheap methods incomparable with
        SMC on any multi-modal problem.  Results produced before that change
        cannot be read as multi-mode except for ``resample='smc'``.
    mode_search_nsamples : int
        Number of prior draws used when searching for secondary modes
        (``n_modes > 1``).  Higher values make mode discovery more
        reliable in high-dimensional spaces, at the cost of more
        likelihood evaluations.  Uses Latin hypercube sampling for
        even coverage.  Default is 500.
    mode_search_subspace : list of str or None
        Restrict the search for secondary modes to these parameters, holding
        the rest at the primary MAP (the polish step still runs in the full
        space, so a candidate is free to move afterwards).  Use it when the
        degeneracy lives in known coordinates: a narrow mode is undiscoverable
        by a Latin hypercube over the full space -- a sky mode a few 0.01 rad
        across is ~1e-4 of the sky -- but the same budget covers two or three
        named dimensions densely.  For a GW sky reflection, for example,
        ``["zenith", "azimuth"]``.  Default None (search all parameters).
    smc_prior_flow : str
        Which distribution aspire anneals *from*. Aspire's tempered target is
        ``(1 - beta) * log q + beta * (log L + log prior)``, so this is not just
        a starting point: ``log q`` is in the MCMC's target at every
        temperature, vanishing only at ``beta = 1``.

        ``"learned"`` (default) is aspire's own behaviour -- train a normalising
        flow on samples drawn from the Laplace proposal and use that.
        ``"laplace"`` hands aspire the Laplace mixture itself, so ``log q`` is
        analytic and exactly the proposal we constructed, mode weights
        included, and flow training stops being a source of run-to-run scatter.

        **Measured substantially worse on a precessing-BBH example** than the
        trained flow, on both divergence-from-reference and tempering-schedule
        length. The reason is not that the mixture is Gaussian: it is that
        :class:`GaussianMixtureFlow` is built on plain ``multivariate_normal``
        and so is *unbounded and non-periodic*, while the trained flow inherits
        aspire's logit transform on bounded coordinates and angular treatment
        of periodic ones. On an example declaring five periodic parameters and
        a mode sitting on the ``phi_12`` boundary, that makes ``log q`` wrong
        exactly where ``(1 - beta) log q`` dominates.

        A useful analytic prior flow would need a correlated mixture that is
        also truncated and wrapped. Neither existing class is that:
        :class:`GaussianMixtureFlow` keeps correlations but ignores the
        boundaries, and :class:`TruncatedMVNProposal` respects them but is
        diagonal, which discards the correlation structure a degenerate
        posterior lives in. Use ``"learned"`` until one exists.
    mode_symmetries : list of (str, float) or None
        Exact symmetries of the posterior, as ``(parameter, shift)`` pairs, used
        to seed the modes they imply instead of relying on the random
        multi-start search to rediscover them. For a posterior that is exactly
        pi-periodic in ``delta_phase``, pass ``[("delta_phase", np.pi)]``. Each
        implied mode is verified -- its log-posterior must match the one it
        mirrors -- and skipped if the symmetry does not hold, so declaring a
        wrong one is safe. Requires ``n_modes > 1``, which is what builds a
        mixture at all. Default None.
    mode_separation_sigma : float
        How far apart two modes must be, in units of the primary mode's
        per-parameter sigma, to count as distinct.  A candidate closer than this
        to a known mode is skipped before polishing, and a polished candidate
        closer than this is discarded as a duplicate.  Default 3.0.

        Lower it when a real secondary mode sits close to the primary -- e.g. a
        sky-position mode only a few sigma away in some coordinates -- since at
        the default such a mode is rejected as a duplicate and the search keeps
        a distant sidelobe instead.  Raise it if one broad mode is
        being split into several.  This is deliberately per-run rather than a
        new default -- the right value depends on how well separated the
        posterior's modes are, which is the thing being discovered.
        How to weight the mixture components when ``n_modes > 1``.  ``'equal'``
        (default) gives every mode the same weight.  ``'laplace'`` weights each
        by its Laplace local evidence -- log-posterior at the mode plus half the
        log-determinant of its covariance -- so a broad shallow mode can outweigh
        a narrow tall one, which weighting by peak height alone would not
        capture.  The weights set the mixture component weights, and hence how
        every resampling method draws: the share of ``inprior`` draws from each
        mode, the density ``rejection``/``importance`` weight against, and the
        share of the initial SMC cloud.  Every mode keeps at least one particle
        in the SMC cloud regardless.
    smc_kwargs : dict or None
        Configuration for SMC sampling (only used when ``resample='smc'``).
        Recognised keys:

        ``sampler`` : str
            Aspire posterior sampler: ``'importance'`` (default), ``'smc'``,
            or any other strategy accepted by
            ``aspire.Aspire.sample_posterior``.
        ``n_initial_samples`` : int
            Number of initial samples drawn from the Laplace proposal and
            passed to ``aspire.fit()`` (default 1000).
        ``n_samples`` : int
            Number of SMC particles, passed as the first positional argument
            to ``aspire.sample_posterior()``.  This is *not* merely a final
            output size: the particles are carried through every tempering
            iteration and mutation step, so this value drives the whole cost
            of the run (roughly, likelihood evaluations scale linearly with
            it).  The final posterior contains this many samples.  Defaults to
            ``target_nsamples`` if not set.  (Note: aspire also exposes a
            distinct ``n_final_samples`` argument for a post-hoc resample to a
            different size; pass it through ``smc_kwargs`` if you need it.)

        Any other keys are forwarded directly to
        ``aspire.Aspire.sample_posterior()``, so all aspire parameters are
        accessible this way.
    emcee_kwargs : dict or None
        Configuration for ``resample='emcee'`` (only used then). Sampling runs
        in batches of ``nsteps``, re-estimating the integrated autocorrelation
        time after each and stopping once the chain holds ``target_nsamples``
        independent samples (or at ``max_nsteps``). Recognised keys:

        ``nwalkers`` : int
            Number of ensemble walkers. Defaults to ``max(4 * ndim, 32)``.

            Worth choosing deliberately: an autocorrelation-time estimate is
            only trustworthy once the chain is ~50 tau long, and at that point
            the run already holds ``50 * nwalkers`` independent samples. Any
            ``nwalkers`` much above ``target_nsamples / 50`` therefore *cannot*
            stop near the target and will oversample by that ratio (a warning
            is logged). ``nwalkers ~= target_nsamples / 50`` makes the two
            conditions coincide, which is the cheapest place to sit -- bounded
            below by needing enough walkers to keep the likelihood pool busy,
            since each half-ensemble move evaluates ``nwalkers / 2`` points.
        ``nsteps`` : int
            Steps per batch, and the length of the first batch. Default 5000.
            Sets the granularity of the convergence check: the check costs no
            likelihood evaluations but is a serial barrier, while overshoot
            past the target is bounded by one batch.
        ``max_nsteps`` : int
            Hard cap on total steps. Defaults to ``nsteps``, i.e. a single
            batch and no growth, so adaptive running is opt-in. Reaching it
            with too few samples logs a warning rather than failing.
        ``target_nsamples`` : int
            Independent samples to collect before stopping. Defaults to the
            sampler-wide ``target_nsamples``, matching every other resampling
            mode.
        ``discard`` : int
            Number of leading steps discarded as burn-in before flattening the
            chain. Default ``nsteps // 2``. Fixed across batches (burn-in does
            not grow with the chain).
        ``autocorr_tol`` : float
            How many autocorrelation times the post-burn-in chain must span
            before its tau estimate is trusted. Default 50 (emcee's own
            ``tol``), which is conservative -- ~10 already gives a usable if
            low-biased estimate. Lowering it shortens a slow-mixing run
            considerably at the cost of a weaker guarantee, so the achieved
            chain-length/tau ratio is logged every batch and a warning is
            issued when the value is below emcee's own bar. Note this also
            rescales the useful walker count (see ``nwalkers``).
        ``backend_file`` : str, bool, or None
            Persist the full un-thinned chain to HDF5 via
            ``emcee.backends.HDFBackend``. ``True`` writes
            ``{outdir}/{label}_emcee_chain.h5``, a string is used as the path,
            and ``None`` (default) keeps it off. The returned samples are
            thinned, so this is the only copy of the raw chain -- worth having
            for analysis the run itself cannot do, above all re-estimating tau
            at several ``discard`` values, which distinguishes a tau measuring
            leftover burn-in drift from a genuinely slow mode. The file holds
            ``nwalkers * nsteps * ndim`` floats, so it is large for a long run,
            and any chain already in it is cleared at startup.
        ``thin`` : int, optional
            Fixed thinning factor. If omitted (default), the post-burn-in
            chain is thinned automatically by the estimated integrated
            autocorrelation time (the largest across parameters), so the
            returned samples are approximately independent -- matching every
            other resampling mode, and what the comparison metrics (JSD/EMD
            against a reference sampler) assume of their input.

        Any other keys are forwarded to ``emcee.EnsembleSampler``. Walkers are
        seeded from the Laplace proposal via the same in-prior draw used to seed
        SMC's initial cloud (:meth:`_draw_inprior_samples`).
    smc_progress : bool
        If True (default) and ``resample='smc'`` with an SMC-family sampler,
        register a per-iteration callback that logs a one-line progress summary
        at INFO (and, when ``smc_plot_every > 0``, re-renders the diagnostic
        figures). Set to False to disable it. To override the callback
        entirely, pass your own ``smc_kwargs=dict(checkpoint_callback=...)``.
    smc_plot_every : int
        How often to re-render the SMC stats and evolution-and-marginals
        figures *during* sampling, in iterations. ``0`` (default) means never:
        they are written once when sampling finishes. Set to ``n > 0`` to watch
        a long run live, at a cost -- the evolution figure fits a
        ``gaussian_kde`` per parameter per iteration, so re-rendering it every
        iteration grows quadratically over a run. Ignored unless
        ``plot_diagnostic=True``, which is also what gates the final render.
    prior_parameters : list or None
        List of parameter names for which initial proposal samples should be
        replaced with independent draws from the prior. Use this for parameters
        with wide posteriors consistent with their prior, where the Hessian
        poorly constrains the proposal covariance. Default is None (no
        replacement).
    resume : bool
        If True (default) and a resume file exists, load it and continue
        from where the previous run left off. The path depends on the
        resampling mode: ``{outdir}/{label}_resume.pickle`` for the batched
        modes (rejection / importance / inprior), and
        ``{outdir}/{label}_smc_resume.h5`` for SMC (written and read by
        aspire). Set to False to ignore any existing resume file and start
        fresh. Resuming a batched mode skips the MAP search and covariance
        estimation; resuming SMC currently still re-runs MAP+covariance and
        restarts aspire from its checkpointed SMC iteration.
    checkpoint_signal : int or None
        Optional additional signal number (e.g. ``signal.SIGUSR1``) that, when
        received, triggers a clean checkpoint and exit with ``exit_code``.
        ``SIGTERM`` / ``SIGINT`` / ``SIGALRM`` are always wired by the base
        class.
    check_point_delta_t : float
        Periodic in-loop checkpoint interval in seconds (default 600) for
        the batched resampling modes (rejection / importance / inprior).
        Set to 0 to disable periodic saves (signal-driven only). SMC uses
        aspire's per-iteration HDF5 checkpoint instead and is not affected
        by this kwarg.
    """

    sampler_name = "laplace"
    sampling_seed_key = "seed"
    default_kwargs = dict(
        # The sampling seed. `sampling_seed_key = "seed"` tells bilby to route
        # `sampling_seed`/`random_seed` here and reseed its own generator with
        # it, but the key has to exist in the defaults or
        # `_verify_kwargs_against_default_kwargs` strips it back out before
        # `run_sampler` ever sees it -- which silently left the SMC stage
        # unseeded. `None` means "draw from OS entropy", matching dynesty.
        seed=None,
        resample="rejection",
        target_nsamples=10000,
        batch_nsamples=1000,
        prior_nsamples=100,
        minimization_method="differential_evolution",
        plot_diagnostic=False,
        cov_scaling=1,
        sampling_cov=None,
        use_injection_for_map=True,
        fail_on_error=True,
        use_unit_cube=True,
        jacobian_cap_scale=1.0,
        hessian_kwargs=None,
        fisher_method="hessian",
        fisher_kwargs=None,
        n_modes=1,
        mode_search_nsamples=500,
        mode_search_subspace=None,
        mode_separation_sigma=3.0,
        mode_symmetries=None,
        smc_prior_flow="learned",
        mode_weights="equal",
        smc_kwargs=None,
        emcee_kwargs=None,
        smc_progress=True,
        smc_plot_every=0,
        max_iterations=1e6,
        prior_parameters=None,
        resume=True,
        checkpoint_signal=None,
        check_point_delta_t=600,
        # Parallelism.  These have to appear here or bilby's
        # `_verify_kwargs_against_default_kwargs` strips them from `self.kwargs`
        # before `_setup_pool` (which reads both) ever sees them.
        npool=None,
        pool=None,
    )

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        plot=False,
        exit_code=77,
        skip_import_verification=True,
        **kwargs,
    ):
        super().__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            exit_code=exit_code,
            **kwargs,
        )

    def _translate_kwargs(self, kwargs):
        """Fold bilby's ``npool`` aliases into a single ``npool`` kwarg.

        ``Sampler.npool`` looks up ``self.kwargs`` before falling back to the
        ``npool`` argument of ``run_sampler``, and ``npool`` now carries a
        default of ``None``, so both the aliases (``n_pool``, ``cores``, ...)
        and the explicit ``run_sampler(npool=...)`` argument have to be written
        into the kwargs here or they would be shadowed by that default.
        """
        if "npool" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["npool"] = kwargs.pop(equiv)
                    break
            else:
                # `_npool` is set by `Sampler.__init__` before the kwargs setter
                # runs, but tests (and any other caller) may build a bare
                # instance without it.
                npool = getattr(self, "_npool", None)
                if npool is not None and npool > 1:
                    kwargs["npool"] = npool
        return super()._translate_kwargs(kwargs)

    # ------------------------------------------------------------------
    # Resume / checkpoint scaffolding
    # ------------------------------------------------------------------
    # The kwargs below are operational and not part of the
    # checkpoint-identity hash (changing them between runs is allowed).
    _CHECKPOINT_IGNORE_KWARGS = frozenset(
        {
            "resume",
            "checkpoint_signal",
            "check_point_delta_t",
            "plot_diagnostic",
            "fail_on_error",
            "smc_progress",
            "max_iterations",
            # Operational parallelism kwargs.  bilby's `_setup_pool` stashes the
            # live multiprocessing pool into `self.kwargs["pool"]`, which cannot
            # be pickled; the worker count is likewise not part of the run identity.
            "pool",
            "npool",
        }
    )

    def _checkpoint_versions(self):
        """Versions of the packages whose drift we report on resume."""
        import bilby

        from . import __version__ as bilby_laplace_version

        return dict(
            bilby_laplace=bilby_laplace_version,
            bilby=bilby.__version__,
            numpy=np.__version__,
        )

    def _checkpoint_kwargs_hash(self):
        """A sha256 hash identifying the kwargs/priors that affect the run.

        Mismatch on resume signals that the configuration has changed and the
        accumulated state is not valid to continue from.
        """
        import dill

        identity = {
            "kwargs": {k: v for k, v in self.kwargs.items() if k not in self._CHECKPOINT_IGNORE_KWARGS},
            "priors": repr(self.priors),
            "search_keys": list(self.search_parameter_keys),
            "injection_parameters": self.injection_parameters,
        }
        return hashlib.sha256(dill.dumps(identity)).hexdigest()

    def _init_checkpoint_state(self, mode, mean, cov):
        """Initialise the in-memory checkpoint payload after MAP+covariance."""
        self._checkpoint_state = dict(mode=mode, mean=mean, cov=cov)

    def _update_checkpoint_state(self, **fields):
        """Merge fields into the in-memory checkpoint payload."""
        if self._checkpoint_state is None:
            return
        self._checkpoint_state.update(fields)

    @staticmethod
    def _resume_accumulator_lists(state, *names):
        """Fresh list copies of checkpointed accumulator fields, by name.

        Every batched resampling loop (in-prior, rejection, importance) only
        ever passes scalar counters to ``_update_checkpoint_state`` on each
        iteration, relying on the accumulator lists themselves already being
        registered. On resume that registration has to be redone with lists
        the loop actually owns -- reusing the checkpoint's own list objects
        here would leave the checkpoint pointing at containers the loop never
        appends to, silently losing every sample accepted after the resume.
        """
        return [list(state[name]) for name in names]

    def _maybe_periodic_checkpoint(self):
        """Save the resume file if `check_point_delta_t` seconds have elapsed."""
        delta_t = float(self.kwargs.get("check_point_delta_t") or 0)
        if delta_t <= 0:
            return
        now = time.time()
        if now - self._last_save_t > delta_t:
            self.write_current_state()
            self._last_save_t = now

    def _maybe_periodic_rejection_diagnostic(self, proposal, accepted, rejected, parameter_names):
        """Re-render the live rejection progress plot every `check_point_delta_t` seconds.

        Overlays the accepted samples (foreground) on the rejected proposal
        draws (background) so the acceptance structure can be watched as the
        run progresses.  Failures are logged but never interrupt sampling.
        """
        delta_t = float(self.kwargs.get("check_point_delta_t") or 0)
        if delta_t <= 0:
            return
        now = time.time()
        if now - self._last_diagnostic_t <= delta_t:
            return
        self._last_diagnostic_t = now
        try:
            self.create_rejection_progress_diagnostic(proposal.mean, proposal.cov, parameter_names, accepted, rejected)
        except Exception as e:  # a diagnostic must never crash the run
            logger.warning(f"Failed to create rejection progress diagnostic plot: {e}")

    def write_current_state(self):
        """Snapshot the current sampler state to the resume file.

        Called by the base class's signal handler (``SIGTERM`` / ``SIGINT`` /
        ``SIGALRM`` and any user-wired ``checkpoint_signal``) and periodically
        from inside the batched resampling loops.  Idempotent: if no
        checkpoint state has been initialised yet (e.g. the run was killed
        before MAP+covariance finished), no file is written.
        """
        if self._checkpoint_state is None:
            return
        import dill
        from bilby.core.utils import (
            check_directory_exists_and_if_not_mkdir,
            safe_file_dump,
        )

        check_directory_exists_and_if_not_mkdir(self.outdir)

        # Update the cumulative sampling time so the resumed run reports
        # honest total wall-time.
        now = datetime.datetime.now()
        sampling_time_s = (now - self.start_time).total_seconds()

        payload = dict(self._checkpoint_state)
        payload["kwargs_hash"] = self._checkpoint_kwargs_hash()
        payload["search_keys"] = list(self.search_parameter_keys)
        payload["versions"] = self._checkpoint_versions()
        payload["rng_state"] = random.rng.bit_generator.state
        payload["sampling_time_s"] = sampling_time_s
        try:
            safe_file_dump(payload, self.resume_file, dill)
            logger.info(f"Wrote checkpoint to {self.resume_file}")
        except Exception as exc:  # never crash the run for a failed checkpoint
            logger.warning(f"Could not write resume file {self.resume_file}: {exc}")

    def _read_saved_state(self):
        """Load and validate the resume file, returning True on success.

        Raises ``SamplerError`` on a kwargs/priors/parameter-name mismatch
        (the run cannot meaningfully continue from a different configuration).
        Version drift in bilby/bilby_laplace/numpy is logged as a warning but
        is not fatal.
        """
        if not os.path.isfile(self.resume_file):
            logger.info(f"No resume file at {self.resume_file}; starting fresh.")
            return False
        if os.stat(self.resume_file).st_size == 0:
            logger.info(f"Resume file {self.resume_file} is empty; starting fresh.")
            return False

        import dill

        try:
            with open(self.resume_file, "rb") as f:
                payload = dill.load(f)
        except Exception as exc:
            raise SamplerError(
                f"Failed to load resume file {self.resume_file}: {exc}. "
                f"Delete the file or pass resume=False to start fresh."
            )

        expected_hash = self._checkpoint_kwargs_hash()
        if payload.get("kwargs_hash") != expected_hash:
            raise SamplerError(
                f"Resume file {self.resume_file} was written with a different "
                "configuration (kwargs / priors / injection_parameters). "
                "Delete the file or pass resume=False to start fresh."
            )
        if payload.get("search_keys") != list(self.search_parameter_keys):
            raise SamplerError(
                f"Resume file {self.resume_file} has different parameter names. "
                "Delete the file or pass resume=False to start fresh."
            )

        stored_versions = payload.get("versions") or {}
        for pkg, ver in self._checkpoint_versions().items():
            old = stored_versions.get(pkg)
            if old is not None and old != ver:
                logger.warning(
                    f"Resume file was written with {pkg}={old}; " f"this run uses {pkg}={ver}. Continuing anyway."
                )

        random.rng.bit_generator.state = payload["rng_state"]
        prior_s = float(payload.get("sampling_time_s") or 0.0)
        # Shift start_time backwards so end - start = prior + current.
        self.start_time = datetime.datetime.now() - datetime.timedelta(seconds=prior_s)
        _meta_keys = (
            "kwargs_hash",
            "search_keys",
            "versions",
            "rng_state",
            "sampling_time_s",
        )
        self._checkpoint_state = {k: payload[k] for k in payload if k not in _meta_keys}
        logger.info(
            f"Resumed from {self.resume_file} "
            f"(mode={self._checkpoint_state.get('mode')!r}, "
            f"prior sampling {prior_s:.1f}s)."
        )
        return True

    def _smc_resume_file_path(self):
        """Canonical SMC HDF5 resume file path (aspire requires .h5/.hdf5)."""
        return f"{self.outdir}/{self.label}_smc_resume.h5"

    def _cleanup_resume_file(self):
        """Remove any resume files after a clean run finishes.

        Removes both the batched-loop pickle file (rejection / importance /
        inprior) and the SMC HDF5 file written by aspire, if present.
        """
        for path in (self.resume_file, self._smc_resume_file_path()):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    logger.info(f"Removed resume file {path}")
            except OSError as exc:
                logger.warning(f"Could not remove resume file {path}: {exc}")

    def _resolve_sampling_cov(self, sampling_cov, parameter_names):
        """Normalize a user-provided sampling covariance to an ndarray.

        Accepts either a pandas DataFrame with parameter-named index/columns,
        a ``(names, cov)`` tuple, or ``None``.  Returns an ``(N, N)`` ndarray
        ordered by ``parameter_names``, or ``None`` when the input is ``None``.

        All mismatches (unknown names, missing names, duplicates, wrong shape,
        non-symmetry, non-PSD) raise ``ValueError`` up-front.
        """
        if sampling_cov is None:
            return None

        if isinstance(sampling_cov, pd.DataFrame):
            cov_df = sampling_cov
        elif isinstance(sampling_cov, tuple) and len(sampling_cov) == 2:
            names, cov = sampling_cov
            names = list(names)
            cov = np.asarray(cov, dtype=float)
            if cov.shape != (len(names), len(names)):
                raise ValueError(
                    f"sampling_cov array has shape {cov.shape}; expected "
                    f"({len(names)}, {len(names)}) for parameter_names of "
                    f"length {len(names)}."
                )
            cov_df = pd.DataFrame(cov, index=names, columns=names)
        else:
            raise TypeError(
                "sampling_cov must be a pandas DataFrame with parameter-named "
                "index/columns, a (parameter_names, cov) tuple, or None. "
                f"Got {type(sampling_cov).__name__}."
            )

        row_names = list(cov_df.index)
        col_names = list(cov_df.columns)

        if len(set(row_names)) != len(row_names):
            dups = sorted({n for n in row_names if row_names.count(n) > 1})
            raise ValueError(f"sampling_cov parameter names contain duplicates: {dups}")

        if set(row_names) != set(col_names):
            row_only = sorted(set(row_names) - set(col_names))
            col_only = sorted(set(col_names) - set(row_names))
            raise ValueError(
                f"sampling_cov DataFrame row index and column index must "
                f"contain the same parameter names. Row-only: {row_only}. "
                f"Column-only: {col_only}."
            )

        expected = set(parameter_names)
        got = set(row_names)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            if missing:
                raise ValueError(f"sampling_cov is missing covariance entries for parameter(s): {missing}")
            if extra:
                raise ValueError(f"sampling_cov contains unknown parameter(s) not in the model: {extra}")

        cov_df = cov_df.loc[list(parameter_names), list(parameter_names)]
        C = np.asarray(cov_df.values, dtype=float)

        asym = float(np.max(np.abs(C - C.T)))
        scale = max(float(np.max(np.abs(C))), 1.0)
        if asym > 1e-8 * scale:
            raise ValueError(
                f"sampling_cov is not symmetric: max asymmetry {asym:.3g} " f"exceeds tolerance {1e-8 * scale:.3g}."
            )
        C = 0.5 * (C + C.T)

        eigvals = np.linalg.eigvalsh(C)
        max_eig = float(eigvals.max())
        tol = 1e-8 * max(max_eig, 1.0)
        if eigvals.min() < -tol:
            raise ValueError(
                f"sampling_cov is not positive semi-definite: minimum "
                f"eigenvalue {eigvals.min():.3g} is below tolerance {-tol:.3g}."
            )

        return C

    #: Reserved dict key setting the scale for parameters not listed explicitly.
    _COV_SCALING_OTHERS_KEY = "others"

    def _resolve_cov_scaling(self, cov_scaling, parameter_names):
        """Normalize ``cov_scaling`` to a per-parameter variance-scale vector.

        Accepts either a scalar (applied to every parameter) or a dict mapping
        parameter names to scales.  Returns an ``(N,)`` ndarray of variance
        multipliers ordered by ``parameter_names``.

        A dict may include the reserved key ``"others"`` to set the scale for
        every parameter not listed explicitly (default ``1.0``); this is not
        treated as a parameter name.

        Each value multiplies the corresponding parameter's *variance*, matching
        the historical scalar behaviour (which multiplied the covariance
        matrix); see ``_apply_cov_scaling`` for how off-diagonal terms follow.
        All values must be finite and strictly positive; unknown dict keys raise
        ``ValueError`` so typos surface immediately.
        """
        n = len(parameter_names)
        if isinstance(cov_scaling, dict):
            others_key = self._COV_SCALING_OTHERS_KEY
            default = float(cov_scaling.get(others_key, 1.0))
            expected = set(parameter_names)
            extra = sorted(set(cov_scaling) - expected - {others_key})
            if extra:
                raise ValueError(f"cov_scaling contains unknown parameter(s) not in the model: {extra}")
            v = np.array([float(cov_scaling.get(name, default)) for name in parameter_names], dtype=float)
            defaulted = [name for name in parameter_names if name not in cov_scaling]
            if defaulted:
                logger.info(f"cov_scaling applying {others_key}={default} to unspecified parameter(s): {defaulted}")
        else:
            v = np.full(n, float(cov_scaling), dtype=float)

        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError(f"cov_scaling values must be finite and strictly positive; got {v.tolist()}.")
        return v

    @staticmethod
    def _apply_cov_scaling(covariance, cov_scaling_vec):
        """Apply a per-parameter variance scaling to a covariance matrix.

        With ``D = diag(sqrt(v))`` this returns ``D @ C @ D``, so the diagonal
        variances scale by ``v`` while off-diagonal entries scale by
        ``sqrt(v_i * v_j)``.  This preserves the correlation structure and
        positive-definiteness, and reduces to ``s * C`` when every ``v_i == s``.
        """
        s = np.sqrt(np.asarray(cov_scaling_vec, dtype=float))
        return np.outer(s, s) * np.asarray(covariance, dtype=float)

    def _replace_with_prior_samples(self, samples, parameter_names):
        """Replace specified parameters with draws from the prior.

        For parameters in self.kwargs['prior_parameters'], replace values
        in the samples DataFrame with independent draws from their priors.
        This is useful for parameters with wide posteriors consistent with
        their prior, where the Hessian poorly constrains the proposal covariance.

        Parameters
        ----------
        samples : pd.DataFrame
            Samples with columns for each parameter.
        parameter_names : list
            Names of all parameters in order.

        Returns
        -------
        samples : pd.DataFrame
            Modified samples with specified parameters replaced by prior draws.
        """
        prior_params = self.kwargs.get("prior_parameters") or []
        if not prior_params:
            return samples

        bad_params = [p for p in prior_params if p in parameter_names]
        unknown_params = [p for p in prior_params if p not in parameter_names]

        if unknown_params:
            logger.warning(f"prior_parameters contains unknown parameters: {unknown_params}")

        for param in bad_params:
            prior_draws = self.priors[param].sample(len(samples))
            samples[param] = prior_draws
            logger.debug(f"Replaced initial {param} samples with {len(samples)} prior draws")

        return samples

    def _periodic_mask(self, parameter_names):
        """Boolean mask of which sampled parameters wrap.

        Same rule ``aspire_bilby`` uses (``boundary == "periodic"``).  Needed by
        the proposal so a periodic coordinate is wrapped rather than truncated,
        and by ``_smc_sample`` so aspire treats it as an angle.
        """
        return np.array(
            [getattr(self.priors[key], "boundary", None) == "periodic" for key in parameter_names],
            dtype=bool,
        )

    def _effective_log_proposal(self, proposal, x, parameter_names):
        """Log proposal density ``log g(x)`` accounting for prior-substituted
        dimensions.

        For each parameter in ``prior_parameters`` the values are drawn from
        the prior rather than the truncated-Gaussian marginal (see
        ``_replace_with_prior_samples``), so the proposal density on that
        dimension IS the prior density.  ``proposal.logpdf`` still returns the
        truncated-normal marginal there, so we swap it for the prior density.

        Without this correction every importance/rejection weight
        ``ln_r = logl + logpi - log_g`` keeps an uncancelled
        ``truncnorm_j / prior_j`` factor on the replaced dimensions, biasing
        both the recovered posterior and the evidence (the factor is not
        constant unless the Laplace sigma happens to equal the prior width).
        With the correction the replaced dimension's ``logpi`` and ``log_g``
        cancel exactly, giving it unit weight as intended.
        """
        log_g = proposal.logpdf(x)
        prior_params = [p for p in (self.kwargs.get("prior_parameters") or []) if p in parameter_names]
        if not prior_params:
            return log_g
        if not hasattr(proposal, "_dists"):
            # Non-truncated-Gaussian proposal: cannot decompose per-marginal.
            logger.warning("Cannot correct proposal density for prior_parameters on this proposal type.")
            return log_g
        parameter_names = list(parameter_names)
        x2 = np.atleast_2d(x)
        for p in prior_params:
            j = parameter_names.index(p)
            log_g = log_g - proposal._dists[j].logpdf(x2[:, j]) + np.asarray(self.priors[p].ln_prob(x2[:, j]))
        return log_g

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Return expected output files/dirs (used by bilby_pipe / HTCondor)."""
        return [], []

    @signal_wrapper
    def run_sampler(self):
        self.start_time = datetime.datetime.now()

        # Checkpoint scaffolding.  resume_file is the on-disk location; the
        # in-memory _checkpoint_state holds whatever the signal handler / the
        # periodic save would dump.  _last_save_t throttles periodic saves.
        self.resume_file = f"{self.outdir}/{self.label}_resume.pickle"
        self._checkpoint_state = None
        self._last_save_t = time.time()
        # Throttles the live rejection-sampling progress diagnostic, which is
        # re-rendered on the same wall-clock cadence as the checkpoint.
        self._last_diagnostic_t = time.time()

        estimator = LaplacePosteriorEstimator(
            likelihood=self.likelihood,
            priors=self.priors,
            minimization_method=self.kwargs["minimization_method"],
            n_prior_samples=self.kwargs["prior_nsamples"],
            use_unit_cube=self.kwargs["use_unit_cube"],
            jacobian_cap_scale=self.kwargs["jacobian_cap_scale"],
            hessian_kwargs=self.kwargs["hessian_kwargs"],
            fisher_method=self.kwargs["fisher_method"],
            fisher_kwargs=self.kwargs["fisher_kwargs"],
            marginalized_reference=self.injection_parameters,
            # Drawn from the same `random.rng` bilby seeds everywhere else
            # (via `sampling_seed_key = "seed"`, see below), rather than left
            # to scipy's own default -- differential_evolution's default seed
            # consults numpy's legacy global RandomState, which reseeding
            # `random.rng` does not touch, so without this the MAP search
            # would be unseeded even in an otherwise fully reproducible run.
            seed=int(random.rng.integers(2**32)),
        )

        # Validate any user-provided sampling covariance up-front (before the
        # MAP search) so naming/shape errors surface immediately.
        user_cov = self._resolve_sampling_cov(self.kwargs["sampling_cov"], estimator.parameter_names)
        cov_scaling = self._resolve_cov_scaling(self.kwargs["cov_scaling"], estimator.parameter_names)
        if user_cov is not None and self.kwargs["n_modes"] > 1:
            raise SamplerError(
                "sampling_cov cannot be combined with n_modes > 1; "
                "multi-mode search builds an independent covariance per mode."
            )

        # Wire any user-requested extra signal (SIGTERM/SIGINT/SIGALRM are
        # already handled by signal_wrapper).
        extra_sig = self.kwargs.get("checkpoint_signal")
        if extra_sig is not None:
            try:
                signal.signal(int(extra_sig), self.write_current_state_and_exit)
                logger.info(f"Wired signal {int(extra_sig)} for checkpoint + exit")
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                logger.warning(f"Could not wire checkpoint_signal={extra_sig}: {exc}")

        # Attempt to resume.  If a valid file exists we skip MAP & covariance
        # and restore (mean, cov, accumulators); a mismatched file is fatal.
        resumed = bool(self.kwargs.get("resume", True)) and self._read_saved_state()

        # scipy's own function-evaluation counts for MAP finding and the
        # Hessian -- not applicable when resuming (neither runs), and
        # hessian_nfev stays None when a user-supplied covariance or the
        # waveform Fisher skips scipy.differentiate.hessian entirely.
        map_nfev = None
        hessian_nfev = None

        if resumed:
            mean = np.asarray(self._checkpoint_state["mean"])
            cov = np.asarray(self._checkpoint_state["cov"])
            map_sample_dict = dict(zip(estimator.parameter_names, mean))
            logger.info("Skipping MAP and covariance estimation (resumed from checkpoint).")
        else:
            # Choose starting point for MAP search
            if self.injection_parameters and self.kwargs["use_injection_for_map"]:
                fallback = self.priors.sample_subset(estimator.parameter_names)
                missing = [k for k in estimator.parameter_names if k not in self.injection_parameters]
                if missing:
                    logger.warning(
                        f"use_injection_for_map=True but the following parameters are not in "
                        f"injection_parameters (using prior samples as fallback): {missing}"
                    )
                initial_sample = {
                    key: self.injection_parameters.get(key, fallback[key]) for key in estimator.parameter_names
                }
            else:
                initial_sample = None

            map_sample_dict = estimator.get_MAP_sample(initial_sample)
            minimization_metadata = getattr(estimator, "minimization_metadata", None)
            if minimization_metadata is not None:
                map_nfev = int(getattr(minimization_metadata, "nfev", 0) or 0)
            mean = np.array(list(map_sample_dict.values()))
            if user_cov is not None:
                logger.info("Using user-provided sampling covariance (skipping Laplace estimate)")
                covariance = user_cov
            else:
                covariance = estimator.calculate_posterior_covariance(map_sample_dict)
                hessian_metadata = getattr(estimator, "hessian_metadata", None)
                if hessian_metadata is not None:
                    # nfev is per-Hessian-entry (an (N, N) array from
                    # scipy.differentiate.hessian), so this sums scipy's own
                    # per-entry bookkeeping rather than counting distinct
                    # calls -- an upper bound if entries share cached
                    # evaluations, but consistent and comparable across runs.
                    hessian_nfev = int(np.sum(hessian_metadata.nfev))
            # Validate (repair) the *estimated* covariance first, then apply the
            # user's cov_scaling last so it is authoritative.  Validation
            # re-derives widths from the likelihood curvature, so scaling before
            # it lets validation silently override the requested scale for
            # poorly-estimated parameters.
            cov = self._validate_covariance(estimator, mean, covariance)
            cov = self._apply_cov_scaling(cov, cov_scaling)

        msg = "Gaussian proposal (MAP +/- 1-sigma):\n " + "\n ".join(
            f"{key}: {val:.5f} +/- {np.sqrt(var):.5f}" for (key, val), var in zip(map_sample_dict.items(), np.diag(cov))
        )
        logger.info(msg)

        # Build the proposal distribution.  For rejection and importance
        # sampling we use a truncated Gaussian (per-marginal independent
        # truncated normals) so that every draw lands within the prior support.
        # This eliminates wasted likelihood evaluations on out-of-bounds
        # samples and is especially important when the Laplace-derived sigma is
        # much larger than the prior width.
        # With ``n_modes > 1`` this also runs the mode search and returns a
        # mixture, which every resampling method below then draws from.  On a
        # resume the modes come back from the checkpoint instead: repeating the
        # search would cost a second multi-start optimisation and could hand
        # the second half of the run a different proposal from the first.
        if resumed:
            modes = self._checkpoint_state.get("modes")
            mode_log_weights = self._checkpoint_state.get("mode_log_weights")
            if modes is None:
                logger.warning(
                    "Checkpoint predates multi-mode proposals and carries no modes; "
                    "resuming from the single Gaussian about the stored MAP."
                )
                modes = [(mean, cov, None)]
            proposal = self._mode_proposal(estimator, modes, mode_log_weights)
        else:
            proposal, modes, mode_log_weights = self._build_proposal(estimator, mean, cov, cov_scaling)

        if self.kwargs["plot_diagnostic"] and not resumed:
            init_samples = self._draw_inprior_samples(proposal, 5000, estimator.parameter_names)
            self.create_proposal_diagnostic(mean, cov, estimator.parameter_names, init_samples)

        target_nsamples = self.kwargs["target_nsamples"]
        resample = self.kwargs["resample"]
        if resample == "None":
            resample = None

        # Laplace evidence (always available).  Compute fresh, or restore from
        # the checkpoint if we're resuming (we no longer have `covariance`).
        if resumed:
            log_evidence_laplace = self._checkpoint_state["log_evidence_laplace"]
        else:
            log_evidence_laplace = estimator.log_evidence_laplace(map_sample_dict, covariance)
            # Initialise the checkpoint payload for resumable modes.  Other
            # modes (`None`, `'smc'`) are not yet checkpointable; leave
            # _checkpoint_state as None so no file is written.
            if resample in ("rejection", "importance", "inprior"):
                self._init_checkpoint_state(mode=resample, mean=mean, cov=cov)
                self._checkpoint_state["log_evidence_laplace"] = log_evidence_laplace
                # The modes as well as the primary MAP: a resumed run has to
                # rebuild the *same* proposal, and re-running the mode search
                # would neither be free nor guaranteed to find the same modes.
                self._checkpoint_state["modes"] = modes
                self._checkpoint_state["mode_log_weights"] = mode_log_weights

        log_evidence = log_evidence_laplace
        log_evidence_err = np.nan

        # Set up the multiprocessing pool (bilby handles npool / a user-supplied
        # ``pool`` kwarg and stashes the likelihood in each worker once).  The
        # estimator uses it to evaluate batches of proposal likelihoods in
        # parallel; all resampling modes route through
        # ``log_likelihood_from_array`` so they all benefit.
        self._setup_pool()
        estimator.pool = self.pool
        estimator.npool = self.npool or 1

        # For most modes the number of likelihood evaluations equals the number
        # of proposal draws (``len(g_samples)``).  SMC is different: the real
        # work happens inside aspire's iterations, invisible to ``g_samples``,
        # so ``_run_smc`` returns the true count explicitly.
        nlikelihood = None
        try:
            if resample is None:
                samples, logl, g_samples, efficiency = self._sample_laplace(mean, cov, estimator, target_nsamples)
            elif resample == "smc":
                (
                    samples,
                    logl,
                    g_samples,
                    efficiency,
                    smc_log_z,
                    smc_log_z_err,
                    nlikelihood,
                ) = self._run_smc(proposal, estimator, modes, mode_log_weights)
                if smc_log_z is not None:
                    log_evidence = float(smc_log_z)
                    log_evidence_err = float(smc_log_z_err)
            elif resample == "rejection":
                samples, logl, g_samples, efficiency, log_evidence, log_evidence_err = self._run_rejection_sampling(
                    proposal, estimator, map_sample_dict
                )
            elif resample == "inprior":
                samples, logl, g_samples, efficiency = self._run_inprior(proposal, estimator)
                log_evidence = np.nan
                log_evidence_err = np.nan
            elif resample == "importance":
                samples, logl, g_samples, efficiency, log_evidence, log_evidence_err = self._run_importance_sampling(
                    proposal, estimator, map_sample_dict
                )
            elif resample == "emcee":
                samples, logl, g_samples, efficiency, nlikelihood = self._run_emcee(proposal, estimator)
                log_evidence = np.nan
                log_evidence_err = np.nan
            else:
                raise ValueError(
                    f"Unknown resample method {resample!r}. "
                    f"Expected one of: None, 'rejection', 'importance', 'inprior', 'smc', 'emcee'."
                )
        finally:
            self._close_pool()
            estimator.pool = None

        end_time = datetime.datetime.now()
        self.sampling_time = end_time - self.start_time

        # Both quantities above are on the *full* likelihood's footing:
        # ``LaplacePosteriorEstimator`` calls ``likelihood.log_likelihood()``
        # regardless of ``use_ratio``, so every evidence here is a full log Z.
        logger.info(
            f"Log-evidence summary (full log Z, noise term included): "
            f"Laplace={log_evidence_laplace:.2f}, "
            f"final={log_evidence:.2f} "
            f"+/- {log_evidence_err:.2f}"
        )

        # Hand bilby what its post-processing expects.  Under ``use_ratio`` it
        # reads the returned evidence as a log Bayes factor and adds the noise
        # evidence back on top (see ``bilby.core.sampler.run_sampler``), so the
        # noise term has to come off here first.  The per-sample values take the
        # identical shift, so both stay on the same (log Bayes factor)
        # convention bilby expects; leaving ``logl`` on the full-log-Z
        # convention here would double-count the noise evidence once bilby
        # adds it back.  Shifting by a constant leaves ``log_evidence_err`` alone.
        if self.use_ratio:
            log_noise_evidence = self.likelihood.noise_log_likelihood()
            logl -= log_noise_evidence
            log_evidence -= log_noise_evidence

        if nlikelihood is None:
            nlikelihood = len(g_samples)

        self._generate_result(
            samples,
            logl,
            log_evidence=log_evidence,
            log_evidence_err=log_evidence_err,
            log_evidence_laplace=log_evidence_laplace,
            efficiency=efficiency,
            nlikelihood=nlikelihood,
            map_nfev=map_nfev,
            hessian_nfev=hessian_nfev,
        )

        # The run finished cleanly; the resume file is no longer needed.
        self._cleanup_resume_file()
        self._checkpoint_state = None

        return self.result

    def _generate_result(
        self,
        samples,
        log_likelihood_evaluations,
        log_evidence=np.nan,
        log_evidence_err=np.nan,
        **run_stats,
    ):
        self.result.samples = samples[self.search_parameter_keys].values
        self.result.log_likelihood_evaluations = log_likelihood_evaluations
        self.result.log_evidence = log_evidence
        self.result.log_evidence_err = log_evidence_err
        # Populate bilby's standard field so `result.num_likelihood_evaluations`
        # reports the true count (the custom run_statistics["nlikelihood"] is
        # kept too for the comparison table).
        nlikelihood = run_stats.get("nlikelihood")
        if nlikelihood is not None:
            self.result.num_likelihood_evaluations = int(nlikelihood)
        run_stats["sampling_time_s"] = self.sampling_time.total_seconds()
        self.result.meta_data["run_statistics"] = run_stats

    def _sample_laplace(self, mean, cov, estimator, target_nsamples):
        """Draw samples directly from the Gaussian approximation without resampling.

        This mode is the raw Laplace approximation, so the draws are *not*
        filtered to the prior support -- that is what ``resample='inprior'``
        is for, and silently filtering here would erase the distinction.  The
        Gaussian has tails outside any bounded prior, so on a bounded or
        constrained prior some fraction of the output has zero prior density.
        That fraction is reported rather than hidden: if it is not small, the
        Gaussian is a poor fit to the constrained posterior and this mode's
        output should not be treated as posterior samples.
        """
        logger.info(f"Drawing {target_nsamples} samples from " f"Gaussian approximation (no resampling)")
        samples_array = random.rng.multivariate_normal(mean, cov, target_nsamples)
        samples = pd.DataFrame(samples_array, columns=estimator.parameter_names)
        samples = self._replace_with_prior_samples(samples, estimator.parameter_names)

        logpi = np.real(np.array(self.priors.ln_prob(samples, axis=0)))
        n_outside = int(np.sum(np.isinf(logpi)))
        if n_outside:
            logger.warning(
                f"{n_outside}/{target_nsamples} "
                f"({100.0 * n_outside / target_nsamples:.1f}%) of the Gaussian draws fall outside "
                f"the prior support (out of range, or violating a Constraint prior). "
                f"resample=None returns them unfiltered; use resample='inprior' to draw only "
                f"in-support samples."
            )

        logl = np.full(target_nsamples, np.nan)
        return samples, logl, samples, 100.0

    def _draw_inprior_samples(self, proposal, n, parameter_names):
        """Draw *n* samples from *proposal* filtered to the prior support.

        Draws in batches, discarding any sample where the full prior
        log-probability is ``-inf`` -- which covers both a parameter falling
        outside its own range and a ``Constraint`` violation, since
        ``PriorDict.ln_prob`` applies both.  No likelihood evaluations are
        performed.  Returns a ``(n, ndim)`` float array.
        """
        batch_nsamples = self.kwargs["batch_nsamples"]
        collected = []
        n_collected = 0
        total_drawn = 0

        while n_collected < n:
            if self._check_iteration_limit("In-prior sampling", total_drawn, n_collected):
                break
            x_batch = proposal.sample(batch_nsamples)
            g_batch = pd.DataFrame(x_batch, columns=parameter_names)
            # Substitute the prior-drawn parameters *before* the support test,
            # as `_run_inprior` and the weighted loops do.  Doing it afterwards
            # would hand back a cloud that was filtered and then perturbed: a
            # fresh draw for one parameter can push an accepted sample back
            # outside the support, which on a constrained prior silently
            # reintroduces exactly the points this filter exists to remove.
            g_batch = self._replace_with_prior_samples(g_batch, parameter_names)
            x_batch = g_batch.values
            logpi = np.real(np.array(self.priors.ln_prob(g_batch, axis=0)))
            in_prior = ~np.isinf(logpi)
            total_drawn += batch_nsamples
            if in_prior.any():
                collected.append(x_batch[in_prior])
                n_collected += int(in_prior.sum())

        if not collected:
            return np.empty((0, len(parameter_names)))
        x_out = np.vstack(collected)[:n]
        logger.debug(
            f"Drew {len(x_out)} in-prior samples from {total_drawn} proposals "
            f"({100.0 * len(x_out) / total_drawn:.1f}% efficiency)"
        )
        return x_out

    @staticmethod
    def _stratified_counts(n, weights):
        """Split *n* draws across components in proportion to *weights*.

        Largest-remainder apportionment, so the counts sum to exactly *n*.
        Every component gets at least one draw: a mode the Laplace weighting
        judges negligible may simply have a poorly estimated covariance, and
        leaving it entirely unrepresented in the initial cloud is not something
        SMC can recover from.
        """
        k = len(weights)
        if n < k:
            raise SamplerError(f"Cannot draw {n} initial samples across {k} modes; need at least one per mode.")
        w = np.asarray(weights, dtype=float)
        w = np.full(k, 1.0 / k) if not np.isfinite(w).all() or w.sum() <= 0 else w / w.sum()

        exact = w * (n - k)  # reserve one per mode, apportion the rest
        counts = np.floor(exact).astype(int)
        for i in np.argsort(exact - counts)[::-1][: int((n - k) - counts.sum())]:
            counts[i] += 1
        return (counts + 1).tolist()

    def _draw_initial_smc_samples(self, proposals, n, parameter_names, weights=None):
        """Draw *n* in-prior samples spread across *proposals*.

        With one proposal this is exactly ``_draw_inprior_samples``.  With
        several -- one per mode found by ``_find_multiple_maps`` -- the draw is
        stratified across modes, which is what makes the flow aspire fits to this
        cloud multimodal.  The split follows *weights* when given (so the cloud
        matches the mixture used as the proposal flow) and is even otherwise.
        Returns an ``(n, ndim)`` float array.
        """
        if len(proposals) == 1:
            return self._draw_inprior_samples(proposals[0], n, parameter_names)

        k = len(proposals)
        if weights is None:
            # Spread the remainder over the leading modes so the total is exactly n.
            counts = [n // k + (1 if i < n % k else 0) for i in range(k)]
        else:
            counts = self._stratified_counts(n, weights)
        chunks = []
        for i, (mode_proposal, n_i) in enumerate(zip(proposals, counts)):
            x_i = self._draw_inprior_samples(mode_proposal, n_i, parameter_names)
            if len(x_i) < n_i:
                logger.warning(f"Mode {i} contributed only {len(x_i)} of the {n_i} requested initial SMC samples")
            chunks.append(x_i)

        x_out = np.vstack(chunks)
        # Shuffle so the cloud is not ordered by mode: aspire's flow training
        # splits it into train/validation sets by position.
        random.rng.shuffle(x_out)
        logger.info(
            f"Initial SMC cloud: {len(x_out)} samples over {k} modes " f"({', '.join(str(len(c)) for c in chunks)})"
        )
        return x_out

    def _check_iteration_limit(self, method_name, n_proposed, n_accepted):
        """Check if iteration limit has been hit and acceptance rate is below 1%.

        Raises SamplerError or logs error depending on fail_on_error setting.
        Returns True if limit was hit, False otherwise.
        """
        max_iterations = int(self.kwargs["max_iterations"])
        if n_proposed < max_iterations:
            return False

        acceptance_rate = 100.0 * n_accepted / n_proposed
        if acceptance_rate >= 1.0:
            return False

        msg = (
            f"{method_name} hit iteration limit "
            f"({max_iterations:.0e} samples) with only "
            f"{n_accepted} accepted samples "
            f"({acceptance_rate:.2f}% < 1%). "
            f"The proposal is poorly matched to the likelihood. "
            f"Consider increasing cov_scaling."
        )
        if self.kwargs["fail_on_error"]:
            raise SamplerError(msg)
        else:
            logger.error(msg)
        return True

    def _run_inprior(self, proposal, estimator):
        """Draw and filter samples to those within prior bounds.

        Draws from the proposal and keeps only samples within the prior support,
        discarding out-of-bounds samples. Returns in-prior samples with their
        likelihoods evaluated.

        Returns ``(samples, logl, g_samples, efficiency)``.
        """
        target_nsamples = self.kwargs["target_nsamples"]
        batch_nsamples = self.kwargs["batch_nsamples"]

        logger.info(f"Drawing samples from proposal and filtering to prior support " f"(target: {target_nsamples})")

        state = self._checkpoint_state
        resumed_loop = state is not None and "samples_list" in state
        if resumed_loop:
            samples_list, logl_list = self._resume_accumulator_lists(state, "samples_list", "logl_list")
            total_drawn = int(state["total_drawn"])
            n_accepted = int(state["n_accepted"])
            logger.info(
                f"Resumed in-prior sampling at {n_accepted}/{target_nsamples} " f"accepted ({total_drawn} drawn)"
            )
            self._update_checkpoint_state(
                samples_list=samples_list,
                logl_list=logl_list,
                total_drawn=total_drawn,
                n_accepted=n_accepted,
            )
        else:
            samples_list = []
            logl_list = []
            total_drawn = 0
            n_accepted = 0
            self._update_checkpoint_state(
                samples_list=samples_list,
                logl_list=logl_list,
                total_drawn=total_drawn,
                n_accepted=n_accepted,
            )

        pbar = tqdm.tqdm(
            total=target_nsamples,
            desc="Filtering to prior",
            unit="sample",
            dynamic_ncols=True,
            initial=min(n_accepted, target_nsamples),
        )

        while n_accepted < target_nsamples:
            if self._check_iteration_limit("In-prior sampling", total_drawn, n_accepted):
                pbar.close()
                break
            x_batch = proposal.sample(batch_nsamples)
            g_batch = pd.DataFrame(x_batch, columns=estimator.parameter_names)

            # Apply prior replacement BEFORE checking if in prior
            g_batch = self._replace_with_prior_samples(g_batch, estimator.parameter_names)
            x_batch = g_batch.values

            logpi = np.real(np.array(self.priors.ln_prob(g_batch, axis=0)))
            in_prior = ~np.isinf(logpi)

            total_drawn += batch_nsamples
            batch_accepted = int(np.sum(in_prior))

            if in_prior.any():
                x_in = x_batch[in_prior]
                logl_in = estimator.log_likelihood_from_array(x_in.T)

                samples_list.append(x_in)
                logl_list.append(logl_in)
                n_accepted += batch_accepted

                samples_to_show = min(batch_accepted, max(0, target_nsamples - pbar.n))
                pbar.update(samples_to_show)
                pbar.set_postfix(
                    {
                        "drawn": total_drawn,
                        "eff": f"{100.0 * n_accepted / total_drawn:.1f}%",
                    }
                )

            self._update_checkpoint_state(n_accepted=n_accepted, total_drawn=total_drawn)
            self._maybe_periodic_checkpoint()

        pbar.close()

        if not samples_list:
            empty = pd.DataFrame(columns=estimator.parameter_names)
            return empty, np.array([]), empty, 0.0

        x_out = np.vstack(samples_list)[:target_nsamples]
        logl_out = np.hstack(logl_list)[:target_nsamples]
        samples = pd.DataFrame(x_out, columns=estimator.parameter_names)

        efficiency = 100.0 * n_accepted / total_drawn if total_drawn else 0.0
        logger.info(f"Filtering complete: kept {len(x_out)} of {total_drawn} samples ({efficiency:.1f}% efficiency)")

        return samples, logl_out, samples, efficiency

    def _mode_proposal(self, estimator, modes, log_weights):
        """The proposal for a set of ``(mean, cov, log_posterior)`` modes.

        A single :class:`TruncatedMVNProposal` for one mode, a
        :class:`TruncatedMVNMixtureProposal` for several.  Shared by the
        first-pass build and the resume path so a resumed run cannot quietly
        continue from a different proposal than it started with.
        """
        components = [
            TruncatedMVNProposal(
                mode_mean,
                mode_cov,
                lower=estimator.prior_bounds_min,
                upper=estimator.prior_bounds_max,
                periodic=self._periodic_mask(estimator.parameter_names),
            )
            for mode_mean, mode_cov, _ in modes
        ]
        if len(components) == 1:
            return components[0]
        weights = None if log_weights is None else np.exp(log_weights - logsumexp(log_weights))
        return TruncatedMVNMixtureProposal(components, weights)

    def _build_proposal(self, estimator, mean, cov, cov_scaling):
        """Find the posterior modes and build the proposal every mode draws from.

        This runs *before* the resampling branch, so ``n_modes`` applies to
        every method rather than to ``smc`` alone: every resampling method is
        seeded from the same mixture proposal, so comparisons between them
        stay controlled.

        Returns ``(proposal, modes, log_weights)`` where *modes* is a list of
        ``(mean, cov, log_posterior)`` triples (one entry, with
        ``log_posterior=None``, when ``n_modes == 1``) and *log_weights* is
        ``None`` for equal weighting or a single mode.
        """
        n_modes = self.kwargs["n_modes"]
        if n_modes <= 1:
            modes = [(np.asarray(mean, dtype=float), np.asarray(cov, dtype=float), None)]
            return self._mode_proposal(estimator, modes, None), modes, None

        modes = self._find_multiple_maps(estimator, n_modes, cov_scaling, mean, cov)
        mode_weighting = self.kwargs["mode_weights"]
        if mode_weighting == "laplace":
            log_weights = self._laplace_mode_log_weights(modes, len(estimator.parameter_names))
            modes, log_weights = self._drop_negligible_modes(modes, log_weights)
        elif mode_weighting == "equal":
            log_weights = None
        else:
            raise SamplerError(f"mode_weights must be 'equal' or 'laplace', got {mode_weighting!r}.")

        proposal = self._mode_proposal(estimator, modes, log_weights)
        weights = getattr(proposal, "weights", np.array([1.0]))
        self._log_mode_summary(modes, estimator.parameter_names, weights)
        return proposal, modes, log_weights

    def _run_smc(self, proposal, estimator, modes, log_weights):
        """Build the Laplace proposal flow and run SMC sampling.

        The modes were found by ``_build_proposal`` before the branch, so what
        happens here is only aspire's half of it: wrapping them in a flow and
        seeding the cloud.  Under the default ``smc_prior_flow="learned"`` the
        modes reach aspire only through the initial cloud its flow is trained
        on; under ``"laplace"`` the mixture is handed to aspire directly as the
        prior flow.

        Returns ``(samples, logl, g_samples, efficiency, smc_log_z,
        smc_log_z_err, nlikelihood)`` where ``smc_log_z`` is ``None`` if the
        aspire result did not carry a log-evidence attribute, and
        ``nlikelihood`` is the true number of likelihood evaluations performed
        by aspire (plus the final output evaluation).
        """
        if len(modes) > 1:
            proposal_flow = GaussianMixtureFlow(
                [m for m, _, _ in modes],
                [c for _, c, _ in modes],
                log_weights=log_weights,
            )
            mode_weight_values = proposal_flow.weights
            init_proposals = list(proposal.components)
        else:
            mode_mean, mode_cov, _ = modes[0]
            proposal_flow = GaussianFlow(mode_mean, mode_cov)
            init_proposals = [proposal]
            mode_weight_values = None

        samples, logl, smc_log_z, smc_log_z_err, nlikelihood = self._smc_sample(
            init_proposals, estimator, mode_weights=mode_weight_values, proposal_flow=proposal_flow
        )

        if self.kwargs["plot_diagnostic"]:
            self.create_smc_diagnostic(samples, proposal_flow)

        # Report efficiency on the same footing as the other resampling modes:
        # final samples per likelihood evaluation.  Unlike rejection/importance,
        # SMC always returns exactly the requested number of samples, so the
        # ratio of output samples to proposal draws is meaningless (always 1);
        # dividing by the *true* internal likelihood-evaluation count instead
        # makes this an apples-to-apples cost metric.
        efficiency = 100.0 * len(samples) / nlikelihood if nlikelihood else np.nan

        return samples, logl, samples, efficiency, smc_log_z, smc_log_z_err, nlikelihood

    def _compute_ln_ratios(self, x, g_df, proposal, estimator):
        """Compute log[L(x)π(x)/g(x)] for a batch of proposal samples.

        Returns ``(ln_r, logl)`` arrays of length ``len(g_df)``.
        Samples outside the prior get ``ln_r = -inf``.
        """
        logpi = np.real(np.array(self.priors.ln_prob(g_df, axis=0)))
        in_prior = ~np.isinf(logpi)
        logl = np.full(len(g_df), -np.inf)
        if in_prior.any():
            logl[in_prior] = estimator.log_likelihood_from_array(x[in_prior].T)
        else:
            msg = "All proposal samples fell outside the prior"
            if self.kwargs["fail_on_error"]:
                raise SamplerError(msg)
            logger.debug(msg)
        log_g = self._effective_log_proposal(proposal, x, list(g_df.columns))
        ln_r = logl + logpi - log_g
        return ln_r, logl

    def _run_rejection_sampling(self, proposal, estimator, map_sample_dict):
        """Draw samples from the proposal using rejection sampling.

        The acceptance probability for a proposal θ is L(θ)π(θ) / (M·g(θ)),
        where M is an upper bound on L(θ)π(θ)/g(θ).  The bound is established
        by a pre-scan batch before any samples are accepted, so that a
        mid-run bound update (which would invalidate earlier acceptances)
        cannot occur.

        Returns ``(samples, logl, g_samples, efficiency, log_evidence, log_evidence_err)``.
        """
        target_nsamples = self.kwargs["target_nsamples"]
        batch_nsamples = self.kwargs["batch_nsamples"]
        mean = proposal.mean

        state = self._checkpoint_state
        resumed_loop = state is not None and "all_samples" in state

        if resumed_loop:
            # Restore accumulators (including the pre-scan ln_M, which must not
            # be recomputed: every prior accept used this bound, and changing
            # it mid-run would invalidate the rejection sample).
            ln_M = float(state["ln_M"])
            all_samples, all_logl, all_g_samples, all_ln_r = self._resume_accumulator_lists(
                state, "all_samples", "all_logl", "all_g_samples", "all_ln_r"
            )
            n_accepted = int(state["n_accepted"])
            n_proposed = int(state["n_proposed"])
            n_bound_violations = int(state.get("n_bound_violations", 0))
            logger.info(
                f"Resumed rejection sampling at {n_accepted}/{target_nsamples} "
                f"accepted ({n_proposed} proposed, ln_M = {ln_M:.2f})"
            )
            self._update_checkpoint_state(
                ln_M=ln_M,
                all_samples=all_samples,
                all_logl=all_logl,
                all_g_samples=all_g_samples,
                all_ln_r=all_ln_r,
                n_accepted=n_accepted,
                n_proposed=n_proposed,
                n_bound_violations=n_bound_violations,
            )
        else:
            # --- Establish the rejection bound ln_M ---
            # Start from the analytic value at the MAP.
            ln_M = (
                float(estimator.log_likelihood_from_array(mean))
                + sum(
                    np.log(max(self.priors[k].prob(float(map_sample_dict[k])), 1e-300))
                    for k in estimator.parameter_names
                )
                - float(self._effective_log_proposal(proposal, mean.reshape(1, -1), estimator.parameter_names)[0])
            )

            # Pre-scan: draw in-prior calibration samples to find the empirical
            # maximum of L(x)π(x)/g(x).  Using in-prior samples avoids wasting
            # calibration evaluations on out-of-bounds points and gives a tighter
            # bound estimate.  These samples are discarded (not accepted/rejected)
            # so the bound is fixed before the main loop begins.
            x_cal = self._draw_inprior_samples(proposal, batch_nsamples, estimator.parameter_names)
            g_cal = pd.DataFrame(x_cal, columns=estimator.parameter_names)
            ln_r_cal, _ = self._compute_ln_ratios(x_cal, g_cal, proposal, estimator)
            finite_cal = np.isfinite(ln_r_cal)
            if finite_cal.any():
                empirical_max = float(np.max(ln_r_cal[finite_cal]))
                if empirical_max > ln_M:
                    logger.info(f"Pre-scan raised rejection bound " f"{ln_M:.2f} → {empirical_max:.2f}")
                    ln_M = empirical_max

            logger.info(
                f"Drawing {target_nsamples} samples using rejection sampling "
                f"(batch size {batch_nsamples}, ln_M = {ln_M:.2f})"
            )

            # --- Main rejection loop ---
            all_samples, all_logl, all_g_samples, all_ln_r = [], [], [], []
            n_accepted = n_proposed = 0
            n_bound_violations = 0
            # Stash the post-pre-scan state so an interruption before the
            # first batch still preserves ln_M.
            self._update_checkpoint_state(
                ln_M=ln_M,
                all_samples=all_samples,
                all_logl=all_logl,
                all_g_samples=all_g_samples,
                all_ln_r=all_ln_r,
                n_accepted=n_accepted,
                n_proposed=n_proposed,
                n_bound_violations=n_bound_violations,
            )

        pbar = tqdm.tqdm(total=target_nsamples, desc="Rejection sampling", file=sys.stdout)

        # In-memory-only accumulator of rejected draws for the live progress
        # diagnostic.  Deliberately not checkpointed (it would bloat the resume
        # file); on resume it simply rebuilds from freshly-drawn batches.
        all_rejected = []

        while n_accepted < target_nsamples:
            if self._check_iteration_limit("Rejection sampling", n_proposed, n_accepted):
                pbar.close()
                break

            x = proposal.sample(batch_nsamples)
            g_df = pd.DataFrame(x, columns=estimator.parameter_names)

            # Apply prior replacement BEFORE computing likelihoods
            g_df = self._replace_with_prior_samples(g_df, estimator.parameter_names)
            x = g_df.values

            ln_r, logl = self._compute_ln_ratios(x, g_df, proposal, estimator)
            finite = np.isfinite(ln_r)

            # Count samples that exceed the bound (accepted with prob 1,
            # which may introduce minor bias).
            if finite.any():
                n_over = int(np.sum(ln_r[finite] > ln_M))
                n_bound_violations += n_over

            # Accept if log(U) < ln_r - ln_M  ⟺  U < L(x)π(x) / (M·g(x))
            log_u = np.log(random.rng.uniform(size=batch_nsamples))
            accepted = finite & (log_u < ln_r - ln_M)

            n_proposed += batch_nsamples
            n_accepted += int(accepted.sum())
            efficiency = 100.0 * n_accepted / n_proposed
            pbar.set_postfix({"acceptance": f"{efficiency:.3f}%"}, refresh=False)
            pbar.update(int(accepted.sum()))

            g_accepted = g_df[accepted].reset_index(drop=True)
            all_samples.append(g_accepted)
            all_logl.append(logl[accepted])
            all_g_samples.append(g_df)
            all_ln_r.append(ln_r)
            if self.kwargs["plot_diagnostic"]:
                all_rejected.append(g_df[~accepted].reset_index(drop=True))

            # Update the checkpoint payload after each batch, then optionally
            # save it (throttled by check_point_delta_t).  A signal handler can
            # fire between batches and dump this state at any time.
            self._update_checkpoint_state(
                n_accepted=n_accepted,
                n_proposed=n_proposed,
                n_bound_violations=n_bound_violations,
            )
            self._maybe_periodic_checkpoint()

            if self.kwargs["plot_diagnostic"] and all_samples:
                self._maybe_periodic_rejection_diagnostic(
                    proposal,
                    pd.concat(all_samples, ignore_index=True),
                    pd.concat(all_rejected, ignore_index=True) if all_rejected else None,
                    estimator.parameter_names,
                )

        pbar.close()

        if not all_samples:
            empty = pd.DataFrame(columns=estimator.parameter_names)
            return empty, np.array([]), empty, 0.0, np.nan, np.nan

        if n_bound_violations > 0:
            logger.warning(
                f"{n_bound_violations} of {n_proposed} proposal samples "
                f"({100.0 * n_bound_violations / n_proposed:.1f}%) exceeded "
                f"the rejection bound and were accepted with probability 1. "
                f"The posterior may be slightly biased — consider increasing "
                f"cov_scaling to widen the proposal."
            )

        samples = pd.concat(all_samples, ignore_index=True)
        logl = np.concatenate(all_logl)
        g_samples = pd.concat(all_g_samples, ignore_index=True)
        ln_r_all = np.concatenate(all_ln_r)
        efficiency = 100.0 * len(samples) / len(g_samples)

        log_evidence, log_evidence_err = self._compute_is_evidence(ln_r_all)

        logger.info(
            f"Rejection sampling complete: {len(samples)} accepted from "
            f"{len(g_samples)} proposals ({efficiency:.1f}%)"
        )

        if self.kwargs["plot_diagnostic"]:
            weights = np.exp(ln_r_all - ln_M)
            try:
                self.create_resample_diagnostic(samples, g_samples, mean, weights, method="rejection")
            except Exception as e:
                logger.warning(f"Failed to create rejection sampling diagnostic plot: {e}")

        return samples, logl, g_samples, efficiency, log_evidence, log_evidence_err

    def _run_importance_sampling(self, proposal, estimator, map_sample_dict):
        """Draw samples from the proposal using importance resampling.

        Accumulates batches until ``target_nsamples`` are collected by
        resampling proportionally to the IS weights L(θ)π(θ)/g(θ).

        Returns ``(samples, logl, g_samples, efficiency, log_evidence, log_evidence_err)``.
        """
        target_nsamples = self.kwargs["target_nsamples"]
        batch_nsamples = self.kwargs["batch_nsamples"]

        state = self._checkpoint_state
        resumed_loop = state is not None and "all_samples" in state
        if resumed_loop:
            all_samples, all_logl, all_g_samples, all_ln_r = self._resume_accumulator_lists(
                state, "all_samples", "all_logl", "all_g_samples", "all_ln_r"
            )
            n_accepted = int(state["n_accepted"])
            n_proposed = int(state["n_proposed"])
            logger.info(
                f"Resumed importance sampling at {n_accepted}/{target_nsamples} " f"drawn ({n_proposed} proposed)"
            )
            self._update_checkpoint_state(
                all_samples=all_samples,
                all_logl=all_logl,
                all_g_samples=all_g_samples,
                all_ln_r=all_ln_r,
                n_accepted=n_accepted,
                n_proposed=n_proposed,
            )
        else:
            all_samples, all_logl, all_g_samples, all_ln_r = [], [], [], []
            n_accepted = n_proposed = 0
            self._update_checkpoint_state(
                all_samples=all_samples,
                all_logl=all_logl,
                all_g_samples=all_g_samples,
                all_ln_r=all_ln_r,
                n_accepted=n_accepted,
                n_proposed=n_proposed,
            )

        logger.info(f"Drawing {target_nsamples} samples using importance resampling " f"(batch size {batch_nsamples})")
        pbar = tqdm.tqdm(total=target_nsamples, desc="Importance sampling", file=sys.stdout)

        while n_accepted < target_nsamples:
            if self._check_iteration_limit("Importance sampling", n_proposed, n_accepted):
                pbar.close()
                break

            x = proposal.sample(batch_nsamples)
            g_df = pd.DataFrame(x, columns=estimator.parameter_names)

            # Apply prior replacement BEFORE computing likelihoods
            g_df = self._replace_with_prior_samples(g_df, estimator.parameter_names)
            x = g_df.values

            ln_r, logl = self._compute_ln_ratios(x, g_df, proposal, estimator)

            finite = np.isfinite(ln_r)
            if not finite.any():
                continue

            # Normalised weights for this batch
            ln_r_finite = ln_r[finite]
            ln_w = ln_r_finite - np.max(ln_r_finite)
            w = np.exp(ln_w)
            w /= w.sum()

            ess = int(np.floor(np.exp(kish_log_effective_sample_size(ln_r_finite))))
            n_draw = min(ess, len(g_df))

            if n_draw == 0:
                pbar.update(0)
                continue

            finite_idx = np.where(finite)[0]
            chosen = random.rng.choice(len(finite_idx), size=n_draw, p=w)
            idx = finite_idx[chosen]

            n_proposed += batch_nsamples
            n_accepted += n_draw
            efficiency = 100.0 * n_accepted / n_proposed
            pbar.set_postfix({"acceptance": f"{efficiency:.3f}%"}, refresh=False)
            pbar.update(n_draw)

            g_selected = g_df.iloc[idx].reset_index(drop=True)
            all_samples.append(g_selected)
            all_logl.append(logl[idx])
            all_g_samples.append(g_df)
            all_ln_r.append(ln_r)

            self._update_checkpoint_state(n_accepted=n_accepted, n_proposed=n_proposed)
            self._maybe_periodic_checkpoint()

        pbar.close()

        if not all_samples:
            empty = pd.DataFrame(columns=estimator.parameter_names)
            return empty, np.array([]), empty, 0.0, np.nan, np.nan

        samples = pd.concat(all_samples, ignore_index=True)
        logl = np.concatenate(all_logl)
        g_samples = pd.concat(all_g_samples, ignore_index=True)
        ln_r_all = np.concatenate(all_ln_r)
        efficiency = 100.0 * len(samples) / len(g_samples)

        log_evidence, log_evidence_err = self._compute_is_evidence(ln_r_all)

        logger.info(
            f"Importance sampling complete: {len(samples)} drawn from "
            f"{len(g_samples)} proposals ({efficiency:.1f}% effective)"
        )

        if self.kwargs["plot_diagnostic"]:
            ln_r_shifted = ln_r_all - np.nanmax(ln_r_all)
            weights = np.where(np.isfinite(ln_r_shifted), np.exp(ln_r_shifted), 0.0)
            self.create_resample_diagnostic(samples, g_samples, proposal.mean, weights, method="importance")

        return samples, logl, g_samples, efficiency, log_evidence, log_evidence_err

    def _compute_is_evidence(self, ln_r):
        """Estimate log Z and its uncertainty from raw IS log-weights.

        Uses ``log Z = logsumexp(ln_r) - log(N)`` with a delta-method variance.
        Returns ``(log_evidence, log_evidence_err)``.
        """
        finite = np.isfinite(ln_r)
        if not np.any(finite):
            return np.nan, np.nan

        ln_w = ln_r[finite]
        n_total = len(ln_r)
        log_z = logsumexp(ln_w) - np.log(n_total)
        log_z2 = logsumexp(2 * ln_w) - 2 * np.log(n_total)
        var_ratio = np.exp(log_z2 - 2 * log_z) - 1.0 / n_total
        log_z_err = np.sqrt(var_ratio) if var_ratio > 0 else 0.0

        logger.info(f"IS log-evidence: {log_z:.2f} +/- {log_z_err:.2f}")
        return log_z, log_z_err

    # Default reliability factor: an integrated-autocorrelation-time estimate
    # is only trustworthy once the chain is this many tau long. This is emcee's
    # own `integrated_time` default (its `tol`), and it warns below it.
    #
    # It is deliberately conservative -- Sokal's guidance is that a chain of
    # ~10 tau already gives a usable estimate, biased low -- so it is
    # overridable via ``emcee_kwargs={"autocorr_tol": ...}``. Lowering it
    # trades a weaker guarantee for a much shorter run, which on a
    # slow-mixing posterior can be the difference between finishing and not;
    # the achieved ratio is always logged, so the weaker guarantee stays
    # visible rather than being hidden by a passing check.
    #
    # This factor also sets the useful walker count. Independent samples are
    # `nwalkers * n_post_burn / tau`, and the bar says `n_post_burn / tau >=
    # tol`, so by the time the estimate can be trusted the run already holds
    # at least `tol * nwalkers` independent samples. Any nwalkers above
    # `target_nsamples / tol` therefore cannot stop before overshooting the
    # target -- at nwalkers=500, tol=50 and a 5000-sample target it must
    # reach ~25000, five times more sampling than asked for. Choosing
    # `nwalkers ~= target_nsamples / tol` makes the two conditions coincide.
    _EMCEE_AUTOCORR_RELIABLE_FACTOR = 50

    def _emcee_autocorr_status(self, ensemble, discard, nwalkers, parameter_names, tol=None):
        """Autocorrelation state of a (possibly still-growing) emcee chain.

        Returns ``(tau, tau_max, n_independent, reliable)``: the per-parameter
        integrated autocorrelation times, the largest of them (the binding
        one -- the chain is only as mixed as its worst coordinate), how many
        approximately independent samples the post-burn-in chain therefore
        holds, and whether it is long enough (*tol* times ``tau_max``, tol
        defaulting to :attr:`_EMCEE_AUTOCORR_RELIABLE_FACTOR`) for that
        estimate to be trusted at all.

        ``tau_max`` is ``nan`` and ``n_independent`` zero when no parameter
        yields a finite estimate, which is the honest answer for a chain too
        short to say anything about -- and, via ``reliable=False``, keeps the
        batch loop in :meth:`_run_emcee` running rather than stopping on a
        number that does not exist yet.
        """
        if tol is None:
            tol = self._EMCEE_AUTOCORR_RELIABLE_FACTOR
        # `quiet=True`: emcee raises `AutocorrError` below *its own* (fixed,
        # 50x) bar. Here that is the normal state of an early batch, not an
        # error -- the whole point of the loop is to keep sampling until our
        # `tol` clears, which may be lower than emcee's.
        tau = np.asarray(ensemble.get_autocorr_time(discard=discard, quiet=True), dtype=float)
        finite_tau = tau[np.isfinite(tau) & (tau > 0)]
        n_post_burn = max(0, ensemble.iteration - discard)
        if finite_tau.size == 0:
            return tau, np.nan, 0, False
        tau_max = float(np.max(finite_tau))
        n_independent = int(nwalkers * n_post_burn / tau_max)
        reliable = n_post_burn >= tol * tau_max
        return tau, tau_max, n_independent, reliable

    def _run_emcee(self, proposal, estimator):
        """Draw samples via emcee's affine-invariant ensemble sampler.

        Walkers are seeded from the Laplace proposal with
        :meth:`_draw_inprior_samples`, the same in-prior draw used to seed
        SMC's initial cloud, so every resampling method starts from an
        identically-constructed cloud.

        No reweighting is applied: emcee's stationary distribution *is* the
        target posterior, unlike the Gaussian proposal the other resampling
        methods correct. There is consequently no evidence estimate here.

        Periodic parameters (``boundary="periodic"``, e.g. a polarisation or
        sky angle) are wrapped into range before every log-probability
        evaluation and in the returned samples, so a walker can cross the
        seam freely instead of the crossing itself reading as leaving the
        prior -- the same treatment :class:`TruncatedMVNProposal` gives every
        other resampling mode.

        Sampling runs in batches of ``nsteps``, re-estimating the integrated
        autocorrelation time after each and continuing only while the chain
        holds fewer than ``target_nsamples`` independent samples, or is still
        too short for that estimate to be trusted (see
        :meth:`_emcee_autocorr_status`). This is what stops the run length
        from having to be guessed in advance: a chain that mixes well stops
        early, and one that does not keeps going to ``max_nsteps``.

        Batching is the whole cost control here. The convergence check itself
        touches no likelihood (it is an FFT over the stored chain), but it is
        a *serial* barrier -- every walker has to finish the batch before it
        runs -- so the batch has to be long enough that the barrier is
        amortised, while short enough that the run does not overshoot the
        target by much. Overshoot is bounded by one batch, so ``nsteps``
        trades a serial stall against wasted sampling.

        Consecutive walker steps are correlated, so the flattened chain is
        thinned by the estimated autocorrelation time before being returned
        -- otherwise the "samples" handed back are not independent draws,
        which is what every other resampling mode returns and what the
        comparison metrics (JSD/EMD against a reference sampler,
        effective-sample counts) assume. Pass ``emcee_kwargs={"thin": ...}``
        to override this with a fixed thinning factor instead.

        ``efficiency`` is reported on the same footing as every other
        resampling mode -- final (thinned) samples per likelihood evaluation
        -- rather than emcee's own mean acceptance fraction (logged
        separately): unlike rejection/importance, a rejected MCMC move still
        keeps (repeats) a sample, so acceptance fraction and
        samples-per-evaluation are not the same number here.

        Returns ``(samples, logl, g_samples, efficiency, nlikelihood)``.
        """
        import emcee

        parameter_names = estimator.parameter_names
        ndim = len(parameter_names)

        emcee_kw = dict(self.kwargs.get("emcee_kwargs") or {})
        nwalkers = int(emcee_kw.pop("nwalkers", max(4 * ndim, 32)))
        nsteps = int(emcee_kw.pop("nsteps", 5000))
        # `max_nsteps == nsteps` (the default) runs exactly one batch, i.e.
        # the previous fixed-length behaviour, so growth is opt-in.
        max_nsteps = int(emcee_kw.pop("max_nsteps", nsteps))
        discard = int(emcee_kw.pop("discard", nsteps // 2))
        thin_override = emcee_kw.pop("thin", None)
        target_nsamples = int(emcee_kw.pop("target_nsamples", self.kwargs["target_nsamples"]))
        autocorr_tol = float(emcee_kw.pop("autocorr_tol", self._EMCEE_AUTOCORR_RELIABLE_FACTOR))
        backend_file = emcee_kw.pop("backend_file", None)
        if discard >= nsteps:
            raise SamplerError(f"emcee_kwargs['discard'] ({discard}) must be less than nsteps ({nsteps}).")
        if max_nsteps < nsteps:
            raise SamplerError(
                f"emcee_kwargs['max_nsteps'] ({max_nsteps}) must be at least nsteps ({nsteps}), "
                f"which is the first batch's length."
            )
        if autocorr_tol <= 0:
            raise SamplerError(f"emcee_kwargs['autocorr_tol'] ({autocorr_tol}) must be positive.")

        logger.info(
            f"Running emcee: {nwalkers} walkers x {nsteps} steps per batch "
            f"(discard {discard}, max {max_nsteps} steps, target {target_nsamples} independent "
            f"samples, autocorr_tol {autocorr_tol:g})"
        )
        if autocorr_tol < self._EMCEE_AUTOCORR_RELIABLE_FACTOR:
            logger.warning(
                f"autocorr_tol={autocorr_tol:g} is below emcee's own {self._EMCEE_AUTOCORR_RELIABLE_FACTOR} "
                f"bar, so the autocorrelation time may be underestimated (and the independent-sample "
                f"count correspondingly overstated). The achieved chain-length/tau ratio is logged each "
                f"batch; check it before trusting the count."
            )
        # The reliability bar alone guarantees `autocorr_tol * nwalkers`
        # independent samples, so a large walker count cannot stop near the
        # target -- flag it rather than silently oversampling by that factor.
        floor_from_walkers = int(autocorr_tol * nwalkers)
        if floor_from_walkers > 2 * target_nsamples:
            logger.warning(
                f"nwalkers={nwalkers} cannot yield fewer than {floor_from_walkers} independent "
                f"samples once the autocorrelation estimate becomes reliable, far above the "
                f"{target_nsamples} requested: the run will overshoot. Consider "
                f"nwalkers~={int(target_nsamples // autocorr_tol)}."
            )

        # Optional persistence of the full (un-thinned) chain. `True` uses the
        # conventional per-run path; a string is taken as an explicit one.
        # Worth it for post-hoc analysis the run itself cannot do -- above all
        # re-estimating tau at several `discard` values, which distinguishes a
        # tau measuring leftover burn-in drift from a genuinely slow mode --
        # and it is the only copy of the raw chain, since the returned samples
        # are thinned. Off by default: the file is nwalkers * nsteps * ndim
        # floats, which is large for a long run.
        backend = None
        if backend_file:
            from bilby.core.utils import check_directory_exists_and_if_not_mkdir

            if backend_file is True:
                backend_file = f"{self.outdir}/{self.label}_emcee_chain.h5"
            check_directory_exists_and_if_not_mkdir(self.outdir)
            backend = emcee.backends.HDFBackend(str(backend_file))
            # Fresh run: `reset` clears any chain left by a previous one, so a
            # stale file cannot be silently appended to or mistaken for this
            # run's output. (Resuming from a backend is not wired up here.)
            backend.reset(nwalkers, ndim)
            logger.info(f"Persisting the full emcee chain to {backend_file}")

        initial_theta = self._draw_inprior_samples(proposal, nwalkers, parameter_names)

        # A periodic coordinate (e.g. psi, azimuth) has no hard edge: 0 and
        # the period are the same point.  emcee's stretch move knows nothing
        # about this and proposes freely in real-valued space, so without
        # wrapping, a proposal that drifts past a periodic parameter's
        # boundary reads as "outside the prior" (-inf) and is rejected --
        # exactly the false rejection-at-the-wall every other resampling mode
        # avoids via `TruncatedMVNProposal`'s wrapping / aspire's declared
        # `periodic_parameters`. Wrapping here lets a walker cross the seam
        # freely instead of piling up against it.
        periodic_mask = self._periodic_mask(parameter_names)
        lower = estimator.prior_bounds_min
        period = estimator.prior_bounds_max - lower

        def wrap_periodic(x):
            if not periodic_mask.any():
                return x
            x = np.array(x, dtype=float, copy=True)
            x[:, periodic_mask] = lower[periodic_mask] + np.mod(
                x[:, periodic_mask] - lower[periodic_mask], period[periodic_mask]
            )
            return x

        # Real likelihood-evaluation count, for cost accounting comparable
        # with the other resampling modes (see `_make_aspire_log_likelihood`,
        # which counts the same way): only in-prior points are ever handed to
        # the likelihood, so this is the work actually done, not nwalkers *
        # nsteps.
        n_likelihood_evaluations = [0]

        def log_prob_batch(x_batch):
            """Vectorised log-posterior for a batch of walker positions.

            Out-of-prior-support points (including ``Constraint``
            violations, via ``PriorDict.ln_prob``) get ``-inf`` and are never
            handed to the likelihood, matching every other resampling mode.
            Periodic parameters are wrapped into range first, so crossing
            their boundary is not itself treated as leaving the prior.
            """
            x_batch = wrap_periodic(np.atleast_2d(np.asarray(x_batch, dtype=float)))
            g_df = pd.DataFrame(x_batch, columns=parameter_names)
            logpi = np.real(np.array(self.priors.ln_prob(g_df, axis=0)))
            in_prior = ~np.isinf(logpi)
            logl = np.full(len(g_df), -np.inf)
            if in_prior.any():
                logl[in_prior] = estimator.log_likelihood_from_array(x_batch[in_prior].T)
                n_likelihood_evaluations[0] += int(in_prior.sum())
            return logl + logpi

        # vectorize=True hands log_prob_batch the whole set of walkers being
        # moved at once (rather than one at a time), which is what lets it
        # route through the estimator's pooled batch path exactly like every
        # other resampling mode.
        ensemble = emcee.EnsembleSampler(nwalkers, ndim, log_prob_batch, vectorize=True, backend=backend, **emcee_kw)

        # --- Batched sampling, growing the chain until it has enough
        # independent samples (or hits `max_nsteps`). ---
        state = initial_theta
        tau = np.full(ndim, np.nan)
        tau_max, n_independent, reliable = np.nan, 0, False
        tau_history = []
        while True:
            n_run = min(nsteps, max_nsteps - ensemble.iteration)
            ensemble.run_mcmc(state, n_run, progress=True)
            # `None` continues from the sampler's stored final state, so the
            # next batch is a seamless continuation, not a restart.
            state = None

            tau, tau_max, n_independent, reliable = self._emcee_autocorr_status(
                ensemble, discard, nwalkers, parameter_names, tol=autocorr_tol
            )
            # Kept for the diagnostic: how tau evolves with chain length is
            # what distinguishes a converging chain (tau flattens) from a
            # drifting one (tau grows with the chain, so the independent-sample
            # count never improves however long the run).
            tau_history.append((ensemble.iteration, np.array(tau, dtype=float)))
            # The chain-length/tau ratio is the achieved reliability, quoted
            # against the bar so a lowered `autocorr_tol` cannot hide how far
            # short of emcee's own 50x the estimate actually is.
            ratio = (ensemble.iteration - discard) / tau_max if np.isfinite(tau_max) and tau_max > 0 else np.nan
            logger.info(
                f"emcee at {ensemble.iteration}/{max_nsteps} steps: "
                f"tau_max={tau_max:.1f}, chain/tau={ratio:.1f} (bar {autocorr_tol:g}), "
                f"{n_independent}/{target_nsamples} independent samples"
                f"{'' if reliable else ' (tau estimate not yet reliable)'}"
            )
            if reliable and n_independent >= target_nsamples:
                break
            if ensemble.iteration >= max_nsteps:
                caveat = (
                    ""
                    if reliable
                    else (
                        ", and the autocorrelation estimate is still below its reliability bar, "
                        "so both that count and the thinning below are uncertain"
                    )
                )
                logger.warning(
                    f"emcee stopped at max_nsteps={max_nsteps} with {n_independent} independent "
                    f"samples against the {target_nsamples} requested{caveat}. "
                    f"Raise max_nsteps, or reduce target_nsamples."
                )
                break

        logger.info(
            "Autocorrelation time per parameter: "
            + ", ".join(f"{name}={t:.1f}" for name, t in zip(parameter_names, tau))
        )

        if thin_override is not None:
            thin = max(1, int(thin_override))
            logger.info(f"Using user-specified emcee thinning: thin={thin}")
        elif not np.isfinite(tau_max):
            thin = 1
            logger.warning(
                "Could not estimate the emcee autocorrelation time (non-finite for every "
                "parameter); returning unthinned samples. Consider raising max_nsteps."
            )
        else:
            thin = max(1, int(np.ceil(tau_max)))

        chain = ensemble.get_chain(discard=discard, thin=thin, flat=True)
        log_prob_flat = ensemble.get_log_prob(discard=discard, thin=thin, flat=True)

        # emcee's stored chain holds the raw (possibly out-of-range) walker
        # positions `log_prob_batch` wrapped only for evaluation; wrap the
        # output too so returned periodic parameters sit in their canonical
        # range, matching every other resampling mode.
        chain = wrap_periodic(chain)
        samples = pd.DataFrame(chain, columns=parameter_names)
        # Recover log-likelihood alone (emcee only tracks the log-posterior)
        # by subtracting the log-prior back off; cheap relative to the
        # likelihood and avoids a second round of likelihood evaluations.
        logpi = np.real(np.array(self.priors.ln_prob(samples, axis=0)))
        logl = log_prob_flat - logpi

        nlikelihood = int(n_likelihood_evaluations[0])
        efficiency = 100.0 * len(samples) / nlikelihood if nlikelihood else np.nan
        logger.info(
            f"emcee complete: {len(samples)} samples kept (thin={thin}) from {nlikelihood} "
            f"likelihood evaluations ({efficiency:.3f}% effective); mean acceptance fraction "
            f"{100.0 * float(np.mean(ensemble.acceptance_fraction)):.1f}%"
        )

        if self.kwargs["plot_diagnostic"]:
            try:
                self.create_emcee_diagnostic(samples, ensemble, discard, thin, tau_history, autocorr_tol)
            except Exception as exc:  # a diagnostic must never crash a completed run
                logger.warning(f"Failed to create emcee diagnostic plot: {exc}")

        return samples, logl, samples, efficiency, nlikelihood

    @staticmethod
    def _make_aspire_log_likelihood(estimator, counter):
        """Return an aspire-compatible batched log-likelihood.

        Aspire hands the likelihood a whole batch of samples at once, but
        ``aspire_bilby``'s wrapper loops over that batch with a ``map_fn``
        keyword defaulting to the builtin (serial) ``map``; unless something
        binds a pool's ``map`` into it (as aspire_bilby's own plugin does via
        ``PoolHandler``), every evaluation runs in a single process.  Rather
        than bind it -- which would require aspire_bilby's module-level globals
        to be populated in every worker, i.e. the pool to be created *after*
        ``get_aspire_functions`` -- route the batch through the estimator.  It
        applies the same fixed-parameter and prior-bounds handling as every
        other resampling mode, and its pooled path pulls the likelihood from
        bilby's per-worker global, so it is correct for any pool start method.

        ``counter`` is a single-element mutable list into which the number of
        likelihood evaluations is accumulated.  Only points with a finite
        log-prior are evaluated (matching ``aspire_bilby``), so the count is
        the work actually done rather than the output-sample count.
        """

        def batched_log_likelihood(samples):
            if getattr(samples, "log_prior", None) is None:
                raise SamplerError("aspire called the log-likelihood before the log-prior was evaluated")
            mask = np.isfinite(np.asarray(samples.log_prior))
            counter[0] += int(np.sum(mask))
            logl = np.full(len(mask), -np.inf)
            if mask.any():
                x = np.asarray(samples.x, dtype=float)[mask, :]
                logl[mask] = estimator.log_likelihood_from_array(x.T)
            return logl

        return batched_log_likelihood

    def _smc_sample(self, init_proposals, estimator, mode_weights=None, proposal_flow=None):
        """Run posterior sampling via aspire, starting from the Laplace proposal.

        *init_proposals* is a list of ``TruncatedMVNProposal`` objects, one per
        discovered posterior mode (a single entry unless ``n_modes > 1``).
        Initial samples are drawn from them and filtered to the prior support,
        matching the inprior/rejection sampling approach.  That cloud is what
        ``aspire.fit()`` trains the flow on, and aspire passes that flow to the
        sampler as its ``prior_flow`` -- so it, and nothing else, sets where the
        annealing path starts.  ``aspire.sample_posterior()`` then refines the
        particles toward the true posterior.
        """
        from aspire import Aspire
        from aspire.samples import Samples
        from aspire_bilby.utils import get_aspire_functions

        parameter_names = estimator.parameter_names

        # Per-parameter box, which is all aspire's logit preconditioning can
        # express -- a Constraint bounds a derived quantity and has no
        # rectangular form.  Constraints still bind, via the log-prior below:
        # aspire_bilby's `log_prior` is `priors.ln_prob(...)`, which returns
        # -inf on a violation, and `_make_aspire_log_likelihood` masks on that
        # before evaluating anything.  So the constrained region is enforced in
        # the target, not in the preconditioner.
        prior_bounds = {key: (self.priors[key].minimum, self.priors[key].maximum) for key in parameter_names}

        # Only the log-prior is taken from aspire_bilby; the likelihood is our
        # own pool-aware wrapper (see ``_make_aspire_log_likelihood``).
        functions = get_aspire_functions(self.likelihood, self.priors, parameter_names)

        n_likelihood_evaluations = [0]
        batched_log_likelihood = self._make_aspire_log_likelihood(estimator, n_likelihood_evaluations)

        # Aspire has to be told which coordinates wrap.  Without this a
        # periodic parameter gets the bounded->logit preconditioning instead of
        # angular treatment and the pCN kernel cannot step across the boundary
        # -- on the HLV example that affects psi and azimuth, and psi was
        # consistently the worst-recovered parameter.  Derived from the priors
        # the same way ``aspire_bilby`` derives it (``boundary == "periodic"``),
        # then restricted to sampled parameters: a marginalised periodic
        # coordinate such as phase is not part of aspire's parameter vector.
        mask = self._periodic_mask(parameter_names)
        periodic_parameters = [key for key, is_periodic in zip(parameter_names, mask) if is_periodic]
        if periodic_parameters:
            logger.info(f"Declaring periodic parameter(s) to aspire: {periodic_parameters}")

        # Flow architecture goes to the Aspire *constructor* (it stores any
        # extra kwargs as flow_kwargs), not to sample_posterior, which would
        # forward them to sample() and raise.  ``n_final_samples`` by contrast
        # is a sample() argument and rides through with the rest of smc_kw.
        flow_kwargs = dict((self.kwargs.get("smc_kwargs") or {}).get("flow_kwargs") or {})
        if flow_kwargs:
            logger.info(f"Flow architecture: {flow_kwargs}")

        # ``smc_prior_flow="laplace"`` hands aspire our analytic mixture as the
        # flow.  ``Aspire.fit`` skips ``init_flow()`` when one is already set
        # and then calls ``flow.fit(...)``, which is a no-op on ours -- so
        # nothing is trained and ``log q`` in the tempered target is the
        # Laplace mixture in closed form.
        prior_flow_mode = self.kwargs.get("smc_prior_flow", "learned")
        if prior_flow_mode not in ("learned", "laplace"):
            raise SamplerError(f"smc_prior_flow must be 'learned' or 'laplace', got {prior_flow_mode!r}.")
        supplied_flow = proposal_flow if prior_flow_mode == "laplace" else None
        if prior_flow_mode == "laplace" and proposal_flow is None:
            raise SamplerError("smc_prior_flow='laplace' requires a proposal flow, which run_sampler did not build.")

        aspire_sampler = Aspire(
            log_likelihood=batched_log_likelihood,
            log_prior=functions.log_prior,
            flow=supplied_flow,
            dims=len(parameter_names),
            parameters=parameter_names,
            prior_bounds=prior_bounds,
            periodic_parameters=periodic_parameters,
            **flow_kwargs,
        )

        # Copy so we can pop without mutating the user's dict
        smc_kw = dict(self.kwargs.get("smc_kwargs") or {})
        sampler_type = smc_kw.pop("sampler", "importance")
        n_initial = smc_kw.pop("n_initial_samples", 1000)
        # ``n_samples`` is aspire's SMC *particle count*: it is carried through
        # every tempering iteration and mutation step, so it drives the whole
        # cost of the run (not merely the size of a final draw).  It maps to the
        # first positional argument of ``sample_posterior``.
        n_samples = smc_kw.pop("n_samples", self.kwargs["target_nsamples"])
        smc_kw.pop("flow_kwargs", None)  # consumed by the Aspire constructor above

        # Seed aspire's own generator from the sampler seed.
        #
        # ``sampling_seed_key = "seed"`` makes bilby reseed *its* global
        # generator, which covers the Laplace stage, but aspire builds a
        # separate RNG of its own and defaults it to OS entropy.  Without this
        # the SMC stage stays unreproducible even when a seed is given -- and
        # since the Laplace stage is deterministic, aspire's flow training and
        # MCMC are where all the run-to-run scatter lives.
        #
        # Only ``numpy`` is assumed: it is the backend these examples run on. A
        # torch/jax run should pass its own ``rng`` through ``smc_kwargs``,
        # which takes precedence.
        seed = self.kwargs.get("seed")
        if seed is not None and smc_kw.get("rng") is None:
            try:
                from orng import RandomGenerator as _AspireRNG  # aspire >= 0.1.0a21
            except ImportError:
                from orng import ArrayRNG as _AspireRNG
            smc_kw["rng"] = _AspireRNG("numpy", seed=int(seed))
            logger.info(f"Seeding aspire RNG with {int(seed)}")

        # Draw initial samples filtered to the prior support, consistent with
        # the inprior/rejection sampling paths.
        initial_theta = self._draw_initial_smc_samples(init_proposals, n_initial, parameter_names, weights=mode_weights)

        if len(init_proposals) > 1 and self.kwargs["plot_diagnostic"]:
            # Overwrite the proposal diagnostic now that the real initial cloud
            # exists.  The copy written by ``run_sampler`` predates the mode
            # search, so it shows only the primary mode and misrepresents where
            # a multi-mode SMC run actually starts.
            self.create_proposal_diagnostic(
                init_proposals[0].mean,
                init_proposals[0].cov,
                parameter_names,
                initial_theta,
            )

        initial_samples = Samples(initial_theta, parameters=parameter_names)
        aspire_sampler.fit(initial_samples)

        # SMC checkpoint/resume integration (only for SMC-family samplers,
        # which support aspire's checkpoint_file_path / resume_from kwargs).
        is_smc = "smc" in sampler_type.lower()
        smc_file_checkpoint = None
        if is_smc and bool(self.kwargs.get("resume", True)):
            smc_file_checkpoint = self._smc_resume_file_path()
            from bilby.core.utils import check_directory_exists_and_if_not_mkdir

            check_directory_exists_and_if_not_mkdir(self.outdir)
            if os.path.isfile(smc_file_checkpoint):
                smc_kw.setdefault("resume_from", smc_file_checkpoint)
                logger.info(f"Resuming SMC from {smc_file_checkpoint}")
            else:
                # Make sure aspire's default file callback fires when no
                # custom callback is provided.
                smc_kw.setdefault("checkpoint_file_path", smc_file_checkpoint)
                smc_kw.setdefault("checkpoint_every", 1)

        # Register a per-iteration progress callback for SMC-family samplers
        # when `smc_progress` is enabled.  When a checkpoint file path is set
        # (resume=True), the callback also writes aspire's HDF5 checkpoint so
        # both behaviours coexist.  The user can override either by passing
        # their own checkpoint_callback / checkpoint_every in smc_kwargs.
        if self.kwargs.get("smc_progress", True) and is_smc:
            smc_kw.setdefault(
                "checkpoint_callback",
                self._make_smc_callback(aspire_sampler, file_checkpoint_path=smc_file_checkpoint),
            )
            smc_kw.setdefault("checkpoint_every", 1)

        logger.info(f"Starting Aspire sampling (sampler: {sampler_type})")
        result, self._smc_history = aspire_sampler.sample_posterior(
            n_samples, sampler=sampler_type, return_history=True, **smc_kw
        )

        self._save_smc_figures(self._smc_history, result)

        x_out = np.asarray(result.x)
        samples = pd.DataFrame(x_out, columns=parameter_names)
        logl = estimator.log_likelihood_from_array(x_out.T)

        # True likelihood-evaluation count: everything aspire evaluated
        # internally, plus the final evaluation of the output samples above.
        nlikelihood = int(n_likelihood_evaluations[0]) + len(x_out)
        logger.info(
            f"SMC used {n_likelihood_evaluations[0]} likelihood evaluations "
            f"internally (+{len(x_out)} for the final output samples); "
            f"total {nlikelihood}"
        )

        smc_log_z = getattr(result, "log_evidence", None)
        smc_log_z_err = getattr(result, "log_evidence_error", np.nan)
        if smc_log_z is not None:
            logger.info(f"Aspire log-evidence: {smc_log_z:.2f} " f"+/- {smc_log_z_err:.2f}")

        return samples, logl, smc_log_z, smc_log_z_err, nlikelihood

    def _make_smc_callback(self, aspire_sampler, file_checkpoint_path=None):
        """Return a ``checkpoint_callback`` for the SMC sampler.

        The callback runs once per iteration: it logs a one-line summary, and
        when ``plot_diagnostic`` is enabled, overwrites the stats and
        evolution-and-marginals figures so they reflect the latest state.

        If ``file_checkpoint_path`` is set, the callback also writes the
        aspire HDF5 checkpoint file for that iteration so the SMC run can be
        resumed from disk via ``resume_from``.  The aspire SMC sampler
        normally installs its default file callback only when
        ``checkpoint_callback`` is ``None``; we have to compose the two by
        hand to keep both behaviours.
        """
        plot_diagnostic = bool(self.kwargs.get("plot_diagnostic", False))
        plot_every = int(self.kwargs.get("smc_plot_every", 0) or 0)
        # Aspire force-calls the callback once more at the end with the same
        # iteration number; track the last-logged iteration so the per-iter log
        # line is not duplicated.  Plotting stays idempotent.
        last_logged_iter = [-1]
        # Aspire's default file callback lives on the inner SMC sampler
        # instance, which only exists once ``sample_posterior`` has built it.
        # Resolve it lazily and cache.
        file_cb_cache = [None]

        def callback(state):
            inner = getattr(aspire_sampler, "sampler", None) or getattr(aspire_sampler, "_sampler", None)
            history = getattr(inner, "history", None)
            if history is None:
                return

            # Write aspire's file checkpoint first so resume state is durable
            # even if logging/plotting raise below.
            if file_checkpoint_path is not None and inner is not None:
                if file_cb_cache[0] is None:
                    try:
                        file_cb_cache[0] = inner.default_file_checkpoint_callback(str(file_checkpoint_path))
                    except Exception as exc:  # never crash the run for a failed checkpoint
                        logger.warning(f"Could not initialise SMC file checkpoint: {exc}")
                        file_cb_cache[0] = False  # sentinel: don't try again
                if file_cb_cache[0] not in (None, False):
                    try:
                        file_cb_cache[0](state)
                    except Exception as exc:
                        logger.warning(f"SMC file checkpoint write failed: {exc}")

            iteration = state.get("iteration", -1)
            if iteration != last_logged_iter[0]:
                try:
                    self._log_smc_iteration(state, history)
                except Exception as exc:  # logging is best-effort
                    logger.debug(f"SMC per-iter logging failed: {exc}")
                last_logged_iter[0] = iteration
            # Re-rendering every iteration is expensive: the evolution figure
            # fits a gaussian_kde per parameter per iteration, so the cost grows
            # quadratically over a run.  Off by default -- the figures are
            # written once at the end instead (see `_save_smc_figures`).
            if not plot_diagnostic or plot_every <= 0 or iteration % plot_every:
                return
            try:
                self._save_smc_stats_figure(history)
                live_samples = state.get("samples")
                if live_samples is not None and history.sample_history:
                    self._save_smc_evolution_marginals_figure(history, live_samples)
            except Exception as exc:  # plotting is best-effort
                logger.warning(f"SMC per-iter plotting failed: {exc}")

        return callback

    def _log_smc_iteration(self, state, history):
        """One-line per-iteration SMC progress summary."""
        iteration = state.get("iteration", -1)
        meta = state.get("meta") or {}
        beta = meta.get("beta", float("nan"))
        live_samples = state.get("samples")

        parts = [f"SMC iter {iteration}", f"β={beta:.3g}"]
        if history.ess:
            parts.append(f"ESS={history.ess[-1]:.0f}")
        if getattr(history, "ess_target", None):
            parts.append(f"target={history.ess_target[-1]:.0f}")
        log_z = getattr(live_samples, "log_evidence", None) if live_samples is not None else None
        if log_z is not None and np.isfinite(log_z):
            parts.append(f"log Z≈{float(log_z):.2f}")
        if getattr(history, "mcmc_acceptance", None):
            parts.append(f"accept={history.mcmc_acceptance[-1]:.2f}")
        if getattr(history, "mcmc_autocorr", None):
            parts.append(f"autocorr={history.mcmc_autocorr[-1]:.1f}")
        logger.info(", ".join(parts))

    def _save_smc_figures(self, history, live_samples):
        """Write the SMC stats and evolution figures.

        Called once after sampling so the figures exist even when the
        per-iteration callback is not rendering them (the default).  Plotting is
        best-effort: a failed figure must not lose a completed run.
        """
        if not self.kwargs.get("plot_diagnostic", False) or history is None:
            return
        try:
            self._save_smc_stats_figure(history)
            if live_samples is not None and getattr(history, "sample_history", None):
                self._save_smc_evolution_marginals_figure(history, live_samples)
        except Exception as exc:
            logger.warning(f"SMC diagnostic plotting failed: {exc}")

    def _save_smc_stats_figure(self, history):
        """Overwrite the SMC stats figure with the current history."""
        import matplotlib.pyplot as plt

        fig_stats, _ = plt.subplots(6, 1, sharex=True, figsize=(8, 14))
        fig_stats = history.plot(fig=fig_stats)
        fig_stats.suptitle("SMC diagnostics")
        # Reserve headroom for the suptitle; a bare tight_layout packs the top
        # panel against it and the title overlaps the first axis.
        fig_stats.tight_layout(rect=[0, 0, 1, 0.99])
        safe_save_figure(
            fig=fig_stats,
            filename=f"{self.outdir}/{self.label}_diagnostic_smc_stats.png",
            dpi=150,
        )
        plt.close(fig_stats)

    def _label_for(self, name):
        """LaTeX axis label for a parameter, taken from the prior's
        ``latex_label`` when one is available, otherwise the parameter name
        with underscores replaced by spaces."""
        prior = None
        try:
            prior = self.priors[name]
        except (KeyError, TypeError):
            prior = None
        label = getattr(prior, "latex_label", None) if prior is not None else None
        return label if label else name.replace("_", " ")

    def _labels_for(self, names):
        """LaTeX axis labels for a sequence of parameters (see ``_label_for``)."""
        return [self._label_for(name) for name in names]

    def _save_smc_evolution_marginals_figure(self, history, live_samples):
        """Overwrite the SMC evolution figure with weighted-particle scatter
        evolution (left) and weighted current 1-D marginals (right) per
        parameter.  The left panel resamples each iteration's particles
        according to their weights so the visible density matches the SMC
        approximation to the posterior at that iteration."""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        parameter_names = list(self.search_parameter_keys)
        # Extra rows for log-likelihood and log-prior
        n_rows = len(parameter_names) + 2
        fig, axs = plt.subplots(
            n_rows,
            2,
            figsize=(11, 2.2 * n_rows),
            gridspec_kw={"width_ratios": [3, 1.5]},
            squeeze=False,
        )
        # Share x only within the left (iteration) column. The right-column
        # marginals each have their own parameter scale and must autoscale
        # independently -- sharing them collapses every marginal onto the
        # widest parameter's range (e.g. log prior), leaving the rest blank.
        for r in range(1, n_rows):
            axs[r, 0].sharex(axs[0, 0])

        # Pre-compute per-iteration data:
        #   - (subsampled, weight-resampled) particle cloud for the sina scatter
        #   - full particle array + normalised weights for the median/interval
        max_per_iter = 500
        rng = random.rng
        per_iter = []  # list of (it, x_for_scatter, x_full, weights_full_or_None)
        for it, smc_samples in enumerate(history.sample_history):
            x_full = np.asarray(smc_samples.x)
            n_pts = len(x_full)
            if n_pts == 0:
                continue
            try:
                log_w = np.asarray(smc_samples.log_weights())
                w = np.exp(log_w - logsumexp(log_w))
            except Exception:
                w = None
            if n_pts > max_per_iter:
                if w is not None:
                    idx = rng.choice(n_pts, size=max_per_iter, replace=True, p=w)
                else:
                    idx = rng.choice(n_pts, size=max_per_iter, replace=False)
                x_scatter = x_full[idx]
            else:
                x_scatter = x_full
            per_iter.append((it, x_scatter, x_full, w))

        def _weighted_quantile(values, weights, q):
            """Linear-interpolated weighted quantile."""
            if weights is None:
                return np.quantile(values, q)
            order = np.argsort(values)
            v = values[order]
            cum_w = np.cumsum(weights[order])
            total = cum_w[-1]
            if total <= 0:
                return np.quantile(values, q)
            return np.interp(q, cum_w / total, v)

        # Live (current) marginals — weighted histogram on the right.
        x_now = np.asarray(live_samples.x)
        try:
            log_w_now = np.asarray(live_samples.log_weights())
            w_now = np.exp(log_w_now - logsumexp(log_w_now))
        except Exception:
            w_now = None

        # Sina-style jitter: horizontal offset proportional to local 1-D
        # density, so each iteration's column bulges where particles are dense.
        # Half-width is kept below 0.5 so adjacent iterations don't overlap.
        from scipy.stats import gaussian_kde

        sina_half_width = 0.4

        # Helper to plot one row (evolution scatter on left, marginal on right).
        def _plot_row(row_idx, vals_per_iter, vals_now, label, is_last, true_val=None):
            ax_left = axs[row_idx, 0]

            # True value line behind scatter.
            if true_val is not None:
                ax_left.axhline(
                    true_val,
                    color="lightgray",
                    ls="--",
                    lw=1.2,
                    zorder=1,
                )

            for it, v_scatter, v_full, w in vals_per_iter:
                n = len(v_scatter)
                if n >= 2 and np.ptp(v_scatter) > 0:
                    try:
                        kde = gaussian_kde(v_scatter)
                        density = kde(v_scatter)
                        peak = density.max()
                        density_norm = density / peak if peak > 0 else np.zeros(n)
                    except Exception:
                        density_norm = np.zeros(n)
                else:
                    density_norm = np.zeros(n)
                jitter = rng.uniform(-1.0, 1.0, size=n) * sina_half_width * density_norm
                ax_left.scatter(
                    it + jitter,
                    v_scatter,
                    s=4,
                    alpha=min(0.8, 250.0 / max(n, 1)),
                    color="C0",
                    edgecolors="none",
                    zorder=2,
                )

                # Weighted median and 90% interval from the full ensemble.
                lo = _weighted_quantile(v_full, w, 0.05)
                med = _weighted_quantile(v_full, w, 0.5)
                hi = _weighted_quantile(v_full, w, 0.95)
                ax_left.errorbar(
                    it,
                    med,
                    yerr=[[med - lo], [hi - med]],
                    fmt="o",
                    color="black",
                    markersize=4,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    elinewidth=1.2,
                    capsize=4,
                    zorder=5,
                )

            ax_left.set_ylabel(label)
            ax_left.xaxis.set_major_locator(mticker.MultipleLocator(1))
            if is_last:
                ax_left.set_xlabel("Iteration")
            else:
                ax_left.tick_params(labelbottom=False)

            ax_right = axs[row_idx, 1]
            ax_right.hist(
                vals_now,
                bins=40,
                weights=w_now,
                density=True,
                color="C0",
                alpha=0.75,
                edgecolor="none",
            )
            ax_right.set_yticks([])
            if is_last:
                ax_right.set_xlabel(label)
            else:
                ax_right.tick_params(labelbottom=False)

        # --- Parameter rows ---
        for i, name in enumerate(parameter_names):
            true_val = None
            if self.injection_parameters:
                true_val = self.injection_parameters.get(name)

            vals_iter = [(it, x_scatter[:, i], x_full[:, i], w) for it, x_scatter, x_full, w in per_iter]
            _plot_row(
                i,
                vals_iter,
                x_now[:, i],
                self._label_for(name),
                is_last=False,
                true_val=true_val,
            )

        # --- Log-likelihood and log-prior rows ---
        for extra_idx, (attr, label) in enumerate([("log_likelihood", "log likelihood"), ("log_prior", "log prior")]):
            row = len(parameter_names) + extra_idx
            is_last = extra_idx == 1

            vals_iter = []
            for it, _x_scatter, _x_full, w in per_iter:
                smc_samples = history.sample_history[it]
                raw = getattr(smc_samples, attr, None)
                if raw is None:
                    continue
                arr_full = np.asarray(raw)
                if len(arr_full) == 0:
                    continue
                n_pts = len(arr_full)
                if n_pts > max_per_iter:
                    if w is not None:
                        idx = rng.choice(n_pts, size=max_per_iter, replace=True, p=w)
                    else:
                        idx = rng.choice(n_pts, size=max_per_iter, replace=False)
                    arr_scatter = arr_full[idx]
                else:
                    arr_scatter = arr_full
                vals_iter.append((it, arr_scatter, arr_full, w))

            raw_now = getattr(live_samples, attr, None)
            if raw_now is None:
                now_arr = np.zeros(len(x_now))
            else:
                now_arr = np.asarray(raw_now)

            _plot_row(row, vals_iter, now_arr, label, is_last=is_last)

        fig.suptitle("SMC parameter evolution and current marginals")
        # Reserve headroom for the suptitle; tight_layout otherwise packs the
        # top row up against it and the title overlaps the first axis.
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        safe_save_figure(
            fig=fig,
            filename=f"{self.outdir}/{self.label}_diagnostic_smc_evolution.png",
            dpi=150,
        )
        plt.close(fig)

    # Smallest log-likelihood drop at 1 sigma that _validate_covariance will
    # treat as measured rather than as numerical noise. Also caps that method's
    # inflation at 0.5 / 0.01 = 50 in variance, ~7x in sigma.
    _MIN_VALIDATION_DROP = 0.01

    # Bounded widening applied to a direction whose probe could not resolve a
    # drop -- 4x in variance, 2x in sigma. Used for a flat likelihood and for a
    # sub-threshold drop alike, because the two are indistinguishable.
    _UNRESOLVED_INFLATION = 4.0

    def _validate_covariance(self, estimator, mean, cov):
        """Validate the covariance by checking likelihood along each principal axis.

        At 1-sigma from the MAP along each eigenvector the log-likelihood should
        drop by 0.5 for a Gaussian. A smaller drop means the posterior is wider
        than the Gaussian predicts, and that eigenvalue is inflated to match.
        Directions are never shrunk.

        Two things keep that from running away.

        It works in *prior-scaled* coordinates, as the rest of the estimator
        does -- the unit-cube path, ``_floor_precision_at_prior``, and the
        preconditioned inversion in ``calculate_posterior_covariance`` all
        non-dimensionalise before touching eigenvalues, and this was the one
        step that did not. In those units a scaled eigenvalue of 1 is "as wide
        as the prior".

        And no direction is inflated past the prior. A posterior cannot be
        wider than its prior, so a scaled eigenvalue above 1 is not a wide
        posterior but a failed probe: on a Fisher with genuinely rank-deficient
        (near-zero) eigenvalues, those directions are already at prior width by
        the time this runs, the probe then finds the likelihood flat along
        them (true but uninformative), and without the cap it would widen them
        past the prior on that alone.
        """
        prior_sd = np.asarray(estimator._prior_standard_deviations(), dtype=float)
        outer_sd = np.outer(prior_sd, prior_sd)
        scaled = 0.5 * (cov / outer_sd + (cov / outer_sd).T)
        eigvals, eigvecs = np.linalg.eigh(scaled)
        logl_peak = float(estimator.log_likelihood_from_array(mean))

        any_inflated = False
        for i in range(len(eigvals)):
            sigma_i = np.sqrt(max(eigvals[i], 1e-30))
            # Unit vector in scaled space, expressed as a parameter-space step.
            direction = prior_sd * eigvecs[:, i]

            # Evaluate at +/- 1 sigma
            logl_plus = float(estimator.log_likelihood_from_array(mean + sigma_i * direction))
            logl_minus = float(estimator.log_likelihood_from_array(mean - sigma_i * direction))

            # Collect finite drops only (skip out-of-bounds)
            drops = []
            if np.isfinite(logl_plus):
                drops.append(logl_peak - logl_plus)
            if np.isfinite(logl_minus):
                drops.append(logl_peak - logl_minus)
            if not drops:
                continue

            # Use the smaller drop (the wider side)
            actual_drop = min(drops)
            expected_drop = 0.5

            # A drop too small to resolve is not evidence of a wide posterior.
            # ``expected / actual`` diverges as the measured drop goes to zero,
            # and the drop goes to zero for two very different reasons: the
            # posterior really is much wider than the Gaussian, or the 1-sigma
            # step is far below the scale on which the likelihood varies
            # smoothly, so the probe is reading its numerical noise floor.
            #
            # The second is what happens on a sharply-measured parameter, whose
            # 1-sigma step is small enough that the measured drop is dominated
            # by floating-point noise rather than genuine curvature. Without a
            # floor, several such directions can all get driven to the same
            # noise-floor drop and be "widened" by an enormous, meaningless
            # factor; truncation to the prior then hides it, leaving a
            # "Laplace proposal" that is the prior in those coordinates.
            #
            # Below the threshold the direction is treated as unresolved and
            # given the same bounded widening as a flat one, rather than an
            # unbounded correction inferred from noise. Genuinely unconstrained
            # directions are not lost by this: they are already bounded at the
            # prior by ``_floor_precision_at_prior``. The threshold also caps
            # the resolved branch, since expected/actual <= 0.5/0.01 = 50 in
            # variance, i.e. ~7x in sigma.
            if actual_drop < self._MIN_VALIDATION_DROP:
                # No evidence, so no change. A drop this small -- often
                # negative, i.e. the likelihood rising into numerical noise --
                # says the probe could not resolve the direction, not that the
                # posterior is wide along it. Widening anyway, even by a
                # "safe" bounded factor, is unsound: a near-null direction can
                # already sit close to prior width, so even a modest widening
                # factor pushes it (and any parameter with a component along
                # it) out to the prior on noise alone.
                #
                # Directions the data genuinely does not constrain are not
                # lost: they already carry large variance from the Fisher, and
                # `_floor_precision_at_prior` bounds them at the prior.
                logger.debug(
                    f"Leaving axis {i} unchanged: log-likelihood drop at 1 "
                    f"sigma ({actual_drop:.2e}) is below the "
                    f"{self._MIN_VALIDATION_DROP:g} needed to distinguish a "
                    f"wide posterior from an unresolved probe."
                )
            elif actual_drop < expected_drop * 0.5:
                inflation = expected_drop / actual_drop
                eigvals[i] = min(eigvals[i] * inflation, 1.0)
                any_inflated = True
                logger.info(
                    f"Widening proposal along axis {i}: "
                    f"posterior is {inflation:.1f}x wider "
                    f"than Gaussian approximation"
                )

        if any_inflated:
            scaled = eigvecs @ np.diag(eigvals) @ eigvecs.T
            cov = 0.5 * (scaled + scaled.T) * outer_sd

        # Guarantee strict positive definiteness for scipy's _PSD check.
        # scipy.stats.multivariate_normal (allow_singular=False) requires all
        # eigenvalues to exceed eps = 1e6 * machine_eps * max_eigval ≈ 2.22e-10 * max.
        # Use 1e-9 * max as the floor to stay comfortably above that threshold.
        #
        # Applied in prior-scaled coordinates. A ridge proportional to the
        # identity in *raw* units is set by the largest-variance parameter and
        # would swamp the smallest, added regardless of whether any direction
        # had actually been inflated. Scaling first gives each parameter a
        # jitter proportional to its own prior width, which is the same
        # convention the rest of the estimator uses.
        scaled_out = 0.5 * (cov / outer_sd + (cov / outer_sd).T)
        eigvals_out = np.linalg.eigvalsh(scaled_out)
        min_floor = max(1e-9 * eigvals_out.max(), 1e-30)
        if eigvals_out.min() < min_floor:
            scaled_out = scaled_out + (min_floor - eigvals_out.min()) * np.eye(len(cov))
            cov = scaled_out * outer_sd

        return cov

    def _latin_hypercube_prior(self, parameter_names, n_samples):
        """Draw *n_samples* from the prior using Latin hypercube
        sampling for even coverage.

        Generates a stratified grid in [0,1]^D, shuffles each
        column independently, then maps through each prior's
        inverse CDF (``rescale``).

        ``rescale`` is per-parameter, so the cloud fills the prior *box*: on a
        constrained prior part of it lands where the prior density is zero.
        Those points are not dropped here -- the caller may still overwrite
        some coordinates (``mode_search_subspace`` pins everything outside the
        subspace at the primary MAP), so the only configuration worth testing
        against the constraints is the one the caller ends up with.  The caller
        filters.
        """
        ndim = len(parameter_names)

        # Stratified uniform in each dimension
        intervals = np.arange(n_samples, dtype=float)
        lhs_unit = np.column_stack([(intervals + random.rng.uniform(size=n_samples)) / n_samples for _ in range(ndim)])
        # Shuffle each column independently
        for j in range(ndim):
            random.rng.shuffle(lhs_unit[:, j])

        # Map [0,1] -> prior support via inverse CDF
        samples = np.empty_like(lhs_unit)
        for j, key in enumerate(parameter_names):
            samples[:, j] = self.priors[key].rescale(lhs_unit[:, j])

        return samples

    def _drop_constraint_violations(self, x, parameter_names, context):
        """Drop rows of an ``(n, ndim)`` cloud that violate a Constraint prior.

        Returns *x* unchanged when the prior carries no constraints.
        """
        if not self.priors.constraint_keys or len(x) == 0:
            return x
        keep = np.asarray(
            self.priors.evaluate_constraints({key: x[:, j] for j, key in enumerate(parameter_names)}),
            dtype=bool,
        )
        if keep.all():
            return x
        logger.info(f"{context}: dropped {int((~keep).sum())}/{len(x)} candidates violating a Constraint prior")
        return x[keep]

    def _find_multiple_maps(self, estimator, n_modes, cov_scaling, primary_mean, primary_cov):
        """Find up to *n_modes* distinct MAP estimates and their covariances.

        The primary mode is *given*: ``primary_mean`` and ``primary_cov`` are the
        MAP and covariance ``run_sampler`` already computed, validated and
        scaled.  Only the search for secondary modes happens here -- multi-start
        local optimisation from prior draws, each candidate polished,
        deduplicated by ``mode_separation_sigma``, and covariance-validated.

        Taking the primary rather than recomputing it is what makes
        ``n_modes > 1`` a superset of ``n_modes = 1`` instead of a different
        analysis: recomputing it here would ignore ``use_injection_for_map``
        and skip the polish that secondary candidates get, degrading the seed
        that later resampling (e.g. SMC's tempering schedule) relies on, as
        well as costing a redundant global optimisation and Hessian.

        Returns a list of ``(mean, cov, log_posterior)`` triples sorted by
        descending log-posterior.
        """
        parameter_names = estimator.parameter_names
        logger.info(f"Searching for up to {n_modes} posterior " f"mode(s)")

        # --- 1. Primary mode: the one run_sampler already found ---
        best_mean = np.asarray(primary_mean, dtype=float)
        cov = np.asarray(primary_cov, dtype=float)
        best_logp = float(estimator.log_posterior_from_array(best_mean))
        std_scale = np.sqrt(np.diag(cov))
        logger.info(f"Primary mode taken from the MAP search: " f"log-posterior = {best_logp:.2f}")

        found_modes = [(best_mean, cov, best_logp)]

        if n_modes <= 1:
            self._log_mode_summary(found_modes, parameter_names)
            return found_modes

        # --- 2. Multi-start search for secondary modes ---
        n_starts = self.kwargs["mode_search_nsamples"]
        subspace = self.kwargs.get("mode_search_subspace")
        separation = float(self.kwargs["mode_separation_sigma"])
        if not np.isfinite(separation) or separation <= 0:
            raise SamplerError(f"mode_separation_sigma must be finite and positive, got {separation!r}.")

        # Latin hypercube in [0,1]^D, then map to prior
        prior_x = self._latin_hypercube_prior(parameter_names, n_starts)

        if subspace:
            unknown = [name for name in subspace if name not in parameter_names]
            if unknown:
                raise SamplerError(
                    f"mode_search_subspace names parameter(s) not being sampled: {unknown}. "
                    f"Sampled parameters are {list(parameter_names)}."
                )
            # Vary only the named coordinates and pin the rest at the primary
            # MAP.  Searching the full space is what makes a narrow secondary
            # mode undiscoverable: a sky mode a few 0.01 rad across is ~1e-4 of
            # the sky, and a Latin hypercube over all D dimensions will not land
            # in it.  Restricted to the coordinates the degeneracy actually
            # lives in, the same budget covers the subspace densely.  The polish
            # below still runs in the full space, so the pinned coordinates are
            # free to move once a candidate is found.
            free = [parameter_names.index(name) for name in subspace]
            pinned = np.tile(best_mean, (len(prior_x), 1))
            pinned[:, free] = prior_x[:, free]
            prior_x = pinned
            logger.info(
                f"Evaluating {n_starts} Latin hypercube samples over "
                f"{list(subspace)} (other parameters pinned at the primary MAP) "
                f"to search for secondary modes"
            )
        else:
            logger.info(f"Evaluating {n_starts} prior samples " f"(Latin hypercube) to search for " f"secondary modes")

        # After any subspace pinning, so the configuration tested is the one
        # actually used as a starting point.  A start in a constraint-forbidden
        # region is a wasted polish: the log-posterior there is a flat -inf,
        # giving the local optimiser nothing to descend.
        prior_x = self._drop_constraint_violations(prior_x, parameter_names, "Mode search")
        if len(prior_x) == 0:
            logger.warning("Mode search: every Latin hypercube start violates a Constraint prior; no secondary modes")
            self._log_mode_summary(found_modes, parameter_names)
            return found_modes

        prior_logp = np.array([float(estimator.log_posterior_from_array(x)) for x in prior_x])

        # Sort descending by posterior
        order = np.argsort(prior_logp)[::-1]

        # Polish top candidates far from existing modes
        max_polish = 10 * n_modes
        n_polished = 0

        for idx in order:
            if len(found_modes) >= n_modes:
                break
            if n_polished >= max_polish:
                break

            x = prior_x[idx]

            # Skip if near an existing mode
            near_existing = any(np.max(np.abs(x - m) / std_scale) < separation for m, _, _ in found_modes)
            if near_existing:
                continue

            # Polish with local optimizer
            n_polished += 1
            sample_dict = dict(zip(parameter_names, x))
            polished = estimator._maximize_posterior_from_initial_sample(sample_dict)
            p_mean = np.array(polished.x)
            p_logp = -polished.fun
            logger.debug(f"Candidate {n_polished}: " f"log-posterior = {p_logp:.2f} " f"after local optimisation")

            # Re-check after polishing
            is_dup = any(np.max(np.abs(p_mean - m) / std_scale) < separation for m, _, _ in found_modes)
            if is_dup:
                logger.debug(f"Candidate {n_polished} converged to " f"a known mode; skipping")
                continue

            try:
                p_dict = dict(zip(parameter_names, p_mean))
                p_covariance = estimator.calculate_posterior_covariance(p_dict)
                # Validate first, then scale last (see run_sampler for rationale).
                p_cov = self._validate_covariance(estimator, p_mean, p_covariance)
                p_cov = self._apply_cov_scaling(p_cov, cov_scaling)
                found_modes.append((p_mean, p_cov, p_logp))
                logger.info(f"Secondary mode {len(found_modes) - 1} " f"found: log-posterior = {p_logp:.2f}")
            except Exception as exc:
                logger.warning(f"Could not compute covariance for " f"candidate {n_polished}: {exc}")

        # --- 2b. Modes implied by an exact symmetry ---
        found_modes = self._add_symmetric_modes(estimator, found_modes, std_scale, separation)

        # --- 3. Sort and summarise ---
        found_modes.sort(key=lambda r: r[2], reverse=True)
        # The caller logs the summary once the mixture weights are known.
        return found_modes

    # A symmetry-implied mode is seeded only when its log-posterior matches the
    # one it mirrors to within this many nats.  An exact symmetry reproduces it
    # to ~1e-9, so the tolerance is not there to be generous: it is what catches
    # a symmetry declared for a problem that does not actually have it, which is
    # then skipped rather than seeding the mixture with a fictitious component.
    _SYMMETRY_LOGP_TOL = 0.5

    def _add_symmetric_modes(self, estimator, found_modes, std_scale, separation):
        """Seed the modes implied by an exact symmetry, rather than hunting them.

        Some posteriors are exactly periodic in one coordinate: on the
        precessing BBH sampled in ``delta_phase`` the log-posterior satisfies
        ``lnP(x + pi) = lnP(x)`` to machine precision, giving two lobes of
        identical mass.  Leaving those to the random multi-start search is a bad
        bet.  The narrower the coordinate, the smaller the target -- and the
        search competes for a fixed ``n_modes`` budget against near-duplicates
        of the mode it already has, which are rejected only when *every*
        coordinate falls inside ``mode_separation_sigma``.  On that example the
        search returned three candidates all at the same ``delta_phase``, the
        mirror was never proposed, and the SMC sampled one lobe of two -- worth
        ~ln(2) in the evidence and a phase posterior at a quarter of its width.

        A declared symmetry is written down instead.  The shift is a pure
        translation in one coordinate, so the local curvature is carried over
        unchanged and the mirrored mode reuses the covariance rather than
        paying for another Hessian.

        The log-posterior is still evaluated at the mirrored point and compared
        with its source: the symmetry is verified, never assumed.
        """
        symmetries = self.kwargs.get("mode_symmetries") or []
        if not symmetries:
            return found_modes

        names = list(estimator.parameter_names)
        lows = np.asarray(estimator.prior_bounds_min, dtype=float)
        highs = np.asarray(estimator.prior_bounds_max, dtype=float)
        out = list(found_modes)
        for param, shift in symmetries:
            if param not in names:
                logger.warning(f"mode_symmetries names {param!r}, which is not a sampled parameter; ignoring it.")
                continue
            index = names.index(param)
            low, period = lows[index], highs[index] - lows[index]
            for mean, cov, logp in list(out):
                mirrored = np.array(mean, dtype=float)
                mirrored[index] = low + np.mod(mirrored[index] + float(shift) - low, period)
                mirrored_logp = float(estimator.log_posterior_from_array(mirrored))
                offset = abs(mirrored_logp - logp)
                if not np.isfinite(mirrored_logp) or offset > self._SYMMETRY_LOGP_TOL:
                    logger.info(
                        f"Symmetry {param} + {float(shift):.4f} does not hold at "
                        f"{mean[index]:.4f} (log-posterior differs by {offset:.2f}); not seeding it."
                    )
                    continue
                if any(np.max(np.abs(mirrored - m) / std_scale) < separation for m, _, _ in out):
                    continue
                out.append((mirrored, cov, mirrored_logp))
                logger.info(
                    f"Symmetry mode seeded: {param} {mean[index]:.4f} -> {mirrored[index]:.4f}, "
                    f"log-posterior = {mirrored_logp:.2f} (matches its mirror to {offset:.2e})"
                )
        return out

    # Mixture components holding less than this share of the total Laplace mass
    # are dropped rather than seeded.  Keeping them is worse than useless: such a
    # mode contributes a particle or two to a 10k cloud -- enough to appear as a
    # stray point in the diagnostics, far too few to be sampled -- and in
    # practice it is a likelihood sidelobe the mode search turned up and the
    # weighting correctly rejected (on the HLV example, one at exactly the
    # primary azimuth plus pi).
    _MIN_MODE_WEIGHT = 1e-3

    def _drop_negligible_modes(self, found_modes, log_weights):
        """Discard mixture components below ``_MIN_MODE_WEIGHT``.

        Returns the surviving ``(mean, cov, logp)`` triples and their log
        weights.  The highest-weight mode is always kept, so this can never
        empty the mixture.
        """
        weights = np.exp(log_weights - logsumexp(log_weights))
        keep = weights >= self._MIN_MODE_WEIGHT
        keep[int(np.argmax(weights))] = True
        if keep.all():
            return found_modes, log_weights

        dropped = [
            f"{np.array2string(np.asarray(mean), precision=4)} (weight {weight:.2e})"
            for (mean, _cov, _logp), weight, kept in zip(found_modes, weights, keep)
            if not kept
        ]
        logger.info(
            f"Dropping {len(dropped)} mixture component(s) below the "
            f"{self._MIN_MODE_WEIGHT:g} weight threshold: " + "; ".join(dropped)
        )
        return [m for m, kept in zip(found_modes, keep) if kept], log_weights[keep]

    @staticmethod
    def _laplace_mode_log_weights(found_modes, ndim):
        """Unnormalised log Laplace evidence of each mode.

        The mass a mode carries is not its peak height: a broad shallow mode can
        hold more posterior mass than a narrow tall one.  The Laplace estimate of
        a mode's local evidence is

            log Z_i = log L_i + log pi_i + (d/2) log(2 pi) + (1/2) log det Sigma_i

        i.e. the log-posterior at the mode plus the log-volume of its covariance.
        Weighting by log-posterior alone would drop the volume term and
        systematically over-weight the narrowest mode.  The ``(d/2) log(2 pi)``
        term is common to every mode and cancels on normalisation; it is kept so
        the values are readable as evidences.

        A mode whose covariance is not positive definite falls back to its
        log-posterior, which at worst reproduces the naive weighting for that
        one component.
        """
        log_weights = []
        for mean, cov, logp in found_modes:
            sign, log_det = np.linalg.slogdet(np.asarray(cov, dtype=float))
            if sign <= 0 or not np.isfinite(log_det):
                logger.warning(
                    "Mode at "
                    + np.array2string(np.asarray(mean), precision=4)
                    + " has a non-positive-definite covariance; weighting it by log-posterior alone."
                )
                log_weights.append(float(logp))
            else:
                log_weights.append(float(logp) + 0.5 * ndim * np.log(2.0 * np.pi) + 0.5 * float(log_det))
        return np.asarray(log_weights, dtype=float)

    @staticmethod
    def _log_mode_summary(found_modes, parameter_names, weights=None):
        """Log a table summarising the discovered modes."""
        header = f"{'Mode':<6} {'log-posterior':>14} {'weight':>8}  " + "  ".join(f"{p:>12}" for p in parameter_names)
        rows = []
        for i, (mean, _cov, logp) in enumerate(found_modes):
            vals = "  ".join(f"{v:>+12.4f}" for v in mean)
            weight = f"{weights[i]:>8.4f}" if weights is not None else f"{'-':>8}"
            rows.append(f"  {i:<4d} {logp:>14.2f} {weight}  {vals}")
        logger.info(f"Summary of {len(found_modes)} mode(s):\n" f"  {header}\n" + "\n".join(rows))

    def create_proposal_diagnostic(self, mean, cov, parameter_names, init_samples=None):
        """Corner plot comparing the Gaussian proposal, initialization samples, and injection parameters.

        Shows:
        - Initialization samples (actual samples drawn from proposal with prior replacement applied)
          as histograms on diagonal and scatter on off-diagonal
        - Gaussian proposal as 1-D curves on diagonal and 2-D contours on off-diagonal
        - MAP estimate and injection parameters (if available) as markers

        Parameters
        ----------
        mean : array
            MAP estimate (mean of proposal)
        cov : array
            Covariance matrix of proposal
        parameter_names : list
            Names of parameters
        init_samples : array, optional
            Initial samples drawn from proposal (shape: N_samples x N_params).
            If None, prior samples are shown instead.
        """
        import corner
        import matplotlib.lines as mpllines
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        labels = self._labels_for(parameter_names)
        ndim = len(parameter_names)

        proposal_sigmas = np.sqrt(np.diag(cov))

        # Use initialization samples if provided, otherwise use prior samples
        if init_samples is not None:
            display_samples = init_samples
            sample_color, sample_ls = "C0", "-"
            sample_label = "Initial samples"
        else:
            n_samples = 10000
            display_samples = np.column_stack([self.priors[k].sample(n_samples) for k in parameter_names])
            sample_color, sample_ls = "C0", "-"
            sample_label = "Prior samples"

        ranges = [(self.priors[k].minimum, self.priors[k].maximum) for k in parameter_names]

        g_color, g_ls = "k", "--"

        panel_size = 2.5
        fig_size = panel_size * ndim
        fig = corner.corner(
            display_samples,
            color=sample_color,
            contour_kwargs={"linestyles": sample_ls, "alpha": 0.8},
            hist_kwargs={"density": True, "ls": sample_ls, "alpha": 0.8},
            no_fill_contours=True,
            plot_density=False,
            plot_datapoints=True,
            plot_contours=False,
            fill_contours=True,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
            bins=50,
            smooth=0.7,
            max_n_ticks=5,
            labels=labels,
            truths=mean,
            truth_color="C1",
            fig=plt.figure(figsize=(fig_size, fig_size)),
            range=ranges,
        )

        axes_grid = np.array(fig.get_axes()).reshape(ndim, ndim)

        # 1-D analytic Gaussian marginals on diagonal panels
        for i in range(ndim):
            ax = axes_grid[i, i]
            lo, hi = ranges[i]
            xs = np.linspace(lo, hi, 300)
            ys = norm.pdf(xs, loc=mean[i], scale=proposal_sigmas[i])
            ax.plot(xs, ys, color=g_color, lw=1.5, ls=g_ls)
            # Extend y-axis to accommodate full Gaussian peak
            y_max = ax.get_ylim()[1]
            ax.set_ylim(top=max(y_max, ys.max() * 1.1))

        # 2-D samples on off-diagonal panels with low alpha,
        # overlaid with Gaussian proposal contours.
        for row in range(ndim):
            for col in range(row):
                ax = axes_grid[row, col]
                ax.scatter(
                    display_samples[:, col],
                    display_samples[:, row],
                    s=1,
                    alpha=0.01,
                    color=sample_color,
                )

        # 2-D analytic Gaussian contours on off-diagonal panels.
        # Contour levels exp(-0.5) and exp(-2) relative to the peak match the
        # 1-sigma and 2-sigma enclosed-mass fractions for a 2-D Gaussian.
        for row in range(ndim):
            for col in range(row):
                ax = axes_grid[row, col]
                x_lo, x_hi = ranges[col]
                y_lo, y_hi = ranges[row]
                xs = np.linspace(x_lo, x_hi, 60)
                ys = np.linspace(y_lo, y_hi, 60)
                X, Y = np.meshgrid(xs, ys)
                pos = np.dstack([X, Y])
                sub_mean = mean[[col, row]]
                sub_cov = cov[np.ix_([col, row], [col, row])]
                try:
                    Z = multivariate_normal(mean=sub_mean, cov=sub_cov).pdf(pos)
                    Z_max = Z.max()
                    if Z_max > 0:
                        ax.contour(
                            X,
                            Y,
                            Z,
                            levels=[Z_max * np.exp(-2), Z_max * np.exp(-0.5)],
                            colors=g_color,
                            linestyles=[g_ls],
                            alpha=0.8,
                        )
                except Exception:
                    pass

        legend_handles = [
            mpllines.Line2D([0], [0], color=sample_color, linestyle=sample_ls),
            mpllines.Line2D([0], [0], color=g_color, linestyle=g_ls),
            mpllines.Line2D(
                [0],
                [0],
                color="C1",
                linestyle=":",
                marker="s",
                markersize=8,
                linewidth=1.5,
            ),
        ]
        legend_labels = [sample_label, "Gaussian proposal", "MAP"]

        # Add injection parameters if available
        if self.injection_parameters:
            legend_handles.append(
                mpllines.Line2D(
                    [0],
                    [0],
                    color="C2",
                    linestyle=":",
                    marker="x",
                    markersize=8,
                    linewidth=1.5,
                )
            )
            legend_labels.append("Injection")

            # Overlay injection parameters on axes
            injection_array = np.array([self.injection_parameters.get(k, np.nan) for k in parameter_names])
            for i in range(ndim):
                ax = axes_grid[i, i]
                if np.isfinite(injection_array[i]):
                    ax.axvline(injection_array[i], color="C2", lw=1.5, ls=":")

            for row in range(ndim):
                for col in range(row):
                    ax = axes_grid[row, col]
                    if np.isfinite(injection_array[col]) and np.isfinite(injection_array[row]):
                        ax.scatter(
                            [injection_array[col]],
                            [injection_array[row]],
                            color="C2",
                            marker="x",
                            s=80,
                            zorder=5,
                            linewidths=1.5,
                        )

        axes_grid[0, 0].legend(
            legend_handles,
            legend_labels,
            fontsize="small",
        )
        fig.suptitle("Gaussian proposal: initial samples, injection, and MAP")

        filename = f"{self.outdir}/{self.label}_diagnostic_proposal.png"
        safe_save_figure(fig=fig, filename=filename, dpi=150)
        plt.close(fig)
        return fig

    def create_rejection_progress_diagnostic(self, mean, cov, parameter_names, accepted, rejected=None):
        """Live corner-style diagnostic for the rejection-sampling loop.

        Rendered periodically while rejection sampling runs (see
        ``_maybe_periodic_rejection_diagnostic``).  Rejected proposal draws are
        shown as a faint background cloud, with the accepted samples drawn on
        top (higher ``zorder`` and opacity) so their distribution stands out
        clearly against the rejected draws.  The Gaussian proposal, MAP, and
        injection (if available) are overlaid for reference.

        Parameters
        ----------
        mean : array
            MAP estimate (mean of the proposal).
        cov : array
            Covariance matrix of the proposal.
        parameter_names : list
            Names of the parameters.
        accepted : pandas.DataFrame
            Accepted samples accumulated so far.
        rejected : pandas.DataFrame, optional
            Rejected proposal draws accumulated so far.
        """
        import matplotlib.lines as mpllines
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        labels = self._labels_for(parameter_names)
        ndim = len(parameter_names)
        sigmas = np.sqrt(np.diag(cov))
        ranges = [(self.priors[k].minimum, self.priors[k].maximum) for k in parameter_names]

        acc = accepted[list(parameter_names)].values
        rej = rejected[list(parameter_names)].values if rejected is not None and len(rejected) else None

        rej_color, acc_color, g_color, g_ls = "0.6", "C0", "k", "--"

        panel_size = 2.5
        fig, axes = plt.subplots(
            ndim,
            ndim,
            figsize=(panel_size * ndim, panel_size * ndim),
            squeeze=False,
        )

        injection_array = None
        if self.injection_parameters:
            injection_array = np.array([self.injection_parameters.get(k, np.nan) for k in parameter_names])

        for row in range(ndim):
            for col in range(ndim):
                ax = axes[row, col]
                if col > row:
                    ax.set_visible(False)
                    continue

                if row == col:
                    lo, hi = ranges[row]
                    bins = np.linspace(lo, hi, 50)
                    if rej is not None:
                        ax.hist(rej[:, row], bins=bins, density=True, color=rej_color, alpha=0.4, zorder=1)
                    ax.hist(acc[:, row], bins=bins, density=True, histtype="step", color=acc_color, lw=1.5, zorder=3)
                    xs = np.linspace(lo, hi, 300)
                    ys = norm.pdf(xs, loc=mean[row], scale=sigmas[row])
                    ax.plot(xs, ys, color=g_color, lw=1.5, ls=g_ls, zorder=4)
                    ax.axvline(mean[row], color="C1", lw=1.2, ls=":", zorder=5)
                    if injection_array is not None and np.isfinite(injection_array[row]):
                        ax.axvline(injection_array[row], color="C2", lw=1.2, ls=":", zorder=5)
                    ax.set_xlim(lo, hi)
                    ax.set_yticklabels([])
                else:
                    # Rejected draws first (background), accepted on top.
                    if rej is not None:
                        ax.scatter(
                            rej[:, col],
                            rej[:, row],
                            s=2,
                            alpha=0.05,
                            color=rej_color,
                            zorder=1,
                            rasterized=True,
                        )
                    ax.scatter(
                        acc[:, col],
                        acc[:, row],
                        s=3,
                        alpha=0.5,
                        color=acc_color,
                        zorder=3,
                        rasterized=True,
                    )

                    # Gaussian proposal 1-sigma / 2-sigma contours.
                    x_lo, x_hi = ranges[col]
                    y_lo, y_hi = ranges[row]
                    X, Y = np.meshgrid(np.linspace(x_lo, x_hi, 60), np.linspace(y_lo, y_hi, 60))
                    sub_mean = mean[[col, row]]
                    sub_cov = cov[np.ix_([col, row], [col, row])]
                    try:
                        Z = multivariate_normal(mean=sub_mean, cov=sub_cov).pdf(np.dstack([X, Y]))
                        if Z.max() > 0:
                            ax.contour(
                                X,
                                Y,
                                Z,
                                levels=[Z.max() * np.exp(-2), Z.max() * np.exp(-0.5)],
                                colors=g_color,
                                linestyles=[g_ls],
                                alpha=0.8,
                                zorder=4,
                            )
                    except Exception:
                        pass
                    ax.scatter([mean[col]], [mean[row]], color="C1", marker="s", s=40, zorder=5)
                    if injection_array is not None and np.isfinite(injection_array[[col, row]]).all():
                        ax.scatter(
                            [injection_array[col]],
                            [injection_array[row]],
                            color="C2",
                            marker="x",
                            s=60,
                            lw=1.5,
                            zorder=6,
                        )
                    ax.set_xlim(x_lo, x_hi)
                    ax.set_ylim(y_lo, y_hi)

                if row == ndim - 1:
                    ax.set_xlabel(labels[col])
                else:
                    ax.set_xticklabels([])
                if col == 0 and row != 0:
                    ax.set_ylabel(labels[row])
                elif col != 0:
                    ax.set_yticklabels([])

        legend_handles = [
            mpllines.Line2D([0], [0], color=acc_color, lw=1.5),
            mpllines.Line2D([0], [0], marker="o", color=rej_color, lw=0, markersize=6),
            mpllines.Line2D([0], [0], color=g_color, ls=g_ls),
            mpllines.Line2D([0], [0], color="C1", ls=":", marker="s", markersize=8, lw=1.2),
        ]
        legend_labels = ["Accepted", "Rejected", "Gaussian proposal", "MAP"]
        if injection_array is not None:
            legend_handles.append(mpllines.Line2D([0], [0], color="C2", ls=":", marker="x", markersize=8, lw=1.2))
            legend_labels.append("Injection")
        axes[0, 0].legend(legend_handles, legend_labels, fontsize="small")

        fig.suptitle(f"Rejection sampling progress: {len(accepted)} accepted")

        filename = f"{self.outdir}/{self.label}_diagnostic_rejection_progress.png"
        safe_save_figure(fig=fig, filename=filename, dpi=150)
        plt.close(fig)
        return fig

    def create_resample_diagnostic(self, samples, raw_samples, mean, weights, method):
        """Produce a corner plot comparing the proposal and resampled posteriors."""
        import corner
        import matplotlib.lines as mpllines
        import matplotlib.pyplot as plt

        labels = self._labels_for(self.search_parameter_keys)
        labels.append("weights")

        corner_kwargs = dict(
            bins=50,
            smooth=0.7,
            max_n_ticks=5,
            truths=np.concatenate((mean, [1])),
            truth_color="C3",
            labels=labels,
        )

        xs = samples[self.search_parameter_keys].values
        xs = np.concatenate((xs, np.random.uniform(0, 1, len(xs)).reshape(-1, 1)), axis=1)
        rxs = raw_samples[self.search_parameter_keys].values
        rxs = np.concatenate((rxs, weights.reshape(-1, 1)), axis=1)

        # Sort by weight for cleaner scatter colouring
        idxs = np.argsort(weights)
        rxs = rxs[idxs]

        g_color, g_ls = "k", "--"
        f_color, f_ls = "C0", "-"

        panel_size = 2.5
        fig_size = panel_size * len(labels)
        lines = []
        fig = corner.corner(
            rxs,
            color=g_color,
            contour_kwargs={"linestyles": g_ls, "alpha": 0.8},
            hist_kwargs={"density": True, "ls": g_ls, "alpha": 0.8},
            data_kwargs={"alpha": 1},
            no_fill_contours=True,
            alpha=0.8,
            plot_density=False,
            plot_datapoints=False,
            fill_contours=False,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
            fig=plt.figure(figsize=(fig_size, fig_size)),
            **corner_kwargs,
        )
        lines.append(mpllines.Line2D([0], [0], color=g_color, linestyle=g_ls))

        if len(xs) > len(samples.keys()):
            fig = corner.corner(
                xs,
                color=f_color,
                contour_kwargs={"linestyles": f_ls, "alpha": 0.8},
                contourf_kwargs={"alpha": 0.8},
                hist_kwargs={"density": True, "ls": f_ls, "alpha": 0.8},
                no_fill_contours=True,
                fig=fig,
                alpha=0.1,
                plot_density=True,
                plot_datapoints=False,
                fill_contours=False,
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
                range=[1] * self.ndim + [(0, 1)],
                **corner_kwargs,
            )
            lines.append(mpllines.Line2D([0], [0], color=f_color, linestyle=f_ls))

        axes = np.array(fig.get_axes())
        labels = ["$g(x)$"] + (["$f(x)$"] if len(lines) > 1 else [])
        axes[0].legend(lines, labels)
        fig.suptitle(f"Resampling method: {method}")

        filename = f"{self.outdir}/{self.label}_diagnostic_resample_{method}.png"
        safe_save_figure(fig=fig, filename=filename, dpi=150)
        plt.close(fig)
        return fig

    def create_smc_diagnostic(self, samples, proposal_flow):
        """Produce a corner plot comparing the Laplace proposal and SMC output.

        Mode locations (MAP estimates) are overlaid as vertical lines on the
        diagonal panels and as scatter points on the off-diagonal panels.
        """
        import corner
        import matplotlib.lines as mpllines
        import matplotlib.pyplot as plt

        labels = self._labels_for(self.search_parameter_keys)
        corner_kwargs = dict(
            bins=50,
            smooth=0.7,
            max_n_ticks=5,
            labels=labels,
        )

        # Extract mode means from the flow
        if isinstance(proposal_flow, GaussianMixtureFlow):
            mode_means = [d.mean for d in proposal_flow._dists]
        else:
            mode_means = [proposal_flow._mean]

        # Draw reference samples from the proposal (Gaussian or mixture)
        n = len(samples)
        laplace_samples, _ = proposal_flow.sample_and_log_prob(n)

        g_color, g_ls = "k", "--"
        f_color, f_ls = "C0", "-"
        mode_colors = [f"C{i + 1}" for i in range(len(mode_means))]

        panel_size = 2.5
        fig_size = panel_size * self.ndim
        fig = corner.corner(
            laplace_samples,
            color=g_color,
            contour_kwargs={"linestyles": g_ls, "alpha": 0.8},
            hist_kwargs={"density": True, "ls": g_ls, "alpha": 0.8},
            no_fill_contours=True,
            plot_density=False,
            plot_datapoints=False,
            fill_contours=False,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
            fig=plt.figure(figsize=(fig_size, fig_size)),
            **corner_kwargs,
        )
        fig = corner.corner(
            samples[self.search_parameter_keys].values,
            color=f_color,
            contour_kwargs={"linestyles": f_ls, "alpha": 0.8},
            contourf_kwargs={"alpha": 0.8},
            hist_kwargs={"density": True, "ls": f_ls, "alpha": 0.8},
            no_fill_contours=True,
            fig=fig,
            plot_density=True,
            plot_datapoints=False,
            fill_contours=False,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
            range=[1] * self.ndim,
            **corner_kwargs,
        )

        # Overlay mode locations on the axes grid
        ndim = self.ndim
        axes_grid = np.array(fig.get_axes()).reshape(ndim, ndim)
        for mode_mean, mc in zip(mode_means, mode_colors):
            for row in range(ndim):
                for col in range(ndim):
                    ax = axes_grid[row, col]
                    if row == col:
                        ax.axvline(mode_mean[col], color=mc, lw=1.5, ls=":")
                    elif row > col:
                        ax.scatter(
                            [mode_mean[col]],
                            [mode_mean[row]],
                            color=mc,
                            marker="+",
                            s=80,
                            zorder=5,
                            linewidths=1.5,
                        )

        # Build legend
        legend_handles = [
            mpllines.Line2D([0], [0], color=g_color, linestyle=g_ls),
            mpllines.Line2D([0], [0], color=f_color, linestyle=f_ls),
        ]
        legend_labels = ["Initial (Laplace)", "Final (SMC)"]
        for i, mc in enumerate(mode_colors):
            legend_handles.append(
                mpllines.Line2D(
                    [0],
                    [0],
                    color=mc,
                    linestyle=":",
                    marker="+",
                    markersize=8,
                    linewidth=1.5,
                )
            )
            legend_labels.append(f"Mode {i}")

        axes_grid[0, 0].legend(legend_handles, legend_labels, fontsize="small")
        fig.suptitle("Resampling method: SMC")

        filename = f"{self.outdir}/{self.label}_diagnostic_smc_samples.png"
        safe_save_figure(fig=fig, filename=filename, dpi=150)
        plt.close(fig)

        # The stats and evolution-and-marginals figures are produced
        # incrementally by the per-iteration callback (see _make_smc_callback),
        # so the files already reflect the final state.

        return fig

    # Walker traces plotted per row are capped at this many, chosen once
    # (the first N of the exchangeable ensemble, no need for a random pick)
    # rather than all of them: a full nwalkers=1000 ensemble would draw a
    # million-plus line segments per row and make the figure both slow to
    # render and unreadable.
    _EMCEE_TRACE_MAX_WALKERS = 200

    def _plot_emcee_tau_row(self, axes_pair, tau_history, parameter_names, discard, tol):
        """Autocorrelation time against chain length (left) and its final
        per-parameter values (right); see :meth:`create_emcee_diagnostic`.

        The worst-mixing parameter is highlighted because it alone sets the
        thinning and the independent-sample count -- the chain is only as
        mixed as its slowest coordinate.
        """
        ax_left, ax_right = axes_pair
        steps = np.array([h[0] for h in tau_history], dtype=float)
        taus = np.vstack([h[1] for h in tau_history])  # (n_checks, ndim)

        final_tau = taus[-1]
        worst = int(np.nanargmax(final_tau)) if np.any(np.isfinite(final_tau)) else 0

        for i, _name in enumerate(parameter_names):
            if i == worst:
                continue
            ax_left.plot(steps, taus[:, i], color="0.75", lw=0.8, zorder=2)
        ax_left.plot(steps, taus[:, worst], color="C3", lw=1.6, zorder=3, label=self._label_for(parameter_names[worst]))
        # Reliability frontier: the estimate is only trustworthy where the
        # post-burn-in chain is `_EMCEE_AUTOCORR_RELIABLE_FACTOR` tau long,
        # i.e. where a tau curve sits *below* this line. A converging chain's
        # tau flattens and the frontier overtakes it; a drifting chain's tau
        # climbs alongside it and they never cross.
        frontier = np.maximum(steps - discard, 0.0) / tol
        ax_left.plot(steps, frontier, color="k", ls="--", lw=1.2, zorder=4, label=f"reliability bar ({tol:g}x)")
        ax_left.set_yscale("log")
        ax_left.set_ylabel("autocorr. time")
        ax_left.set_xlabel("Step")
        ax_left.legend(fontsize="x-small", loc="upper left")

        order = np.argsort(final_tau)
        ax_right.barh(
            np.arange(len(order)),
            final_tau[order],
            color=["C3" if i == worst else "0.75" for i in order],
        )
        ax_right.set_yticks(np.arange(len(order)))
        ax_right.set_yticklabels([self._label_for(parameter_names[i]) for i in order], fontsize="xx-small")
        ax_right.set_xlabel("final autocorr. time")

    def create_emcee_diagnostic(self, samples, ensemble, discard, thin, tau_history=None, autocorr_tol=None):
        """Trace of the emcee ensemble (left) and the final independent
        posterior (right), one row per parameter plus log-likelihood/log-prior.

        Loosely modelled on :meth:`_save_smc_evolution_marginals_figure`: the
        same left/right, one-row-per-parameter(+log-likelihood/log-prior)
        layout, with the SMC version's per-iteration weighted scatter and
        quantile machinery replaced by a plain per-step walker trace -- there
        are no SMC-style particle weights or annealing iterations here, just
        one un-tempered chain per walker.

        *samples* is the final (burn-in-discarded, thinned) DataFrame
        :meth:`_run_emcee` is about to return -- exactly what the right-hand
        marginals are histograms of. *ensemble* is the ``emcee.EnsembleSampler``
        that produced it, from which the left-hand trace is drawn thinned by
        the same *thin* used for ``samples`` -- both so the two panels are on
        a comparable footing, and because the raw (unthinned) trace is mostly
        redundant, highly-correlated ink that is slow to render at any
        realistic ``nwalkers``. *discard* marks the burn-in cutoff on the trace.

        *tau_history*, when given, is the ``[(step, tau_vector), ...]`` record
        :meth:`_run_emcee` accumulates across its batches, and adds a final
        row tracking the autocorrelation estimate against chain length. That
        row is the one that says whether the run is converging at all: a
        chain that is mixing has tau flatten while the reliability frontier
        (``(step - discard) / 50``, drawn dashed) climbs past it, whereas a
        drifting chain has tau grow roughly in step with the chain, so the
        two never cross and the independent-sample count stops improving.
        """
        import matplotlib.pyplot as plt

        parameter_names = list(samples.columns)
        ndim = len(parameter_names)

        # Periodic parameters are wrapped for the same reason `log_prob_batch`
        # wraps them: emcee's raw chain can wander outside a periodic
        # parameter's canonical range, and plotting that raw drift would
        # obscure the actual (wrapped) trajectory with a spurious trend.
        lower = np.array([self.priors[name].minimum for name in parameter_names])
        upper = np.array([self.priors[name].maximum for name in parameter_names])
        periodic_mask = self._periodic_mask(parameter_names)
        period = upper - lower

        def wrap(x):
            if not periodic_mask.any():
                return x
            x = np.array(x, dtype=float, copy=True)
            x[..., periodic_mask] = lower[periodic_mask] + np.mod(
                x[..., periodic_mask] - lower[periodic_mask], period[periodic_mask]
            )
            return x

        # Thin the trace itself by the same `thin` used for `samples` -- not
        # just for consistency, but because plotting every one of nsteps *
        # nwalkers raw points (correlated at the same timescale `thin`
        # corrects for) is both slow to render and mostly redundant ink.
        total_steps = ensemble.iteration
        chain = wrap(ensemble.get_chain(thin=thin))  # (n_thinned, nwalkers, ndim)
        log_prob = ensemble.get_log_prob(thin=thin)  # (n_thinned, nwalkers)
        n_thinned, nwalkers, _ = chain.shape
        # Actual step indices of the thinned points (matches emcee's own
        # `chain[thin-1::thin]` slicing), so the x-axis and the `discard`
        # cutoff below stay in real step units regardless of `thin`.
        step = np.arange(thin - 1, total_steps, thin)
        n_shown = min(nwalkers, self._EMCEE_TRACE_MAX_WALKERS)

        # Decompose the full trace's log-posterior into likelihood + prior,
        # matching SMC's two extra rows. Prior evaluation is cheap (no
        # likelihood calls), so this costs nothing next to the sampling itself.
        flat_for_prior = pd.DataFrame(chain.reshape(-1, ndim), columns=parameter_names)
        logpi_full = np.real(np.array(self.priors.ln_prob(flat_for_prior, axis=0))).reshape(n_thinned, nwalkers)
        logl_full = log_prob - logpi_full

        show_tau = bool(tau_history)
        n_rows = ndim + 2 + (1 if show_tau else 0)
        fig, axs = plt.subplots(
            n_rows,
            2,
            figsize=(11, 2.0 * n_rows),
            gridspec_kw={"width_ratios": [3, 1.5]},
            squeeze=False,
        )
        for r in range(1, n_rows):
            axs[r, 0].sharex(axs[0, 0])

        def _plot_row(row_idx, trace, final_vals, label, is_last, true_val=None):
            ax_left = axs[row_idx, 0]
            ax_left.axvspan(0, discard, color="lightgray", alpha=0.4, zorder=0)
            # Markers, not connected lines: adjacent *plotted* points are
            # `thin` real steps apart, so a line between them would draw a
            # trajectory the walker never actually took. A 2-D y-array plots
            # one marker series per column in a single call -- far cheaper
            # than looping over `n_shown` individual plot() calls.
            ax_left.plot(step, trace[:, :n_shown], color="C0", alpha=0.15, marker=".", markersize=2, linestyle="none")
            if true_val is not None:
                ax_left.axhline(true_val, color="k", ls="--", lw=1.0, zorder=2)
            ax_left.set_ylabel(label)
            if is_last:
                ax_left.set_xlabel("Step")
            else:
                ax_left.tick_params(labelbottom=False)

            ax_right = axs[row_idx, 1]
            ax_right.hist(final_vals, bins=40, density=True, color="C0", alpha=0.75, edgecolor="none")
            ax_right.set_yticks([])
            if is_last:
                ax_right.set_xlabel(label)
            else:
                ax_right.tick_params(labelbottom=False)

        for i, name in enumerate(parameter_names):
            true_val = self.injection_parameters.get(name) if self.injection_parameters else None
            _plot_row(
                i,
                chain[:, :, i],
                samples[name].values,
                self._label_for(name),
                is_last=False,
                true_val=true_val,
            )

        # Same (discard, thin) `_run_emcee` used to build `samples`, so this
        # reproduces its exact log-likelihood/log-prior rather than
        # recomputing anything from scratch.
        final_logpi = np.real(np.array(self.priors.ln_prob(samples, axis=0)))
        final_logpost = ensemble.get_log_prob(discard=discard, thin=thin, flat=True)
        final_logl = final_logpost - final_logpi

        _plot_row(ndim, logl_full, final_logl, "log likelihood", is_last=False)
        _plot_row(ndim + 1, logpi_full, final_logpi, "log prior", is_last=not show_tau)

        if show_tau:
            self._plot_emcee_tau_row(
                axs[n_rows - 1],
                tau_history,
                parameter_names,
                discard,
                self._EMCEE_AUTOCORR_RELIABLE_FACTOR if autocorr_tol is None else autocorr_tol,
            )

        fig.suptitle("emcee walker trace and final posterior")
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        safe_save_figure(
            fig=fig,
            filename=f"{self.outdir}/{self.label}_diagnostic_emcee_evolution.png",
            dpi=150,
        )
        plt.close(fig)
