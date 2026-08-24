from functools import partial

import numpy as np
import pandas as pd
import scipy.differentiate as sd
import tqdm
from bilby.core.prior import PriorDict
from bilby.core.utils import logger, random
from scipy.optimize import OptimizeResult, differential_evolution, minimize

# Convergence threshold for the global MAP search, in nats.
#
# ``scipy.optimize.differential_evolution`` stops when
#
#     std(population energies) <= atol + tol * |mean(population energies)|
#
# and its defaults are ``atol=0, tol=0.01`` -- a purely *relative* criterion.
# That is fine for an objective of order unity but wrong here: an unnormalised
# log-posterior carries the (arbitrary) noise evidence as an additive offset,
# so a relative criterion stops once the population spread is a fixed
# fraction of that offset rather than of the actual posterior structure --
# which can trigger convergence after a single generation, well short of the
# true MAP.
#
# The criterion therefore has to be absolute: stop only once the population's
# log-posterior spread itself is below a nat, independent of the offset.
DE_ATOL = 1.0


def array_to_dict(keys, array):
    return dict(zip(keys, array))


def _pool_log_likelihood(param_keys, fixed_parameters, x_col):
    """Evaluate the log-likelihood for a single parameter vector in a pool worker.

    The (heavy) likelihood is not shipped with each task: it is pulled from
    bilby's per-worker global, populated once when the pool is created via
    :meth:`bilby...Sampler._setup_pool`.  Only the small static arguments
    (parameter names, fixed values) ride along, bound in with
    ``functools.partial``.

    No prior-support handling here.  The estimator's
    ``log_likelihood_from_array`` screens the whole batch -- box *and*
    constraints -- before dispatch, so every column that arrives is already in
    support.  That is both cheaper (one vectorised test per batch instead of
    one scalar test per task) and stricter than a per-worker box test would be
    (it also covers ``Constraint`` priors, not just parameter bounds).
    """
    from bilby.core.sampler.base_sampler import _sampling_convenience_dump

    likelihood = _sampling_convenience_dump.likelihood
    x = np.asarray(x_col, dtype=float)
    parameters = {**fixed_parameters, **dict(zip(param_keys, x))}
    return likelihood.log_likelihood(parameters=parameters)


class LaplacePosteriorEstimator:
    def __init__(
        self,
        likelihood,
        priors,
        parameters=None,
        minimization_method="Nelder-Mead",
        n_prior_samples=100,
        use_unit_cube=True,
        jacobian_cap_scale=1.0,
        hessian_kwargs=None,
        fisher_method="hessian",
        fisher_kwargs=None,
        marginalized_reference=None,
        seed=None,
    ):
        """Estimate posteriors using the Laplace approximation.

        Finds the MAP (maximum a posteriori) estimate, computes the Hessian of
        the log-posterior at the MAP, and uses its inverse as a Gaussian
        approximation to the posterior covariance.

        Parameters
        ----------
        likelihood: bilby.core.likelihood.Likelihood
            A bilby likelihood object.
        priors: bilby.core.prior.PriorDict
            A bilby prior object.
        parameters: list
            Names of parameters to sample in.
        minimization_method: str
            The method to use in scipy.optimize.minimize for MAP finding.
            Default is ``'Nelder-Mead'``.
        n_prior_samples: int
            The number of prior samples to draw and use as starting points
            for the MAP search (multi-start mode).
        use_unit_cube: bool
            If True (default), compute the Hessian in unit-cube space via the
            prior CDFs. This avoids boundary clipping when the MAP is near a
            prior edge, giving unbiased curvature estimates. In this space the
            precision is floored at the prior precision (the variance of a
            uniform prior is 1/12) before inversion, so a noisy, indefinite, or
            non-finite curvature estimate can never yield a posterior broader
            than the prior; unconstrained directions default to prior width.
        jacobian_cap_scale: float
            Scales the Jacobian cap applied when transforming the unit-cube
            Hessian back to parameter space. The cap is
            ``jacobian_cap_scale / prior_width``. The default of 1.0 caps at
            the uniform-prior Jacobian. Values < 1 apply a tighter cap,
            widening the proposal for prior-dominated parameters.
        hessian_kwargs: dict, optional
            Keyword arguments forwarded to ``scipy.differentiate.hessian``.
            Defaults are ``{"initial_step": 0.5}`` in parameter space and
            ``{"initial_step": 0.001, "step_factor": 2, "maxiter": 10}`` in
            unit-cube space. Any key provided here overrides the corresponding
            default. Note that ``scipy.differentiate.hessian`` *shrinks* the
            step each iteration (``initial_step / step_factor**k``); raising
            ``maxiter`` drives the step toward zero, where a noisy objective
            (e.g. a marginalised likelihood) becomes dominated by round-off and
            the curvature estimate degrades. More iterations is not necessarily
            better.
        fisher_method: str
            How to estimate the posterior precision. ``'hessian'`` (default)
            finite-differences the scalar log-posterior. ``'waveform'`` builds
            the genuine Fisher matrix from gravitational-wave waveform
            derivatives (likelihood Fisher plus prior precision); it requires a
            ``GravitationalWaveTransient``-like likelihood and works directly in
            parameter space (``use_unit_cube`` and ``jacobian_cap_scale`` are
            ignored). Relative-binning likelihoods are supported (the Fisher is
            built on the full frequency grid, of which relative binning is an
            approximation); ROQ and multi-banded likelihoods are not. If the
            likelihood analytically marginalises over phase,
            time, and/or distance, the Fisher is built over the augmented set
            (sampled parameters plus the marginalised ones) and the marginalised
            block is removed via its Schur complement -- equivalent to
            marginalising, not conditioning, over those parameters. The result is
            floored at the prior precision (as the unit-cube path is), so no
            marginal variance exceeds the prior.
        fisher_kwargs: dict, optional
            Keyword arguments forwarded to the waveform-Fisher computation when
            ``fisher_method='waveform'`` (e.g. ``eps``, ``eps_mass``,
            ``eps_time``).
        marginalized_reference: dict, optional
            Point values for analytically-marginalised parameters (phase, time,
            distance) at which to evaluate the waveform Fisher. Typically the
            injection. Any marginalised parameter absent here is reconstructed
            from the marginalised likelihood at the MAP. Only used by
            ``fisher_method='waveform'``.
        seed: int, optional
            Seeds ``scipy.optimize.differential_evolution``'s own internal
            random stream (used for MAP finding when
            ``minimization_method='differential_evolution'`` and no
            ``initial_sample`` is given). scipy's default seeding consults
            numpy's legacy global ``RandomState``, which
            ``bilby.core.utils.random.seed()`` does not touch (it manages its
            own ``Generator`` instance, ``random.rng``), so without this
            ``differential_evolution`` is effectively unseeded even when the
            rest of a bilby run is reproducible. The sampler passes a seed
            drawn from ``random.rng`` here, so reseeding ``random.rng`` (e.g.
            via a bilby ``sampling_seed``) reproduces the MAP search too.
            Other minimisation paths are unaffected: the multi-start
            Nelder-Mead starting points are drawn via ``priors.sample_subset``,
            which already goes through ``random.rng``.
        """
        self.likelihood = likelihood

        # Optional multiprocessing pool for vectorised likelihood evaluation.
        # Set by the sampler after the pool is created; ``None`` means serial.
        self.pool = None
        self.npool = 1

        if not isinstance(priors, PriorDict):
            priors = PriorDict(priors)

        if parameters is None:
            self.parameter_names = priors.non_fixed_keys
        else:
            self.parameter_names = parameters
        self.minimization_method = minimization_method
        self.n_prior_samples = n_prior_samples
        self.seed = seed
        self.use_unit_cube = use_unit_cube
        self.jacobian_cap_scale = jacobian_cap_scale
        self.hessian_kwargs = hessian_kwargs if hessian_kwargs is not None else {}
        self.fisher_method = fisher_method
        self.fisher_kwargs = fisher_kwargs if fisher_kwargs is not None else {}
        self.marginalized_reference = marginalized_reference if marginalized_reference is not None else {}
        if fisher_method == "waveform":
            from .gw_fisher import validate_waveform_likelihood

            validate_waveform_likelihood(likelihood)
        elif fisher_method != "hessian":
            raise ValueError(f"fisher_method must be 'hessian' or 'waveform', got {fisher_method!r}.")
        self.N = len(self.parameter_names)
        # The full PriorDict, kept for `Constraint` priors.  A constraint is a
        # bound on a *derived* quantity (mass_1/mass_2 from chirp_mass and
        # mass_ratio, say), so it cannot be expressed per-parameter: enforcing
        # it needs the dict's `conversion_function` and `evaluate_constraints`,
        # neither of which survives the per-parameter `priors_dict` below.
        # `PriorDict.ln_prob` applies both the per-parameter support and the
        # constraints in one call, which is what `log_prior` uses.
        self.priors = priors
        # Per-parameter view of the sampled priors, for the many places that
        # legitimately need one marginal at a time (cdf/rescale, widths,
        # per-parameter precision).  Constraint priors are absent by
        # construction: `non_fixed_keys` excludes them.
        self.priors_dict = {key: priors[key] for key in self.parameter_names}
        # Per-prior bound on the precision it may contribute; see
        # ``_prior_precision_cap``.  Depends only on the prior, so it is cached
        # across the repeated precision evaluations of a multi-mode search.
        self._prior_precision_cap_cache = {}
        self._prior_std_cache = None
        # scipy's own OptimizeResult / _RichResult from MAP finding and the
        # Hessian, kept for their `nfev` -- the true likelihood-evaluation
        # counts for each stage, read by the sampler after each call. None
        # until the corresponding method has actually run.
        self.minimization_metadata = None
        self.hessian_metadata = None

        # Starting points for the multi-start MAP search.  Drawn *constrained*:
        # `sample_subset` ignores Constraint priors, so on a constrained prior
        # a fraction of the starts would begin in a forbidden region where the
        # log-posterior is -inf and the local optimiser has no gradient to
        # follow.  Skip when using differential_evolution, which doesn't need
        # starting points.
        if minimization_method != "differential_evolution":
            # `sample_subset_constrained` deletes constraint keys from the list
            # it is given, in place -- pass a copy so `parameter_names` is safe.
            self.prior_samples = [
                priors.sample_subset_constrained(list(self.parameter_names)) for _ in range(n_prior_samples)
            ]
        self.prior_bounds_min = np.array([priors[key].minimum for key in self.parameter_names])
        self.prior_bounds_max = np.array([priors[key].maximum for key in self.parameter_names])
        self.prior_bounds = list(zip(self.prior_bounds_min, self.prior_bounds_max))

        self.prior_width_dict = {}
        for key in self.parameter_names:
            width = priors[key].width
            if np.isnan(width):
                raise ValueError(f"Prior width is ill-formed for {key}")
            self.prior_width_dict[key] = width

        self.fixed_parameters = {}
        for key, val in priors.items():
            if key in self.parameter_names:
                continue
            if isinstance(val, (int, float)):
                self.fixed_parameters[key] = float(val)
            elif hasattr(val, "peak"):  # DeltaFunction prior
                self.fixed_parameters[key] = float(val.peak)

    def log_likelihood(self, sample):
        if not isinstance(sample, dict):
            if isinstance(sample, pd.DataFrame) and len(sample) == 1:
                sample = sample.to_dict()
            else:
                raise ValueError("sample must be a dict or single-row DataFrame")
        # Merge fixed values first so that sampled values always take priority.
        return self.likelihood.log_likelihood(parameters={**self.fixed_parameters, **sample})

    def log_prior(self, sample):
        """Evaluate log-prior for a parameter dict (sampled parameters only).

        Delegates to ``PriorDict.ln_prob`` rather than summing the marginals
        by hand.  That is what makes every stage built on the log-posterior --
        the MAP search, the parameter-space Hessian, the mode search and the
        Laplace evidence -- respect ``Constraint`` priors: a constraint bounds
        a *derived* quantity, so no product of per-parameter terms can see it.
        ``ln_prob`` returns ``-inf`` both outside a parameter's own range and
        for a constraint violation, and adds bilby's
        ``normalize_constraint_factor`` so the density is normalised on the
        constrained support (the same normalisation the rejection/importance/
        SMC paths already get by calling ``ln_prob`` themselves).
        """
        return float(self.priors.ln_prob({k: float(sample[k]) for k in self.parameter_names}))

    def log_posterior(self, sample):
        """Evaluate log-posterior = log-likelihood + log-prior."""
        lp = self.log_prior(sample)
        if not np.isfinite(lp):
            return -np.inf
        return self.log_likelihood(sample) + lp

    def constraint_mask(self, x_array):
        """Boolean mask of which columns satisfy the ``Constraint`` priors.

        ``x_array`` is ``(N_params, N_samples)``, column-stacked to match
        :meth:`log_likelihood_from_array`.  Returns a ``(N_samples,)`` bool
        array; all-``True`` when the prior carries no constraints, without
        touching the conversion function at all.

        Deliberately vectorised over the whole batch.  bilby's dynesty wrapper
        calls ``evaluate_constraints`` once per likelihood call, which on this
        BNS prior costs ~23 us a point; the same check across a batch costs
        ~0.06-0.19 us a point, because the conversion function and the
        constraint comparisons are pure numpy broadcasts.  Since every
        resampling mode here already hands the likelihood whole batches, the
        constraint belongs at batch level too.
        """
        n = x_array.shape[1]
        if not self.priors.constraint_keys:
            return np.ones(n, dtype=bool)
        sample = {key: x_array[i] for i, key in enumerate(self.parameter_names)}
        # `evaluate_constraints` returns floats (1.0/0.0), not bools.
        return np.asarray(self.priors.evaluate_constraints(sample), dtype=bool)

    def log_likelihood_from_array(self, x_array, clip_to_bounds=False):
        """Log-likelihood for one parameter vector or a column-stacked batch.

        Accepts ``(N_params,)`` or ``(N_params, N_samples)`` and returns a
        scalar or ``(N_samples,)`` respectively.  Points outside the prior
        support -- either outside a parameter's own range or violating a
        ``Constraint`` -- return ``-inf`` and are never handed to the
        likelihood.

        ``clip_to_bounds`` clips into the per-parameter box, as before.  It
        cannot rescue a constraint violation: a constraint bounds a derived
        quantity, so there is no per-parameter projection onto the allowed
        region, and such points still return ``-inf``.
        """
        x_array = np.asarray(x_array, dtype=float)
        single = x_array.ndim == 1
        x2 = x_array[:, None] if single else x_array

        if clip_to_bounds:
            x2 = np.clip(x2, self.prior_bounds_min[:, None], self.prior_bounds_max[:, None])
            keep = np.ones(x2.shape[1], dtype=bool)
        else:
            keep = np.all(
                (x2 >= self.prior_bounds_min[:, None]) & (x2 <= self.prior_bounds_max[:, None]),
                axis=0,
            )

        # Constraints are only meaningful for in-box points: the conversion
        # function can be undefined (or merely nonsense) outside a parameter's
        # own range, so screen the box first and constrain the survivors.
        if keep.any():
            keep[np.flatnonzero(keep)] = self.constraint_mask(x2[:, keep])

        logl = np.full(x2.shape[1], -np.inf)
        if keep.any():
            logl[keep] = self._log_likelihood_for_columns(x2[:, keep])
        return logl[0] if single else logl

    def _log_likelihood_for_columns(self, x_array):
        """Evaluate every column of ``(N_params, N_samples)``, in support already."""
        # Parallel path: the likelihood evaluation is the dominant cost in the
        # resampling loops and is embarrassingly parallel (no RNG in the
        # workers), so the result is numerically identical to the serial path.
        if self.pool is not None and x_array.shape[1] > 1:
            return self._log_likelihood_from_array_pool(x_array)
        return np.array(
            [
                self.log_likelihood(array_to_dict(self.parameter_names, x_array[:, j]))
                for j in range(x_array.shape[1])
            ],
            dtype=float,
        )

    def _log_likelihood_from_array_pool(self, x_array):
        """Evaluate ``log_likelihood_from_array`` over a pool for a 2-D batch.

        ``x_array`` is ``(N_params, N_samples)``; each column is one parameter
        vector, already screened against the prior support by the caller.
        Returns a ``(N_samples,)`` array matching the serial path.
        """
        columns = [x_array[:, j] for j in range(x_array.shape[1])]
        worker = partial(
            _pool_log_likelihood,
            list(self.parameter_names),
            dict(self.fixed_parameters),
        )
        # Aim for a few chunks per worker so IPC overhead is amortised without
        # starving workers at the tail of the batch.
        npool = max(1, int(self.npool))
        chunksize = max(1, len(columns) // (4 * npool))
        results = self.pool.map(worker, columns, chunksize=chunksize)
        return np.asarray(results, dtype=float)

    def log_posterior_from_array(self, x_array):
        """Evaluate log-posterior from a parameter array (or column-stacked arrays).

        No explicit bounds test: ``log_posterior`` -> ``log_prior`` ->
        ``PriorDict.ln_prob`` already returns ``-inf`` outside a parameter's
        range *and* on a constraint violation, and ``log_posterior``
        short-circuits before touching the likelihood.  The box test this used
        to carry was both redundant and weaker than what it guarded.
        """

        def wrapped(x_array):
            return self.log_posterior(array_to_dict(self.parameter_names, x_array))

        return np.apply_along_axis(wrapped, 0, x_array)

    def _to_unit_cube(self, x_array):
        return np.array([self.priors_dict[k].cdf(float(x_array[i])) for i, k in enumerate(self.parameter_names)])

    def _from_unit_cube(self, u_array):
        return np.array(
            [
                self.priors_dict[k].rescale(float(np.clip(u_array[i], 0.0, 1.0)))
                for i, k in enumerate(self.parameter_names)
            ]
        )

    def _jacobian_diag(self, x_array):
        """Diagonal of dθ/du = 1/p(θ) at the given parameter values.

        If the MAP sits exactly on a prior boundary where p(θ)=0, the
        parameter is nudged inward by a small fraction of its prior
        width so that the Jacobian remains finite.
        """
        result = np.empty(self.N)
        for i, k in enumerate(self.parameter_names):
            p = self.priors_dict[k].prob(float(x_array[i]))
            if p == 0:
                nudge = 1e-6 * self.prior_width_dict[k]
                x_lo = x_array[i] + nudge
                x_hi = x_array[i] - nudge
                p = max(
                    self.priors_dict[k].prob(float(x_lo)),
                    self.priors_dict[k].prob(float(x_hi)),
                )
                if p == 0:
                    raise ValueError(
                        f"Prior probability is zero for {k}={x_array[i]:.6g} "
                        f"even after nudging; the MAP may be outside the prior"
                    )
                logger.warning(
                    f"MAP value {k}={x_array[i]:.6g} is on the prior "
                    f"boundary (p=0); nudging for Jacobian computation"
                )
            result[i] = 1.0 / p
        return result

    def log_posterior_in_unit_cube(self, u_array):
        """log p(θ(u)|d) = log L(θ(u)) + log π(θ(u)); in unit-cube coords."""

        def wrapped(u):
            x = self._from_unit_cube(u)
            return self.log_posterior(array_to_dict(self.parameter_names, x))

        return np.apply_along_axis(wrapped, 0, u_array)

    def calculate_posterior_precision(self, sample):
        if self.fisher_method == "waveform":
            return self._calculate_precision_waveform(sample)
        if self.use_unit_cube:
            return self._calculate_precision_unit_cube(sample)
        return self._calculate_precision_parameter_space(sample)

    # Analytically-marginalised parameters this path can reinstate in the
    # Fisher and then marginalise out (calibration is excluded -- it marginalises
    # a discrete index, not a continuous direction, and is refused at validation).
    _SUPPORTED_MARGINALIZED = ("geocent_time", "phase", "luminosity_distance")

    def _calculate_precision_waveform(self, sample):
        """Posterior precision from the GW waveform Fisher plus prior precision.

        The likelihood Fisher ``F_ij = (d_i h | d_j h)`` plus the diagonal prior
        precision has the same "negative Hessian of the log-posterior" meaning as
        the other paths.

        If the likelihood analytically marginalises over phase/time/distance,
        those parameters are reinstated: the full precision is built over the
        augmented set (sampled + marginalised) and the marginalised block ``m``
        is removed via its Schur complement,
        ``P_r = P_rr - P_rm P_mm^{-1} P_mr``.  This equals the sampled-parameter
        sub-block of the full covariance ``(P^{-1})_rr`` -- i.e. it marginalises
        over the reconstructed parameters (accounting for their degeneracies),
        rather than conditioning on (fixing) them.
        """
        from .gw_fisher import waveform_fisher_matrix

        marg_names = self._supported_marginalized_names()
        base = {
            **getattr(self.likelihood, "parameters", {}),
            **self.fixed_parameters,
            **sample,
        }

        if not marg_names:
            fisher = waveform_fisher_matrix(self.likelihood, self.parameter_names, base, **self.fisher_kwargs)
            precision = fisher + np.diag(self._prior_precision_diag(sample))
            return self._floor_precision_at_prior(precision)

        marg_values = self._resolve_marginalized_values(sample, marg_names)
        full_names = list(self.parameter_names) + marg_names
        base = {**base, **marg_values}
        logger.info(
            f"Waveform Fisher: reinstating marginalised parameter(s) {marg_names}, "
            f"then marginalising them out via the Schur complement."
        )
        fisher = waveform_fisher_matrix(self.likelihood, full_names, base, **self.fisher_kwargs)
        prior_prec = np.concatenate(
            [
                self._prior_precision_diag(sample),
                self._prior_precision_diag(marg_values, marg_names),
            ]
        )
        precision_full = fisher + np.diag(prior_prec)

        nr = len(self.parameter_names)
        p_rr = precision_full[:nr, :nr]
        p_rm = precision_full[:nr, nr:]
        p_mm = precision_full[nr:, nr:]
        p_mr = precision_full[nr:, :nr]
        try:
            schur_term = p_rm @ np.linalg.solve(p_mm, p_mr)
        except np.linalg.LinAlgError:
            schur_term = p_rm @ np.linalg.pinv(p_mm) @ p_mr
        precision = p_rr - schur_term
        return self._floor_precision_at_prior(0.5 * (precision + precision.T))

    # Quantiles used to evaluate each prior's standard deviation for the floor.
    # Deterministic (a midpoint grid through the inverse CDF) rather than random
    # draws: the proposal covariance must not jitter between runs, and this
    # neither consumes the global RNG nor introduces Monte-Carlo error.
    _PRIOR_STD_NQUANTILES = 4096

    def _prior_standard_deviations(self):
        """Per-parameter prior standard deviation, estimated from draws.

        Cached: depends only on the priors, not on where the precision is
        evaluated.  Evaluated rather than assumed, because the obvious closed
        form -- ``width / sqrt(12)``, the std of a *uniform* prior -- is wrong
        for every non-uniform prior, and this scale is exactly what sets the
        width of a direction the data does not constrain.

        Diagonal only: the estimator does not retain the full ``PriorDict``, so
        correlations a constraint prior would induce are not captured.  Any
        prior whose inverse CDF cannot be evaluated falls back to the uniform
        form.
        """
        if getattr(self, "_prior_std_cache", None) is None:
            quantiles = (np.arange(self._PRIOR_STD_NQUANTILES) + 0.5) / self._PRIOR_STD_NQUANTILES
            stds = []
            for key in self.parameter_names:
                try:
                    values = np.asarray(self.priors_dict[key].rescale(quantiles), dtype=float)
                    stds.append(float(np.std(values[np.isfinite(values)])))
                except Exception as exc:  # pragma: no cover - prior-specific
                    logger.debug(f"Could not evaluate prior {key!r} for the precision floor: {exc}")
                    stds.append(np.nan)
            stds = np.asarray(stds, dtype=float)
            fallback = np.array([self.prior_width_dict[k] for k in self.parameter_names]) / np.sqrt(12.0)
            unusable = ~np.isfinite(stds) | (stds <= 0)
            if unusable.any():
                stds[unusable] = fallback[unusable]
            self._prior_std_cache = stds
        return self._prior_std_cache

    # Rescaled-eigenvalue guard for the marginal cap; see
    # _floor_precision_at_prior. Numerical only, never binding on the result.
    _EIGENVALUE_GUARD = 1e-8

    def _floor_precision_at_prior(self, precision):
        """Bound the parameter-space precision below by the prior precision.

        The waveform Fisher (unlike the unit-cube Hessian) is not automatically
        prior-bounded, so a direction the data leaves unconstrained -- e.g. a
        polarisation angle under phase marginalisation, whose Schur complement
        can be near-singular -- would otherwise invert to a runaway variance.

        Rescaling by the prior standard deviation maps the prior precision to
        the identity, so in those units "no wider than the prior" is
        ``marginal variance <= 1``.  That bound is applied to the *marginals*
        directly: any coordinate whose rescaled marginal variance exceeds 1 is
        shrunk to exactly 1 by a congruence ``C -> S C S`` with
        ``S = diag(min(1, 1/sqrt(diag(C))))``.  Coordinates already inside the
        prior get ``s = 1`` and are left untouched; correlations are unchanged
        because a diagonal congruence rescales rows and columns together, and
        positive-definiteness is preserved for the same reason.  Skipped if any
        prior width is non-finite (an unbounded prior cannot bound the
        posterior).

        The bound is applied to the marginals rather than to the rescaled
        *eigenvalues*: flooring eigenvalues only ever adds precision, which can
        only shrink every marginal variance, so it narrows already-narrow
        directions along with the runaway ones instead of leaving them alone.
        Capping the marginals directly leaves already-narrow directions
        untouched by construction while still killing the runaway.

        The scale comes from ``_prior_standard_deviations`` rather than the
        uniform-prior ``width / sqrt(12)``, which is wrong for any non-uniform
        prior (see that method).
        """
        precision = 0.5 * (precision + precision.T)
        widths = np.array([self.prior_width_dict[k] for k in self.parameter_names])
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
            logger.warning(
                "Non-finite or non-positive prior width(s); skipping the prior " "bound on the waveform precision."
            )
            return precision

        prior_std = self._prior_standard_deviations()
        outer_std = np.outer(prior_std, prior_std)
        scaled = precision * outer_std  # D P D, with D = diag(prior_std)
        scaled = 0.5 * (scaled + scaled.T)

        # Numerical guard only. A non-positive eigenvalue inverts to a negative
        # or infinite variance and would make the marginal cap below undefined.
        # 1e-8 in rescaled units is a variance 1e8 times the prior -- far wider
        # than the cap allows -- so unlike flooring at 1 this never binds on
        # the returned precision.
        eigvals, eigvecs = np.linalg.eigh(scaled)
        n_guarded = int(np.sum(eigvals < self._EIGENVALUE_GUARD))
        if n_guarded:
            logger.debug(
                f"Guarding {n_guarded} non-positive rescaled eigenvalue(s) "
                f"(min {eigvals.min():.3g}) before the marginal cap."
            )
        eigvals = np.maximum(eigvals, self._EIGENVALUE_GUARD)
        scaled_cov = (eigvecs / eigvals) @ eigvecs.T

        marginal = np.diag(scaled_cov).copy()
        over = marginal > 1.0
        if np.any(over):
            names = [n for n, flag in zip(self.parameter_names, over) if flag]
            logger.info(
                f"Capping {int(over.sum())} marginal variance(s) at the prior: "
                + ", ".join(f"{n} ({marginal[i]:.3g}x prior var)" for i, n in zip(np.flatnonzero(over), names))
            )
            shrink = np.where(over, 1.0 / np.sqrt(np.maximum(marginal, self._EIGENVALUE_GUARD)), 1.0)
            scaled_cov = scaled_cov * np.outer(shrink, shrink)

        scaled = np.linalg.inv(0.5 * (scaled_cov + scaled_cov.T))
        precision = scaled / outer_std
        return 0.5 * (precision + precision.T)

    def _supported_marginalized_names(self):
        """Marginalised parameters (excluding any already sampled) to reinstate."""
        marg = list(getattr(self.likelihood, "marginalized_parameters", []) or [])
        return [n for n in marg if n in self._SUPPORTED_MARGINALIZED and n not in self.parameter_names]

    def _resolve_marginalized_values(self, sample, names):
        """Point values for the marginalised parameters: reference (injection)
        where finite, otherwise reconstructed from the likelihood at the MAP."""
        values = {}
        missing = []
        for name in names:
            ref = self.marginalized_reference.get(name)
            if ref is not None and np.isfinite(float(ref)):
                values[name] = float(ref)
            else:
                missing.append(name)
        if missing:
            values.update(self._reconstruct_marginalized(sample, missing))
        return values

    def _reconstruct_marginalized(self, sample, names):
        """Reconstruct marginalised parameters at the MAP via bilby's built-in
        conditional draw (a single, RNG-seeded sample)."""
        base = {
            **getattr(self.likelihood, "parameters", {}),
            **self.fixed_parameters,
            **sample,
        }
        reconstructed = self.likelihood.generate_posterior_sample_from_marginalized_likelihood(dict(base))
        values = {name: float(reconstructed[name]) for name in names}
        logger.info(
            f"Reconstructed marginalised parameter(s) at the MAP: " f"{ {k: round(v, 6) for k, v in values.items()} }"
        )
        return values

    def _prior_precision_diag(self, sample, names=None):
        """Diagonal prior precision ``-d^2/dtheta^2 log pi_i`` at the MAP.

        Computed by central differences on each 1-D prior log-density (priors
        are smooth and cheap to difference).  Returns zeros for flat priors and
        where the density is non-finite (e.g. at a boundary).  ``names`` defaults
        to the sampled parameters; pass an explicit list (e.g. marginalised
        parameters, whose priors live on ``likelihood.priors``) to evaluate
        those instead.
        """
        if names is None:
            names = self.parameter_names
        precision = np.zeros(len(names))
        for i, key in enumerate(names):
            # Sampled-parameter priors live on priors_dict; marginalised-parameter
            # priors (phase/time/distance) live on the likelihood.
            prior = self.priors_dict.get(key) or self.likelihood.priors[key]
            width = self.prior_width_dict.get(key)
            if width is None or not np.isfinite(width) or width == 0:
                width = getattr(prior, "maximum", 1.0) - getattr(prior, "minimum", 0.0)
                width = width if np.isfinite(width) and width != 0 else 1.0
            x = float(sample[key])
            h = 1e-4 * width
            with np.errstate(divide="ignore", invalid="ignore"):
                lp = np.log(prior.prob(x))
                lp_plus = np.log(prior.prob(x + h))
                lp_minus = np.log(prior.prob(x - h))
            d2 = (lp_plus - 2.0 * lp + lp_minus) / h**2
            value = -d2 if np.isfinite(d2) else 0.0
            # Bound the contribution.  Evaluated pointwise this curvature is
            # unbounded in both directions: it diverges wherever the prior
            # density has a cusp (e.g. a log-divergent density at a boundary),
            # which can collapse that parameter's estimated width to near zero
            # even with no likelihood information; it can also go negative
            # wherever log pi is locally convex, which subtracts information
            # and pushes the precision matrix towards indefiniteness.  A
            # one-dimensional prior can supply neither more information than
            # its own inverse variance nor less than none, so clamp to that
            # range.
            precision[i] = float(np.clip(value, 0.0, self._prior_precision_cap(key, prior)))
        return precision

    # Draws used to bound the prior precision.  Sampling rather than an analytic
    # variance because bilby priors do not expose one uniformly; 20k keeps the
    # bound's Monte-Carlo error well under a percent.
    _PRIOR_PRECISION_CAP_NSAMPLES = 20000

    def _prior_precision_cap(self, key, prior):
        """Largest precision the 1-D prior on *key* may contribute.

        ``1 / Var(prior)``: a prior cannot pin a parameter down more tightly
        than its own spread.  A prior that cannot be sampled is left unbounded
        rather than guessed at, which preserves the previous behaviour for it.
        """
        if key not in self._prior_precision_cap_cache:
            try:
                draws = np.asarray(prior.sample(self._PRIOR_PRECISION_CAP_NSAMPLES), dtype=float)
                variance = float(np.var(draws[np.isfinite(draws)]))
            except Exception as exc:  # pragma: no cover - prior-specific failure
                logger.debug(f"Could not sample prior {key!r} to bound its precision: {exc}")
                variance = np.nan
            usable = np.isfinite(variance) and variance > 0
            self._prior_precision_cap_cache[key] = 1.0 / variance if usable else np.inf
        return self._prior_precision_cap_cache[key]

    def _calculate_precision_parameter_space(self, sample):
        logger.info("Computing Hessian of log-posterior (scipy.differentiate)")
        point = np.array([sample[key] for key in self.parameter_names])
        kw = {"initial_step": 0.5, **self.hessian_kwargs}
        res = sd.hessian(self.log_posterior_from_array, point, **kw)
        self.hessian_metadata = res
        precision = -res.ddf
        logger.debug(f"Estimated Hessian:\n{precision}")
        return precision

    # In unit-cube coordinates the prior is uniform on [0, 1], whose variance
    # is 1/12.  A bounded posterior can never be broader than its prior, so
    # 12 is the smallest physically-meaningful precision along any direction.
    # Flooring the unit-cube precision eigenvalues here bounds the posterior
    # covariance by the prior: directions the data does not constrain (or that
    # are corrupted by finite-difference noise, giving spurious near-zero or
    # negative curvature) gracefully fall back to prior width instead of
    # blowing up.  Well-constrained directions have precision >> 12 and are
    # untouched.
    PRIOR_PRECISION_UNIT_CUBE = 12.0

    def _calculate_precision_unit_cube(self, sample):
        x_array = np.array([sample[key] for key in self.parameter_names])
        u_map = self._to_unit_cube(x_array)

        kw = {
            "initial_step": 0.001,
            "step_factor": 2,
            "maxiter": 10,
            **self.hessian_kwargs,
        }
        logger.info(f"Computing Hessian of log-posterior in unit cube (scipy.differentiate) with {kw}")
        res = sd.hessian(self.log_posterior_in_unit_cube, u_map, **kw)
        self.hessian_metadata = res
        logger.debug(f"Hessian computed: success={res.success}, status={res.status}, nfev={res.nfev}")

        # scipy.differentiate reports per-element convergence.  On a noisy
        # objective (e.g. a marginalised GW likelihood) most entries routinely
        # fail its convergence test even when the estimate is perfectly usable,
        # so this is logged at info level rather than warned: it is a diagnostic
        # of estimate quality, not an error.  The prior floor below is what
        # actually bounds the covariance if the estimate is poor.
        success = np.asarray(res.success)
        n_failed = int(success.size - np.count_nonzero(success))
        if n_failed > 0:
            logger.info(
                f"scipy.differentiate.hessian reported non-convergence for "
                f"{n_failed}/{success.size} entries (status codes: "
                f"{np.unique(np.asarray(res.status)).tolist()}); curvature "
                f"estimate may be noisy. The prior floor bounds the covariance."
            )

        precision_u = -res.ddf
        logger.debug(f"Hessian (unit cube):\n{precision_u}")

        # A too-large step can push evaluation points outside the unit cube (or
        # otherwise fail), leaving non-finite Hessian entries.  Zero them so the
        # affected directions collapse to near-zero curvature and are lifted to
        # the prior precision by the floor below, rather than crashing the
        # eigendecomposition.
        n_nonfinite = int(np.sum(~np.isfinite(precision_u)))
        if n_nonfinite > 0:
            logger.warning(
                f"Unit-cube Hessian has {n_nonfinite} non-finite entry(ies) "
                f"(the step may be too large); zeroing them so those directions "
                f"default to prior width."
            )
            precision_u = np.where(np.isfinite(precision_u), precision_u, 0.0)

        # Floor the unit-cube precision at the prior precision (see
        # PRIOR_PRECISION_UNIT_CUBE).  Symmetrise, then floor the eigenvalues:
        # this simultaneously enforces positive-definiteness (removing spurious
        # negative curvature) and bounds the covariance by the prior.
        precision_u = 0.5 * (precision_u + precision_u.T)
        eigvals, eigvecs = np.linalg.eigh(precision_u)
        n_floored = int(np.sum(eigvals < self.PRIOR_PRECISION_UNIT_CUBE))
        if n_floored > 0:
            logger.info(
                f"Flooring {n_floored} unit-cube precision eigenvalue(s) at the "
                f"prior precision ({self.PRIOR_PRECISION_UNIT_CUBE:g}; "
                f"min was {eigvals.min():.3g}). Those directions default to "
                f"prior width."
            )
            eigvals = np.maximum(eigvals, self.PRIOR_PRECISION_UNIT_CUBE)
            precision_u = (eigvecs * eigvals) @ eigvecs.T
            precision_u = 0.5 * (precision_u + precision_u.T)

        J_inv = 1.0 / self._jacobian_diag(x_array)  # = p(θ_MAP)

        # Cap the Jacobian at the uniform-prior value (1/width) for each
        # parameter.  When the prior is strongly peaked (p(θ) >> 1/width),
        # the uncapped J_inv amplifies the unit-cube precision and collapses
        # the parameter-space covariance.  Capping prevents this for prior-
        # dominated parameters while leaving likelihood-constrained
        # parameters unaffected.
        uniform_cap = np.array([self.jacobian_cap_scale / self.prior_width_dict[k] for k in self.parameter_names])
        capped = J_inv > uniform_cap
        if np.any(capped):
            names = [k for k, c in zip(self.parameter_names, capped) if c]
            logger.info(f"Capping Jacobian for prior-dominated parameter(s): " f"{', '.join(names)}")
            J_inv = np.minimum(J_inv, uniform_cap)

        return J_inv[:, None] * precision_u * J_inv[None, :]

    def log_evidence_laplace(self, sample, covariance):
        """Laplace approximation to the log evidence.

        Parameters
        ----------
        sample : dict
            MAP parameter values.
        covariance : array
            Inverse of the negative Hessian of the log-posterior (the
            posterior covariance at the MAP).

        Returns
        -------
        log_evidence : float
            log Z ≈ log L(θ_MAP) + log π(θ_MAP) + (d/2) log(2π)
                    + (1/2) log det(Σ)
        """
        d = len(self.parameter_names)
        log_l_map = self.log_likelihood(sample)
        # Via `log_prior` (i.e. `PriorDict.ln_prob`), not a hand-rolled product
        # of marginals: on a constrained prior the two differ by
        # `log(normalize_constraint_factor)` -- 0.50 nats on the BNS_3G prior,
        # for instance.  The rejection/importance/SMC evidences all come from
        # `ln_prob` already, so summing marginals here left this estimate on a
        # different normalisation from every other log Z the sampler reports.
        log_pi_map = self.log_prior(sample)
        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:
            logger.warning("covariance has non-positive determinant; " "Laplace evidence estimate may be unreliable")
        log_z = log_l_map + log_pi_map + 0.5 * d * np.log(2 * np.pi) + 0.5 * log_det
        logger.info(
            f"Laplace log-evidence: {log_z:.2f} "
            f"(log L_MAP={log_l_map:.2f}, "
            f"log π_MAP={log_pi_map:.2f}, "
            f"det term={0.5 * log_det:.2f})"
        )
        return log_z

    # Eigenvalues of the preconditioned precision below this fraction of the
    # largest are floored before inversion, capping the condition number of the
    # inverted matrix at 1 / COVARIANCE_REL_FLOOR.
    #
    # This is a numerical guard, not a statistical bound. The statistical bound
    # -- no direction wider than the prior -- is `_floor_precision_at_prior`,
    # and this floor must stay looser than that one: if it fires first on a
    # poorly-constrained direction, it silently overrides the prior bound
    # instead of leaving that direction to it, understating the true width of
    # exactly the parameters the data does not constrain. Set low enough that
    # it fires only on genuinely zero or negative eigenvalues, relying on the
    # diagonal preconditioning above to have already removed most of the
    # parameter-scale-driven ill-conditioning.
    COVARIANCE_REL_FLOOR = 1e-14

    def calculate_posterior_covariance(self, sample):
        precision = self.calculate_posterior_precision(sample)

        # Force the precision to be symmetric by averaging off-diagonal estimates.
        upper_off_diagonal_average = 0.5 * (np.triu(precision, 1) + np.triu(precision.T, 1))
        precision = np.diag(np.diag(precision)) + upper_off_diagonal_average + upper_off_diagonal_average.T

        # Diagonal preconditioning: rescale to (approximately) unit diagonal so
        # the eigen-flooring and inversion below act on a well-conditioned
        # matrix.  This is the shared normalisation trick from GWFish/gwfast and
        # removes most of the parameter-scale-driven ill-conditioning.  The
        # observed-information matrix can be indefinite (unlike a pure Fisher),
        # so fall back to no scaling if the diagonal is not strictly positive.
        diag = np.diag(precision)
        if np.all(np.isfinite(diag)) and np.all(diag > 0):
            scale = np.sqrt(diag)
        else:
            logger.warning("Precision diagonal is not strictly positive; skipping preconditioning.")
            scale = np.ones(self.N)
        outer_scale = np.outer(scale, scale)
        precision_norm = precision / outer_scale

        cond_raw = np.linalg.cond(precision)
        cond_norm = np.linalg.cond(precision_norm)
        logger.info(f"Precision condition number: {cond_raw:.3g} -> {cond_norm:.3g} after preconditioning")

        # Invert via the eigendecomposition of the normalised matrix, flooring
        # small or negative eigenvalues at a relative threshold.  Flooring the
        # *precision* eigenvalues up to a positive value yields large-but-finite
        # variance in poorly-constrained or indefinite directions -- the right
        # behaviour for a proposal covariance (wide where the data does not
        # constrain).  This is deliberately NOT GWFish-style truncation, which
        # would instead zero out those directions.
        eigvals, eigvecs = np.linalg.eigh(precision_norm)
        max_abs = max(np.max(np.abs(eigvals)), 1e-30)
        threshold = self.COVARIANCE_REL_FLOOR * max_abs
        n_floored = int(np.sum(eigvals < threshold))
        if n_floored > 0:
            logger.warning(
                f"{n_floored} preconditioned precision eigenvalue(s) below "
                f"{threshold:.2g} (min={eigvals.min():.3g}); flooring before "
                f"inversion. Those directions are poorly constrained or indefinite."
            )
            eigvals = np.maximum(eigvals, threshold)

        # M^{-1} = V diag(1/λ) Vᵀ, then undo the scaling: Σ = D⁻¹ M⁻¹ D⁻¹.
        # All eigenvalues are now positive, so Σ is positive definite by
        # construction.
        covariance_norm = (eigvecs / eigvals) @ eigvecs.T
        covariance = covariance_norm / outer_scale
        covariance = 0.5 * (covariance + covariance.T)

        return covariance

    def sample_array(self, sample, n=1):
        if sample == "maxL":
            sample = self.get_maximum_likelihood_sample()

        self.mean = np.array(list(sample.values()))
        self.covariance = self.calculate_posterior_covariance(sample)
        return random.rng.multivariate_normal(self.mean, self.covariance, n)

    def sample_dataframe(self, sample, n=1):
        samples = self.sample_array(sample, n)
        return pd.DataFrame(samples, columns=self.parameter_names)

    def _maximize_posterior_differential_evolution(self):
        """Global MAP search: differential evolution, then a local polish.

        Both departures from scipy's defaults are load-bearing; see
        :data:`DE_ATOL` for the measurements behind them.
        """

        def neg_log_post(x):
            return -self.log_posterior_from_array(x)

        out = differential_evolution(
            neg_log_post,
            bounds=self.prior_bounds,
            seed=self.seed,
            # Absolute, not relative: see DE_ATOL.
            tol=0,
            atol=DE_ATOL,
            # scipy's own polish is L-BFGS-B, whose finite-difference gradients
            # are useless on a likelihood whose parameters span 1e-2 to 5e3 (it
            # returned the input unchanged after 36 evaluations, and scipy's
            # numdiff warned of invalid subtractions). Nelder-Mead below does
            # the job, and is the local method this class uses everywhere else.
            polish=False,
        )
        polished = minimize(neg_log_post, out.x, bounds=self.prior_bounds, method="Nelder-Mead")
        # A fresh result rather than a mutated leg: `nfev` has to price the
        # whole search (`run_statistics` quotes it as the MAP's cost), and
        # editing scipy's return value in place would leave the caller holding
        # an object whose own count no longer means what it says.
        best = OptimizeResult(**(polished if polished.fun <= out.fun else out))
        best.nfev = out.nfev + polished.nfev
        return best

    def _maximize_posterior_from_initial_sample(self, initial_sample):
        x0 = list(initial_sample.values())

        def neg_log_post(x):
            return -self.log_posterior_from_array(x)

        # differential_evolution is not a valid method for scipy.optimize.minimize;
        # fall back to Nelder-Mead when used with an initial starting point.
        local_method = (
            "Nelder-Mead" if self.minimization_method == "differential_evolution" else self.minimization_method
        )
        return minimize(
            neg_log_post,
            x0,
            bounds=self.prior_bounds,
            method=local_method,
        )

    def get_maximum_likelihood_sample(self, initial_sample=None):
        """Alias for :meth:`get_MAP_sample`, used internally by
        :meth:`sample_array` for ``sample='maxL'``."""
        return self.get_MAP_sample(initial_sample)

    def get_MAP_sample(self, initial_sample=None):
        """Find the maximum a posteriori (MAP) estimate.

        Maximizes log-likelihood + log-prior. By default uses differential
        evolution, a global optimizer that searches the full prior-bounded
        space. If ``initial_sample`` is provided, a single local minimization
        is run from that starting point.

        When ``minimization_method`` is not ``'differential_evolution'`` and no
        ``initial_sample`` is given, a multi-start local-optimizer strategy is
        used instead: ``n_prior_samples`` random prior draws are each used as
        starting points for a local optimizer and the best result is returned.
        """
        if initial_sample:
            logger.info("Finding MAP from initial parameters")
            minout = self._maximize_posterior_from_initial_sample(initial_sample)
        elif self.minimization_method == "differential_evolution":
            logger.info("Finding MAP using differential evolution")
            minout = self._maximize_posterior_differential_evolution()
        else:
            logger.info(f"Finding MAP from " f"{self.n_prior_samples} starting points")
            max_logP = -np.inf
            logP_list = []
            successes = 0
            for sample in tqdm.tqdm(self.prior_samples):
                out = self._maximize_posterior_from_initial_sample(sample)
                logP = -out.fun
                logP_list.append(logP)
                if out.success:
                    successes += 1
                if logP > max_logP:
                    max_logP = logP
                    minout = out

            if np.isinf(max_logP):
                raise ValueError("Maximisation of the posterior failed")

            logger.info(
                f"Optimisation complete: "
                f"{100 * successes / self.n_prior_samples:.0f}% "
                f"of starts converged, "
                f"best log-posterior = {max_logP:.4f}"
            )

        self.minimization_metadata = minout
        map_sample = {key: val for key, val in zip(self.parameter_names, minout.x)}
        log_l = self.log_likelihood(map_sample)
        log_pi = self.log_prior(map_sample)
        logger.info(f"MAP found: log-posterior = {-minout.fun:.4f} " f"(log-L = {log_l:.4f}, log-prior = {log_pi:.4f})")
        return map_sample
