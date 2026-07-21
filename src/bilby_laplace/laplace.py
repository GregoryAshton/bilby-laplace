import numpy as np
import pandas as pd
import scipy.differentiate as sd
import tqdm
from bilby.core.prior import PriorDict
from bilby.core.utils import logger, random
from scipy.optimize import differential_evolution, minimize


def array_to_dict(keys, array):
    return dict(zip(keys, array))


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
            ignored). If the likelihood analytically marginalises over phase,
            time, and/or distance, the Fisher is built over the augmented set
            (sampled parameters plus the marginalised ones) and the marginalised
            block is removed via its Schur complement -- equivalent to
            marginalising, not conditioning, over those parameters. The result is
            floored at the prior precision (as the unit-cube path is), so no
            marginal variance exceeds the prior.
        fisher_kwargs: dict, optional
            Keyword arguments forwarded to the waveform-Fisher computation when
            ``fisher_method='waveform'`` (e.g. ``eps``, ``eps_mass``).
        marginalized_reference: dict, optional
            Point values for analytically-marginalised parameters (phase, time,
            distance) at which to evaluate the waveform Fisher. Typically the
            injection. Any marginalised parameter absent here is reconstructed
            from the marginalised likelihood at the MAP. Only used by
            ``fisher_method='waveform'``.
        """
        self.likelihood = likelihood

        if not isinstance(priors, PriorDict):
            priors = PriorDict(priors)

        if parameters is None:
            self.parameter_names = priors.non_fixed_keys
        else:
            self.parameter_names = parameters
        self.minimization_method = minimization_method
        self.n_prior_samples = n_prior_samples
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
        self.priors_dict = {key: priors[key] for key in self.parameter_names}

        # Construct prior samples at initialisation so that the prior is not stored.
        # Skip when using differential_evolution, which doesn't need starting points.
        if minimization_method != "differential_evolution":
            self.prior_samples = [priors.sample_subset(self.parameter_names) for _ in range(n_prior_samples)]
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
        """Evaluate log-prior for a parameter dict (sampled parameters only)."""
        return sum(np.log(self.priors_dict[k].prob(float(sample[k]))) for k in self.parameter_names)

    def log_posterior(self, sample):
        """Evaluate log-posterior = log-likelihood + log-prior."""
        lp = self.log_prior(sample)
        if not np.isfinite(lp):
            return -np.inf
        return self.log_likelihood(sample) + lp

    def log_likelihood_from_array(self, x_array, clip_to_bounds=False):
        def wrapped_logl(x_array):
            if clip_to_bounds:
                x_array = x_array.copy()
                idxs = x_array < self.prior_bounds_min
                x_array[idxs] = self.prior_bounds_min[idxs]
                idxs = x_array > self.prior_bounds_max
                x_array[idxs] = self.prior_bounds_max[idxs]
            else:
                if np.any(x_array < self.prior_bounds_min) or np.any(x_array > self.prior_bounds_max):
                    return -np.inf

            return self.log_likelihood(array_to_dict(self.parameter_names, x_array))

        def wrapped_logl_arb(x_array):
            return np.apply_along_axis(wrapped_logl, 0, x_array)

        return wrapped_logl_arb(x_array)

    def log_posterior_from_array(self, x_array):
        """Evaluate log-posterior from a parameter array (or column-stacked arrays)."""

        def wrapped(x_array):
            if np.any(x_array < self.prior_bounds_min) or np.any(x_array > self.prior_bounds_max):
                return -np.inf
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

    def log_likelihood_in_unit_cube(self, u_array):
        """L̃(u) = L(θ(u)); same shape contract as log_likelihood_from_array."""

        def wrapped(u):
            x = self._from_unit_cube(u)
            return self.log_likelihood(array_to_dict(self.parameter_names, x))

        return np.apply_along_axis(wrapped, 0, u_array)

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

    def _floor_precision_at_prior(self, precision):
        """Bound the parameter-space precision below by the prior precision.

        The waveform Fisher (unlike the unit-cube Hessian) is not automatically
        prior-bounded, so a direction the data leaves unconstrained -- e.g. the
        polarisation angle under phase marginalisation, whose Schur complement
        can be near-singular -- would otherwise invert to a runaway variance.

        Rescaling by the prior standard deviation (``width / sqrt(12)``, the std
        of a uniform prior over the support) maps the prior precision to the
        identity; flooring the rescaled precision eigenvalues at 1 then enforces
        ``posterior precision >= prior precision`` in every direction, i.e. no
        marginal variance exceeds the prior. Well-constrained directions
        (precision >> prior) are untouched. Skipped if any prior width is
        non-finite (an unbounded prior cannot bound the posterior).
        """
        precision = 0.5 * (precision + precision.T)
        widths = np.array([self.prior_width_dict[k] for k in self.parameter_names])
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
            logger.warning(
                "Non-finite or non-positive prior width(s); skipping the prior " "bound on the waveform precision."
            )
            return precision

        prior_std = widths / np.sqrt(12.0)
        outer_std = np.outer(prior_std, prior_std)
        scaled = precision * outer_std  # D P D, with D = diag(prior_std)
        eigvals, eigvecs = np.linalg.eigh(scaled)
        n_floored = int(np.sum(eigvals < 1.0))
        if n_floored > 0:
            logger.info(
                f"Flooring {n_floored} waveform precision eigenvalue(s) at the "
                f"prior (min rescaled eigenvalue {eigvals.min():.3g}); those "
                f"directions default to prior width."
            )
            eigvals = np.maximum(eigvals, 1.0)
            scaled = (eigvecs * eigvals) @ eigvecs.T
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
            precision[i] = -d2 if np.isfinite(d2) else 0.0
        return precision

    def _calculate_precision_parameter_space(self, sample):
        logger.info("Computing Hessian of log-posterior (scipy.differentiate)")
        point = np.array([sample[key] for key in self.parameter_names])
        kw = {"initial_step": 0.5, **self.hessian_kwargs}
        res = sd.hessian(self.log_posterior_from_array, point, **kw)
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
        log_pi_map = sum(np.log(self.priors_dict[k].prob(float(sample[k]))) for k in self.parameter_names)
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
    # largest are floored before inversion.  This caps the condition number of
    # the inverted matrix at 1 / COVARIANCE_REL_FLOOR.
    COVARIANCE_REL_FLOOR = 1e-10

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
        def neg_log_post(x):
            return -self.log_posterior_from_array(x)

        return differential_evolution(neg_log_post, bounds=self.prior_bounds)

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
        """Deprecated alias for :meth:`get_MAP_sample`."""
        return self.get_MAP_sample(initial_sample)

    def get_MAP_sample(self, initial_sample=None):
        """Find the maximum a posteriori (MAP) estimate.

        Maximizes log-likelihood + log-prior. By default uses differential
        evolution, a global optimizer that searches the full prior-bounded
        space. If ``initial_sample`` is provided, a single local minimization
        is run from that starting point.

        When ``minimization_method`` is not ``'differential_evolution'`` and no
        ``initial_sample`` is given, the legacy multi-start Nelder-Mead strategy
        is used: ``n_prior_samples`` random prior draws are each used as starting
        points for a local optimizer and the best result is returned.
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
