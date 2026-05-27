import numpy as np
import pandas as pd
import scipy.differentiate as sd
import scipy.linalg
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
            prior edge, giving unbiased curvature estimates.
        jacobian_cap_scale: float
            Scales the Jacobian cap applied when transforming the unit-cube
            Hessian back to parameter space. The cap is
            ``jacobian_cap_scale / prior_width``. The default of 1.0 caps at
            the uniform-prior Jacobian. Values < 1 apply a tighter cap,
            widening the proposal for prior-dominated parameters.
        hessian_kwargs: dict, optional
            Keyword arguments forwarded to ``scipy.differentiate.hessian``.
            Defaults are ``{"initial_step": 0.5}`` in parameter space and
            ``{"initial_step": 0.1}`` in unit-cube space. Any key provided
            here overrides the corresponding default.
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
        if self.use_unit_cube:
            return self._calculate_precision_unit_cube(sample)
        return self._calculate_precision_parameter_space(sample)

    def _calculate_precision_parameter_space(self, sample):
        logger.info("Computing Hessian of log-posterior (scipy.differentiate)")
        point = np.array([sample[key] for key in self.parameter_names])
        kw = {"initial_step": 0.5, **self.hessian_kwargs}
        res = sd.hessian(self.log_posterior_from_array, point, **kw)
        precision = -res.ddf
        logger.debug(f"Estimated Hessian:\n{precision}")
        return precision

    def _calculate_precision_unit_cube(self, sample):
        x_array = np.array([sample[key] for key in self.parameter_names])
        u_map = self._to_unit_cube(x_array)

        kw = {"initial_step": 0.001, "step_factor": 2, "maxiter": 20, **self.hessian_kwargs}
        logger.info(f"Computing Hessian of log-posterior in unit cube (scipy.differentiate) with {kw}")
        res = sd.hessian(self.log_posterior_in_unit_cube, u_map, **kw)
        logger.debug(f"Hessian computed: success={res.success}, status={res.status}, nfev={res.nfev}")

        precision_u = -res.ddf
        logger.debug(f"Hessian (unit cube):\n{precision_u}")

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

    def calculate_posterior_covariance(self, sample):
        precision = self.calculate_posterior_precision(sample)

        # Force the precision to be symmetric by averaging off-diagonal estimates
        upper_off_diagonal_average = 0.5 * (np.triu(precision, 1) + np.triu(precision.T, 1))
        precision = np.diag(np.diag(precision)) + upper_off_diagonal_average + upper_off_diagonal_average.T

        # Regularise a near-singular precision by flooring small/negative
        # eigenvalues before inversion.  This prevents LinAlgError on singular
        # matrices when some parameters are poorly constrained by the data.
        prec_eigvals, prec_eigvecs = np.linalg.eigh(precision)
        max_abs = max(np.max(np.abs(prec_eigvals)), 1e-30)
        threshold = 1e-10 * max_abs
        n_small = int(np.sum(prec_eigvals < threshold))
        if n_small > 0:
            logger.warning(
                f"Precision has {n_small} eigenvalue(s) below {threshold:.2g} "
                f"(min={prec_eigvals.min():.3g}); some parameters appear poorly "
                f"constrained. Flooring to {threshold:.2g} before inversion."
            )
            prec_eigvals = np.maximum(prec_eigvals, threshold)
            precision = prec_eigvecs @ np.diag(prec_eigvals) @ prec_eigvecs.T
            precision = 0.5 * (precision + precision.T)

        covariance = scipy.linalg.inv(precision)

        # Ensure the covariance is positive definite.  Apply the 1e-6 * max
        # floor unconditionally: inversion of an ill-conditioned precision
        # (condition number ~1e10) can leave tiny-but-positive eigenvalues that
        # are still below scipy's _PSD tolerance (eps ≈ 2.22e-10 * max).
        # Flooring at 1e-6 * max keeps the condition number at most 1e6, well
        # inside that threshold.
        eigvals, eigvecs = np.linalg.eigh(covariance)
        n_neg = int(np.sum(eigvals < 0))
        if n_neg > 0:
            logger.warning(
                f"Covariance matrix has {n_neg} negative "
                f"eigenvalue(s); projecting to nearest "
                f"positive-definite matrix"
            )
        floor = 1e-6 * np.max(eigvals)
        if eigvals.min() < floor:
            eigvals = np.maximum(eigvals, floor)
            covariance = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Re-symmetrise to remove floating-point drift
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
