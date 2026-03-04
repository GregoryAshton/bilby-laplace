import scipy
from packaging import version

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import minimize, differential_evolution
import tqdm

from bilby.core.utils import random, logger
from bilby.core.prior import PriorDict


def array_to_dict(keys, array):
    return dict(zip(keys, array))


class FisherMatrixPosteriorEstimator:
    def __init__(
        self,
        likelihood,
        priors,
        parameters=None,
        minimization_method="Nelder-Mead",
        fd_eps=1e-6,
        n_prior_samples=100,
        use_unit_cube=True,
    ):
        """A class to estimate posteriors using a Fisher Information Matrix approach

        Parameters
        ----------
        likelihood: bilby.core.likelihood.Likelihood
            A bilby likelihood object
        priors: bilby.core.prior.PriorDict
            A bilby prior object
        parameters: list
            Names of parameters to sample in
        minimization_method: str (Nelder-Mead)
            The method to use in scipy.optimize.minimize
        fd_eps: float
            A parameter to control the size of perturbation used when finite
            differencing the likelihood
        n_prior_samples: int
            The number of prior samples to draw and use to attempt estimation
            of the maximum likelihood sample.
        use_unit_cube: bool
            If True (default), compute the FIM in unit-cube space via the prior
            CDFs. This avoids boundary clipping when the MAP is near a prior
            edge, giving unbiased curvature estimates.
        """
        self.likelihood = likelihood

        if not isinstance(priors, PriorDict):
            priors = PriorDict(priors)

        if parameters is None:
            self.parameter_names = priors.non_fixed_keys
        else:
            self.parameter_names = parameters
        self.minimization_method = minimization_method
        self.fd_eps = fd_eps
        self.n_prior_samples = n_prior_samples
        self.use_unit_cube = use_unit_cube
        self.N = len(self.parameter_names)
        self.priors_dict = {key: priors[key] for key in self.parameter_names}

        # Construct prior samples at initialisation so that the prior is not stored.
        # Skip when using differential_evolution, which doesn't need starting points.
        if minimization_method != "differential_evolution":
            self.prior_samples = [
                priors.sample_subset(self.parameter_names) for _ in range(n_prior_samples)
            ]
        self.prior_bounds_min = np.array(
            [priors[key].minimum for key in self.parameter_names]
        )
        self.prior_bounds_max = np.array(
            [priors[key].maximum for key in self.parameter_names]
        )
        self.prior_bounds = list(zip(self.prior_bounds_min, self.prior_bounds_max))

        self.prior_width_dict = {}
        for key in self.parameter_names:
            width = priors[key].width
            if np.isnan(width):
                raise ValueError(f"Prior width is ill-formed for {key}")
            self.prior_width_dict[key] = width

        # Collect fixed parameter values (floats or DeltaFunction priors) so
        # that every likelihood call receives a complete parameter dict.  This
        # is required when the likelihood uses internal marginalisation (e.g.
        # bilby's GravitationalWaveTransient), which still needs the fixed
        # reference values (e.g. geocent_time) even though they are not sampled.
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
        return self.likelihood.log_likelihood(
            parameters={**self.fixed_parameters, **sample}
        )

    def log_likelihood_from_array(self, x_array):
        def wrapped_logl(x_array):
            # Map points outside the bounds to the bounds
            x_array = x_array.copy()
            idxs = x_array < self.prior_bounds_min
            x_array[idxs] = self.prior_bounds_min[idxs]

            idxs = x_array > self.prior_bounds_max
            x_array[idxs] = self.prior_bounds_max[idxs]

            return self.log_likelihood(array_to_dict(self.parameter_names, x_array))

        def wrapped_logl_arb(x_array):
            return np.apply_along_axis(wrapped_logl, 0, x_array)

        return wrapped_logl_arb(x_array)

    def _to_unit_cube(self, x_array):
        return np.array([self.priors_dict[k].cdf(float(x_array[i]))
                         for i, k in enumerate(self.parameter_names)])

    def _from_unit_cube(self, u_array):
        return np.array([self.priors_dict[k].rescale(float(np.clip(u_array[i], 0.0, 1.0)))
                         for i, k in enumerate(self.parameter_names)])

    def _jacobian_diag(self, x_array):
        """Diagonal of dθ/du = 1/p(θ) at the given parameter values."""
        return np.array([1.0 / self.priors_dict[k].prob(float(x_array[i]))
                         for i, k in enumerate(self.parameter_names)])

    def log_likelihood_in_unit_cube(self, u_array):
        """L̃(u) = L(θ(u)); same shape contract as log_likelihood_from_array."""
        def wrapped(u):
            x = self._from_unit_cube(u)
            return self.log_likelihood(array_to_dict(self.parameter_names, x))
        return np.apply_along_axis(wrapped, 0, u_array)

    def _second_deriv_unit_cube(self, u_map, ii, jj):
        """Finite-difference second derivative of L̃ in unit-cube coords."""
        h = self.fd_eps
        ei = np.zeros(self.N); ei[ii] = h
        ej = np.zeros(self.N); ej[jj] = h
        if ii == jj:
            return (self.log_likelihood_in_unit_cube(u_map + ei)
                    - 2 * self.log_likelihood_in_unit_cube(u_map)
                    + self.log_likelihood_in_unit_cube(u_map - ei)) / h**2
        else:
            return (self.log_likelihood_in_unit_cube(u_map + ei + ej)
                    - self.log_likelihood_in_unit_cube(u_map + ei - ej)
                    - self.log_likelihood_in_unit_cube(u_map - ei + ej)
                    + self.log_likelihood_in_unit_cube(u_map - ei - ej)) / (4 * h**2)

    def calculate_FIM(self, sample):
        if self.use_unit_cube:
            return self._calculate_FIM_unit_cube(sample)
        return self._calculate_FIM_parameter_space(sample)

    def _calculate_FIM_parameter_space(self, sample):
        if version.parse(scipy.__version__) < version.parse("1.15"):
            logger.info("Scipy version < 1.15, using finite-difference fallback")
            FIM = np.zeros((self.N, self.N))
            for ii, ii_key in enumerate(self.parameter_names):
                for jj, jj_key in enumerate(self.parameter_names):
                    FIM[ii, jj] = -self.get_second_order_derivative(
                        sample, ii_key, jj_key
                    )
            return FIM
        else:
            import scipy.differentiate as sd

            logger.info(
                "Using scipy.differentiate to estimate the Fisher information matrix (FIM)"
            )
            point = np.array([sample[key] for key in self.parameter_names])
            res = sd.hessian(self.log_likelihood_from_array, point, initial_step=0.5)
            FIM = -res.ddf
            logger.debug(f"Estimated FIM:\n{FIM}")
            return FIM

    def _calculate_FIM_unit_cube(self, sample):
        x_array = np.array([sample[key] for key in self.parameter_names])
        u_map = self._to_unit_cube(x_array)

        if version.parse(scipy.__version__) < version.parse("1.15"):
            logger.info("Scipy < 1.15: finite-difference FIM in unit cube")
            FIM_u = np.zeros((self.N, self.N))
            for ii in range(self.N):
                for jj in range(self.N):
                    FIM_u[ii, jj] = -self._second_deriv_unit_cube(u_map, ii, jj)
        else:
            import scipy.differentiate as sd
            logger.info("scipy.differentiate hessian in unit cube")
            res = sd.hessian(self.log_likelihood_in_unit_cube, u_map, initial_step=0.5)
            FIM_u = -res.ddf
            logger.debug(f"FIM (unit cube):\n{FIM_u}")

        J_inv = 1.0 / self._jacobian_diag(x_array)   # = p(θ_MAP)
        return J_inv[:, None] * FIM_u * J_inv[None, :]

    def calculate_iFIM(self, sample):
        FIM = self.calculate_FIM(sample)

        # Force the FIM to be symmetric by averaging off-diagonal estimates
        upper_off_diagonal_average = 0.5 * (np.triu(FIM, 1) + np.triu(FIM.T, 1))
        FIM = (
            np.diag(np.diag(FIM))
            + upper_off_diagonal_average
            + upper_off_diagonal_average.T
        )

        iFIM = scipy.linalg.inv(FIM)

        # Ensure iFIM is positive definite
        min_eig = np.min(np.real(np.linalg.eigvals(iFIM)))
        if min_eig < 0:
            logger.warning("Scaling the iFIM to ensure it is positive definite")
            iFIM -= 10 * min_eig * np.eye(*iFIM.shape)

        return iFIM

    def sample_array(self, sample, n=1):
        if sample == "maxL":
            sample = self.get_maximum_likelihood_sample()

        self.mean = np.array(list(sample.values()))
        self.iFIM = self.calculate_iFIM(sample)
        return random.rng.multivariate_normal(self.mean, self.iFIM, n)

    def sample_dataframe(self, sample, n=1):
        samples = self.sample_array(sample, n)
        return pd.DataFrame(samples, columns=self.parameter_names)

    def get_second_order_derivative(self, sample, ii, jj):
        if ii == jj:
            return self.get_finite_difference_xx(sample, ii)
        else:
            return self.get_finite_difference_xy(sample, ii, jj)

    def get_finite_difference_xx(self, sample, ii):
        p = self._shift_sample_x(sample, ii, 1)
        m = self._shift_sample_x(sample, ii, -1)

        dx = 0.5 * (p[ii] - m[ii])

        loglp = self.log_likelihood(p)
        logl = self.log_likelihood(sample)
        loglm = self.log_likelihood(m)

        return (loglp - 2 * logl + loglm) / dx**2

    def get_finite_difference_xy(self, sample, ii, jj):
        pp = self._shift_sample_xy(sample, ii, 1, jj, 1)
        pm = self._shift_sample_xy(sample, ii, 1, jj, -1)
        mp = self._shift_sample_xy(sample, ii, -1, jj, 1)
        mm = self._shift_sample_xy(sample, ii, -1, jj, -1)

        dx = 0.5 * (pp[ii] - mm[ii])
        dy = 0.5 * (pp[jj] - mm[jj])

        loglpp = self.log_likelihood(pp)
        loglpm = self.log_likelihood(pm)
        loglmp = self.log_likelihood(mp)
        loglmm = self.log_likelihood(mm)

        return (loglpp - loglpm - loglmp + loglmm) / (4 * dx * dy)

    def _shift_sample_x(self, sample, x_key, x_coef):
        vx = sample[x_key]
        dvx = self.fd_eps * self.prior_width_dict[x_key]
        shift_sample = sample.copy()
        shift_sample[x_key] = vx + x_coef * dvx
        return shift_sample

    def _shift_sample_xy(self, sample, x_key, x_coef, y_key, y_coef):
        vx = sample[x_key]
        vy = sample[y_key]
        dvx = self.fd_eps * self.prior_width_dict[x_key]
        dvy = self.fd_eps * self.prior_width_dict[y_key]
        shift_sample = sample.copy()
        shift_sample[x_key] = vx + x_coef * dvx
        shift_sample[y_key] = vy + y_coef * dvy
        return shift_sample

    def _maximize_likelihood_differential_evolution(self):
        def neg_log_like(x):
            return -self.log_likelihood_from_array(x)

        return differential_evolution(neg_log_like, bounds=self.prior_bounds)

    def _maximize_likelihood_from_initial_sample(self, initial_sample):
        x0 = list(initial_sample.values())

        def neg_log_like(x):
            return -self.log_likelihood_from_array(x)

        # differential_evolution is not a valid method for scipy.optimize.minimize;
        # fall back to Nelder-Mead when used with an initial starting point.
        local_method = (
            "Nelder-Mead"
            if self.minimization_method == "differential_evolution"
            else self.minimization_method
        )
        return minimize(
            neg_log_like,
            x0,
            bounds=self.prior_bounds,
            method=local_method,
        )

    def get_maximum_likelihood_sample(self, initial_sample=None):
        """Attempt optimization of the maximum likelihood.

        By default uses differential evolution, a global optimizer that searches
        the full prior-bounded space and does not require a starting point. This
        makes it robust on real data where the posterior peak may be far from
        random prior draws.

        If ``initial_sample`` is provided, a single local minimization is run
        from that starting point using ``self.minimization_method``.

        When ``minimization_method`` is not ``'differential_evolution'`` and no
        ``initial_sample`` is given, the legacy multi-start Nelder-Mead strategy
        is used: ``n_prior_samples`` random prior draws are each used as starting
        points for a local optimizer and the best result is returned.
        """
        if initial_sample:
            logger.info(
                f"Maximising the likelihood from initial sample {initial_sample}"
            )
            minout = self._maximize_likelihood_from_initial_sample(initial_sample)
        elif self.minimization_method == "differential_evolution":
            logger.info("Maximising the likelihood using differential evolution")
            minout = self._maximize_likelihood_differential_evolution()
            logger.info(f"Differential evolution result: {minout.message}")
        else:
            logger.info(
                f"Maximising the likelihood using {self.n_prior_samples} prior samples"
            )
            max_logL = -np.inf
            logL_list = []
            successes = 0
            for sample in tqdm.tqdm(self.prior_samples):
                out = self._maximize_likelihood_from_initial_sample(sample)
                logL = -out.fun
                logL_list.append(logL)
                if out.success:
                    successes += 1
                if logL > max_logL:
                    max_logL = logL
                    minout = out

            if np.isinf(max_logL):
                raise ValueError("Maximisation of the likelihood failed")

            logger.info(
                f"Finished with {100 * successes / self.n_prior_samples:.1f}% success rate | "
                f"Maximum log-likelihood {max_logL:.4f} | "
                f"(max-mu)/sigma= {(max_logL - np.mean(logL_list)) / np.std(logL_list):.2f}"
            )

        self.minimization_metadata = minout
        logger.info(f"Maximum likelihood estimation: {minout.message}")
        return {key: val for key, val in zip(self.parameter_names, minout.x)}
