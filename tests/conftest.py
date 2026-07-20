"""Shared fixtures for the bilby_laplace test suite.

The central fixture is a 2-D correlated Gaussian likelihood with a known mean
and covariance (mirroring ``examples/gaussian/run.py``).  Because the Laplace
approximation is exact for a Gaussian posterior, the analytic mean/covariance
double as ground truth for the maths in :mod:`bilby_laplace.laplace`.
"""

import bilby
import numpy as np
import pytest

# Ground-truth parameters of the correlated Gaussian likelihood.
MU = np.array([1.0, -0.5])
SIGMA_X, SIGMA_Y, RHO = 0.3, 0.5, 0.7
TRUE_COV = np.array(
    [
        [SIGMA_X**2, RHO * SIGMA_X * SIGMA_Y],
        [RHO * SIGMA_X * SIGMA_Y, SIGMA_Y**2],
    ]
)
PRIOR_MIN, PRIOR_MAX = -5.0, 5.0


class CorrelatedGaussianLikelihood(bilby.core.likelihood.Likelihood):
    """Normalised 2-D correlated Gaussian in parameters ``x`` and ``y``."""

    def __init__(self, mu=MU, cov=TRUE_COV):
        super().__init__(parameters=dict(x=None, y=None))
        self.mu = np.asarray(mu, dtype=float)
        self._inv_cov = np.linalg.inv(cov)
        self._log_norm = -0.5 * np.log(np.linalg.det(2 * np.pi * cov))

    def log_likelihood(self, parameters=None):
        p = parameters if parameters is not None else self.parameters
        d = np.array([p["x"], p["y"]]) - self.mu
        return -0.5 * d @ self._inv_cov @ d + self._log_norm


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed bilby's global RNG so every test is deterministic."""
    bilby.core.utils.random.seed(1234)


@pytest.fixture
def gaussian_likelihood():
    return CorrelatedGaussianLikelihood()


@pytest.fixture
def gaussian_priors():
    return bilby.core.prior.PriorDict(
        dict(
            x=bilby.core.prior.Uniform(PRIOR_MIN, PRIOR_MAX, "x"),
            y=bilby.core.prior.Uniform(PRIOR_MIN, PRIOR_MAX, "y"),
        )
    )


@pytest.fixture
def gaussian_injection():
    return {"x": float(MU[0]), "y": float(MU[1])}


@pytest.fixture
def true_cov():
    return TRUE_COV.copy()


@pytest.fixture
def estimator(gaussian_likelihood, gaussian_priors):
    """A ready-built estimator on the Gaussian (unit-cube path, default)."""
    from bilby_laplace.laplace import LaplacePosteriorEstimator

    return LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)


@pytest.fixture
def sampler(gaussian_likelihood, gaussian_priors, tmp_path):
    """A ``Laplace`` sampler instance for exercising helper methods.

    ``run_sampler`` is not called; this is only for the small pure helpers
    (covariance validation, evidence, sampling-cov resolution, ...).
    """
    from bilby_laplace.sampler import Laplace

    return Laplace(
        likelihood=gaussian_likelihood,
        priors=gaussian_priors,
        outdir=str(tmp_path),
        label="test",
    )
