"""The MAP search must cross a periodic parameter's seam, not stop at it.

A periodic parameter's prior edges are the same point.  Bounded there, the
local Nelder-Mead steps (the polish after each differential evolution, and
the search from an initial sample) get pinned against the edge when the peak
lies just across it.  map_validation found exactly this on a GW posterior: a
polish stuck at phi_12 = 0 and psi = pi/2 while the top of the same hill sat
across the wrap.
"""

import bilby
import numpy as np
import pytest

from bilby_laplace.laplace import LaplacePosteriorEstimator

PEAK = (2 * np.pi - 0.1, 0.3)


class _SeamLikelihood(bilby.core.likelihood.Likelihood):
    """A narrow peak in a periodic *phi*, just below the top of its range."""

    def __init__(self, sigma=0.05):
        super().__init__(parameters=dict(phi=None, y=None))
        self.sigma = sigma

    def log_likelihood(self, parameters=None):
        p = parameters if parameters is not None else self.parameters
        dphi = np.angle(np.exp(1j * (p["phi"] - PEAK[0])))  # circular distance
        return -0.5 * (dphi**2 + (p["y"] - PEAK[1]) ** 2) / self.sigma**2


@pytest.fixture
def seam_priors():
    return bilby.core.prior.PriorDict(
        dict(
            phi=bilby.core.prior.Uniform(0, 2 * np.pi, "phi", boundary="periodic"),
            y=bilby.core.prior.Uniform(-1, 1, "y"),
        )
    )


def _circular_error(phi):
    return abs(np.angle(np.exp(1j * (phi - PEAK[0]))))


def test_a_local_search_crosses_the_seam(seam_priors):
    """Started just above phi = 0, the peak is 0.15 away across the wrap."""
    est = LaplacePosteriorEstimator(_SeamLikelihood(), seam_priors, use_unit_cube=False)
    found = est.get_MAP_sample(initial_sample=dict(phi=0.05, y=0.0))
    assert _circular_error(found["phi"]) < 1e-2
    assert 0 <= found["phi"] <= 2 * np.pi


def test_the_global_search_returns_a_wrapped_in_range_map(seam_priors):
    est = LaplacePosteriorEstimator(
        _SeamLikelihood(), seam_priors, minimization_method="differential_evolution", use_unit_cube=False, seed=1
    )
    found = est.get_MAP_sample()
    assert _circular_error(found["phi"]) < 1e-2
    assert 0 <= found["phi"] <= 2 * np.pi


def test_wrap_periodic_handles_single_points_and_batches(seam_priors):
    est = LaplacePosteriorEstimator(_SeamLikelihood(), seam_priors, use_unit_cube=False)
    np.testing.assert_allclose(est._wrap_periodic([-0.1, 5.0]), [2 * np.pi - 0.1, 5.0])  # y untouched
    batch = est._wrap_periodic(np.array([[-0.1, 7.0, 1.0], [5.0, -5.0, 0.0]]))
    np.testing.assert_allclose(batch, [[2 * np.pi - 0.1, 7.0 - 2 * np.pi, 1.0], [5.0, -5.0, 0.0]])
