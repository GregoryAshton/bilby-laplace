"""Tests for wrapping of periodic parameters in the proposal.

Truncation renormalises a Gaussian inside the prior range and puts *zero* mass
on the far side of the boundary.  For an angle that is wrong: the mass below the
lower edge belongs just under the upper edge.  On the precessing BBH example the
``phi_jl`` MAP sits at 0.835 with a proposal sigma near 1.4, so about a quarter
of its Gaussian falls below zero; truncating lost that tail, leaving 1.6% of the
posterior above 0.85*2pi where dynesty and the prior-seeded control both have
~12%.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from bilby_laplace.sampler import TruncatedMVNProposal

TWO_PI = 2 * np.pi
LOWER = np.array([0.0, 0.0])
UPPER = np.array([TWO_PI, TWO_PI])
# Mean near the lower edge with a sigma comparable to the range -- the regime
# where truncation and wrapping differ most.
MEAN = np.array([0.835, 3.14])
COV = np.diag([1.4**2, 1.4**2])


def _proposal(periodic):
    return TruncatedMVNProposal(MEAN, COV, lower=LOWER, upper=UPPER, periodic=periodic)


def test_wrapping_puts_mass_beyond_the_far_edge():
    """The regression: truncation leaves the upper edge empty."""
    wrapped = _proposal([True, False]).sample(20000)[:, 0]
    truncated = _proposal([False, False]).sample(20000)[:, 0]

    assert np.mean(wrapped > 0.85 * TWO_PI) > 0.05
    assert np.mean(truncated > 0.85 * TWO_PI) < 0.005


def test_draws_stay_inside_the_prior_range():
    x = _proposal([True, True]).sample(20000)

    assert x.min() >= 0.0
    assert x.max() <= TWO_PI


def test_wrapped_density_is_normalised():
    p = _proposal([True, False])
    # Integrate the 1-D wrapped marginal over its period.
    integral, _ = quad(lambda v: np.exp(p._wrapped_logpdf(np.array([v]), 0))[0], 0.0, TWO_PI, limit=200)

    assert integral == pytest.approx(1.0, abs=1e-6)


def test_wrapped_density_is_periodic():
    p = _proposal([True, False])
    eps = 1e-9

    near_lower = p._wrapped_logpdf(np.array([eps]), 0)[0]
    near_upper = p._wrapped_logpdf(np.array([TWO_PI - eps]), 0)[0]

    assert near_lower == pytest.approx(near_upper, abs=1e-6)


def test_samples_match_the_density():
    """Sampling and logpdf must describe the same distribution."""
    p = _proposal([True, False])
    x = p.sample(40000)[:, 0]
    edges = np.linspace(0, TWO_PI, 13)
    empirical, _ = np.histogram(x, bins=edges, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    predicted = np.exp(p._wrapped_logpdf(centres, 0))

    np.testing.assert_allclose(empirical, predicted, atol=0.02)


def test_non_periodic_dimension_is_unchanged():
    """Wrapping one coordinate must not disturb the others."""
    both = _proposal([True, False]).sample(20000)[:, 1]

    assert both.min() >= 0.0 and both.max() <= TWO_PI
    assert both.mean() == pytest.approx(MEAN[1], abs=0.05)


def test_periodic_defaults_to_none_and_truncates():
    p = TruncatedMVNProposal(MEAN, COV, lower=LOWER, upper=UPPER)

    assert not p._periodic.any()


def test_periodic_mask_shape_is_validated():
    with pytest.raises(ValueError, match="one entry per parameter"):
        TruncatedMVNProposal(MEAN, COV, lower=LOWER, upper=UPPER, periodic=[True])


def test_logpdf_sums_over_dimensions():
    p = _proposal([True, False])
    x = np.array([[1.0, 3.0]])

    total = p.logpdf(x)[0]
    expected = p._wrapped_logpdf(np.array([1.0]), 0)[0] + p._dists[1].logpdf(3.0)

    assert total == pytest.approx(expected)
