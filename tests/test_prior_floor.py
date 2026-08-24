"""Tests for the prior bound on the waveform precision.

A direction the data does not constrain must fall back to the prior's width.
The scale used for that fallback matters: ``width / sqrt(12)`` is the std of a
*uniform* prior, so it overstates the width of any non-uniform prior (e.g. a
``Sine`` or ``AlignedSpin``), which inflates the fallback beyond the true
prior width.  Whitening by the sampled prior covariance instead makes the
floor exact for any prior.
"""

import bilby
import numpy as np
import pytest
from bilby.core.prior import PriorDict, Sine, Uniform

from bilby_laplace.laplace import LaplacePosteriorEstimator


@pytest.fixture
def sine_estimator(gaussian_likelihood):
    """One Sine prior (non-uniform) and one Uniform, so the two scales differ."""
    priors = PriorDict(dict(x=Sine(name="x"), y=Uniform(-5, 5, "y")))
    return LaplacePosteriorEstimator(gaussian_likelihood, priors)


def test_prior_standard_deviations_match_the_priors(sine_estimator):
    sd = sine_estimator._prior_standard_deviations()

    # Sine on [0, pi] has variance pi^2/4 - 2; Uniform(-5,5) has 100/12.
    assert sd[0] == pytest.approx(np.sqrt(np.pi**2 / 4 - 2), rel=0.02)
    assert sd[1] == pytest.approx(10 / np.sqrt(12), rel=0.02)


def test_uniform_proxy_overstates_a_sine_prior(sine_estimator):
    """The reason for the change, stated as a test."""
    sd = sine_estimator._prior_standard_deviations()
    proxy = np.pi / np.sqrt(12.0)

    assert proxy / sd[0] == pytest.approx(1.32, rel=0.02)


def test_unconstrained_direction_falls_back_to_the_prior(sine_estimator):
    """A precision of ~0 must come back as the prior covariance, not wider."""
    floored = sine_estimator._floor_precision_at_prior(np.diag([1e-12, 1e-12]))
    cov = np.linalg.inv(floored)
    prior_cov = np.diag(sine_estimator._prior_standard_deviations() ** 2)

    np.testing.assert_allclose(cov, prior_cov, rtol=0.05, atol=1e-8)


def test_well_constrained_directions_are_untouched(sine_estimator):
    """Precision far above the prior's must pass through unchanged."""
    tight = np.diag([1e6, 1e6])

    floored = sine_estimator._floor_precision_at_prior(tight)

    np.testing.assert_allclose(floored, tight, rtol=1e-6)


def test_no_marginal_variance_exceeds_the_prior(sine_estimator):
    """The bound is on the marginals: diag(cov) <= diag(prior_cov)."""
    rng = np.random.default_rng(0)
    prior_var = sine_estimator._prior_standard_deviations() ** 2
    for _ in range(20):
        a = rng.normal(size=(2, 2))
        p = a @ a.T + 1e-9 * np.eye(2)
        cov = np.linalg.inv(sine_estimator._floor_precision_at_prior(p))

        assert np.all(np.diag(cov) <= prior_var * (1 + 1e-6))


def test_the_bound_never_narrows_a_marginal_already_inside_the_prior(sine_estimator):
    """The property the eigenvalue floor did not have, and the reason it changed.

    Flooring rescaled *eigenvalues* at 1 only ever adds precision, so it could
    only shrink marginals -- including ones already narrower than the prior,
    which it would narrow further with no justification. Capping the
    marginals instead must leave those untouched.
    """
    rng = np.random.default_rng(1)
    prior_var = sine_estimator._prior_standard_deviations() ** 2
    for _ in range(20):
        a = rng.normal(size=(2, 2))
        p = a @ a.T + 1e-9 * np.eye(2)
        before = np.diag(np.linalg.inv(p))
        after = np.diag(np.linalg.inv(sine_estimator._floor_precision_at_prior(p)))

        inside = before <= prior_var
        np.testing.assert_allclose(after[inside], before[inside], rtol=1e-6)


def test_correlations_survive_the_cap(sine_estimator):
    """The cap is a diagonal congruence, so it rescales rows and columns
    together and cannot change the correlation matrix."""
    p = np.linalg.inv(np.array([[400.0, 18.0], [18.0, 4.0]]))  # both marginals over prior
    before = np.linalg.inv(p)
    after = np.linalg.inv(sine_estimator._floor_precision_at_prior(p))

    def corr(c):
        d = np.sqrt(np.diag(c))
        return c / np.outer(d, d)

    np.testing.assert_allclose(corr(after), corr(before), rtol=1e-6)


def test_result_is_symmetric(sine_estimator):
    out = sine_estimator._floor_precision_at_prior(np.array([[2.0, 0.3], [0.29, 1.0]]))

    np.testing.assert_allclose(out, out.T, rtol=1e-12)


def test_prior_standard_deviations_are_cached(sine_estimator):
    first = sine_estimator._prior_standard_deviations()
    sine_estimator.priors_dict = {}  # would break a second evaluation

    np.testing.assert_array_equal(sine_estimator._prior_standard_deviations(), first)


def test_non_finite_prior_width_skips_the_bound(sine_estimator):
    sine_estimator.prior_width_dict["x"] = np.inf
    p = np.diag([1e-12, 1e-12])

    np.testing.assert_array_equal(sine_estimator._floor_precision_at_prior(p), p)


def test_uniform_scale_is_exact(gaussian_likelihood):
    """The deterministic grid must reproduce the closed form, not approximate it
    -- a sampled estimate carried ~0.3% Monte-Carlo error and made the proposal
    covariance jitter between runs."""
    priors = PriorDict(dict(x=Uniform(-5, 5, "x"), y=Uniform(-5, 5, "y")))
    est = LaplacePosteriorEstimator(gaussian_likelihood, priors)

    sd = est._prior_standard_deviations()

    assert sd[0] == pytest.approx(10 / np.sqrt(12), rel=1e-6)


def test_scale_is_deterministic(gaussian_likelihood):
    """Two estimators built from the same priors must agree exactly."""

    def build():
        return LaplacePosteriorEstimator(gaussian_likelihood, PriorDict(dict(x=Sine(name="x"), y=Uniform(-5, 5, "y"))))

    bilby.core.utils.random.seed(1)
    a = build()._prior_standard_deviations()
    bilby.core.utils.random.seed(2)
    b = build()._prior_standard_deviations()

    np.testing.assert_array_equal(a, b)
