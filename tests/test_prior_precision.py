"""Tests for the bound on the prior's contribution to the precision.

``_prior_precision_diag`` differences ``-d^2 log pi / dtheta^2`` at the MAP.
Pointwise that curvature is unbounded in both directions, and on a real GW prior
both bites: an ``AlignedSpin`` density diverges logarithmically at chi = 0, the
MAP of a log-posterior is drawn onto that cusp, and differencing there returns
~1.5e7 -- which collapsed chi_2's proposal width by a factor of ~800 on the HLV
example, with the likelihood Fisher contributing nothing to that number.  On the
flank of the same cusp the curvature goes negative, subtracting information.

The bound is ``1 / Var(prior)``: a prior cannot pin a parameter down more
tightly than its own spread, nor supply negative information.  It has no free
parameter, so these tests assert the property rather than a tuned value.
"""

import bilby
import numpy as np
import pytest
from bilby.core.prior import Normal, PriorDict, Uniform
from bilby.gw.prior import AlignedSpin

from bilby_laplace.laplace import LaplacePosteriorEstimator


def _estimator(x_prior, gaussian_likelihood):
    """Estimator whose first sampled parameter carries *x_prior*."""
    priors = PriorDict(dict(x=x_prior, y=Uniform(-5, 5, "y")))
    return LaplacePosteriorEstimator(gaussian_likelihood, priors)


@pytest.fixture
def aligned_spin_estimator(gaussian_likelihood):
    return _estimator(AlignedSpin(name="x", a_prior=Uniform(minimum=0, maximum=0.99)), gaussian_likelihood)


def test_cusp_does_not_blow_up_the_precision(aligned_spin_estimator):
    """The regression: at chi = 0 this used to return ~1.5e7."""
    est = aligned_spin_estimator
    cap = est._prior_precision_cap("x", est.priors_dict["x"])

    precision = est._prior_precision_diag({"x": 0.0, "y": 0.0})

    assert np.isfinite(precision[0])
    assert precision[0] <= cap * (1.0 + 1e-9)
    # The bound is ~9 for this prior; the unclamped value was six orders up.
    assert precision[0] < 1e3


@pytest.mark.parametrize("x", [0.0, 1e-6, 1e-3, 0.05, 0.2, 0.5, 0.9])
def test_precision_stays_within_the_bound_everywhere(aligned_spin_estimator, x):
    """Both pathologies at once: never negative, never above 1/Var."""
    est = aligned_spin_estimator
    cap = est._prior_precision_cap("x", est.priors_dict["x"])

    precision = est._prior_precision_diag({"x": x, "y": 0.0})

    assert np.isfinite(precision[0])
    assert 0.0 <= precision[0] <= cap * (1.0 + 1e-9)


def test_negative_curvature_is_clamped_to_zero(aligned_spin_estimator):
    """Somewhere on the cusp's flank the raw curvature is negative.

    A negative prior precision subtracts information from the Fisher and drives
    the precision matrix towards indefiniteness, which is how a naive cap on the
    upper side alone produced non-positive variances.
    """
    est = aligned_spin_estimator
    flank = [0.02, 0.05, 0.1, 0.15]

    values = [est._prior_precision_diag({"x": x, "y": 0.0})[0] for x in flank]

    assert all(v >= 0.0 for v in values)


def test_flat_prior_contributes_nothing(gaussian_likelihood):
    """A uniform prior has no curvature, so it must add exactly zero."""
    est = _estimator(Uniform(-5, 5, "x"), gaussian_likelihood)

    precision = est._prior_precision_diag({"x": 1.0, "y": 0.0})

    assert precision[0] == pytest.approx(0.0, abs=1e-6)


def test_gaussian_prior_keeps_its_own_precision(gaussian_likelihood):
    """The bound must not clip a legitimate value.

    For a Normal prior the curvature is exactly ``1/sigma^2`` and the bound is
    also ``1/Var`` -- they coincide, so the clamp is a no-op up to the Monte
    Carlo error of the variance estimate.
    """
    sigma = 0.5
    est = _estimator(Normal(mu=0.0, sigma=sigma, name="x"), gaussian_likelihood)

    precision = est._prior_precision_diag({"x": 0.0, "y": 0.0})

    assert precision[0] == pytest.approx(1.0 / sigma**2, rel=0.05)


def test_cap_is_the_prior_inverse_variance(gaussian_likelihood):
    est = _estimator(Uniform(-5, 5, "x"), gaussian_likelihood)

    cap = est._prior_precision_cap("x", est.priors_dict["x"])

    assert cap == pytest.approx(1.0 / (10.0**2 / 12.0), rel=0.05)


def test_cap_is_cached_per_parameter(gaussian_likelihood):
    """A multi-mode search re-evaluates the precision once per mode; the bound
    depends only on the prior, so it must not be resampled each time."""
    est = _estimator(Uniform(-5, 5, "x"), gaussian_likelihood)

    first = est._prior_precision_cap("x", est.priors_dict["x"])
    est.priors_dict["x"] = Uniform(-500, 500, "x")  # would give a very different cap
    second = est._prior_precision_cap("x", est.priors_dict["x"])

    assert second == first


def test_unsamplable_prior_is_left_unbounded(gaussian_likelihood):
    """Rather than guess a bound, fall back to the previous behaviour."""
    est = _estimator(Uniform(-5, 5, "x"), gaussian_likelihood)

    class _Unsamplable:
        def sample(self, *args, **kwargs):
            raise RuntimeError("no sampling here")

    assert est._prior_precision_cap("weird", _Unsamplable()) == np.inf


def test_marginalized_names_are_bounded_too(gaussian_likelihood):
    """The waveform path evaluates marginalised priors through the same helper
    with an explicit ``names`` list, so the bound has to apply there as well."""
    priors = PriorDict(dict(x=Uniform(-5, 5, "x"), y=Uniform(-5, 5, "y")))
    est = LaplacePosteriorEstimator(gaussian_likelihood, priors)
    est.likelihood.priors = {"z": AlignedSpin(name="z", a_prior=Uniform(minimum=0, maximum=0.99))}

    precision = est._prior_precision_diag({"z": 0.0}, names=["z"])

    assert np.isfinite(precision[0])
    assert 0.0 <= precision[0] <= est._prior_precision_cap("z", est.likelihood.priors["z"]) * (1.0 + 1e-9)


def test_bilby_import_is_available():
    """Guard against the AlignedSpin import silently changing module."""
    assert bilby.gw.prior.AlignedSpin is AlignedSpin
