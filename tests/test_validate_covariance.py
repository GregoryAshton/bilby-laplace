"""Tests for the likelihood probe that validates a Laplace covariance.

The probe exists because the Fisher is not the posterior curvature: ``F =
(dh|dh)`` is the expected information under the linear-signal approximation,
and that approximation breaks down first along the *best-measured* directions,
where the quadratic expansion is valid over the smallest range. So the
covariance can claim a constraint the likelihood does not have, and only a
direct probe of the likelihood can catch it.

Two failure modes of a fixed 1-sigma probe are pinned here, both measured on a
13-parameter precessing-BBH posterior:

* a direction whose sigma is orders of magnitude below the prior width cannot
  produce a resolvable drop at 1 sigma however wrong it is, and used to be
  waved through as "unresolved";
* a 1-sigma step can land outside the prior on *both* sides, and the axis used
  to be skipped entirely -- five of thirteen axes went unvalidated there.
"""

import bilby
import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform

from bilby_laplace.laplace import LaplacePosteriorEstimator
from bilby_laplace.sampler import Laplace

from conftest import MU, TRUE_COV


@pytest.fixture
def validator(sampler, estimator):
    """``(fn, estimator, mean)`` -- the probe, ready to call with a covariance."""
    return sampler._validate_covariance, estimator, MU.copy()


def test_exact_covariance_is_left_alone(validator):
    """The Laplace approximation is exact for a Gaussian, so nothing to widen."""
    validate, estimator, mean = validator

    out = validate(estimator, mean, TRUE_COV.copy())

    assert out == pytest.approx(TRUE_COV, rel=0.05)


def test_a_spuriously_narrow_direction_is_widened(validator):
    """The gap that mattered: a covariance far too narrow must be corrected.

    A 1e-6 shrink in variance is 1e-3 in sigma, comparable to the ~30-50x
    under-estimate measured on the GW problem but deliberately worse.
    """
    validate, estimator, mean = validator

    out = validate(estimator, mean, TRUE_COV * 1e-6)

    # Recovered, not merely nudged: for a genuinely Gaussian likelihood the
    # inferred width is exact up to the discreteness of the doubling.
    assert out == pytest.approx(TRUE_COV, rel=0.3)


def test_the_fixed_step_probe_could_not_have_resolved_it(validator):
    """Why the old code missed it, pinned so the regression cannot come back.

    At the *claimed* 1 sigma the drop is far below ``_MIN_VALIDATION_DROP``, so
    the old fixed-step probe took its "unresolved, leave alone" branch -- which
    is exactly where a wrongly narrow direction hides.
    """
    _, estimator, mean = validator
    narrow = TRUE_COV * 1e-6

    prior_sd = np.asarray(estimator._prior_standard_deviations(), dtype=float)
    scaled = narrow / np.outer(prior_sd, prior_sd)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (scaled + scaled.T))
    peak = float(estimator.log_likelihood_from_array(mean))

    for i in range(len(eigvals)):
        step = np.sqrt(eigvals[i]) * (prior_sd * eigvecs[:, i])
        drop = peak - float(estimator.log_likelihood_from_array(mean + step))
        assert drop < Laplace._MIN_VALIDATION_DROP


def test_an_out_of_bounds_probe_is_measured_not_skipped(validator, monkeypatch):
    """A 1-sigma step outside the prior must shrink until it fits.

    The old code hit ``if not drops: continue`` and never looked at the axis
    again. Counting evaluations is the direct way to assert the shrink loop
    runs, since for a too-wide direction the *outcome* is correctly "leave
    alone" either way -- the probe never shrinks a covariance.
    """
    validate, estimator, mean = validator
    calls = []
    original = estimator.log_likelihood_from_array
    monkeypatch.setattr(
        estimator,
        "log_likelihood_from_array",
        lambda x: (calls.append(np.asarray(x, dtype=float).copy()), original(x))[1],
    )

    # Sigma ~30 on a prior of half-width 5: both +/- 1 sigma are far outside.
    out = validate(estimator, mean, np.diag([900.0, 900.0]))

    probes = [c for c in calls if not np.allclose(c, mean)]
    assert probes, "no probe steps were evaluated at all"
    assert any(
        np.all(np.abs(c) <= 5.0) for c in probes
    ), "every probe stayed outside the prior; the step never shrank"
    assert np.all(np.isfinite(out))


def test_a_boundary_shrunk_probe_does_not_narrow_the_axis(validator):
    """A direction at prior scale must not be "widened" to its shrunken probe.

    Caught on a real run: sigma just under ``_PROBE_UNCONSTRAINED_SIGMA`` put
    the 1-sigma step outside the prior, the probe halved its way inward by ~250x
    to get back in, resolved no drop there, and the axis was then classed as
    wrongly narrow and "widened" to a step far *below* what it already claimed.
    Having had to shrink is itself evidence the direction is at prior scale.
    """
    validate, estimator, mean = validator
    prior_sd = np.asarray(estimator._prior_standard_deviations(), dtype=float)

    # y at 0.4 of prior width -- under the 0.5 threshold, but its 1-sigma step
    # still leaves the box; x kept at its true width.
    cov = np.diag([TRUE_COV[0, 0], (0.4 * prior_sd[1]) ** 2])

    out = validate(estimator, mean, cov.copy())

    assert out[1, 1] >= cov[1, 1] * 0.99, "a prior-scale direction was narrowed"


def test_a_genuinely_unconstrained_direction_is_not_widened():
    """The invariant the original guard protected, kept intact.

    A flat likelihood along a direction already at prior width is true but
    uninformative. Widening it anyway would push it past the prior on noise.
    """

    class FlatInY(bilby.core.likelihood.Likelihood):
        """Constrains ``x`` only; ``y`` is exactly flat."""

        def __init__(self):
            super().__init__(parameters=dict(x=None, y=None))

        def log_likelihood(self, parameters=None):
            p = parameters if parameters is not None else self.parameters
            return -0.5 * (p["x"] / 0.3) ** 2

    priors = PriorDict(dict(x=Uniform(-5, 5, "x"), y=Uniform(-5, 5, "y")))
    estimator = LaplacePosteriorEstimator(FlatInY(), priors)
    sampler = Laplace(likelihood=FlatInY(), priors=priors, outdir="/tmp", label="flat")

    prior_sd = float(np.asarray(estimator._prior_standard_deviations())[1])
    cov = np.diag([0.3**2, prior_sd**2])

    out = sampler._validate_covariance(estimator, np.zeros(2), cov)

    assert out[1, 1] == pytest.approx(cov[1, 1], rel=0.05)
