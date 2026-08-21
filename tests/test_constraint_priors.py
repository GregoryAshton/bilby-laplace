"""Tests for ``Constraint`` prior support.

A ``Constraint`` prior bounds a *derived* quantity -- ``mass_1``/``mass_2``
computed from a sampled ``chirp_mass`` and ``mass_ratio``, say.  bilby routes
it through the ``PriorDict``'s ``conversion_function``, so it cannot be
expressed as a per-parameter range and no product of marginals can see it:
``PriorDict.ln_prob`` is the only thing that applies both.

The fixture below is a deliberately small stand-in for the GW case: two
sampled parameters ``x`` and ``y`` and a constraint on their sum, which is
exactly the structure of the mass case (derived quantity, non-rectangular
allowed region) without a waveform in sight.
"""

import numpy as np
import pytest
from bilby.core.prior import Constraint, PriorDict, Uniform

from bilby_laplace.laplace import LaplacePosteriorEstimator

# The constraint keeps x + y <= SUM_MAX, cutting the corner off the [0,4]^2
# box.  The allowed region is a triangle: no per-parameter range describes it.
SUM_MAX = 4.0


def _conversion(sample):
    out = sample.copy()
    out["total"] = out["x"] + out["y"]
    return out


@pytest.fixture
def constrained_priors():
    return PriorDict(
        dictionary=dict(
            x=Uniform(0, 4, "x"),
            y=Uniform(0, 4, "y"),
            total=Constraint(minimum=0, maximum=SUM_MAX, name="total"),
        ),
        conversion_function=_conversion,
    )


class FlatLikelihood:
    """Constant log-likelihood, so the prior alone decides the support."""

    def log_likelihood(self, parameters=None):
        return 0.0


@pytest.fixture
def constrained_estimator(constrained_priors):
    return LaplacePosteriorEstimator(
        FlatLikelihood(),
        constrained_priors,
        minimization_method="Nelder-Mead",
        n_prior_samples=5,
    )


# x + y = 7 > 4: inside the box, outside the constraint.
VIOLATING = dict(x=3.5, y=3.5)
# x + y = 2 <= 4: allowed.
ALLOWED = dict(x=1.0, y=1.0)


def test_constraint_keys_are_not_sampled(constrained_estimator):
    """The constraint is on a derived quantity, so it is never a coordinate."""
    assert constrained_estimator.parameter_names == ["x", "y"]
    assert "total" not in constrained_estimator.priors_dict


def test_log_prior_rejects_a_constraint_violation(constrained_estimator):
    """The regression this suite exists for: a hand-rolled product of
    marginals returns a finite value here, because neither marginal is out of
    range."""
    assert np.isfinite(constrained_estimator.log_prior(ALLOWED))
    assert constrained_estimator.log_prior(VIOLATING) == -np.inf


def test_log_prior_still_rejects_out_of_range(constrained_estimator):
    """Delegating to ``ln_prob`` must not lose the per-parameter support."""
    assert constrained_estimator.log_prior(dict(x=99.0, y=1.0)) == -np.inf


def test_log_posterior_from_array_rejects_a_violation(constrained_estimator):
    x = np.array([[ALLOWED["x"], VIOLATING["x"]], [ALLOWED["y"], VIOLATING["y"]]])
    got = constrained_estimator.log_posterior_from_array(x)
    assert np.isfinite(got[0])
    assert got[1] == -np.inf


def test_constraint_mask_is_vectorised(constrained_estimator):
    x = np.array([[1.0, 3.5, 0.5], [1.0, 3.5, 0.5]])  # sums 2, 7, 1
    assert list(constrained_estimator.constraint_mask(x)) == [True, False, True]


def test_constraint_mask_is_all_true_without_constraints(estimator):
    """An unconstrained prior must not pay for, or be filtered by, this."""
    x = np.array([[1.0, 2.0], [-0.5, 0.5]])
    assert list(estimator.constraint_mask(x)) == [True, True]


def test_likelihood_array_returns_minus_inf_for_a_violation(constrained_estimator):
    """The likelihood is flat at 0.0, so a -inf can only come from the support
    test -- and only the constraint can reject this in-box point."""
    x = np.column_stack([[ALLOWED["x"], ALLOWED["y"]], [VIOLATING["x"], VIOLATING["y"]]])
    got = constrained_estimator.log_likelihood_from_array(x)
    assert got[0] == 0.0
    assert got[1] == -np.inf


def test_likelihood_array_single_vector_shape(constrained_estimator):
    """The ``(N_params,)`` input contract still returns a scalar."""
    allowed = constrained_estimator.log_likelihood_from_array(np.array([ALLOWED["x"], ALLOWED["y"]]))
    violating = constrained_estimator.log_likelihood_from_array(np.array([VIOLATING["x"], VIOLATING["y"]]))
    assert float(allowed) == 0.0
    assert float(violating) == -np.inf


def test_clip_to_bounds_cannot_rescue_a_constraint(constrained_estimator):
    """Clipping projects onto the box; the constraint region is not a box, so
    an in-box violation must still be rejected."""
    x = np.array([VIOLATING["x"], VIOLATING["y"]])
    got = constrained_estimator.log_likelihood_from_array(x, clip_to_bounds=True)
    assert float(got) == -np.inf


def test_clip_to_bounds_still_clips_out_of_range(constrained_estimator):
    """An out-of-box point that is allowed once clipped is evaluated."""
    x = np.array([-5.0, 1.0])  # clips to x=0 -> total=1, allowed
    got = constrained_estimator.log_likelihood_from_array(x, clip_to_bounds=True)
    assert float(got) == 0.0


def test_map_starting_points_satisfy_the_constraint(constrained_estimator):
    """``sample_subset`` ignores constraints; ``sample_subset_constrained``
    does not.  A start in the forbidden region gives the optimiser a flat
    -inf."""
    for sample in constrained_estimator.prior_samples:
        assert sample["x"] + sample["y"] <= SUM_MAX


def test_map_search_respects_the_constraint(constrained_priors):
    """With a likelihood peaked in the forbidden corner, the MAP must stay in
    the allowed region rather than chase the peak."""

    class CornerPeakedLikelihood:
        """Maximised at x=y=3.5, i.e. total=7, outside the constraint."""

        def log_likelihood(self, parameters=None):
            p = parameters
            return -((p["x"] - 3.5) ** 2 + (p["y"] - 3.5) ** 2)

    est = LaplacePosteriorEstimator(
        CornerPeakedLikelihood(),
        constrained_priors,
        minimization_method="differential_evolution",
        seed=1234,
    )
    map_sample = est.get_MAP_sample()
    assert map_sample["x"] + map_sample["y"] <= SUM_MAX + 1e-6


def test_evidence_uses_the_constrained_normalisation(constrained_estimator, constrained_priors):
    """``log_evidence_laplace`` must be on the same normalisation as the
    rejection/importance/SMC evidences, which come from ``ln_prob``."""
    covariance = np.eye(2) * 0.01
    log_z = constrained_estimator.log_evidence_laplace(ALLOWED, covariance)

    ratio = constrained_priors.normalize_constraint_factor(("x", "y"))
    assert ratio > 1.0  # the constraint really does remove prior volume

    naive_log_pi = sum(np.log(constrained_priors[k].prob(ALLOWED[k])) for k in ("x", "y"))
    sign, log_det = np.linalg.slogdet(covariance)
    naive = 0.0 + naive_log_pi + 0.5 * 2 * np.log(2 * np.pi) + 0.5 * log_det

    assert log_z == pytest.approx(naive + np.log(ratio))
