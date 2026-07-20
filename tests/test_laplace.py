"""Unit tests for :mod:`bilby_laplace.laplace`.

The Laplace approximation is exact for a Gaussian posterior, so the correlated
Gaussian fixture provides analytic ground truth for the MAP, covariance, and
evidence.
"""

import bilby
import numpy as np
import pytest
from conftest import MU, PRIOR_MAX, PRIOR_MIN, TRUE_COV

from bilby_laplace.laplace import LaplacePosteriorEstimator, array_to_dict


def test_array_to_dict():
    assert array_to_dict(["a", "b"], [1, 2]) == {"a": 1, "b": 2}


def test_parameter_names_default_to_non_fixed(estimator):
    assert estimator.parameter_names == ["x", "y"]
    assert estimator.N == 2


def test_prior_bounds(estimator):
    np.testing.assert_array_equal(estimator.prior_bounds_min, [PRIOR_MIN, PRIOR_MIN])
    np.testing.assert_array_equal(estimator.prior_bounds_max, [PRIOR_MAX, PRIOR_MAX])


# ---------------------------------------------------------------------------
# log-probability evaluation
# ---------------------------------------------------------------------------
def test_log_prior_uniform(estimator):
    # Uniform(-5, 5) has density 1/10 per parameter -> log = 2 * log(0.1).
    lp = estimator.log_prior({"x": 0.0, "y": 0.0})
    assert lp == pytest.approx(2 * np.log(0.1))


def test_log_posterior_is_likelihood_plus_prior(estimator):
    sample = {"x": 1.0, "y": -0.5}
    expected = estimator.log_likelihood(sample) + estimator.log_prior(sample)
    assert estimator.log_posterior(sample) == pytest.approx(expected)


def test_log_posterior_out_of_prior_is_neg_inf(estimator):
    assert estimator.log_posterior({"x": 100.0, "y": 0.0}) == -np.inf


def test_log_likelihood_rejects_bad_input(estimator):
    with pytest.raises(ValueError):
        estimator.log_likelihood([1.0, 2.0])


def test_fixed_parameters_merged(gaussian_priors):
    """Parameters fixed by a DeltaFunction prior are passed to the likelihood."""

    class NeedsZ(bilby.core.likelihood.Likelihood):
        def __init__(self):
            super().__init__(parameters=dict(x=None, z=None))

        def log_likelihood(self, parameters=None):
            p = parameters if parameters is not None else self.parameters
            return p["z"]  # return the fixed value so we can assert on it

    priors = bilby.core.prior.PriorDict(
        dict(
            x=bilby.core.prior.Uniform(-5, 5, "x"),
            z=bilby.core.prior.DeltaFunction(peak=3.0, name="z"),
        )
    )
    est = LaplacePosteriorEstimator(NeedsZ(), priors)
    assert est.parameter_names == ["x"]
    assert est.fixed_parameters == {"z": 3.0}
    assert est.log_likelihood({"x": 0.0}) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# vectorised array evaluation
# ---------------------------------------------------------------------------
def test_log_likelihood_from_array_matches_scalar(estimator):
    x = np.array([1.0, -0.5])
    got = float(estimator.log_likelihood_from_array(x))
    assert got == pytest.approx(estimator.log_likelihood({"x": 1.0, "y": -0.5}))


def test_log_likelihood_from_array_vectorised_shape(estimator):
    # Column-stacked (N_params, N_samples).
    x = np.array([[1.0, 1.1, 0.9], [-0.5, -0.4, -0.6]])
    out = estimator.log_likelihood_from_array(x)
    assert out.shape == (3,)


def test_log_likelihood_from_array_out_of_bounds(estimator):
    assert estimator.log_likelihood_from_array(np.array([100.0, 0.0])) == -np.inf


def test_log_likelihood_from_array_clip_to_bounds(estimator):
    """With clipping, an out-of-bounds point evaluates at the clipped edge."""
    oob = np.array([100.0, 0.0])
    clipped = np.array([PRIOR_MAX, 0.0])
    got = float(estimator.log_likelihood_from_array(oob, clip_to_bounds=True))
    expected = float(estimator.log_likelihood_from_array(clipped))
    assert got == pytest.approx(expected)


def test_log_posterior_from_array(estimator):
    x = np.array([1.0, -0.5])
    assert float(estimator.log_posterior_from_array(x)) == pytest.approx(estimator.log_posterior({"x": 1.0, "y": -0.5}))


# ---------------------------------------------------------------------------
# unit-cube transforms and Jacobian
# ---------------------------------------------------------------------------
def test_unit_cube_round_trip(estimator):
    x = np.array([1.0, -0.5])
    u = estimator._to_unit_cube(x)
    assert np.all((u >= 0) & (u <= 1))
    np.testing.assert_allclose(estimator._from_unit_cube(u), x, atol=1e-9)


def test_unit_cube_midpoint(estimator):
    # Centre of Uniform(-5, 5) maps to u = 0.5.
    u = estimator._to_unit_cube(np.array([0.0, 0.0]))
    np.testing.assert_allclose(u, [0.5, 0.5])


def test_jacobian_diag_uniform(estimator):
    # dtheta/du = width = 10 for Uniform(-5, 5).
    j = estimator._jacobian_diag(np.array([0.0, 0.0]))
    np.testing.assert_allclose(j, [10.0, 10.0])


def test_jacobian_diag_boundary_nudges(estimator):
    """On the prior boundary (p=0) the value is nudged inward, not raised."""
    j = estimator._jacobian_diag(np.array([PRIOR_MAX, 0.0]))
    np.testing.assert_allclose(j, [10.0, 10.0])


# ---------------------------------------------------------------------------
# MAP finding
# ---------------------------------------------------------------------------
def test_get_map_from_initial_sample(estimator):
    m = estimator.get_MAP_sample({"x": 0.9, "y": -0.4})
    assert m["x"] == pytest.approx(MU[0], abs=1e-3)
    assert m["y"] == pytest.approx(MU[1], abs=1e-3)


def test_get_map_differential_evolution(gaussian_likelihood, gaussian_priors):
    est = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors, minimization_method="differential_evolution")
    m = est.get_MAP_sample()
    assert m["x"] == pytest.approx(MU[0], abs=1e-2)
    assert m["y"] == pytest.approx(MU[1], abs=1e-2)


def test_get_map_multistart_nelder_mead(gaussian_likelihood, gaussian_priors):
    est = LaplacePosteriorEstimator(
        gaussian_likelihood,
        gaussian_priors,
        minimization_method="Nelder-Mead",
        n_prior_samples=20,
    )
    m = est.get_MAP_sample()
    assert m["x"] == pytest.approx(MU[0], abs=1e-2)
    assert m["y"] == pytest.approx(MU[1], abs=1e-2)


def test_maximum_likelihood_alias(estimator):
    """Deprecated alias forwards to get_MAP_sample."""
    m = estimator.get_maximum_likelihood_sample({"x": 0.9, "y": -0.4})
    assert m["x"] == pytest.approx(MU[0], abs=1e-3)


# ---------------------------------------------------------------------------
# covariance and evidence (exact for a Gaussian)
# ---------------------------------------------------------------------------
@pytest.fixture
def map_sample():
    return {"x": float(MU[0]), "y": float(MU[1])}


def test_covariance_recovers_truth_unit_cube(estimator, map_sample):
    cov = estimator.calculate_posterior_covariance(map_sample)
    np.testing.assert_allclose(cov, TRUE_COV, atol=1e-3)


def test_covariance_recovers_truth_parameter_space(gaussian_likelihood, gaussian_priors, map_sample):
    est = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors, use_unit_cube=False)
    cov = est.calculate_posterior_covariance(map_sample)
    np.testing.assert_allclose(cov, TRUE_COV, atol=1e-3)


def test_covariance_is_symmetric(estimator, map_sample):
    cov = estimator.calculate_posterior_covariance(map_sample)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_precision_is_inverse_covariance(estimator, map_sample):
    prec = estimator.calculate_posterior_precision(map_sample)
    np.testing.assert_allclose(np.linalg.inv(prec), TRUE_COV, atol=1e-3)


def test_log_evidence_is_exact_for_gaussian(estimator, map_sample):
    """Z = integral(L * pi) = (1/prior_area) since L is normalised -> -log(100)."""
    cov = estimator.calculate_posterior_covariance(map_sample)
    log_z = estimator.log_evidence_laplace(map_sample, cov)
    prior_area = (PRIOR_MAX - PRIOR_MIN) ** 2
    assert log_z == pytest.approx(-np.log(prior_area), abs=1e-3)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def test_sample_array_shape(estimator, map_sample):
    samples = estimator.sample_array(map_sample, n=500)
    assert samples.shape == (500, 2)


def test_sample_dataframe_columns_and_stats(estimator, map_sample):
    df = estimator.sample_dataframe(map_sample, n=20000)
    assert list(df.columns) == ["x", "y"]
    # Drawn from N(MAP, Sigma): recover the mean and marginal variances.
    np.testing.assert_allclose(df.mean().values, MU, atol=0.02)
    np.testing.assert_allclose(df.var().values, np.diag(TRUE_COV), rtol=0.1)


# ---------------------------------------------------------------------------
# construction-time validation
# ---------------------------------------------------------------------------
def test_invalid_fisher_method_raises(gaussian_likelihood, gaussian_priors):
    with pytest.raises(ValueError, match="fisher_method"):
        LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors, fisher_method="bogus")


def test_ill_formed_prior_width_raises(gaussian_likelihood):
    bad_y = bilby.core.prior.Uniform(-5, 5, "y")
    bad_y.maximum = np.nan  # ill-formed bound -> width is nan
    priors = bilby.core.prior.PriorDict(dict(x=bilby.core.prior.Uniform(-5, 5, "x"), y=bad_y))
    with pytest.raises(ValueError, match="Prior width"):
        LaplacePosteriorEstimator(gaussian_likelihood, priors)
