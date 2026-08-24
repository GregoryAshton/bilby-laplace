"""Tests for the ``cov_scaling`` resolve/apply pair.

``_resolve_cov_scaling`` normalizes the user-facing ``cov_scaling`` kwarg
(scalar or per-parameter dict) to a variance-scale vector; ``_apply_cov_scaling``
then applies it to a covariance matrix. Documented in ``Laplace.default_kwargs``
as load-bearing but had no dedicated tests.
"""

import numpy as np
import pytest

from bilby_laplace.sampler import Laplace


def test_scalar_broadcasts_to_every_parameter(sampler):
    v = sampler._resolve_cov_scaling(4.0, ["x", "y", "z"])

    np.testing.assert_allclose(v, [4.0, 4.0, 4.0])


def test_dict_sets_named_parameters_and_defaults_the_rest(sampler):
    v = sampler._resolve_cov_scaling({"x": 4.0}, ["x", "y"])

    np.testing.assert_allclose(v, [4.0, 1.0])


def test_dict_others_key_sets_the_default_for_unlisted_parameters(sampler):
    v = sampler._resolve_cov_scaling({"x": 4.0, "others": 9.0}, ["x", "y", "z"])

    np.testing.assert_allclose(v, [4.0, 9.0, 9.0])


def test_dict_with_an_unknown_parameter_name_raises(sampler):
    with pytest.raises(ValueError, match="unknown parameter"):
        sampler._resolve_cov_scaling({"not_a_parameter": 2.0}, ["x", "y"])


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_values_raise(sampler, bad_value):
    with pytest.raises(ValueError, match="finite and strictly positive"):
        sampler._resolve_cov_scaling({"x": bad_value}, ["x", "y"])


def test_apply_with_a_uniform_scale_reduces_to_scalar_multiplication():
    cov = np.array([[2.0, 0.5], [0.5, 3.0]])

    scaled = Laplace._apply_cov_scaling(cov, [4.0, 4.0])

    np.testing.assert_allclose(scaled, 4.0 * cov)


def test_apply_scales_diagonal_by_v_and_off_diagonal_by_geometric_mean():
    cov = np.array([[2.0, 0.5], [0.5, 3.0]])
    v = [4.0, 9.0]

    scaled = Laplace._apply_cov_scaling(cov, v)

    np.testing.assert_allclose(np.diag(scaled), np.diag(cov) * v)
    assert scaled[0, 1] == pytest.approx(cov[0, 1] * np.sqrt(v[0] * v[1]))


def test_apply_preserves_positive_definiteness():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + 1e-6 * np.eye(3)

    scaled = Laplace._apply_cov_scaling(cov, [1.0, 4.0, 9.0])

    np.testing.assert_array_less(0.0, np.linalg.eigvalsh(scaled))
