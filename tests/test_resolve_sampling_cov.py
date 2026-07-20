"""Tests for ``Laplace._resolve_sampling_cov`` — the user-supplied covariance
validation and normalisation logic."""

import numpy as np
import pandas as pd
import pytest

NAMES = ["x", "y"]
COV = np.array([[0.09, 0.105], [0.105, 0.25]])


def test_none_passthrough(sampler):
    assert sampler._resolve_sampling_cov(None, NAMES) is None


def test_dataframe_input(sampler):
    df = pd.DataFrame(COV, index=NAMES, columns=NAMES)
    out = sampler._resolve_sampling_cov(df, NAMES)
    np.testing.assert_allclose(out, COV)


def test_tuple_input(sampler):
    out = sampler._resolve_sampling_cov((NAMES, COV), NAMES)
    np.testing.assert_allclose(out, COV)


def test_dataframe_reordered_to_parameter_order(sampler):
    # Supply the covariance in reversed order; expect it reordered to NAMES.
    rev = ["y", "x"]
    df = pd.DataFrame(COV, index=rev, columns=rev)
    out = sampler._resolve_sampling_cov(df, NAMES)
    # out[0,0] should be var(x); in the reversed frame that is COV[1, 1].
    assert out[0, 0] == pytest.approx(COV[1, 1])
    assert out[1, 1] == pytest.approx(COV[0, 0])


def test_tuple_wrong_shape_raises(sampler):
    bad = np.eye(3)
    with pytest.raises(ValueError, match="shape"):
        sampler._resolve_sampling_cov((NAMES, bad), NAMES)


def test_wrong_type_raises(sampler):
    with pytest.raises(TypeError):
        sampler._resolve_sampling_cov([[1, 0], [0, 1]], NAMES)


def test_duplicate_names_raise(sampler):
    df = pd.DataFrame(COV, index=["x", "x"], columns=["x", "x"])
    with pytest.raises(ValueError, match="duplicate"):
        sampler._resolve_sampling_cov(df, NAMES)


def test_row_col_mismatch_raises(sampler):
    df = pd.DataFrame(COV, index=["x", "y"], columns=["x", "z"])
    with pytest.raises(ValueError, match="same parameter names"):
        sampler._resolve_sampling_cov(df, NAMES)


def test_missing_parameter_raises(sampler):
    df = pd.DataFrame([[0.1]], index=["x"], columns=["x"])
    with pytest.raises(ValueError, match="missing"):
        sampler._resolve_sampling_cov(df, NAMES)


def test_unknown_parameter_raises(sampler):
    names = ["x", "y", "z"]
    cov3 = np.diag([0.09, 0.25, 1.0])
    df = pd.DataFrame(cov3, index=names, columns=names)
    with pytest.raises(ValueError, match="unknown"):
        sampler._resolve_sampling_cov(df, NAMES)


def test_non_symmetric_raises(sampler):
    asym = np.array([[0.09, 0.2], [0.0, 0.25]])
    df = pd.DataFrame(asym, index=NAMES, columns=NAMES)
    with pytest.raises(ValueError, match="symmetric"):
        sampler._resolve_sampling_cov(df, NAMES)


def test_non_psd_raises(sampler):
    # Symmetric but indefinite (negative eigenvalue).
    npsd = np.array([[1.0, 2.0], [2.0, 1.0]])
    df = pd.DataFrame(npsd, index=NAMES, columns=NAMES)
    with pytest.raises(ValueError, match="positive semi-definite"):
        sampler._resolve_sampling_cov(df, NAMES)


def test_symmetrises_tiny_asymmetry(sampler):
    """A covariance within tolerance of symmetric is accepted and symmetrised."""
    almost = COV.copy()
    almost[0, 1] += 1e-12
    df = pd.DataFrame(almost, index=NAMES, columns=NAMES)
    out = sampler._resolve_sampling_cov(df, NAMES)
    np.testing.assert_allclose(out, out.T, atol=1e-15)
