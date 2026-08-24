"""Tests for the per-parameter agreement metrics (JSD, EMD) and the periodic
recentring they rely on.

These are the numbers every example's comparison table is built from, so their
correctness matters independently of the plotting/table code that consumes
them.
"""

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from bilby_laplace.comparison import _emd, _jsd, _recentre_periodic


def _samples(rng, mean, std, n=3000):
    return rng.normal(mean, std, n)


def test_jsd_is_small_for_two_draws_from_the_same_distribution():
    rng = np.random.default_rng(0)
    a, b = _samples(rng, 0.0, 1.0), _samples(rng, 0.0, 1.0)

    assert _jsd(a, b) == pytest.approx(0.0, abs=0.02)


def test_jsd_saturates_at_log_2_for_disjoint_distributions():
    """JSD is bounded by log(2) nats, reached once the densities no longer overlap."""
    rng = np.random.default_rng(1)
    a, b = _samples(rng, 0.0, 1.0), _samples(rng, 50.0, 1.0)

    assert _jsd(a, b) == pytest.approx(np.log(2.0), abs=1e-6)


def test_jsd_is_symmetric():
    rng = np.random.default_rng(2)
    a, b = _samples(rng, 0.0, 1.0), _samples(rng, 3.0, 2.0)

    assert _jsd(a, b) == pytest.approx(_jsd(b, a))


def test_jsd_is_nan_for_a_degenerate_sample_set():
    rng = np.random.default_rng(3)
    a = _samples(rng, 0.0, 1.0)
    degenerate = np.full(50, 3.0)

    assert np.isnan(_jsd(a, degenerate))


def test_emd_matches_wasserstein_distance_scaled_by_reference_sigma():
    rng = np.random.default_rng(4)
    a, b = _samples(rng, 0.0, 1.0), _samples(rng, 1.0, 2.0)

    expected = wasserstein_distance(a, b) / np.std(b)
    assert _emd(a, b) == pytest.approx(expected)


def test_emd_is_nan_when_the_reference_has_no_spread():
    rng = np.random.default_rng(5)
    a = _samples(rng, 0.0, 1.0)
    degenerate_reference = np.full(50, 3.0)

    assert np.isnan(_emd(a, degenerate_reference))


def test_emd_grows_with_the_displacement():
    rng = np.random.default_rng(6)
    reference = _samples(rng, 0.0, 1.0)
    near = _samples(rng, 1.0, 1.0)
    far = _samples(rng, 10.0, 1.0)

    assert _emd(far, reference) > _emd(near, reference)


def test_recentre_periodic_pulls_a_boundary_straddling_mode_together():
    """A single lump straddling the wrap point looks like two lumps at the
    edges of the raw range; recentring must put it back in one piece."""
    bounds = (0.0, 10.0)
    # A cluster straddling the 0/10 wrap point, split across both edges.
    samples = np.array([0.2, 0.5, 0.8, 9.2, 9.5, 9.8])

    (recentred,) = _recentre_periodic([samples], bounds)

    assert np.ptp(recentred) < 2.0


def test_recentre_periodic_is_a_common_shift_across_all_sample_sets():
    """The same shift must be applied to every set, so it cannot move two
    posteriors relative to each other -- only choose where to cut the circle."""
    bounds = (0.0, 10.0)
    a = np.array([0.5, 9.5])
    b = a + 1.0  # shifted, but still within bounds via wraparound

    recentred_a, recentred_b = _recentre_periodic([a, b], bounds)

    np.testing.assert_allclose(np.mod(recentred_b - recentred_a, 10.0), np.mod(b - a, 10.0))
