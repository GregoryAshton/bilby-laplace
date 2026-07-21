"""Tests for pool-parallelised likelihood evaluation.

The pool only parallelises ``log_likelihood_from_array`` for a column-stacked
batch; drawing and accept/reject decisions stay in the main process, so a
pooled run must be numerically identical to a serial one.  These tests use a
``ThreadPool`` (same ``.map`` interface as ``multiprocessing.Pool``) so they
exercise the new branch without depending on the ``spawn`` start method or
worker-importable fixtures.  The heavy-likelihood-via-worker-global path is
verified separately by the example runs.
"""

from multiprocessing.pool import ThreadPool

import numpy as np
import pytest
from bilby.core.sampler.base_sampler import _initialize_global_variables

from bilby_laplace.laplace import LaplacePosteriorEstimator


@pytest.fixture
def pooled_estimator(gaussian_likelihood, gaussian_priors):
    """Estimator wired to a 2-thread pool, with bilby's worker global set.

    ``_pool_log_likelihood`` pulls the likelihood from bilby's per-worker
    global; with threads that global is shared in-process, so we populate it
    directly rather than through pool creation.
    """
    _initialize_global_variables(
        likelihood=gaussian_likelihood,
        priors=gaussian_priors,
        search_parameter_keys=["x", "y"],
        use_ratio=False,
        parameters={},
    )
    est = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    pool = ThreadPool(2)
    est.pool = pool
    est.npool = 2
    yield est
    pool.close()
    pool.join()


def test_pool_matches_serial(gaussian_likelihood, gaussian_priors, pooled_estimator):
    """A pooled batch evaluation equals the serial evaluation exactly."""
    serial = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    rng = np.random.default_rng(0)
    # Column-stacked (N_params, N_samples) batch within the prior support.
    x = rng.uniform(-1.0, 1.0, size=(2, 37))

    logl_serial = serial.log_likelihood_from_array(x)
    logl_pool = pooled_estimator.log_likelihood_from_array(x)

    assert logl_pool.shape == (37,)
    np.testing.assert_array_equal(logl_pool, logl_serial)


def test_pool_out_of_bounds_is_neg_inf(pooled_estimator):
    """Out-of-prior columns return -inf, matching the serial contract."""
    x = np.array([[0.0, 100.0], [0.0, 0.0]])  # second column is out of bounds
    logl = pooled_estimator.log_likelihood_from_array(x)
    assert np.isfinite(logl[0])
    assert logl[1] == -np.inf


def test_pool_clip_to_bounds(gaussian_likelihood, gaussian_priors, pooled_estimator):
    """With clip_to_bounds, out-of-range columns are clipped, not -inf."""
    serial = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    x = np.array([[0.0, 100.0], [0.0, 0.0]])

    logl_pool = pooled_estimator.log_likelihood_from_array(x, clip_to_bounds=True)
    logl_serial = serial.log_likelihood_from_array(x, clip_to_bounds=True)

    assert np.all(np.isfinite(logl_pool))
    np.testing.assert_array_equal(logl_pool, logl_serial)


def test_single_column_uses_serial_path(pooled_estimator):
    """A single-column (or 1-D) input bypasses the pool and stays serial."""
    x1d = np.array([0.1, -0.2])
    val = pooled_estimator.log_likelihood_from_array(x1d)
    assert np.isscalar(val) or val.shape == ()
