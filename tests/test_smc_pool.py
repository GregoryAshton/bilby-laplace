"""Tests for the pool wiring of the SMC (aspire) resampling path.

Aspire never sees bilby's worker pool by itself: ``aspire_bilby``'s likelihood
wrapper loops over each batch with a ``map_fn`` that defaults to the builtin
serial ``map``, so an unbound wrapper pins the run to one core.  We hand aspire
our own batched wrapper instead, which routes through
``LaplacePosteriorEstimator.log_likelihood_from_array`` and therefore inherits
the (already tested) pooled path.  These tests pin that wrapper's contract:
values identical to serial, out-of-prior points skipped, the pool actually
exercised, and the evaluation counter tracking real work.

A ``ThreadPool`` stands in for ``multiprocessing.Pool`` (same ``.map``
interface) so the tests do not depend on a start method or on
worker-importable fixtures -- the same approach as ``test_pool.py``.
"""

from multiprocessing.pool import ThreadPool

import numpy as np
import pytest
from bilby.core.sampler.base_sampler import SamplerError, _initialize_global_variables

from bilby_laplace.laplace import LaplacePosteriorEstimator
from bilby_laplace.sampler import Laplace


class _FakeSamples:
    """Minimal stand-in for ``aspire.samples.Samples``."""

    def __init__(self, x, log_prior):
        self.x = np.asarray(x, dtype=float)
        self.log_prior = np.asarray(log_prior, dtype=float)


class _CountingPool:
    """``ThreadPool`` wrapper that records how many times ``map`` was called."""

    def __init__(self, processes):
        self._pool = ThreadPool(processes)
        self.n_map_calls = 0

    def map(self, *args, **kwargs):
        self.n_map_calls += 1
        return self._pool.map(*args, **kwargs)

    def close(self):
        self._pool.close()

    def join(self):
        self._pool.join()


@pytest.fixture
def pooled_estimator(gaussian_likelihood, gaussian_priors):
    """Estimator wired to a 2-thread counting pool, with bilby's global set."""
    _initialize_global_variables(
        likelihood=gaussian_likelihood,
        priors=gaussian_priors,
        search_parameter_keys=["x", "y"],
        use_ratio=False,
        parameters={},
    )
    est = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    pool = _CountingPool(2)
    est.pool = pool
    est.npool = 2
    yield est
    pool.close()
    pool.join()


def test_aspire_likelihood_matches_serial_and_uses_the_pool(gaussian_likelihood, gaussian_priors, pooled_estimator):
    """The wrapper's values equal the serial ones and go through the pool."""
    serial = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(23, 2))
    samples = _FakeSamples(x, np.zeros(len(x)))

    counter = [0]
    logl = Laplace._make_aspire_log_likelihood(pooled_estimator, counter)(samples)

    assert logl.shape == (23,)
    np.testing.assert_array_equal(logl, serial.log_likelihood_from_array(x.T))
    assert pooled_estimator.pool.n_map_calls == 1
    assert counter[0] == 23


def test_aspire_likelihood_skips_and_does_not_count_out_of_prior_points(pooled_estimator):
    """Points with an infinite log-prior get -inf and are never evaluated."""
    x = np.array([[0.0, 0.0], [1.0, -0.5], [0.2, 0.3]])
    samples = _FakeSamples(x, [0.0, -np.inf, 0.0])

    counter = [0]
    logl = Laplace._make_aspire_log_likelihood(pooled_estimator, counter)(samples)

    assert np.isfinite(logl[[0, 2]]).all()
    assert logl[1] == -np.inf
    assert counter[0] == 2


def test_aspire_likelihood_accumulates_the_counter_across_batches(pooled_estimator):
    """The counter is cumulative -- it is the run's total evaluation count."""
    counter = [0]
    log_likelihood = Laplace._make_aspire_log_likelihood(pooled_estimator, counter)
    log_likelihood(_FakeSamples(np.zeros((4, 2)), np.zeros(4)))
    log_likelihood(_FakeSamples(np.zeros((6, 2)), np.zeros(6)))
    assert counter[0] == 10


def test_aspire_likelihood_all_out_of_prior_is_all_neg_inf(pooled_estimator):
    """An entirely out-of-prior batch short-circuits without touching the pool."""
    samples = _FakeSamples(np.zeros((3, 2)), np.full(3, -np.inf))

    counter = [0]
    logl = Laplace._make_aspire_log_likelihood(pooled_estimator, counter)(samples)

    assert np.all(logl == -np.inf)
    assert counter[0] == 0
    assert pooled_estimator.pool.n_map_calls == 0


def test_aspire_likelihood_requires_the_log_prior(pooled_estimator):
    """Aspire must evaluate the log-prior first; without it we cannot mask."""
    samples = _FakeSamples(np.zeros((3, 2)), np.zeros(3))
    samples.log_prior = None

    with pytest.raises(SamplerError, match="log-prior"):
        Laplace._make_aspire_log_likelihood(pooled_estimator, [0])(samples)


def test_serial_estimator_needs_no_pool(gaussian_likelihood, gaussian_priors):
    """With no pool the wrapper still works (the estimator falls back)."""
    est = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    samples = _FakeSamples(np.array([[0.0, 0.0], [1.0, -0.5]]), np.zeros(2))

    logl = Laplace._make_aspire_log_likelihood(est, [0])(samples)

    assert np.all(np.isfinite(logl))


# --------------------------------------------------------------------------
# npool plumbing: the aliases and the explicit run_sampler argument both have
# to reach `Sampler.npool`, which reads `self.kwargs` before `self._npool`.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alias", ["npool", "n_pool", "cores", "threads", "queue_size"])
def test_npool_aliases_reach_the_npool_property(gaussian_likelihood, gaussian_priors, tmp_path, alias):
    sampler = Laplace(
        likelihood=gaussian_likelihood,
        priors=gaussian_priors,
        outdir=str(tmp_path),
        label="pool",
        **{alias: 3},
    )
    assert sampler.npool == 3


def test_explicit_npool_argument_is_not_shadowed_by_the_default(gaussian_likelihood, gaussian_priors, tmp_path):
    """``run_sampler(npool=...)`` arrives as the `npool` init argument, not a
    kwarg, so it must be folded into kwargs or the `npool=None` default hides it."""
    sampler = Laplace(
        likelihood=gaussian_likelihood,
        priors=gaussian_priors,
        outdir=str(tmp_path),
        label="pool",
        npool=4,
    )
    assert sampler.npool == 4


def test_npool_defaults_to_no_pool(sampler):
    assert not sampler.npool or sampler.npool == 1
    sampler._setup_pool()
    assert sampler.pool is None


# --------------------------------------------------------------------------
# Periodic parameters.  `aspire_bilby`'s own plugin passes these to Aspire; our
# `_smc_sample` did not, so a wrapping coordinate got the bounded->logit
# preconditioning instead of angular treatment and the pCN kernel could not
# step across the boundary.  On the HLV example that affects psi and azimuth.
# The derivation is duplicated here because it is the contract under test.
# --------------------------------------------------------------------------


def _periodic(priors, parameter_names):
    return [k for k in parameter_names if getattr(priors[k], "boundary", None) == "periodic"]


def test_periodic_parameters_are_detected(gaussian_priors):
    import bilby

    gaussian_priors["x"] = bilby.core.prior.Uniform(0, 1, "x", boundary="periodic")

    assert _periodic(gaussian_priors, ["x", "y"]) == ["x"]


def test_non_periodic_boundaries_are_not_included(gaussian_priors):
    import bilby

    gaussian_priors["x"] = bilby.core.prior.Uniform(0, 1, "x", boundary="reflective")

    assert _periodic(gaussian_priors, ["x", "y"]) == []


def test_marginalised_periodic_parameters_are_excluded(gaussian_priors):
    """A marginalised coordinate is not part of aspire's parameter vector, so
    including it would misalign the indices."""
    import bilby

    gaussian_priors["phase"] = bilby.core.prior.Uniform(0, 1, "phase", boundary="periodic")

    assert "phase" not in _periodic(gaussian_priors, ["x", "y"])


def test_matches_aspire_bilby_helper(gaussian_priors):
    """Our derivation must agree with the plugin's, restricted to sampled keys."""
    import bilby

    pytest.importorskip("aspire_bilby")
    from aspire_bilby.utils import get_periodic_parameters

    gaussian_priors["x"] = bilby.core.prior.Uniform(0, 1, "x", boundary="periodic")
    names = ["x", "y"]

    theirs = [p for p in get_periodic_parameters(gaussian_priors) if p in names]

    assert _periodic(gaussian_priors, names) == theirs
