"""The global MAP search must not be fooled by the log-posterior's offset.

``scipy.optimize.differential_evolution`` converges when

    std(population energies) <= atol + tol * |mean(population energies)|

with defaults ``atol=0, tol=0.01``.  An unnormalised log-posterior carries the
noise evidence as an arbitrary additive constant, so a purely relative
criterion can converge after a single iteration, well short of the optimum,
just because that offset is large.  The criterion has to be absolute instead;
these tests pin that, using a likelihood whose only unusual feature is a
large constant added to it.
"""

import bilby
import numpy as np
import pytest

from bilby_laplace.laplace import DE_ATOL, LaplacePosteriorEstimator

OFFSET = 2.0e5  # the noise evidence's scale on a 3G BNS


class _OffsetGaussianLikelihood(bilby.core.likelihood.Likelihood):
    """A sharp Gaussian peak sitting on a large constant.

    The peak is narrow relative to the prior, so a search that gives up early
    lands far from it; the constant is what a relative convergence tolerance
    reads as "close enough".
    """

    def __init__(self, mu=(1.7, -2.3), sigma=0.05, offset=OFFSET):
        super().__init__(parameters=dict(x=None, y=None))
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = sigma
        self.offset = offset
        self.n_calls = 0

    def log_likelihood(self, parameters=None):
        p = parameters if parameters is not None else self.parameters
        self.n_calls += 1
        d = np.array([p["x"], p["y"]]) - self.mu
        return -0.5 * np.sum(d**2) / self.sigma**2 - self.offset


@pytest.fixture
def offset_priors():
    return bilby.core.prior.PriorDict(
        dict(
            x=bilby.core.prior.Uniform(-5, 5, "x"),
            y=bilby.core.prior.Uniform(-5, 5, "y"),
        )
    )


def _estimator(likelihood, priors):
    return LaplacePosteriorEstimator(
        likelihood, priors, minimization_method="differential_evolution", use_unit_cube=False
    )


def test_the_map_is_found_despite_a_large_constant_offset(offset_priors):
    likelihood = _OffsetGaussianLikelihood()
    estimator = _estimator(likelihood, offset_priors)

    found = estimator.get_MAP_sample()

    np.testing.assert_allclose([found["x"], found["y"]], likelihood.mu, atol=1e-3)


def test_the_offset_does_not_change_where_the_search_lands(offset_priors):
    """The same problem with and without the constant must give the same MAP.

    This is the regression proper: under a relative tolerance the offset alone
    decided how hard the optimiser tried.
    """
    with_offset = _estimator(_OffsetGaussianLikelihood(offset=OFFSET), offset_priors).get_MAP_sample()
    without = _estimator(_OffsetGaussianLikelihood(offset=0.0), offset_priors).get_MAP_sample()

    np.testing.assert_allclose(
        [with_offset["x"], with_offset["y"]], [without["x"], without["y"]], atol=1e-3
    )


@pytest.fixture
def spied_optimisers(monkeypatch):
    """Record the calls into scipy, delegating to the real implementations."""
    from bilby_laplace import laplace as module

    calls = {"de": [], "minimize": []}

    real_de, real_minimize = module.differential_evolution, module.minimize

    def de(*args, **kwargs):
        result = real_de(*args, **kwargs)
        calls["de"].append((kwargs, result))
        return result

    def mini(*args, **kwargs):
        result = real_minimize(*args, **kwargs)
        calls["minimize"].append((kwargs, result))
        return result

    monkeypatch.setattr(module, "differential_evolution", de)
    monkeypatch.setattr(module, "minimize", mini)
    return calls


def test_the_convergence_criterion_is_absolute(offset_priors, spied_optimisers):
    """``tol=0`` is the whole point: it removes the |mean energy| term.

    Asserted on the call rather than on an evaluation count, because how many
    evaluations an absolute threshold needs is a property of the problem -- this
    2-D toy converges in a few hundred, the 11-D BNS in ~50k.
    """
    _estimator(_OffsetGaussianLikelihood(), offset_priors).get_MAP_sample()

    kwargs, _ = spied_optimisers["de"][0]
    assert kwargs["tol"] == 0
    assert kwargs["atol"] == DE_ATOL


def test_the_search_polishes_with_nelder_mead(offset_priors, spied_optimisers):
    """scipy's own polish is disabled: L-BFGS-B's finite-difference gradients
    return the input unchanged on an ill-scaled GW likelihood."""
    _estimator(_OffsetGaussianLikelihood(), offset_priors).get_MAP_sample()

    de_kwargs, _ = spied_optimisers["de"][0]
    polish_kwargs, _ = spied_optimisers["minimize"][-1]
    assert de_kwargs["polish"] is False
    assert polish_kwargs["method"] == "Nelder-Mead"


def test_reported_cost_covers_both_legs_of_the_search(offset_priors, spied_optimisers):
    """``nfev`` prices the whole MAP search, DE plus the polish.

    ``run_statistics`` quotes it, so dropping either leg's share would
    understate the cost of the stage this fix made expensive.
    """
    estimator = _estimator(_OffsetGaussianLikelihood(), offset_priors)

    estimator.get_MAP_sample()

    (_, de_result), (_, polish_result) = spied_optimisers["de"][0], spied_optimisers["minimize"][-1]
    assert estimator.minimization_metadata.nfev == de_result.nfev + polish_result.nfev


def test_the_threshold_is_absolute_and_in_nats():
    assert DE_ATOL == pytest.approx(1.0)
