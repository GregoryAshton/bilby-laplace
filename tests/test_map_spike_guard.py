"""The MAP search must not return a needle spike of the log-posterior.

IMRPhenomXPHM's SpinTaylor precession angles leave points that stand several
nats above everything within ~1e-5 of the unit cube around them, and that are
unchanged along the parameters the spike does not depend on (sheets).  An
optimiser keeps any higher point it touches, so best-of-restarts selection
prefers them.  The guard drops such candidates from the selection.  These tests
pin it on a synthetic surface with exactly that shape: a smooth Gaussian peak,
and next to it a sheet 5 nats high that is 1e-6 of the unit cube wide in ``x``
and unbounded in ``y``.
"""

import bilby
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from bilby_laplace.laplace import MAP_SPIKE_DELTA, MAP_SPIKE_DROP, LaplacePosteriorEstimator

MU = np.array([1.7, -2.3])
SIGMA = 0.5  # 0.05 of the unit cube: far wider than MAP_SPIKE_DELTA
NEEDLE_X = 1.9
NEEDLE_HALF_WIDTH = 1e-5  # parameter units; 1e-6 of the unit cube
NEEDLE_HEIGHT = 5.0


class _SurfaceLikelihood(bilby.core.likelihood.Likelihood):
    """A Gaussian peak, plus optionally a needle sheet or a genuine step in ``x``."""

    def __init__(self, needle=True, step=False):
        super().__init__(parameters=dict(x=None, y=None))
        self.needle = needle
        self.step = step
        self.n_calls = 0

    def log_likelihood(self, parameters=None):
        p = parameters if parameters is not None else self.parameters
        self.n_calls += 1
        x, y = p["x"], p["y"]
        out = -0.5 * ((x - MU[0]) ** 2 + (y - MU[1]) ** 2) / SIGMA**2
        if self.needle and abs(x - NEEDLE_X) < NEEDLE_HALF_WIDTH:
            out += NEEDLE_HEIGHT
        if self.step and x > NEEDLE_X:
            out += 2.0
        return out


@pytest.fixture
def priors():
    return bilby.core.prior.PriorDict(
        dict(x=bilby.core.prior.Uniform(-5, 5, "x"), y=bilby.core.prior.Uniform(-5, 5, "y"))
    )


def _estimator(likelihood, priors, **kwargs):
    kwargs.setdefault("map_restarts", 1)
    return LaplacePosteriorEstimator(
        likelihood, priors, minimization_method="differential_evolution", use_unit_cube=False, seed=3, **kwargs
    )


def _neg(est):
    return lambda x: -est.log_posterior_from_array(np.asarray(x, dtype=float))


def _candidate(est, x):
    x = np.asarray(x, dtype=float)
    return OptimizeResult(x=x, fun=-est.log_posterior_from_array(x), nfev=0)


def test_a_needle_sheet_is_flagged(priors):
    """Narrow in x, unbounded in y: the test must catch it along x alone."""
    est = _estimator(_SurfaceLikelihood(), priors)
    on_needle = np.array([NEEDLE_X, MU[1]])

    excess, x_off, _ = est._spike_test(on_needle, _neg(est))

    assert excess == pytest.approx(NEEDLE_HEIGHT, abs=0.01)
    assert abs(x_off[0] - NEEDLE_X) > NEEDLE_HALF_WIDTH


def test_a_smooth_peak_is_not_flagged(priors):
    est = _estimator(_SurfaceLikelihood(), priors)

    excess, _, _ = est._spike_test(MU, _neg(est))

    assert excess < 1e-3


def test_a_peak_narrower_than_delta_is_not_flagged(priors):
    """A loud signal's peak can be narrower than the offset; that is not a spike.

    Width 5e-5 of the unit cube, a twentieth of MAP_SPIKE_DELTA: both neighbours
    are 200 nats down, which a plain "above both neighbours" test would call a
    spike.
    The curvature correction must see it as the quadratic it is.
    """
    narrow = 5e-5 * 10  # parameter units on the (-5, 5) prior
    likelihood = _SurfaceLikelihood(needle=False)
    likelihood.log_likelihood = lambda parameters=None: -0.5 * (
        ((parameters or likelihood.parameters)["x"] - MU[0]) ** 2 / narrow**2
        + ((parameters or likelihood.parameters)["y"] - MU[1]) ** 2 / SIGMA**2
    )
    est = _estimator(likelihood, priors)

    excess, _, _ = est._spike_test(MU, _neg(est))

    assert abs(excess) < 1e-3


def test_the_high_side_of_a_step_is_not_flagged(priors):
    """A step has volume on its high side; that is structure, not a spike."""
    est = _estimator(_SurfaceLikelihood(needle=False, step=True), priors)
    just_above = np.array([NEEDLE_X + 1e-6, MU[1]])

    excess, _, _ = est._spike_test(just_above, _neg(est))

    assert excess < MAP_SPIKE_DROP


def test_a_point_beside_a_step_is_not_flagged(priors):
    """A step within 2 delta breaks the curvature correction; it must not flag.

    On a slope rising at s nats per delta to a step down of D nats, with both
    +delta and +2 delta beyond the step, the plain excess is D - s but the
    corrected one is D - 2 s / 3: at s = 1.5, D = 2.2 that is 0.7 against 1.2,
    so the correction alone flags a point that stands above its neighbours by
    under a nat. This is the shape beside GW150914's largest spike, where a
    ~1.5-nat step lies within delta of a smooth slope.
    """
    likelihood = _SurfaceLikelihood(needle=False, step=False)
    delta_x = MAP_SPIKE_DELTA * 10  # one delta, in parameter units (the prior is 10 wide)
    step_at, drop, slope = MU[0], 2.2, 1.5 / delta_x

    def log_likelihood(parameters=None):
        p = parameters or likelihood.parameters
        out = slope * (p["x"] - step_at) - 0.5 * (p["y"] - MU[1]) ** 2 / SIGMA**2
        return out - drop if p["x"] > step_at else out

    likelihood.log_likelihood = log_likelihood
    est = _estimator(likelihood, priors)
    beside = np.array([step_at - 0.5 * delta_x, MU[1]])

    excess, _, _ = est._spike_test(beside, _neg(est))

    assert excess < MAP_SPIKE_DROP


def test_a_spike_loses_the_selection_to_a_lower_smooth_peak(priors, caplog):
    """Best-of-restarts would take the spike, which scores higher; the guard must not."""
    est = _estimator(_SurfaceLikelihood(), priors)
    spike = _candidate(est, [NEEDLE_X, MU[1]])
    peak = _candidate(est, MU)
    assert spike.fun < peak.fun  # the spike really is the higher point

    best, n_spikes = est._select_map([spike, peak], _neg(est))

    assert best is peak
    assert n_spikes == 1
    assert "landed on a likelihood spike" in caplog.text


def test_a_lone_spike_falls_back_to_the_point_beside_it(priors, caplog):
    """No non-spike candidate: step off the spike rather than polish, and say so."""
    est = _estimator(_SurfaceLikelihood(), priors)
    spike = _candidate(est, [NEEDLE_X, MU[1]])

    best, n_spikes = est._select_map([spike], _neg(est))

    assert n_spikes == 1
    assert abs(best.x[0] - NEEDLE_X) == pytest.approx(MAP_SPIKE_DELTA * 10)  # one delta, in parameter units
    assert best.fun == pytest.approx(-est.log_posterior_from_array(best.x))
    assert best.fun > spike.fun
    assert "Every MAP candidate was a likelihood spike" in caplog.text


def test_selection_without_spikes_is_the_unguarded_selection(priors):
    """First of equal values kept, exactly as the unguarded loop did."""
    est = _estimator(_SurfaceLikelihood(needle=False), priors)
    a, b = _candidate(est, MU), _candidate(est, MU)

    best, n_spikes = est._select_map([a, b], _neg(est))

    assert best is a
    assert n_spikes == 0


def test_the_guard_does_not_move_a_search_it_does_not_fire_on(priors):
    """Guard on and off must give the same MAP, bit for bit, when no spike is hit.

    The guard draws no random numbers and returns a non-spike unchanged, so
    turning it on may change only the evaluation count.
    """
    on = _estimator(_SurfaceLikelihood(needle=False), priors, map_restarts=2)
    off = _estimator(_SurfaceLikelihood(needle=False), priors, map_restarts=2, map_spike_guard=False)

    r_on = on._maximize_posterior_differential_evolution()
    r_off = off._maximize_posterior_differential_evolution()

    np.testing.assert_array_equal(r_on.x, r_off.x)
    assert r_on.fun == r_off.fun
    assert r_on.n_spikes == 0
    assert r_on.nfev == r_off.nfev + 2 * 4 * 2  # 2 restarts x 4N neighbours, N = 2


def test_the_guard_is_counted_in_nfev(priors):
    likelihood = _SurfaceLikelihood()
    est = _estimator(likelihood, priors, map_restarts=2)

    result = est._maximize_posterior_differential_evolution()

    assert result.nfev == likelihood.n_calls


def test_the_default_offset_is_in_the_unit_cube():
    assert MAP_SPIKE_DELTA == pytest.approx(1e-3)
