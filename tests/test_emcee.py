"""Tests for the ``resample='emcee'`` path (``Laplace._run_emcee``).

The 2-D correlated Gaussian fixture (see ``conftest.py``) makes the Laplace
proposal exactly the true posterior, so a short emcee run seeded from it
should stay close to the known mean/covariance -- that is the main
correctness check here, alongside the plumbing (kwargs, wiring, error
handling).
"""

import os

import bilby
import numpy as np
import pandas as pd
import pytest
from bilby.core.sampler.base_sampler import SamplerError

from bilby_laplace.laplace import LaplacePosteriorEstimator
from bilby_laplace.sampler import Laplace, TruncatedMVNProposal

emcee = pytest.importorskip("emcee")

from conftest import MU, PRIOR_MAX, PRIOR_MIN, TRUE_COV  # noqa: E402


@pytest.fixture
def emcee_setup(gaussian_likelihood, gaussian_priors, tmp_path):
    sampler = Laplace(
        likelihood=gaussian_likelihood,
        priors=gaussian_priors,
        outdir=str(tmp_path),
        label="emcee-test",
    )
    estimator = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    lower = np.array([PRIOR_MIN, PRIOR_MIN])
    upper = np.array([PRIOR_MAX, PRIOR_MAX])
    proposal = TruncatedMVNProposal(MU, TRUE_COV, lower, upper)
    return sampler, estimator, proposal


def test_emcee_recovers_the_known_gaussian(emcee_setup):
    sampler, estimator, proposal = emcee_setup
    nwalkers, nsteps, discard = 32, 3000, 500
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=nwalkers, nsteps=nsteps, discard=discard)

    samples, logl, g_samples, efficiency, nlikelihood = sampler._run_emcee(proposal, estimator)

    assert list(samples.columns) == list(estimator.parameter_names)
    assert g_samples is samples
    assert len(samples) == len(logl)
    # Auto-thinning by the autocorrelation time must actually reduce the
    # sample count below the raw (unthinned) chain -- otherwise this is just
    # the every-step correlated chain, not independent draws.
    assert 0 < len(samples) < nwalkers * (nsteps - discard)
    assert np.all(np.isfinite(logl))
    assert 0.0 <= efficiency <= 100.0
    # Every walker/step pair proposes a point (plus one initial evaluation of
    # the starting ensemble), so the true likelihood-eval count is at most
    # nwalkers * (nsteps + 1) -- less only for out-of-prior draws, none
    # expected here since the fixture prior is a wide box.
    assert nlikelihood <= nwalkers * (nsteps + 1)
    assert nlikelihood >= len(samples)

    # Loose tolerances: just checking it lands near the known posterior
    # rather than testing MCMC convergence in fine detail.
    np.testing.assert_allclose(samples.mean().values, MU, atol=0.08)
    np.testing.assert_allclose(np.cov(samples.values, rowvar=False), TRUE_COV, atol=0.05)


def test_emcee_default_nwalkers_and_nsteps(emcee_setup):
    """No emcee_kwargs at all should still run, using the built-in defaults."""
    sampler, estimator, proposal = emcee_setup
    # Trim the defaults so the test stays fast without touching the
    # production defaults documented on the class.
    sampler.kwargs["emcee_kwargs"] = dict(nsteps=200, discard=50)

    samples, logl, g_samples, efficiency, nlikelihood = sampler._run_emcee(proposal, estimator)

    ndim = len(estimator.parameter_names)
    expected_nwalkers = max(4 * ndim, 32)
    assert 0 < len(samples) <= expected_nwalkers * (200 - 50)
    assert nlikelihood <= expected_nwalkers * (200 + 1)


def test_emcee_thin_override_is_respected(emcee_setup):
    """An explicit ``thin`` skips the automatic autocorrelation-time estimate."""
    sampler, estimator, proposal = emcee_setup
    nwalkers, nsteps, discard, thin = 8, 60, 10, 4
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=nwalkers, nsteps=nsteps, discard=discard, thin=thin)

    samples, *_ = sampler._run_emcee(proposal, estimator)

    # Matches emcee's own thinned-slice indexing (`chain[discard+thin-1::thin]`
    # per walker), not a naive `(nsteps - discard) / thin`.
    n_kept_per_walker = len(range(discard + thin - 1, nsteps, thin))
    assert len(samples) == nwalkers * n_kept_per_walker


def test_emcee_rejects_discard_at_or_past_nsteps(emcee_setup):
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=8, nsteps=50, discard=50)

    with pytest.raises(SamplerError, match="discard"):
        sampler._run_emcee(proposal, estimator)


def test_emcee_out_of_prior_points_get_zero_weight(emcee_setup):
    """log_prob_batch must never hand the likelihood an out-of-support point."""
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=8, nsteps=20, discard=0)

    samples, logl, _, _, _ = sampler._run_emcee(proposal, estimator)
    # Every kept sample must be inside the prior box (the fixture prior has
    # no Constraint, so the box is the whole story here).
    assert (samples["x"].between(PRIOR_MIN, PRIOR_MAX)).all()
    assert (samples["y"].between(PRIOR_MIN, PRIOR_MAX)).all()


def test_emcee_is_a_recognised_resample_option():
    assert "emcee_kwargs" in Laplace.default_kwargs


# --------------------------------------------------------------------------
# Adaptive batched running.  The chain grows in `nsteps` batches until it
# holds `target_nsamples` approximately independent samples *and* is long
# enough for the autocorrelation estimate behind that count to be trusted --
# so the run length need not be guessed in advance.
# --------------------------------------------------------------------------


def _count_batches(sampler, monkeypatch):
    """Record one entry per convergence check, i.e. per completed batch."""
    checks = []
    original = sampler._emcee_autocorr_status

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        checks.append(result)
        return result

    monkeypatch.setattr(sampler, "_emcee_autocorr_status", spy)
    return checks


def test_the_chain_grows_until_the_target_is_met(emcee_setup, monkeypatch):
    """More than one batch must run, and the result must clear the target."""
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["target_nsamples"] = 3000
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=60, nsteps=400, max_nsteps=20000, discard=200)
    checks = _count_batches(sampler, monkeypatch)

    samples, *_ = sampler._run_emcee(proposal, estimator)

    # Growth actually happened (the first batch alone cannot clear the bar at
    # this tau), and stopping was driven by the target rather than the cap.
    assert len(checks) > 1
    _, _, n_independent, reliable = checks[-1]
    assert reliable and n_independent >= 3000
    assert len(samples) >= 3000
    np.testing.assert_allclose(samples.mean().values, MU, atol=0.1)


def test_a_single_batch_is_the_default(emcee_setup, monkeypatch):
    """`max_nsteps` defaulting to `nsteps` keeps the old fixed-length behaviour."""
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["target_nsamples"] = 10**9  # unreachable, so only the cap can stop it
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=16, nsteps=200, discard=50)
    checks = _count_batches(sampler, monkeypatch)

    sampler._run_emcee(proposal, estimator)

    assert len(checks) == 1


def test_hitting_max_nsteps_warns_but_still_returns_samples(emcee_setup, caplog):
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["target_nsamples"] = 10**7  # unreachable within the cap
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=20, nsteps=100, max_nsteps=300, discard=50)

    with caplog.at_level("WARNING"):
        samples, *_ = sampler._run_emcee(proposal, estimator)

    assert len(samples) > 0
    assert "stopped at max_nsteps=300" in caplog.text


def test_too_many_walkers_to_reach_the_target_warns(emcee_setup, caplog):
    """The reliability bar alone forces >= 50 * nwalkers independent samples,
    so a large nwalkers cannot stop near a small target."""
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["target_nsamples"] = 100
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=200, nsteps=100, discard=20)

    with caplog.at_level("WARNING"):
        sampler._run_emcee(proposal, estimator)

    assert "cannot yield fewer than 10000 independent samples" in caplog.text


def test_a_lower_autocorr_tol_stops_the_run_sooner(emcee_setup, monkeypatch):
    """With the sample count met early, the reliability bar is what binds --
    so lowering it must end the run in fewer batches."""
    sampler, estimator, proposal = emcee_setup
    # nwalkers * tol >> target, so `n_independent >= target` is satisfied
    # almost immediately and only the ratio bar can hold the loop open.
    base = dict(nwalkers=60, nsteps=200, max_nsteps=12000, discard=100)
    sampler.kwargs["target_nsamples"] = 200

    sampler.kwargs["emcee_kwargs"] = dict(base, autocorr_tol=5)
    lax = _count_batches(sampler, monkeypatch)
    sampler._run_emcee(proposal, estimator)

    monkeypatch.undo()

    sampler.kwargs["emcee_kwargs"] = dict(base, autocorr_tol=50)
    strict = _count_batches(sampler, monkeypatch)
    sampler._run_emcee(proposal, estimator)

    assert len(lax) < len(strict)


def test_autocorr_tol_must_be_positive(emcee_setup):
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=8, nsteps=50, discard=10, autocorr_tol=0)

    with pytest.raises(SamplerError, match="autocorr_tol"):
        sampler._run_emcee(proposal, estimator)


def test_autocorr_status_honours_an_explicit_tol(emcee_setup):
    """The same chain reads as unreliable at a strict tol and reliable at a lax one."""
    sampler, _, _ = emcee_setup

    class _Stub:
        iteration = 1000

        @staticmethod
        def get_autocorr_time(**kwargs):
            return np.array([50.0])  # chain/tau = 1000/50 = 20

    assert sampler._emcee_autocorr_status(_Stub(), 0, 10, ["x"], tol=50)[3] is False
    assert sampler._emcee_autocorr_status(_Stub(), 0, 10, ["x"], tol=10)[3] is True


def test_backend_file_persists_the_full_unthinned_chain(emcee_setup):
    """The returned samples are thinned, so the backend is the only record of
    the raw chain -- and must hold every step, not the thinned subset."""
    sampler, estimator, proposal = emcee_setup
    nwalkers, nsteps = 16, 300
    sampler.kwargs["emcee_kwargs"] = dict(
        nwalkers=nwalkers, nsteps=nsteps, discard=50, backend_file=True
    )

    samples, *_ = sampler._run_emcee(proposal, estimator)

    expected = f"{sampler.outdir}/{sampler.label}_emcee_chain.h5"
    assert os.path.isfile(expected)

    backend = emcee.backends.HDFBackend(expected, read_only=True)
    chain = backend.get_chain()
    assert chain.shape == (nsteps, nwalkers, len(estimator.parameter_names))
    # Un-thinned: strictly more raw steps than returned samples.
    assert chain.shape[0] * nwalkers > len(samples)


def test_backend_file_accepts_an_explicit_path(emcee_setup, tmp_path):
    sampler, estimator, proposal = emcee_setup
    target = tmp_path / "nested" / "chain.h5"
    target.parent.mkdir()
    sampler.kwargs["emcee_kwargs"] = dict(
        nwalkers=8, nsteps=100, discard=20, backend_file=str(target)
    )

    sampler._run_emcee(proposal, estimator)

    assert target.is_file()


def test_no_backend_file_by_default(emcee_setup):
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=8, nsteps=100, discard=20)

    sampler._run_emcee(proposal, estimator)

    assert not os.path.isfile(f"{sampler.outdir}/{sampler.label}_emcee_chain.h5")


def test_max_nsteps_below_nsteps_is_rejected(emcee_setup):
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=8, nsteps=100, max_nsteps=50, discard=10)

    with pytest.raises(SamplerError, match="max_nsteps"):
        sampler._run_emcee(proposal, estimator)


def test_target_nsamples_defaults_to_the_sampler_wide_value(emcee_setup, monkeypatch):
    """With no explicit `target_nsamples` in emcee_kwargs, the sampler-wide
    value drives the stopping rule -- so raising it must lengthen the run."""
    sampler, estimator, proposal = emcee_setup
    base_kwargs = dict(nwalkers=40, nsteps=300, max_nsteps=9000, discard=100)

    sampler.kwargs["target_nsamples"] = 2000
    sampler.kwargs["emcee_kwargs"] = dict(base_kwargs)
    few = _count_batches(sampler, monkeypatch)
    sampler._run_emcee(proposal, estimator)

    # Restore the real method before spying again, or the second spy would
    # wrap the first and both lists would collect the second run's checks.
    monkeypatch.undo()

    sampler.kwargs["target_nsamples"] = 20000
    sampler.kwargs["emcee_kwargs"] = dict(base_kwargs)
    many = _count_batches(sampler, monkeypatch)
    sampler._run_emcee(proposal, estimator)

    assert len(many) > len(few)


def test_explicit_target_nsamples_overrides_the_sampler_wide_value(emcee_setup, monkeypatch):
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["target_nsamples"] = 10**9  # would never be reached
    sampler.kwargs["emcee_kwargs"] = dict(
        nwalkers=40, nsteps=300, max_nsteps=9000, discard=100, target_nsamples=2000
    )
    checks = _count_batches(sampler, monkeypatch)

    sampler._run_emcee(proposal, estimator)

    # Stopped on the explicit (reachable) target, not the sampler-wide one,
    # so the cap was never reached.
    _, _, n_independent, reliable = checks[-1]
    assert reliable and n_independent >= 2000


def test_autocorr_status_reports_the_binding_parameter(emcee_setup):
    """`tau_max` is the largest tau, and the sample count follows from it."""
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=16, nsteps=400, discard=50)
    sampler._run_emcee(proposal, estimator)

    # Rebuild a short ensemble directly to inspect the helper's arithmetic.
    nwalkers, discard = 16, 50

    class _Stub:
        iteration = 450

        @staticmethod
        def get_autocorr_time(**kwargs):
            return np.array([10.0, 25.0])

    tau, tau_max, n_independent, reliable = sampler._emcee_autocorr_status(
        _Stub(), discard, nwalkers, ["x", "y"]
    )

    assert tau_max == 25.0  # the worst-mixing coordinate binds
    assert n_independent == int(nwalkers * (450 - discard) / 25.0)
    assert reliable is bool((450 - discard) >= 50 * 25.0)


def test_autocorr_status_handles_a_chain_too_short_to_estimate(emcee_setup):
    """Non-finite tau must read as 'not reliable', never as convergence."""
    sampler, _, _ = emcee_setup

    class _Stub:
        iteration = 10

        @staticmethod
        def get_autocorr_time(**kwargs):
            return np.array([np.nan, np.nan])

    tau, tau_max, n_independent, reliable = sampler._emcee_autocorr_status(_Stub(), 0, 8, ["x", "y"])

    assert np.isnan(tau_max)
    assert n_independent == 0
    assert reliable is False


def test_emcee_diagnostic_plot_is_written_when_requested(emcee_setup):
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["plot_diagnostic"] = True
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=16, nsteps=100, discard=20)

    samples, *_ = sampler._run_emcee(proposal, estimator)

    expected = f"{sampler.outdir}/{sampler.label}_diagnostic_emcee_evolution.png"
    assert os.path.isfile(expected)
    assert os.path.getsize(expected) > 0


class _StubEnsemble:
    """Minimal stand-in for ``emcee.EnsembleSampler``'s read-back surface."""

    # `iteration` must equal n_thinned * thin (the tests use thin=1), or the
    # step axis and the thinned chain disagree in length.
    n_thinned, nwalkers, iteration = 4, 3, 4

    def __init__(self, ndim):
        self.ndim = ndim

    def get_chain(self, **kwargs):
        return np.zeros((self.n_thinned, self.nwalkers, self.ndim))

    def get_log_prob(self, **kwargs):
        n = self.n_thinned * self.nwalkers
        return np.zeros(n) if kwargs.get("flat") else np.zeros((self.n_thinned, self.nwalkers))


@pytest.mark.parametrize("with_history", [False, True])
def test_emcee_diagnostic_gains_a_row_for_the_tau_history(emcee_setup, monkeypatch, with_history):
    """The tau row is what shows whether the chain is converging, so the
    figure must carry exactly one extra row when a history is supplied."""
    import bilby_laplace.sampler as sampler_module

    sampler, estimator, _ = emcee_setup
    ndim = len(estimator.parameter_names)
    ensemble = _StubEnsemble(ndim)
    samples = pd.DataFrame(
        np.zeros((ensemble.n_thinned * ensemble.nwalkers, ndim)), columns=estimator.parameter_names
    )
    tau_history = [(2, np.full(ndim, 5.0)), (4, np.arange(1.0, ndim + 1.0))] if with_history else None

    captured = {}

    def capture(fig, filename, **kwargs):
        # Read the grid before `create_emcee_diagnostic` closes the figure.
        captured["nrows"] = fig.axes[0].get_gridspec().nrows

    monkeypatch.setattr(sampler_module, "safe_save_figure", capture)

    sampler.create_emcee_diagnostic(samples, ensemble, 0, 1, tau_history)

    assert captured["nrows"] == ndim + 2 + (1 if with_history else 0)


def test_no_emcee_diagnostic_plot_by_default(emcee_setup):
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=16, nsteps=100, discard=20)

    sampler._run_emcee(proposal, estimator)

    expected = f"{sampler.outdir}/{sampler.label}_diagnostic_emcee_evolution.png"
    assert not os.path.isfile(expected)


def test_a_failed_emcee_diagnostic_does_not_lose_the_run(emcee_setup, monkeypatch, caplog):
    """Diagnostic plotting is best-effort: a failure must not lose the samples."""
    sampler, estimator, proposal = emcee_setup
    sampler.kwargs["plot_diagnostic"] = True
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=16, nsteps=100, discard=20)

    def _boom(*args, **kwargs):
        raise RuntimeError("plotting exploded")

    monkeypatch.setattr(sampler, "create_emcee_diagnostic", _boom)

    with caplog.at_level("WARNING"):
        samples, logl, *_ = sampler._run_emcee(proposal, estimator)

    assert len(samples) > 0
    assert "Failed to create emcee diagnostic plot" in caplog.text


# --------------------------------------------------------------------------
# Periodic parameters.  Every other resampling mode wraps them via
# `TruncatedMVNProposal`; `_run_emcee`'s own `log_prob_batch` must too, or a
# walker that drifts past a periodic boundary reads as leaving the prior and
# gets rejected -- a false rejection-at-the-wall the target never actually has.
# --------------------------------------------------------------------------

PERIOD = 2 * np.pi
# Mean near the lower edge with a sigma comparable to the whole period, same
# regime as test_periodic_proposal.py: a meaningful fraction of the true
# density sits just past 0, wrapping to just under PERIOD.
WRAPPED_TRUE_VALUE = 0.835
WRAPPED_SIGMA = 1.4


class _WrappedGaussianLikelihood(bilby.core.likelihood.Likelihood):
    """1-D likelihood whose density is periodic with period ``PERIOD``."""

    def __init__(self):
        super().__init__(parameters={"x": None})

    def log_likelihood(self, parameters=None):
        p = parameters if parameters is not None else self.parameters
        d = np.mod(p["x"] - WRAPPED_TRUE_VALUE + PERIOD / 2, PERIOD) - PERIOD / 2
        return -0.5 * (d / WRAPPED_SIGMA) ** 2


@pytest.fixture
def periodic_emcee_setup(tmp_path):
    likelihood = _WrappedGaussianLikelihood()
    priors = bilby.core.prior.PriorDict(dict(x=bilby.core.prior.Uniform(0, PERIOD, "x", boundary="periodic")))
    estimator = LaplacePosteriorEstimator(likelihood, priors)
    sampler = Laplace(likelihood=likelihood, priors=priors, outdir=str(tmp_path), label="periodic-emcee-test")
    proposal = TruncatedMVNProposal(
        np.array([WRAPPED_TRUE_VALUE]),
        np.array([[WRAPPED_SIGMA**2]]),
        lower=np.array([0.0]),
        upper=np.array([PERIOD]),
        periodic=np.array([True]),
    )
    return sampler, estimator, proposal


def test_periodic_parameters_are_wrapped_before_scoring(periodic_emcee_setup):
    """Every point handed to the prior/likelihood must be wrapped into range.

    Without wrapping, a walker that drifts past the periodic boundary is
    scored at its raw (out-of-range) coordinate -- which ``PriorDict.ln_prob``
    then correctly reports as ``-inf``, an artificial rejection the wrapped
    target does not actually call for.
    """
    sampler, estimator, proposal = periodic_emcee_setup

    scored = []
    original_ln_prob = sampler.priors.ln_prob

    def spy(*args, **kwargs):
        scored.append(args[0].copy())
        return original_ln_prob(*args, **kwargs)

    sampler.priors.ln_prob = spy
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=32, nsteps=200, discard=50)

    sampler._run_emcee(proposal, estimator)

    x_scored = pd.concat(scored)["x"]
    assert len(x_scored) > 0
    assert x_scored.min() >= -1e-9
    assert x_scored.max() <= PERIOD + 1e-9


def test_periodic_samples_recover_the_wrapped_density(periodic_emcee_setup):
    """The returned samples must show the density's wrapped-around tail.

    Regression check in the same style as
    ``test_periodic_proposal.py::test_wrapping_puts_mass_beyond_the_far_edge``:
    with sigma comparable to the period, a real fraction of the mass sits just
    under ``PERIOD`` (the wrapped image of the lower-edge tail).
    """
    sampler, estimator, proposal = periodic_emcee_setup
    sampler.kwargs["emcee_kwargs"] = dict(nwalkers=32, nsteps=2000, discard=500)

    samples, *_ = sampler._run_emcee(proposal, estimator)
    x = samples["x"].values

    assert x.min() >= -1e-9
    assert x.max() <= PERIOD + 1e-9
    assert np.mean(x > 0.85 * PERIOD) > 0.05

    wrapped_deviation = np.mod(x - WRAPPED_TRUE_VALUE + PERIOD / 2, PERIOD) - PERIOD / 2
    assert wrapped_deviation.mean() == pytest.approx(0.0, abs=0.15)
    assert wrapped_deviation.std() == pytest.approx(WRAPPED_SIGMA, abs=0.3)

