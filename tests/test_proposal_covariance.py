"""``proposal_covariance``: the independent default, and the full-covariance option."""

import bilby
import numpy as np
import pytest
from conftest import RHO, CorrelatedGaussianLikelihood

import bilby_laplace.sampler as sampler_module
from bilby_laplace.sampler import CorrelatedTruncatedMVNProposal, TruncatedMVNProposal

MEAN = np.array([1.0, -0.5])
COV = np.array([[0.09, 0.105], [0.105, 0.25]])  # correlation 0.7


def _grid_integral(proposal, lower, upper, n=400):
    gx = np.linspace(lower[0], upper[0], n)
    gy = np.linspace(lower[1], upper[1], n)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    density = np.exp(proposal.logpdf(np.column_stack([xx.ravel(), yy.ravel()]))).reshape(n, n)
    return np.trapezoid(np.trapezoid(density, gy, axis=1), gx)


def test_full_keeps_the_correlation_and_the_box():
    lower, upper = np.array([0.6, -1.0]), np.array([1.4, 0.2])  # cuts the Gaussian
    prop = CorrelatedTruncatedMVNProposal(MEAN, COV, lower, upper)
    x = prop.sample(20000)
    assert x.shape == (20000, 2)
    assert np.all(x >= lower) and np.all(x <= upper)
    # Truncation lowers the correlation, so compare against a brute-force chop.
    ref = np.random.default_rng(0).multivariate_normal(MEAN, COV, 400_000)
    ref = ref[np.all((ref >= lower) & (ref <= upper), axis=1)]
    assert np.corrcoef(x.T)[0, 1] == pytest.approx(np.corrcoef(ref.T)[0, 1], abs=0.03)
    # The diagonal proposal on the same box has none.
    assert abs(np.corrcoef(TruncatedMVNProposal(MEAN, COV, lower, upper).sample(20000).T)[0, 1]) < 0.03


def test_full_logpdf_is_normalised_on_the_box():
    lower, upper = np.array([0.6, -1.0]), np.array([1.4, 0.2])
    prop = CorrelatedTruncatedMVNProposal(MEAN, COV, lower, upper, n_box_estimate=400_000)
    assert 0.3 < prop.p_box < 0.9
    assert _grid_integral(prop, lower, upper) == pytest.approx(1.0, abs=0.01)
    assert np.isneginf(prop.logpdf(np.array([[2.0, 0.0]]))[0])


def test_full_logpdf_is_normalised_with_a_periodic_coordinate():
    # A mean near the seam of a periodic x, so the wrapped images matter.
    lower, upper = np.array([0.0, -3.0]), np.array([1.0, 2.0])
    mean = np.array([0.05, -0.5])
    prop = CorrelatedTruncatedMVNProposal(mean, COV, lower, upper, periodic=[True, False], n_box_estimate=400_000)
    x = prop.sample(5000)
    assert np.all((x[:, 0] >= 0) & (x[:, 0] < 1))
    assert _grid_integral(prop, lower, upper) == pytest.approx(1.0, abs=0.01)


def test_full_refuses_a_box_holding_none_of_the_gaussian():
    with pytest.raises(sampler_module.SamplerError):
        CorrelatedTruncatedMVNProposal(MEAN, COV * 1e-4, np.array([3.0, 3.0]), np.array([4.0, 4.0]))


def _run(tmp_path, resample, **kwargs):
    priors = bilby.core.prior.PriorDict(
        dict(x=bilby.core.prior.Uniform(-5, 5, "x"), y=bilby.core.prior.Uniform(-5, 5, "y"))
    )
    return bilby.run_sampler(
        likelihood=CorrelatedGaussianLikelihood(),
        priors=priors,
        sampler="laplace",
        outdir=str(tmp_path),
        label=f"{resample}_{kwargs.get('proposal_covariance', 'default')}",
        resample=resample,
        target_nsamples=2000,
        plot_diagnostic=False,
        resume=False,
        plot=False,
        save=False,
        **kwargs,
    )


def _record_proposal_class(monkeypatch):
    built = []
    original = sampler_module.Laplace._proposal_class
    monkeypatch.setattr(
        sampler_module.Laplace, "_proposal_class", lambda self: built.append(original(self)) or built[-1]
    )
    return built


def test_inprior_defaults_to_full(tmp_path, monkeypatch):
    # inprior returns the proposal as the posterior, so it gets the Laplace
    # approximation itself.
    built = _record_proposal_class(monkeypatch)
    result = _run(tmp_path, "inprior")
    assert built == [CorrelatedTruncatedMVNProposal]
    assert result.meta_data["proposal_covariance"] == "full"
    assert np.corrcoef(result.posterior[["x", "y"]].to_numpy().T)[0, 1] == pytest.approx(RHO, abs=0.05)


def test_seeding_methods_default_to_diagonal(tmp_path, monkeypatch):
    # Every other method corrects the proposal, where a wider seed is better.
    built = _record_proposal_class(monkeypatch)
    result = _run(tmp_path, "rejection")
    assert built == [TruncatedMVNProposal]
    assert result.meta_data["proposal_covariance"] == "diagonal"


def test_explicit_diagonal_inprior_drops_the_correlation(tmp_path):
    result = _run(tmp_path, "inprior", proposal_covariance="diagonal")
    assert result.meta_data["proposal_covariance"] == "diagonal"
    assert abs(np.corrcoef(result.posterior[["x", "y"]].to_numpy().T)[0, 1]) < 0.1


def test_inprior_default_falls_back_to_diagonal_with_prior_parameters(tmp_path, monkeypatch):
    built = _record_proposal_class(monkeypatch)
    result = _run(tmp_path, "inprior", prior_parameters=["y"])
    assert built == [TruncatedMVNProposal]
    assert result.meta_data["proposal_covariance"] == "diagonal"


def test_full_inprior_recovers_the_gaussian_correlation(tmp_path):
    # The Laplace approximation is exact here, so `inprior` from the full
    # covariance is the posterior itself.
    result = _run(tmp_path, "inprior", proposal_covariance="full")
    assert np.corrcoef(result.posterior[["x", "y"]].to_numpy().T)[0, 1] == pytest.approx(RHO, abs=0.05)


def test_full_rejection_is_more_efficient_on_a_gaussian(tmp_path):
    diagonal = _run(tmp_path, "rejection")
    full = _run(tmp_path, "rejection", proposal_covariance="full")
    eff = lambda r: r.meta_data["run_statistics"]["efficiency"]  # noqa: E731
    assert eff(full) > 2 * eff(diagonal)
    assert full.log_evidence == pytest.approx(diagonal.log_evidence, abs=0.1)


def test_invalid_choice_and_prior_parameters_are_refused(tmp_path):
    with pytest.raises(sampler_module.SamplerError, match="proposal_covariance"):
        _run(tmp_path, "inprior", proposal_covariance="banana")
    with pytest.raises(sampler_module.SamplerError, match="prior_parameters"):
        _run(tmp_path, "inprior", proposal_covariance="full", prior_parameters=["y"])
