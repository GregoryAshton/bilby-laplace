"""Unit tests for the proposal / flow helpers in :mod:`bilby_laplace.sampler`."""

import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform

from bilby_laplace.sampler import (
    GaussianFlow,
    GaussianMixtureFlow,
    Laplace,
    TruncatedMVNProposal,
    kish_log_effective_sample_size,
)

MEAN = np.array([1.0, -0.5])
COV = np.array([[0.09, 0.105], [0.105, 0.25]])
LOWER = np.array([-5.0, -5.0])
UPPER = np.array([5.0, 5.0])


# ---------------------------------------------------------------------------
# TruncatedMVNProposal
# ---------------------------------------------------------------------------
@pytest.fixture
def proposal():
    return TruncatedMVNProposal(MEAN, COV, LOWER, UPPER)


def test_truncated_sample_shape(proposal):
    x = proposal.sample(100)
    assert x.shape == (100, 2)


def test_truncated_sample_within_bounds():
    # A tight prior that truncates a wide Gaussian: every draw must land inside.
    lower = np.array([0.8, -0.7])
    upper = np.array([1.2, -0.3])
    prop = TruncatedMVNProposal(MEAN, COV, lower, upper)
    x = prop.sample(5000)
    assert np.all(x >= lower) and np.all(x <= upper)


def test_truncated_logpdf_shape(proposal):
    x = proposal.sample(50)
    assert proposal.logpdf(x).shape == (50,)


def test_truncated_logpdf_single_point(proposal):
    lp = proposal.logpdf(MEAN.reshape(1, -1))
    assert lp.shape == (1,)
    assert np.isfinite(lp[0])


def test_truncated_marginals_recover_mean_and_sigma():
    """With bounds far from the mean the marginals are untruncated normals."""
    prop = TruncatedMVNProposal(MEAN, COV, LOWER, UPPER)
    x = prop.sample(50000)
    np.testing.assert_allclose(x.mean(axis=0), MEAN, atol=0.02)
    # Only the diagonal (marginal std) is used by this proposal.
    np.testing.assert_allclose(x.std(axis=0), np.sqrt(np.diag(COV)), rtol=0.05)


# ---------------------------------------------------------------------------
# _effective_log_proposal (prior_parameters density correction)
# ---------------------------------------------------------------------------
def _bare_laplace(prior_parameters):
    """A Laplace instance with just the attributes _effective_log_proposal
    needs, bypassing the heavy Sampler.__init__."""
    obj = Laplace.__new__(Laplace)
    obj.kwargs = {"prior_parameters": prior_parameters}
    obj.priors = PriorDict(dict(x=Uniform(-5, 5, "x"), y=Uniform(-5, 5, "y")))
    return obj


def test_effective_log_proposal_no_prior_params_is_plain_logpdf(proposal):
    """With no prior_parameters the effective density is the proposal logpdf."""
    lap = _bare_laplace(prior_parameters=[])
    x = proposal.sample(20)
    np.testing.assert_allclose(
        lap._effective_log_proposal(proposal, x, ["x", "y"]),
        proposal.logpdf(x),
    )


def test_effective_log_proposal_cancels_replaced_dimension(proposal):
    """For a prior-replaced dimension, logpi - log_g must cancel on that dim,
    leaving only the non-replaced dims' importance weight.

    The pre-fix bug used the truncated-normal marginal for the replaced dim,
    leaving a spurious truncnorm_y/prior_y factor in the weight."""
    import pandas as pd

    lap = _bare_laplace(prior_parameters=["y"])
    x = proposal.sample(20)

    logpi = np.array(lap.priors.ln_prob(pd.DataFrame(x, columns=["x", "y"]), axis=0))
    log_g_eff = lap._effective_log_proposal(proposal, x, ["x", "y"])
    weight_term = logpi - log_g_eff  # what enters ln_r beyond logl

    # y cancels: only the x (non-replaced) dim should remain.
    prior_x = lap.priors["x"].ln_prob(x[:, 0])
    truncnorm_x = proposal._dists[0].logpdf(x[:, 0])
    np.testing.assert_allclose(weight_term, prior_x - truncnorm_x, atol=1e-9)

    # And this genuinely differs from the buggy (uncorrected) version, which
    # would carry a non-zero y contribution.
    buggy_term = logpi - proposal.logpdf(x)
    assert not np.allclose(weight_term, buggy_term)


# ---------------------------------------------------------------------------
# GaussianFlow
# ---------------------------------------------------------------------------
def test_gaussian_flow_log_prob_matches_scipy():
    from scipy.stats import multivariate_normal

    flow = GaussianFlow(MEAN, COV)
    x = np.array([[1.0, -0.5], [1.2, -0.3]])
    expected = multivariate_normal(mean=MEAN, cov=COV).logpdf(x)
    np.testing.assert_allclose(flow.log_prob(x), expected)


def test_gaussian_flow_sample_and_log_prob():
    flow = GaussianFlow(MEAN, COV)
    x, log_prob = flow.sample_and_log_prob(1000)
    assert x.shape == (1000, 2)
    assert log_prob.shape == (1000,)
    np.testing.assert_allclose(x.mean(axis=0), MEAN, atol=0.05)


# ---------------------------------------------------------------------------
# GaussianMixtureFlow
# ---------------------------------------------------------------------------
def test_mixture_flow_log_prob_is_logsumexp_of_components():
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    means = [MEAN, np.array([-2.0, 2.0])]
    covs = [COV, COV]
    flow = GaussianMixtureFlow(means, covs)

    x = np.array([[1.0, -0.5], [-2.0, 2.0]])
    comp = np.array([multivariate_normal(mean=m, cov=c).logpdf(x) for m, c in zip(means, covs)])
    expected = logsumexp(comp - np.log(2), axis=0)
    np.testing.assert_allclose(flow.log_prob(x), expected)


def test_mixture_flow_sample_and_log_prob_shapes():
    means = [MEAN, np.array([-2.0, 2.0])]
    flow = GaussianMixtureFlow(means, [COV, COV])
    x, log_prob = flow.sample_and_log_prob(500)
    assert x.shape == (500, 2)
    assert log_prob.shape == (500,)


# ---------------------------------------------------------------------------
# kish_log_effective_sample_size
# ---------------------------------------------------------------------------
def test_kish_uniform_weights_equal_n():
    ln_w = np.zeros(100)  # all weights equal -> ESS = N
    assert np.exp(kish_log_effective_sample_size(ln_w)) == pytest.approx(100.0)


def test_kish_degenerate_weight_near_one():
    # One dominant weight, the rest negligible -> ESS approaches 1.
    ln_w = np.array([0.0, -50.0, -50.0, -50.0])
    assert np.exp(kish_log_effective_sample_size(ln_w)) == pytest.approx(1.0, abs=1e-3)


def test_kish_all_neg_inf_returns_neg_inf():
    ln_w = np.full(5, -np.inf)
    assert kish_log_effective_sample_size(ln_w) == -np.inf


def test_kish_ignores_neg_inf_entries():
    # Two finite equal weights plus -inf padding -> ESS = 2.
    ln_w = np.array([0.0, 0.0, -np.inf, -np.inf])
    assert np.exp(kish_log_effective_sample_size(ln_w)) == pytest.approx(2.0)
