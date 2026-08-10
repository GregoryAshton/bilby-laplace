"""Tests that the reported evidence does not depend on ``use_ratio``.

The estimator always evaluates the *full* log-likelihood, so every evidence the
sampler computes is a full log Z.  bilby, however, post-processes what a sampler
returns according to ``use_ratio``: under ``use_ratio=True`` it treats the
returned value as a log Bayes factor and adds the noise evidence back on top.
Returning a full log Z under that convention inflates the result by one whole
noise evidence -- on a GW likelihood that is ~13000 nats, which silently made
``log_evidence`` incomparable with dynesty's.

These tests run the real ``bilby.run_sampler`` so bilby's own post-processing is
in the loop; testing the sampler in isolation would miss the half of the
round-trip where the bug lived.
"""

import bilby
import numpy as np
import pytest

from conftest import MU, PRIOR_MAX, PRIOR_MIN, TRUE_COV, CorrelatedGaussianLikelihood

# A big, distinctive noise evidence: large enough that an omitted or doubled
# term cannot hide inside sampling scatter, and not a round number that could
# coincide with something else.
NOISE_LOG_L = -1234.5

# The likelihood is normalised and the prior is uniform over a square, so
# Z = \int L pi = 1 / (prior area) exactly.
PRIOR_AREA = (PRIOR_MAX - PRIOR_MIN) ** 2
TRUE_LOG_Z = -np.log(PRIOR_AREA)


class NoisyGaussianLikelihood(CorrelatedGaussianLikelihood):
    """The shared Gaussian, with a non-trivial noise evidence.

    The base fixture inherits bilby's default ``noise_log_likelihood`` of NaN,
    which would swallow exactly the arithmetic under test.
    """

    def noise_log_likelihood(self):
        return NOISE_LOG_L


@pytest.fixture
def noisy_likelihood():
    return NoisyGaussianLikelihood()


def _run(likelihood, priors, tmp_path, use_ratio):
    return bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="laplace",
        outdir=str(tmp_path),
        label=f"ratio_{use_ratio}",
        use_ratio=use_ratio,
        resample="importance",
        target_nsamples=2000,
        plot_diagnostic=False,
        resume=False,
        plot=False,
        save=False,
    )


@pytest.mark.parametrize("use_ratio", [False, True])
def test_log_evidence_is_the_full_evidence_either_way(noisy_likelihood, gaussian_priors, tmp_path, use_ratio):
    """``result.log_evidence`` is the full log Z under both conventions."""
    result = _run(noisy_likelihood, gaussian_priors, tmp_path, use_ratio)

    assert result.log_evidence == pytest.approx(TRUE_LOG_Z, abs=0.05)


@pytest.mark.parametrize("use_ratio", [False, True])
def test_log_bayes_factor_is_the_evidence_less_the_noise(noisy_likelihood, gaussian_priors, tmp_path, use_ratio):
    """The two bilby fields stay consistent with each other and with the noise."""
    result = _run(noisy_likelihood, gaussian_priors, tmp_path, use_ratio)

    assert result.log_noise_evidence == NOISE_LOG_L
    assert result.log_bayes_factor == pytest.approx(result.log_evidence - NOISE_LOG_L, abs=1e-6)
    assert result.log_bayes_factor == pytest.approx(TRUE_LOG_Z - NOISE_LOG_L, abs=0.05)


def test_the_two_conventions_agree(noisy_likelihood, gaussian_priors, tmp_path):
    """The headline invariant: `use_ratio` must not move the evidence.

    Before the fix these differed by exactly one noise evidence.
    """
    full = _run(noisy_likelihood, gaussian_priors, tmp_path, False)
    ratio = _run(noisy_likelihood, gaussian_priors, tmp_path, True)

    assert full.log_evidence == pytest.approx(ratio.log_evidence, abs=0.05)
    assert abs(full.log_evidence - ratio.log_evidence) < abs(NOISE_LOG_L) / 100


@pytest.mark.parametrize("use_ratio", [False, True])
def test_stored_log_likelihoods_match_the_convention(noisy_likelihood, gaussian_priors, tmp_path, use_ratio):
    """Per-sample values must sit on the same footing as the evidence.

    Shifting one without the other is precisely how the two drifted apart.
    """
    result = _run(noisy_likelihood, gaussian_priors, tmp_path, use_ratio)

    logl = np.asarray(result.log_likelihood_evaluations, dtype=float)
    peak_full = noisy_likelihood.log_likelihood(dict(zip(("x", "y"), MU)))
    offset = -NOISE_LOG_L if use_ratio else 0.0

    # The peak of the sampled log-likelihoods approaches the analytic peak from
    # below; a whole noise evidence out would be unmissable against this bound.
    assert logl.max() < peak_full + offset + 1e-6
    assert logl.max() > peak_full + offset - 1.0


def test_covariance_is_unaffected_by_the_convention(noisy_likelihood, gaussian_priors, tmp_path):
    """A constant offset cannot move the MAP or the curvature.

    This is what licenses the fix being a shift at the end rather than a switch
    to ``log_likelihood_ratio`` throughout the estimator.
    """
    full = _run(noisy_likelihood, gaussian_priors, tmp_path, False)
    ratio = _run(noisy_likelihood, gaussian_priors, tmp_path, True)

    for key, truth, sigma in (("x", MU[0], np.sqrt(TRUE_COV[0, 0])), ("y", MU[1], np.sqrt(TRUE_COV[1, 1]))):
        assert full.posterior[key].mean() == pytest.approx(truth, abs=0.1)
        assert ratio.posterior[key].mean() == pytest.approx(truth, abs=0.1)
        assert ratio.posterior[key].std() == pytest.approx(sigma, rel=0.2)
