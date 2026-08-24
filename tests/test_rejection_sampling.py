"""End-to-end tests for ``_run_rejection_sampling`` itself.

Every other test exercising ``resample="rejection"`` monkeypatches this method
away (to check what proposal it is handed, not what it does), so the pre-scan
bound, the main accept/reject loop, and the IS-style evidence estimate it
returns had no test that actually ran them.
"""

import numpy as np
import pytest
from conftest import MU, PRIOR_MAX, PRIOR_MIN, TRUE_COV

from bilby_laplace.laplace import LaplacePosteriorEstimator
from bilby_laplace.sampler import Laplace, TruncatedMVNProposal


@pytest.fixture
def rejection_setup(gaussian_likelihood, gaussian_priors, tmp_path):
    sampler = Laplace(
        likelihood=gaussian_likelihood,
        priors=gaussian_priors,
        outdir=str(tmp_path),
        label="rejection",
    )
    sampler.kwargs["check_point_delta_t"] = 0
    sampler.kwargs["batch_nsamples"] = 500
    estimator = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    lower = np.array([PRIOR_MIN, PRIOR_MIN])
    upper = np.array([PRIOR_MAX, PRIOR_MAX])
    # TruncatedMVNProposal is diagonal (per-marginal), so it only approximates
    # this correlated posterior -- a deliberately imperfect proposal, since
    # rejection sampling must still recover the exact posterior from it.
    proposal = TruncatedMVNProposal(MU, TRUE_COV, lower, upper)
    map_sample_dict = {"x": float(MU[0]), "y": float(MU[1])}
    return sampler, estimator, proposal, map_sample_dict


def test_rejection_sampling_recovers_the_posterior(rejection_setup):
    sampler, estimator, proposal, map_sample_dict = rejection_setup
    sampler._init_checkpoint_state(mode="rejection", mean=MU, cov=TRUE_COV)
    sampler.kwargs["target_nsamples"] = 4000

    samples, logl, g_samples, efficiency, log_evidence, log_evidence_err = sampler._run_rejection_sampling(
        proposal, estimator, map_sample_dict
    )

    assert len(samples) >= 4000
    assert len(logl) == len(samples)
    assert len(g_samples) >= len(samples)
    # The diagonal proposal mismatches the correlated target, so acceptance
    # is well below 100%, but rejection sampling must still be running (not
    # stalled against the iteration limit).
    assert 0.0 < efficiency < 100.0

    recovered_mean = samples[["x", "y"]].mean().to_numpy()
    recovered_cov = np.cov(samples[["x", "y"]].to_numpy(), rowvar=False)
    np.testing.assert_allclose(recovered_mean, MU, atol=0.05)
    np.testing.assert_allclose(recovered_cov, TRUE_COV, atol=0.05)

    assert np.isfinite(log_evidence)
    assert np.isfinite(log_evidence_err)
    assert log_evidence_err >= 0.0


def test_rejection_sampling_resume_reregisters_accumulators(rejection_setup):
    """After a resumed run the checkpoint payload must reflect the samples
    appended during the resume -- not just the pre-resume ones (see the
    identical invariant tested for ``_run_inprior`` in test_checkpoint_resume.py)."""
    sampler, estimator, proposal, map_sample_dict = rejection_setup
    sampler._init_checkpoint_state(mode="rejection", mean=MU, cov=TRUE_COV)

    sampler.kwargs["target_nsamples"] = 200
    sampler._run_rejection_sampling(proposal, estimator, map_sample_dict)
    state = sampler._checkpoint_state
    accepted_after_first_run = state["n_accepted"]
    assert accepted_after_first_run >= 200

    assert "all_samples" in sampler._checkpoint_state  # triggers the resume path
    sampler.kwargs["target_nsamples"] = 500
    samples, *_ = sampler._run_rejection_sampling(proposal, estimator, map_sample_dict)

    state = sampler._checkpoint_state
    assert state["n_accepted"] >= 500
    assert state["n_accepted"] > accepted_after_first_run
    # The regression assertion: the samples reachable from the checkpoint must
    # match the counter. Before the fix the payload still referenced the
    # pre-resume list, so this count would lag n_accepted.
    assert sum(len(a) for a in state["all_samples"]) == state["n_accepted"]
    assert len(samples) >= 500
