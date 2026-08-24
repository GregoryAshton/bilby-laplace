"""Regression tests for checkpoint/resume of the batched resampling loops.

These guard against an accumulator-desync bug: on resume the sample
accumulator lists are rebuilt as fresh copies, so they must be re-registered
into ``_checkpoint_state`` -- otherwise the checkpoint keeps pointing at the
loaded (pre-resume) lists and silently drops everything appended after the
resume, corrupting a subsequent resume's samples and evidence.
"""

import numpy as np
import pytest
from conftest import MU, PRIOR_MAX, PRIOR_MIN, TRUE_COV

from bilby_laplace.laplace import LaplacePosteriorEstimator
from bilby_laplace.sampler import Laplace, TruncatedMVNProposal


@pytest.fixture
def inprior_setup(gaussian_likelihood, gaussian_priors, tmp_path):
    sampler = Laplace(
        likelihood=gaussian_likelihood,
        priors=gaussian_priors,
        outdir=str(tmp_path),
        label="ckpt",
    )
    # Disable periodic file checkpointing; these tests exercise the in-memory
    # state invariant, not the on-disk pickle.
    sampler.kwargs["check_point_delta_t"] = 0
    sampler.kwargs["batch_nsamples"] = 10
    estimator = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    lower = np.array([PRIOR_MIN, PRIOR_MIN])
    upper = np.array([PRIOR_MAX, PRIOR_MAX])
    proposal = TruncatedMVNProposal(MU, TRUE_COV, lower, upper)
    return sampler, estimator, proposal


def _samples_in_state(state):
    """Total accepted samples currently reachable from the checkpoint payload."""
    return sum(len(a) for a in state["samples_list"])


def test_inprior_resume_reregisters_accumulators(inprior_setup):
    """After a resumed run the checkpoint payload must reflect the samples
    appended during the resume -- not just the pre-resume ones."""
    sampler, estimator, proposal = inprior_setup

    # --- Stage A: initial run to a small target (as run_sampler would) ---
    sampler._init_checkpoint_state(mode="inprior", mean=MU, cov=TRUE_COV)
    sampler.kwargs["target_nsamples"] = 20
    sampler._run_inprior(proposal, estimator)

    state = sampler._checkpoint_state
    assert state["n_accepted"] >= 20
    # Sanity: pre-resume the payload is already consistent.
    assert _samples_in_state(state) == state["n_accepted"]
    accepted_after_A = state["n_accepted"]

    # --- Stage B: resume (checkpoint state still present) to a larger target ---
    assert "samples_list" in sampler._checkpoint_state  # triggers the resume path
    sampler.kwargs["target_nsamples"] = 50
    samples, _, _, _ = sampler._run_inprior(proposal, estimator)

    state = sampler._checkpoint_state
    assert state["n_accepted"] >= 50
    assert state["n_accepted"] > accepted_after_A  # the resume actually drew more

    # The regression assertion: the samples reachable from the checkpoint must
    # match the counter. Before the fix the payload still referenced the
    # pre-resume list, so this count would lag n_accepted.
    assert _samples_in_state(state) == state["n_accepted"]

    # And the returned posterior has the full requested size.
    assert len(samples) == 50
