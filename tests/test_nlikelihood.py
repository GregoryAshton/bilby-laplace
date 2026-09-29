"""``run_statistics["nlikelihood"]`` is every likelihood call the run made.

It used to be the resampling stage's count alone (the proposal draws, or the
count SMC and emcee return), which left out the MAP search, the Hessian,
covariance validation, the mode search and the Laplace evidence. On a GW
problem the MAP search alone was ~90x the 5000 draws ``inprior`` reported. The
estimator now counts every call, whatever stage makes it, and these tests hold
that count to the likelihood's own tally of how often it was called.
"""

import datetime
from multiprocessing.pool import ThreadPool

import bilby
import numpy as np
import pytest
from bilby.core.sampler.base_sampler import _initialize_global_variables
from conftest import CorrelatedGaussianLikelihood

from bilby_laplace.laplace import LaplacePosteriorEstimator


class CountingLikelihood(CorrelatedGaussianLikelihood):
    """The Gaussian, tallying its own calls: the ground truth for the count."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def log_likelihood(self, parameters=None):
        self.calls += 1
        return super().log_likelihood(parameters)


@pytest.mark.parametrize("resample", ["None", "inprior", "rejection", "importance"])
def test_nlikelihood_is_every_call(resample, gaussian_priors, tmp_path):
    """The recorded total equals the calls the likelihood saw, and covers the MAP search."""
    likelihood = CountingLikelihood()
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=gaussian_priors,
        sampler="laplace",
        outdir=str(tmp_path),
        label=f"count_{resample}",
        resample=resample,
        target_nsamples=500,
        plot_diagnostic=False,
        resume=False,
        plot=False,
        save=False,
    )
    stats = result.meta_data["run_statistics"]
    # bilby's base class makes calls of its own before the sampler starts:
    # ``_time_likelihood`` times 100 of them, and ``_verify_parameters`` a
    # couple more. dynesty's count leaves those out too, so the sampler should,
    # and the likelihood's tally exceeds the total by about that much.
    assert 100 <= likelihood.calls - stats["nlikelihood"] <= 110
    assert result.num_likelihood_evaluations == stats["nlikelihood"]
    # The resampling stage's own count is kept alongside, and the total is
    # larger than it by at least the MAP search.
    assert stats["sampling_nfev"] < stats["nlikelihood"]
    assert stats["map_nfev"] > 0
    assert stats["nlikelihood"] - stats["sampling_nfev"] > 0.5 * stats["map_nfev"]


def test_pooled_batch_is_counted(gaussian_priors):
    """The pool's workers bypass ``log_likelihood``, so the batch is counted once, in full."""
    likelihood = CountingLikelihood()
    _initialize_global_variables(
        likelihood=likelihood,
        priors=gaussian_priors,
        search_parameter_keys=["x", "y"],
        use_ratio=False,
        parameters={},
    )
    est = LaplacePosteriorEstimator(likelihood, gaussian_priors)
    with ThreadPool(2) as pool:
        est.pool, est.npool = pool, 2
        x = np.random.default_rng(0).uniform(-1.0, 1.0, size=(2, 37))
        # Two columns outside the prior never reach the likelihood, and are not counted.
        x[0, :2] = 10.0
        est.log_likelihood_from_array(x)
    assert est.n_likelihood_evaluations == 35
    assert likelihood.calls == 35


def test_count_survives_a_resume(gaussian_likelihood, gaussian_priors, tmp_path):
    """A resumed run does not repeat the MAP search, so its calls come back from the checkpoint."""
    from bilby_laplace.sampler import Laplace

    def make():
        sampler = Laplace(likelihood=gaussian_likelihood, priors=gaussian_priors, outdir=str(tmp_path), label="resume")
        # Set by ``run_sampler`` before the resume check; set here the same way.
        sampler.resume_file = f"{tmp_path}/resume_resume.pickle"
        return sampler

    first = make()
    first.start_time = datetime.datetime.now()
    first._estimator = LaplacePosteriorEstimator(gaussian_likelihood, gaussian_priors)
    first._estimator.n_likelihood_evaluations = 123
    first._resumed_nlikelihood = 7
    first._checkpoint_state = dict(mode="inprior", mean=np.zeros(2), cov=np.eye(2))
    first.write_current_state()

    second = make()
    second._resumed_nlikelihood = 0
    assert second._read_saved_state()
    assert second._resumed_nlikelihood == 130
    assert "nlikelihood_total" not in second._checkpoint_state
