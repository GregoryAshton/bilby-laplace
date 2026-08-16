"""Tests for the multi-mode proposal every resampling method draws from.

Until 2026-08 the mode search lived inside ``_run_smc``, so ``n_modes`` was in
practice an SMC-only setting: an ``inprior`` (or ``rejection``, or
``importance``) run built a single Gaussian on the primary MAP however many
modes were requested.  On a multi-modal problem that made the cheap methods
incomparable with SMC -- they were not the same proposal uncorrected, they were
a *different* proposal.  ``Laplace._build_proposal`` now runs before the
resampling branch and returns a :class:`TruncatedMVNMixtureProposal`, and these
tests pin both the mixture's own behaviour and the fact that every method gets
it.
"""

import numpy as np
import pandas as pd
import pytest

from bilby_laplace.sampler import (
    Laplace,
    TruncatedMVNMixtureProposal,
    TruncatedMVNProposal,
)

LOWER = np.array([-5.0, -5.0])
UPPER = np.array([5.0, 5.0])
COV = np.diag([0.05, 0.05])
MODE_MEANS = [np.array([-3.0, -3.0]), np.array([3.0, 3.0])]


def _components(means=MODE_MEANS, cov=COV):
    return [TruncatedMVNProposal(m, cov, lower=LOWER, upper=UPPER) for m in means]


def _nearest_mode(x, means=MODE_MEANS):
    d = np.linalg.norm(x[:, None, :] - np.array(means)[None, :, :], axis=2)
    return np.argmin(d, axis=1)


# --------------------------------------------------------------------------
# The mixture proposal itself
# --------------------------------------------------------------------------


def test_equal_weights_by_default():
    np.testing.assert_allclose(TruncatedMVNMixtureProposal(_components()).weights, [0.5, 0.5])


def test_weights_are_normalised():
    mixture = TruncatedMVNMixtureProposal(_components(), weights=[1.0, 3.0])

    np.testing.assert_allclose(mixture.weights, [0.25, 0.75])


@pytest.mark.parametrize("bad", [[0.5], [0.5, 0.5, 0.0]])
def test_weights_must_match_component_count(bad):
    with pytest.raises(ValueError, match="one entry per component"):
        TruncatedMVNMixtureProposal(_components(), weights=bad)


@pytest.mark.parametrize("bad", [[-0.5, 1.5], [0.0, 0.0], [np.nan, 1.0]])
def test_unusable_weights_are_rejected(bad):
    with pytest.raises(ValueError, match="finite, non-negative"):
        TruncatedMVNMixtureProposal(_components(), weights=bad)


def test_an_empty_mixture_is_rejected():
    with pytest.raises(ValueError, match="at least one component"):
        TruncatedMVNMixtureProposal([])


def test_sampling_follows_the_weights():
    mixture = TruncatedMVNMixtureProposal(_components(), weights=[0.25, 0.75])

    share = _nearest_mode(mixture.sample(4000)).mean()

    assert share == pytest.approx(0.75, abs=0.03)


def test_draws_are_not_ordered_by_component():
    """Callers truncate a batch to the count they wanted.

    ``_run_inprior`` and ``_draw_inprior_samples`` both concatenate batches and
    cut to ``target_nsamples``.  If a batch came out as per-component blocks,
    that truncation would drop the last components preferentially and silently
    reweight the mixture towards the first.
    """
    mixture = TruncatedMVNMixtureProposal(_components())

    first_half = _nearest_mode(mixture.sample(2000)[:1000]).mean()

    assert first_half == pytest.approx(0.5, abs=0.05)


def test_log_prob_is_the_weighted_sum_of_its_components():
    components = _components()
    mixture = TruncatedMVNMixtureProposal(components, weights=[0.25, 0.75])
    x = np.array([[-3.0, -3.0], [3.0, 3.0], [0.0, 0.0]])

    expected = np.log(0.25 * np.exp(components[0].logpdf(x)) + 0.75 * np.exp(components[1].logpdf(x)))

    np.testing.assert_allclose(mixture.logpdf(x), expected, rtol=1e-10)


def test_density_integrates_to_one():
    mixture = TruncatedMVNMixtureProposal(_components(), weights=[0.2, 0.8])
    grid = np.linspace(-5, 5, 400)
    xx, yy = np.meshgrid(grid, grid)
    cell = (grid[1] - grid[0]) ** 2

    total = np.exp(mixture.logpdf(np.column_stack([xx.ravel(), yy.ravel()]))).sum() * cell

    assert total == pytest.approx(1.0, abs=1e-3)


def test_mean_and_cov_name_the_heaviest_component():
    """Not the first, and not necessarily the primary MAP.

    ``mean`` seeds the rejection bound and the diagnostic plots. Modes are
    weighted by Laplace evidence, so a secondary found by the mode search can
    outweigh the point the optimiser started from.
    """
    heavy_cov = np.diag([0.2, 0.2])
    components = [
        TruncatedMVNProposal(MODE_MEANS[0], COV, lower=LOWER, upper=UPPER),
        TruncatedMVNProposal(MODE_MEANS[1], heavy_cov, lower=LOWER, upper=UPPER),
    ]
    mixture = TruncatedMVNMixtureProposal(components, weights=[0.1, 0.9])

    np.testing.assert_allclose(mixture.mean, MODE_MEANS[1])
    np.testing.assert_allclose(mixture.cov, heavy_cov)


def test_a_zero_weight_component_is_legal_and_never_drawn():
    mixture = TruncatedMVNMixtureProposal(_components(), weights=[0.0, 1.0])

    assert _nearest_mode(mixture.sample(500)).all()
    assert np.isfinite(mixture.logpdf(np.array([[3.0, 3.0]]))).all()


# --------------------------------------------------------------------------
# _build_proposal: the mixture reaches every resampling method
# --------------------------------------------------------------------------


def _sampler_with_modes(gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, n_modes, **kwargs):
    """A sampler whose mode search returns two known modes, without optimising."""
    sampler = Laplace(
        likelihood=gaussian_likelihood,
        priors=gaussian_priors,
        outdir=str(tmp_path),
        label="test",
        n_modes=n_modes,
        **kwargs,
    )
    modes = [(m, COV, -float(i)) for i, m in enumerate(MODE_MEANS)]
    monkeypatch.setattr(Laplace, "_find_multiple_maps", lambda *a, **k: list(modes))
    return sampler


def test_single_mode_still_gives_a_plain_proposal(gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, estimator):
    sampler = _sampler_with_modes(gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, n_modes=1)

    proposal, modes, log_weights = sampler._build_proposal(estimator, MODE_MEANS[0], COV, 1)

    assert isinstance(proposal, TruncatedMVNProposal)
    assert len(modes) == 1 and log_weights is None


def test_multiple_modes_give_a_mixture(gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, estimator):
    sampler = _sampler_with_modes(
        gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, n_modes=2, mode_weights="equal"
    )

    proposal, modes, _ = sampler._build_proposal(estimator, MODE_MEANS[0], COV, 1)

    assert isinstance(proposal, TruncatedMVNMixtureProposal)
    assert len(modes) == 2
    # Both modes are actually drawn from -- the regression this file exists for
    # is a proposal that quietly covered only the primary.
    assert set(np.unique(_nearest_mode(proposal.sample(500)))) == {0, 1}


def test_the_search_runs_for_inprior_not_only_for_smc(
    gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, estimator
):
    """`_build_proposal` is called before the resampling branch, so the method
    in use cannot change which proposal is built."""
    built = []
    for resample in ("inprior", "rejection", "importance", "smc"):
        sampler = _sampler_with_modes(
            gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, n_modes=2,
            resample=resample, mode_weights="equal",
        )
        proposal, _, _ = sampler._build_proposal(estimator, MODE_MEANS[0], COV, 1)
        built.append(type(proposal))

    assert built == [TruncatedMVNMixtureProposal] * 4


def test_a_single_surviving_mode_collapses_to_a_plain_proposal(
    gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, estimator
):
    """`_drop_negligible_modes` can leave one component; that is not a mixture.

    It also need not be the primary, so the surviving mode -- not the MAP the
    search started from -- is what the proposal must be centred on.
    """
    sampler = _sampler_with_modes(
        gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, n_modes=2, mode_weights="laplace"
    )
    monkeypatch.setattr(
        Laplace, "_drop_negligible_modes", lambda self, modes, lw: ([modes[1]], np.array([lw[1]]))
    )

    proposal, modes, _ = sampler._build_proposal(estimator, MODE_MEANS[0], COV, 1)

    assert isinstance(proposal, TruncatedMVNProposal)
    assert len(modes) == 1
    np.testing.assert_allclose(proposal.mean, MODE_MEANS[1])


def test_every_resampling_path_is_handed_the_same_proposal(
    gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch
):
    """The invariant, enforced end to end through ``run_sampler``.

    Given the same inputs, ``resample`` must change only what is *done* with
    the proposal, never how it is built. This has now been violated twice --
    the mode search was written inside the SMC branch when it was introduced
    (5b63551), and the 2026-08-05 fix that wired the mixture into aspire's
    initial cloud (b91bd6b) wired it into aspire *only*, leaving inprior,
    rejection and importance on the primary Gaussian. Both times the symptom
    was silent: the kwargs said ``n_modes=3`` and the run did something else.

    So this test does not inspect ``_build_proposal``; it captures the proposal
    object each path actually receives and demands they agree.
    """
    from bilby_laplace.sampler import Laplace as _Laplace

    seen = {}

    def _capture(name, n_returns):
        def fake(self, proposal, *args, **kwargs):
            seen[name] = proposal
            samples = pd.DataFrame({k: np.zeros(4) for k in self.search_parameter_keys})
            logl = np.zeros(4)
            # (samples, logl, g_samples, efficiency) plus whatever else the
            # caller unpacks: evidence pair, or SMC's evidence pair + count.
            return (samples, logl, samples, 100.0, *([0.0] * (n_returns - 4)))

        return fake

    monkeypatch.setattr(_Laplace, "_run_inprior", _capture("inprior", 4))
    monkeypatch.setattr(_Laplace, "_run_rejection_sampling", _capture("rejection", 6))
    monkeypatch.setattr(_Laplace, "_run_importance_sampling", _capture("importance", 6))
    monkeypatch.setattr(_Laplace, "_run_smc", _capture("smc", 7))
    modes = [(m, COV, -float(i)) for i, m in enumerate(MODE_MEANS)]
    monkeypatch.setattr(_Laplace, "_find_multiple_maps", lambda *a, **k: list(modes))

    for resample in ("inprior", "rejection", "importance", "smc"):
        sampler = _Laplace(
            likelihood=gaussian_likelihood,
            priors=gaussian_priors,
            outdir=str(tmp_path),
            label=f"invariant_{resample}",
            resample=resample,
            n_modes=2,
            mode_weights="equal",
            plot_diagnostic=False,
            npool=1,
        )
        sampler.run_sampler()

    assert set(seen) == {"inprior", "rejection", "importance", "smc"}
    reference = seen["inprior"]
    # "All four agree" is necessary but not sufficient: under the old code they
    # agreed on the *single* primary Gaussian, and SMC built its mixture
    # privately afterwards. So demand that what they agree on is the mixture.
    assert isinstance(reference, TruncatedMVNMixtureProposal)
    assert len(reference.components) == 2
    for name, proposal in seen.items():
        assert type(proposal) is type(reference), f"{name} got a different kind of proposal"
        np.testing.assert_allclose(proposal.weights, reference.weights, err_msg=name)
        np.testing.assert_allclose(
            [c.mean for c in proposal.components],
            [c.mean for c in reference.components],
            err_msg=name,
        )


def test_modes_rebuild_the_same_proposal_on_resume(
    gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, estimator
):
    """A resumed run must continue from the proposal it started with.

    The checkpoint carries the modes, not just the primary MAP: re-running the
    search would cost a second multi-start optimisation and is not guaranteed
    to return the same modes, which would change the proposal halfway through a
    run whose earlier samples came from the first one.
    """
    sampler = _sampler_with_modes(
        gaussian_likelihood, gaussian_priors, tmp_path, monkeypatch, n_modes=2, mode_weights="equal"
    )
    proposal, modes, log_weights = sampler._build_proposal(estimator, MODE_MEANS[0], COV, 1)

    rebuilt = sampler._mode_proposal(estimator, modes, log_weights)

    np.testing.assert_allclose(rebuilt.weights, proposal.weights)
    np.testing.assert_allclose(
        [c.mean for c in rebuilt.components], [c.mean for c in proposal.components]
    )
