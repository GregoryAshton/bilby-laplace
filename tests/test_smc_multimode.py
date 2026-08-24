"""Tests for the multi-mode initial cloud handed to the SMC (aspire) path.

Aspire offers no way to inject a mixture proposal directly: ``sample_posterior``
uses whatever flow ``fit`` trained as its ``prior_flow``, and that flow is
trained on the initial samples we supply.  So the modes found by
``_find_multiple_maps`` reach the sampler through that cloud and nowhere else --
if the draw is unimodal, the annealing path starts unimodal no matter how many
modes were discovered.  These tests pin the stratification.
"""

import numpy as np
import pytest

from bilby_laplace.sampler import TruncatedMVNProposal

LOWER = np.array([-5.0, -5.0])
UPPER = np.array([5.0, 5.0])
COV = np.diag([0.05, 0.05])
MODE_MEANS = [np.array([-3.0, -3.0]), np.array([0.0, 0.0]), np.array([3.0, 3.0])]


def _proposals(means):
    return [TruncatedMVNProposal(m, COV, lower=LOWER, upper=UPPER) for m in means]


def _assign_to_modes(x, means):
    """Index of the nearest mode centre for each row of *x*."""
    d = np.linalg.norm(x[:, None, :] - np.array(means)[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def test_single_proposal_is_unchanged(sampler):
    """One mode must behave exactly like the pre-existing unimodal draw."""
    proposals = _proposals(MODE_MEANS[:1])

    x = sampler._draw_initial_smc_samples(proposals, 200, ["x", "y"])

    assert x.shape == (200, 2)
    # Everything sits on the single mode, a few sigma wide at most.
    assert np.all(np.abs(x - MODE_MEANS[0]) < 1.0)


@pytest.mark.parametrize("n", [300, 301, 302])
def test_every_mode_is_represented_evenly(sampler, n):
    """Each mode contributes an equal share, and the total is exactly *n*."""
    proposals = _proposals(MODE_MEANS)

    x = sampler._draw_initial_smc_samples(proposals, n, ["x", "y"])

    assert x.shape == (n, 2)
    counts = np.bincount(_assign_to_modes(x, MODE_MEANS), minlength=3)
    # The modes are far apart relative to COV, so the assignment is exact and
    # the counts are the deterministic n // k split plus the remainder.
    expected = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    np.testing.assert_array_equal(counts, expected)


def test_the_cloud_is_not_ordered_by_mode(sampler):
    """Aspire splits the cloud into train/validation by position, so a
    mode-ordered array would train the flow on a subset of the modes."""
    proposals = _proposals(MODE_MEANS)

    x = sampler._draw_initial_smc_samples(proposals, 300, ["x", "y"])

    assignment = _assign_to_modes(x, MODE_MEANS)
    # Every contiguous third must see more than one mode; a mode-ordered cloud
    # would give one distinct value per block.
    for block in np.array_split(assignment, 3):
        assert len(np.unique(block)) > 1


def test_out_of_prior_mode_is_reported_but_not_fatal(sampler, caplog):
    """A mode whose proposal cannot produce in-prior draws must warn rather
    than silently shrink the cloud without explanation."""
    # Second mode sits on the prior edge; the iteration limit stops the draw.
    means = [MODE_MEANS[0], np.array([4.999, 4.999])]
    proposals = [
        TruncatedMVNProposal(means[0], COV, lower=LOWER, upper=UPPER),
        # Bounds that exclude the prior entirely => no in-prior draws.
        TruncatedMVNProposal(means[1], COV, lower=UPPER + 1.0, upper=UPPER + 2.0),
    ]
    sampler.kwargs["max_iterations"] = 1
    sampler.kwargs["fail_on_error"] = False

    with caplog.at_level("WARNING"):
        x = sampler._draw_initial_smc_samples(proposals, 100, ["x", "y"])

    assert len(x) < 100
    assert "Mode 1 contributed only" in caplog.text


# --------------------------------------------------------------------------
# Diagnostic figure cadence.  The stats and evolution figures used to be
# produced only from inside the per-iteration callback, so there was no way to
# get them once at the end -- and re-rendering every iteration is expensive
# (the evolution figure fits a gaussian_kde per parameter per iteration).
# --------------------------------------------------------------------------


def test_plot_every_defaults_to_end_only(sampler):
    assert sampler.kwargs["smc_plot_every"] == 0


def test_figures_are_written_once_at_the_end(sampler, monkeypatch):
    """The end-of-run render must fire even though the callback never plots."""
    calls = []
    monkeypatch.setattr(sampler, "_save_smc_stats_figure", lambda h: calls.append("stats"))
    monkeypatch.setattr(sampler, "_save_smc_evolution_marginals_figure", lambda h, s: calls.append("evolution"))
    sampler.kwargs["plot_diagnostic"] = True

    class _History:
        sample_history = [object()]

    sampler._save_smc_figures(_History(), object())

    assert calls == ["stats", "evolution"]


def test_no_figures_when_diagnostics_are_off(sampler, monkeypatch):
    monkeypatch.setattr(sampler, "_save_smc_stats_figure", lambda h: pytest.fail("should not plot"))
    sampler.kwargs["plot_diagnostic"] = False

    sampler._save_smc_figures(object(), object())  # must simply return


def test_missing_history_is_not_an_error(sampler):
    sampler.kwargs["plot_diagnostic"] = True

    sampler._save_smc_figures(None, None)  # e.g. a run that produced no history


def test_a_failed_figure_does_not_lose_the_run(sampler, monkeypatch, caplog):
    """Plotting is best-effort: a completed run must survive a broken figure."""

    def _boom(*args, **kwargs):
        raise RuntimeError("matplotlib exploded")

    monkeypatch.setattr(sampler, "_save_smc_stats_figure", _boom)
    sampler.kwargs["plot_diagnostic"] = True

    with caplog.at_level("WARNING"):
        sampler._save_smc_figures(object(), object())

    assert "SMC diagnostic plotting failed" in caplog.text
