"""Tests for the sampler-family colour mapping used by the examples."""

import numpy as np
import pytest

from bilby_laplace.comparison import (
    DEFAULT_SAMPLER_COLOUR,
    SAMPLER_COLOURS,
    colours_for_results,
    sampler_family,
    sampler_labels,
)


class _FakeResult:
    def __init__(self, label):
        self.label = label


@pytest.mark.parametrize(
    "label, expected",
    [
        ("hlv_rejection", "rejection"),
        ("hlv_rejection_user", "rejection"),  # config variant folds into base family
        ("rosenbrock_smc", "smc"),
        ("rb-smc", "smc"),
        ("rb-smc-fast", "smc"),  # example-local variants stay in the base family
        ("gaussian_dynesty", "dynesty"),
        ("bns_laplace", "laplace"),
        ("x_inprior", "inprior"),
        ("roq_importance", "importance"),
        ("something_weird", None),
    ],
)
def test_sampler_family(label, expected):
    assert sampler_family(label) == expected


def test_family_extraction_is_path_and_case_insensitive():
    assert sampler_family("/tmp/outdir/HLV_Rejection") == "rejection"


def test_colours_for_results_aligns_and_defaults():
    results = [
        _FakeResult("hlv_rejection"),
        _FakeResult("hlv_smc"),
        _FakeResult("hlv_mystery"),
    ]
    assert colours_for_results(results) == [
        SAMPLER_COLOURS["rejection"],
        SAMPLER_COLOURS["smc"],
        DEFAULT_SAMPLER_COLOUR,
    ]


def test_colour_overrides_beat_the_palette_without_touching_siblings():
    # A single example can distinguish a variant of a family (here an
    # under-converged "smc-fast" run) without that variant entering the shared
    # palette.  The override key carries no run prefix, so it matches whichever
    # likelihood the example was run with.
    results = [_FakeResult("rb-smc"), _FakeResult("rb-smc-fast"), _FakeResult("mb-smc-fast")]
    assert colours_for_results(results, overrides={"smc-fast": "#785EF0"}) == [
        SAMPLER_COLOURS["smc"],
        "#785EF0",
        "#785EF0",
    ]


def test_colour_overrides_prefer_the_longest_match():
    results = [_FakeResult("rb-smc-fast")]
    overrides = {"smc": "#111111", "smc-fast": "#222222"}
    assert colours_for_results(results, overrides=overrides) == ["#222222"]


def test_colour_overrides_require_contiguous_tokens():
    # "smc-fast" must not match a label that merely contains both tokens apart.
    results = [_FakeResult("rb-smc-rejection-fast")]
    assert colours_for_results(results, overrides={"smc-fast": "#785EF0"}) == [
        SAMPLER_COLOURS["rejection"]
    ]


def test_palette_is_hex_and_unique():
    values = list(SAMPLER_COLOURS.values()) + [DEFAULT_SAMPLER_COLOUR]
    assert all(v.startswith("#") and len(v) == 7 for v in values)
    assert len(set(values)) == len(values)  # every family visually distinct


def test_sampler_labels_strip_shared_prefix_and_prettify():
    results = [
        _FakeResult("gaussian_laplace"),
        _FakeResult("gaussian_rejection"),
        _FakeResult("gaussian_rejection_user"),  # variant stays distinct
        _FakeResult("gaussian_smc"),  # acronym upper-cased
        _FakeResult("gaussian_dynesty"),
    ]
    assert sampler_labels(results) == [
        "Laplace",
        "Rejection",
        "Rejection User",
        "SMC",
        "Dynesty",
    ]


def test_sampler_labels_single_result_has_no_prefix_to_strip():
    # With one result there is no shared prefix; the full basename is kept.
    assert sampler_labels([_FakeResult("gaussian_smc")]) == ["Gaussian SMC"]


# --------------------------------------------------------------------------
# Likelihood-evaluation counts in the comparison table.  `run_statistics` is
# written by this package's sampler and by bilby's dynesty, but not by
# third-party plugins such as aspire's, which populate only bilby's standard
# `num_likelihood_evaluations`.  Reading one field alone printed a dash for a
# count the result was carrying.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (1330000, 1330000.0),
        (1330000.0, 1330000.0),
        ("1330000", 1330000.0),
        (None, None),  # not recorded
        (0, None),  # resumed run / untracked plugin -- never a real measurement
        (-5, None),
        ("nonsense", None),
        (float("nan"), None),
    ],
)
def test_as_count_rejects_unusable_values(value, expected):
    from bilby_laplace.comparison import _as_count

    result = _as_count(value)
    if expected is None:
        assert not np.isfinite(result)
    else:
        assert result == expected
