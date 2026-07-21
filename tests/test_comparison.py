"""Tests for the sampler-family colour mapping used by the examples."""

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
