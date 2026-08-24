"""Tests for the sampler-family colour mapping used by the examples."""

import numpy as np
import pytest

from bilby_laplace.comparison import (
    DEFAULT_SAMPLER_COLOUR,
    METRICS,
    MISSING,
    SAMPLER_COLOURS,
    colours_for_results,
    comparison_table,
    sampler_family,
    sampler_labels,
)


class _FakeResult:
    def __init__(self, label, sampler_kwargs=None):
        self.label = label
        self.sampler_kwargs = sampler_kwargs or {}


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
    assert colours_for_results(results, overrides={"smc-fast": "#785EF0"}) == [SAMPLER_COLOURS["rejection"]]


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


# --------------------------------------------------------------------------
# Table cells.  The printed table and the README table are rendered from one
# set of formatted cells, so these cover both.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (10.081, "10.1s"),
        (59.9, "59.9s"),
        (60.0, "1.0m"),
        (3599.0, "60.0m"),
        (12570.9, "3.5h"),
        (37321.0, "10.4h"),
        (200000.0, "2.3d"),
        (float("nan"), MISSING),
    ],
)
def test_format_time_picks_a_readable_unit(seconds, expected):
    from bilby_laplace.comparison import _format_time

    assert _format_time(seconds) == expected


@pytest.mark.parametrize(
    "n_like, expected",
    [
        (48330000, "48.3"),
        (29213481, "29.2"),
        (5000, "0.005"),  # a fixed 2 decimals would round this to 0.01
        (float("nan"), MISSING),
    ],
)
def test_format_mevals_keeps_small_counts_visible(n_like, expected):
    from bilby_laplace.comparison import _format_mevals

    assert _format_mevals(n_like) == expected


def test_settings_summary_reads_each_samplers_own_layout():
    from bilby_laplace.comparison import _settings_summary

    # The Laplace sampler nests aspire's settings under smc_kwargs...
    laplace_smc = _FakeResult(
        "hlv_smc",
        dict(
            resample="smc",
            smc_kwargs=dict(n_samples=10000, sampler_kwargs=dict(n_steps=100)),
        ),
    )
    # ...while the no-Laplace control goes through aspire's own plugin.
    aspire = _FakeResult(
        "hlv_aspire",
        dict(
            n_samples=10000,
            sample_kwargs=dict(sampler_kwargs=dict(n_steps=100)),
        ),
    )
    assert _settings_summary(laplace_smc) == "nsamples=10000, nsteps=100"
    assert _settings_summary(aspire) == "nsamples=10000, nsteps=100"
    assert _settings_summary(_FakeResult("hlv_dynesty", dict(nlive=1000))) == "nlive=1000"
    # Drawing straight from the proposal has no cost setting worth quoting.
    assert (
        _settings_summary(_FakeResult("hlv_inprior", dict(resample="inprior", smc_kwargs=None, target_nsamples=5000)))
        == ""
    )


def _table_row(**overrides):
    row = dict(
        name="hlv_smc",
        log_z=-12118.41,
        log_z_err=0.03,
        mevals="29.2",
        efficiency="0.034%",
        time="1.9h",
        settings="nsamples=10000",
        metrics={m: dict(mean=3.5, worst="tilt_1", worst_value=12.3) for m in METRICS},
    )
    row.update(overrides)
    return row


def test_comparison_table_has_a_metric_pair_per_metric():
    headers, cells, align = comparison_table([_table_row()], "hlv_dynesty")
    assert len(headers) == len(cells[0]) == len(align)
    # method, log Z, +-, Mevals, efficiency, time, two columns per metric, settings.
    assert len(headers) == 6 + 2 * len(METRICS) + 1
    assert cells[0][-1] == "nsamples=10000"
    assert "tilt_1 (12.30)" in cells[0]


def test_comparison_table_drops_the_metrics_without_a_reference():
    headers, cells, align = comparison_table([_table_row()], None)
    assert len(headers) == len(cells[0]) == len(align) == 7
    assert not any("JSD" in h or "EMD" in h for h in headers)


def test_comparison_table_marks_undefined_cells():
    row = _table_row(
        log_z=float("nan"),
        log_z_err=float("nan"),
        settings="",
        metrics={m: dict(mean=float("nan"), worst=None, worst_value=float("nan")) for m in METRICS},
    )
    _headers, cells, _align = comparison_table([row], "hlv_dynesty")
    assert cells[0].count(MISSING) == 2 + 2 * len(METRICS) + 1


def test_printed_and_markdown_tables_render_the_same_cells():
    from bilby_laplace.comparison import _render_markdown_table, _render_text_table

    headers, cells, align = comparison_table([_table_row()], "hlv_dynesty")
    text, width = _render_text_table(headers, cells, align)
    markdown = _render_markdown_table(headers, cells)
    # Same values in both renderings, whatever the padding and the markdown's
    # code formatting.
    assert all(c in text[-1] for c in cells[0])
    assert [v.strip().strip("`") for v in markdown[-1].strip("|").split("|")] == cells[0]
    assert width == len(text[1])  # the rule spans the whole table
