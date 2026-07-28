"""Shared comparison utilities for Laplace example scripts."""

import glob
import os
import re

import bilby
import numpy as np

# ---------------------------------------------------------------------------
# Fixed, colourblind-safe colours per sampler family
# ---------------------------------------------------------------------------
# Giving each sampler a fixed colour lets a reader recognise the same method at
# a glance across every example plot.  The palette is Okabe-Ito, designed to
# stay distinguishable under all common forms of colour-vision deficiency; its
# low-contrast yellow is deliberately omitted.  We avoid matplotlib's default
# "C0"/"C1"... cycle so the mapping is stable regardless of the order in which
# results happen to load.
SAMPLER_COLOURS = {
    "laplace": "#E69F00",  # orange          - raw Laplace/Gaussian approximation
    "inprior": "#CC79A7",  # reddish purple
    "rejection": "#D55E00",  # vermillion
    "importance": "#56B4E9",  # sky blue
    "smc": "#009E73",  # bluish green
    "dynesty": "#0072B2",  # blue            - reference nested sampler
}

# Colour for any result whose sampler family is not recognised.
DEFAULT_SAMPLER_COLOUR = "#999999"  # neutral grey

# Colour for injection/truth overlay lines; black keeps them distinct from
# every sampler colour above.
TRUTH_COLOUR = "#000000"


def _label_tokens(label):
    """Lower-cased alphanumeric tokens of a label's basename.

    ``"/tmp/out/RB-SMC-fast"`` -> ``["rb", "smc", "fast"]``.
    """
    return [t for t in re.split(r"[^a-z0-9]+", os.path.basename(str(label)).lower()) if t]


def _override_colour(label, overrides):
    """Colour from *overrides* whose key matches *label*, or ``None``.

    A key matches when its tokens appear as a contiguous run in the label's
    tokens, so ``"smc-fast"`` matches ``"rb-smc-fast"`` and ``"std-smc-fast"``
    but not ``"rb-smc"``.  This keeps an override written in one example free of
    that example's run prefix.  The longest matching key wins, so a specific
    override beats a more general one.
    """
    tokens = _label_tokens(label)
    best = None
    for key, colour in overrides.items():
        key_tokens = _label_tokens(key)
        n = len(key_tokens)
        if n and any(tokens[i : i + n] == key_tokens for i in range(len(tokens) - n + 1)):
            if best is None or n > best[0]:
                best = (n, colour)
    return None if best is None else best[1]


def sampler_family(label):
    """Return the sampler-family key for a result label, or ``None``.

    The examples label results ``"{base}_{method}"`` (e.g. ``"hlv_rejection"``,
    ``"rosenbrock_smc"``) or ``"{base}-{method}"`` (e.g. ``"rb-smc"``).
    Configuration variants collapse to their base family --
    ``"..._rejection_user"`` maps to ``"rejection"`` -- so a method keeps a
    single colour regardless of how it was configured, and either ``_`` or
    ``-`` may separate the tokens.

    The *last* matching token wins: the method follows the base in every label
    (``"{base}_{method}"``), and config-variant suffixes (e.g. ``"user"``) are
    not family keys, so this correctly resolves a base that happens to be named
    after another family (``"laplace_smc"`` -> ``"smc"``, not ``"laplace"``).
    """
    family = None
    for token in _label_tokens(label):
        if token in SAMPLER_COLOURS:
            family = token
    return family


def colours_for_results(results, overrides=None):
    """List of plot colours aligned to *results*, keyed by sampler family.

    Pass directly to ``bilby.core.result.plot_multiple(..., colours=...)``.
    Unrecognised samplers fall back to a neutral grey.

    Parameters
    ----------
    results : list
        Loaded bilby ``Result`` objects.
    overrides : dict, optional
        Maps a label fragment to a colour, taking precedence over the shared
        palette -- e.g. ``{"smc-fast": "#785EF0"}`` colours ``"rb-smc-fast"``
        without touching ``"rb-smc"``.  Use this for a run that is specific to
        one example (a variant of a method, a one-off configuration) so it stays
        distinguishable there without claiming a colour in the global palette,
        which is reserved for methods every example can produce.
    """
    colours = []
    for r in results:
        colour = _override_colour(r.label, overrides) if overrides else None
        if colour is None:
            colour = SAMPLER_COLOURS.get(sampler_family(r.label), DEFAULT_SAMPLER_COLOUR)
        colours.append(colour)
    return colours


def _prettify_method(method):
    """Turn a method token like ``"rejection_user"`` into ``"Rejection user"``.

    ``"smc"`` is upper-cased as an acronym; ``"inprior"`` becomes
    ``"Laplace in-prior"`` (it is the Laplace approximation drawn within the
    prior support); other tokens are capitalised.
    """
    text = method.replace("_", " ").replace("-", " ").strip()
    if not text:
        return method

    def _word(w):
        lw = w.lower()
        if lw == "smc":
            return "SMC"
        if lw == "inprior":
            return "Laplace in-prior"
        return w.capitalize()

    return " ".join(_word(w) for w in text.split())


def _format_duration(secs):
    """Human-readable run time as a bracketed suffix, e.g. ``" (1.2 min)"``."""
    if secs >= 3600:
        return f" ({secs / 3600:.1f} hr)"
    if secs >= 60:
        return f" ({secs / 60:.1f} min)"
    return f" ({secs:.0f} s)"


def _format_efficiency(eff):
    """Format an efficiency percentage without collapsing small values to 0.

    A fixed single decimal (``{:.1f}%``) rounds anything below 0.05% to
    ``0.0%``, making a genuinely tiny-but-nonzero efficiency indistinguishable
    from an exact zero.  Below 0.1% we switch to two significant figures (e.g.
    ``0.0034%``) so the value stays visible; a true zero still prints ``0.0%``.
    """
    if eff <= 0:
        return "0.0%"
    if eff < 0.1:
        return f"{eff:.2g}%"
    return f"{eff:.1f}%"


def sampler_labels(results):
    """Short legend labels naming just the sampler for each result.

    Strips the shared ``{base}_`` prefix that every result in an example carries
    (e.g. ``"gaussian_rejection"`` -> ``"Rejection"``) while keeping
    configuration variants distinct (``"gaussian_rejection_user"`` ->
    ``"Rejection user"``).  Falls back to the full basename when there is no
    common prefix to strip.
    """
    names = [os.path.basename(str(r.label)) for r in results]
    if len(names) > 1:
        prefix = os.path.commonprefix(names)
        # Cut at the last separator (``_`` or ``-``) of the shared prefix, so
        # both ``gaussian_rejection`` and ``rb-rejection`` drop their prefix.
        cut = max(prefix.rfind("_"), prefix.rfind("-")) + 1  # 0 when no shared prefix
        names = [n[cut:] if len(n) > cut else n for n in names]
    return [_prettify_method(n) for n in names]


def overlay_injection_lines(fig, parameters, injection_parameters):
    """Overlay injection truth values as dashed lines on a corner plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The corner plot figure.
    parameters : list
        List of parameter names in the plot.
    injection_parameters : dict
        Dictionary mapping parameter names to their true values.
    """
    if not injection_parameters:
        return

    truths = [injection_parameters.get(p) for p in parameters]
    ndim = len(parameters)
    axes = fig.get_axes()
    if len(axes) != ndim * ndim:
        return

    axes_grid = np.array(axes).reshape(ndim, ndim)
    for row in range(ndim):
        for col in range(ndim):
            ax = axes_grid[row, col]
            if row == col:
                if truths[col] is not None:
                    ax.axvline(truths[col], color=TRUTH_COLOUR, ls="--", lw=1.0)
            elif row > col:
                if truths[col] is not None:
                    ax.axvline(truths[col], color=TRUTH_COLOUR, ls="--", lw=0.8, alpha=0.7)
                if truths[row] is not None:
                    ax.axhline(truths[row], color=TRUTH_COLOUR, ls="--", lw=0.8, alpha=0.7)


def compare(pattern, filename, injection_parameters=None, sampler_only_labels=False, colour_overrides=None):
    """Load result files matching pattern, print comparison table, and create corner plot.

    Parameters
    ----------
    pattern : str
        Glob pattern for result files (e.g., ``/path/to/*_result.*``).
    filename : str
        Path for output corner plot.
    injection_parameters : dict, optional
        Dictionary of injection parameter values to overlay on the plot.
        If None, will be extracted from the first result object if available.
    sampler_only_labels : bool
        If True, legend labels name just the sampler (e.g. ``"Rejection"``),
        dropping the shared example prefix and the run time.  Useful when the
        default ``"{Base}_{method} (time)"`` labels are long enough to crowd the
        legend.  Default False.
    colour_overrides : dict, optional
        Per-example colour overrides, passed to :func:`colours_for_results`.

    Returns
    -------
    results : list
        List of loaded bilby Result objects.
    labels : list
        List of formatted labels for each result.
    """
    logger = bilby.core.utils.logger

    result_files = sorted([f for f in glob.glob(pattern) if not f.endswith(".old")])
    if not result_files:
        logger.warning(f"No result files found matching {pattern}")
        return [], []

    results = []
    labels = []
    for f in result_files:
        try:
            r = bilby.core.result.read_in_result(filename=f)
            results.append(r)
            label = os.path.basename(r.label).capitalize() + _format_duration(r.sampling_time.total_seconds())
            labels.append(label)
            logger.info(f"Loaded {f} ({label})")
        except Exception as exc:
            logger.warning(f"Could not load {f}: {exc}")

    # Optionally shorten the legend to just the sampler name, dropping the shared
    # example prefix but keeping the run time (computed after loading so the
    # shared prefix can be detected across all results).
    if sampler_only_labels and results:
        labels = [
            name + _format_duration(r.sampling_time.total_seconds())
            for name, r in zip(sampler_labels(results), results)
        ]

    # Extract injection parameters from first result if not provided
    if injection_parameters is None and results:
        injection_parameters = getattr(results[0], "injection_parameters", None)

    # Comparison table
    W = 75
    print("\n" + "=" * W)
    print("Comparison")
    print("=" * W)
    print(f"{'Method':<20} {'log Z':>10} {'± σ':>8} {'n_like':>8} {'effic.':>8} {'time':>10}")
    print("-" * W)
    for r, _lab in zip(results, labels):
        # Coerce a missing/None evidence to NaN, but preserve a genuine 0.0 --
        # `x or np.nan` would wrongly turn a real 0.0 evidence into NaN.
        log_z = getattr(r, "log_evidence", None)
        log_z = np.nan if log_z is None else log_z
        log_z_err = getattr(r, "log_evidence_err", None)
        log_z_err = np.nan if log_z_err is None else log_z_err
        secs = r.sampling_time.total_seconds()
        run_stats = r.meta_data.get("run_statistics", {})
        n_like = run_stats.get("nlikelihood", np.nan)
        # The Laplace sampler records its own "efficiency" (final samples per
        # likelihood evaluation).  Other samplers (e.g. dynesty) don't, but
        # bilby stores nlikelihood and neffsamples, so we reconstruct the same
        # quantity: effective independent samples per likelihood evaluation.
        # For the Laplace family the draws are iid, so neff ~= n and the two
        # definitions coincide.
        eff = run_stats.get("efficiency", np.nan)
        if not np.isfinite(eff):
            neff = run_stats.get("neffsamples", np.nan)
            if np.isfinite(neff) and np.isfinite(n_like) and n_like:
                eff = 100.0 * neff / n_like
        name = os.path.basename(r.label)
        n_like_str = f"{int(n_like):>8}" if np.isfinite(n_like) else f"{'—':>8}"
        eff_str = f"{_format_efficiency(eff):>8}" if np.isfinite(eff) else f"{'—':>8}"
        print(f"{name:<20} {log_z:>10.2f} {log_z_err:>8.2f} {n_like_str} {eff_str} {secs:>9.1f}s")
    print("=" * W + "\n")

    import matplotlib.pyplot as plt

    fig = bilby.core.result.plot_multiple(
        results,
        labels=labels,
        colours=colours_for_results(results, overrides=colour_overrides),
        filename=filename,
        titles=False,
        save=False,
    )

    # Overlay injection truth values if provided
    if injection_parameters:
        overlay_injection_lines(fig, results[0].search_parameter_keys, injection_parameters)

    fig.savefig(filename, dpi=400)
    plt.close(fig)
    logger.info(f"Comparison corner plot saved to {filename}")

    return results, labels
