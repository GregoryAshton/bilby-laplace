"""Shared comparison utilities for Laplace example scripts."""

import glob
import os
import re

import bilby
import numpy as np
from scipy.stats import entropy, gaussian_kde, wasserstein_distance

# JS divergences are a few thousandths of a nat on these problems, which is
# unreadable; millibits is the unit the numbers are quoted in.
MBITS_PER_NAT = 1000.0 / np.log(2)

# The two per-parameter agreement metrics, in the order they appear in the
# table. JSD is scale-free by construction; the earth-mover distance is not, so
# it is divided by the reference posterior's standard deviation for that
# parameter and reported in units of the reference sigma -- a shift of the whole
# posterior by one sigma scores 1.0, whatever the parameter's units.
#
# They answer different questions and disagree usefully: JSD is dominated by
# shape and width mismatch and saturates once two densities barely overlap,
# while the EMD keeps growing with the displacement and so still distinguishes
# "badly offset" from "hopelessly offset".
METRICS = ("jsd", "emd")
METRIC_UNITS = {"jsd": "mbits", "emd": "sigma"}

# Printed for any table cell whose value was not recorded or is undefined.
MISSING = "—"

# Samples per side of every JS divergence, fixed for every result and every
# example. The estimator is biased upward at finite N (roughly as 1/N), so a
# dynesty posterior (~4500 samples) compared with an SMC one (10000) at their
# natural sizes is penalised by about a factor of two on sample count alone.
#
# A constant rather than "the smallest posterior present": deriving it from the
# results made the number depend on which runs happened to be on disk, so a
# value could not be compared with the same example's value from last week, let
# alone with another example's.
#
# 2000 rather than 1000: at 1000 the noise floor is 1.3-2.6 mbits, which is
# larger than the divergences on the examples that agree well -- every gaussian
# value sat below its own floor, and the ranking of the four samplers reversed
# purely from the sample count. 2000 roughly halves that floor while still
# fitting inside every posterior here, the smallest being a 2727-sample dynesty
# run. It also matches the configuration study in paper/, which settled on the
# same value for the same reason.
#
# Results shorter than this are reported as "—" rather than compared at a
# smaller size, which would silently reintroduce exactly that problem.
JSD_N = 2000

# Fixed, so re-running the comparison on unchanged results reproduces the table.
_JSD_RNG_SEED = 20260810

SAMPLER_COLOURS = {
    "laplace": "#009E73",  # bluish green    - raw Laplace/Gaussian approximation
    "inprior": "#CC79A7",  # reddish purple
    "rejection": "#D55E00",  # vermillion
    "importance": "#56B4E9",  # sky blue
    "smc": "#E69F00",  # orange              - the headline method
    "aspire": "#785EF0",  # violet           - SMC from the prior, no-Laplace control
    "dynesty": "#0072B2",  # blue            - reference nested sampler
}

# Colour for any result whose sampler family is not recognised.
DEFAULT_SAMPLER_COLOUR = "#999999"  # neutral grey

# Colour for injection/truth overlay lines; black keeps them distinct from
# every sampler colour above.
TRUTH_COLOUR = "#000000"

# Distribution names (as pip/PyPI knows them, not import names -- aspire's
# import name is `aspire`, not `aspire-inference`), pulled per-row out of each
# result's own `meta_data["environment_packages"]` (bilby's snapshot of the
# conda/pip environment active when *that* result was produced). Motivated
# directly by the minipcn regression this project has already hit once: a
# same-numbered package reinstall silently changed SMC's behaviour, and
# nothing about the example's own output would have caught it without the
# exact version being on record next to the row it produced. Per-row, not a
# single environment-wide query at `make compare` time, because different
# rows can come from results generated at different times (or copied in from
# elsewhere) under different environments -- a single query would silently
# misattribute those versions to every row.
VERSION_PACKAGES = (
    "bilby", "bilby-laplace", "dynesty", "aspire-inference", "aspire-bilby",
    "minipcn", "numpy", "scipy",
)


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
    """Turn a method token like ``"rejection_user"`` into ``"Rejection User"``.

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


def _as_count(value):
    """Coerce a likelihood-evaluation count to a float, or NaN if unusable.

    Missing, ``None``, non-numeric and non-positive values all mean "not
    recorded".  No sampler genuinely performs zero likelihood evaluations, so a
    zero is a resumed run or an untracked plugin; printing it as ``0`` would
    read as a real measurement, and it would divide by zero downstream.
    """
    try:
        count = float(value)
    except (TypeError, ValueError):
        return np.nan
    return count if count > 0 else np.nan


def _format_efficiency(eff):
    """Format an efficiency percentage without collapsing small values to 0.

    A fixed single decimal (``{:.1f}%``) rounds anything below 0.05% to
    ``0.0%``, making a genuinely tiny-but-nonzero efficiency indistinguishable
    from an exact zero.  Below 0.1% we switch to two significant figures (e.g.
    ``0.0034%``) so the value stays visible; a true zero still prints ``0.0%``.
    """
    if not np.isfinite(eff):
        return MISSING
    if eff <= 0:
        return "0.0%"
    if eff < 0.1:
        return f"{eff:.2g}%"
    return f"{eff:.1f}%"


def _format_time(seconds):
    """A duration in the largest unit that leaves it above 1, e.g. ``"10.4h"``.

    These runs span four orders of magnitude in cost -- ten seconds for the
    in-prior draw against ten hours for dynesty -- and quoting all of them in
    seconds makes the expensive ones unreadable strings of digits.
    """
    if not np.isfinite(seconds):
        return MISSING
    for scale, suffix in ((86400.0, "d"), (3600.0, "h"), (60.0, "m")):
        if seconds >= scale:
            return f"{seconds / scale:.1f}{suffix}"
    return f"{seconds:.1f}s"


def _format_mevals(n_like):
    """Likelihood evaluations in millions, to three significant figures.

    Three significant figures rather than a fixed number of decimals because
    the same column carries both a 48.3 million evaluation SMC run and a 0.005
    million in-prior draw, and a fixed ``{:.2f}`` would print the latter as
    ``0.01``.
    """
    return MISSING if not np.isfinite(n_like) else f"{n_like / 1e6:.3g}"


def _format_metric(value):
    """A metric value to two decimals, or the missing marker."""
    return MISSING if not np.isfinite(value) else f"{value:.2f}"


def _format_worst(summary):
    """``"tilt_1 (15.81)"`` -- the parameter a metric is worst on, and its value."""
    return MISSING if not summary["worst"] else f"{summary['worst']} ({summary['worst_value']:.2f})"


def _settings_summary(result):
    """The few sampler settings worth putting beside a row's numbers.

    Just enough to tell two runs of the same method apart: the SMC cloud size
    and mutation length for anything that runs aspire (whether seeded by Laplace
    or not), the live-point count for dynesty, and nothing at all for the
    methods that draw straight from the Laplace proposal -- their cost is set by
    ``target_nsamples``, which is the number of samples asked for rather than a
    tuning choice.

    Read from ``sampler_kwargs``, so it reports what the run actually used
    rather than what the example script currently says.
    """
    kwargs = getattr(result, "sampler_kwargs", None) or {}
    if not isinstance(kwargs, dict):
        return ""
    if sampler_family(getattr(result, "label", "")) == "dynesty":
        nlive = kwargs.get("nlive")
        return "" if nlive is None else f"nlive={int(nlive)}"

    # The Laplace sampler nests its aspire settings under ``smc_kwargs``; the
    # no-Laplace control goes through aspire's own plugin, which puts
    # ``n_samples`` at the top level and the mutation settings under
    # ``sample_kwargs``.  Non-SMC Laplace runs have ``smc_kwargs=None`` and so
    # fall through to the top level, where neither key exists.
    smc = kwargs.get("smc_kwargs") or kwargs
    if not isinstance(smc, dict):
        return ""
    inner = smc.get("sampler_kwargs")
    if not isinstance(inner, dict):
        outer = smc.get("sample_kwargs")
        inner = outer.get("sampler_kwargs") if isinstance(outer, dict) else None
    n_samples = smc.get("n_samples")
    n_steps = inner.get("n_steps") if isinstance(inner, dict) else None
    parts = []
    if n_samples is not None:
        parts.append(f"nsamples={int(n_samples)}")
    if n_steps is not None:
        parts.append(f"nsteps={int(n_steps)}")
    return ", ".join(parts)


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


def _periodic_bounds(result, key):
    """``(minimum, maximum)`` if this parameter wraps, else None."""
    prior = result.priors.get(key) if getattr(result, "priors", None) else None
    if prior is None or getattr(prior, "boundary", None) != "periodic":
        return None
    return float(prior.minimum), float(prior.maximum)


def _recentre_periodic(samples_list, bounds):
    """Wrap every sample set onto a branch centred on the pooled circular mean.

    A posterior straddling the wrap point is one mode on a circle but two lumps
    on a line, and the KDE only sees the line: the density gets evaluated on a
    grid spanning the full prior range with a trough through the middle of what
    is really a single peak. Recentring first puts the mass in one piece.

    Every set is shifted by the *same* amount, derived from the pooled mean, so
    this cannot move two posteriors relative to each other -- it only chooses
    where to cut the circle, and cuts it as far from the mass as possible.
    """
    low, high = bounds
    period = high - low
    pooled = np.concatenate(samples_list)
    angles = 2 * np.pi * (pooled - low) / period
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    centre = low + (mean_angle % (2 * np.pi)) * period / (2 * np.pi)
    # Map onto [centre - period/2, centre + period/2).
    return [low + np.mod(s - centre + period / 2, period) for s in samples_list]


def _jsd(a, b, n_grid=100):
    """Jensen-Shannon divergence in nats between two 1-D sample sets.

    A gaussian KDE per set, both evaluated on a shared grid spanning the union
    of their supports, normalised, then the symmetrised KL. This reproduces
    ``pesummary.utils.utils.jensen_shannon_divergence_from_samples``, so the
    numbers are comparable with those quoted elsewhere in the GW literature.

    NaN rather than an exception when a set is degenerate (a delta-function
    posterior on a fixed parameter makes the KDE's covariance singular).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    try:
        ka, kb = gaussian_kde(a), gaussian_kde(b)
    except np.linalg.LinAlgError:
        return float("nan")
    x = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), n_grid)
    pa, pb = ka(x), kb(x)
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
        return float("nan")
    pa, pb = pa / pa.sum(), pb / pb.sum()
    m = 0.5 * (pa + pb)
    return float(0.5 * entropy(pa, m) + 0.5 * entropy(pb, m))


def _emd(a, b):
    """Earth-mover distance between two 1-D sample sets, in units of *b*'s sigma.

    *b* is the reference sample, so the scale is the same for every result in a
    column and a value reads as "this posterior is displaced by this fraction of
    a reference standard deviation". NaN when the reference has no spread, which
    would make the normalisation meaningless rather than merely large.

    Unlike the JSD this needs no density estimate: it is an exact functional of
    the two empirical CDFs, so it carries none of the KDE's bandwidth choice.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = np.std(b)
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(wasserstein_distance(a, b) / scale)


def _metric_summary(per_key):
    """``{"mean", "worst", "worst_value"}`` over a parameter -> value mapping."""
    finite = {k: v for k, v in per_key.items() if np.isfinite(v)}
    if not finite:
        return dict(mean=float("nan"), worst=None, worst_value=float("nan"))
    worst = max(finite, key=finite.get)
    return dict(mean=float(np.mean(list(finite.values()))),
                worst=worst, worst_value=finite[worst])


def _empty_metrics():
    """A row of undefined metrics, one entry per :data:`METRICS`."""
    return {m: dict(mean=float("nan"), worst=None, worst_value=float("nan")) for m in METRICS}


def _compare_pair(draw, reference_draw, keys, bounds):
    """Both metrics of *draw* against *reference_draw*, summarised over *keys*."""
    per_key = {m: {} for m in METRICS}
    for k in keys:
        pair = [draw[k].to_numpy(), reference_draw[k].to_numpy()]
        if bounds.get(k):
            pair = _recentre_periodic(pair, bounds[k])
        per_key["jsd"][k] = _jsd(*pair) * MBITS_PER_NAT
        per_key["emd"][k] = _emd(*pair)
    return {m: _metric_summary(per_key[m]) for m in METRICS}


def _reference_index(results):
    """Index of the dynesty result, or None if the set has no reference run."""
    for i, r in enumerate(results):
        if sampler_family(getattr(r, "label", "")) == "dynesty":
            return i
    return None


def divergence_from_reference(results):
    """Agreement of each result's 1-D marginals with the dynesty run.

    Returns ``(rows, reference_index, n_samples)``, where *rows* is a list
    aligned to *results* of ``{metric: {"mean", "worst", "worst_value"}}`` for
    each of :data:`METRICS` (values NaN and *worst* None where undefined), and
    *n_samples* is :data:`JSD_N`, the fixed size every metric is evaluated at.
    ``reference_index`` is None when no dynesty result is present, or when the
    reference is itself too short, in which case every row is empty.

    Parameters are those the reference sampled, kept only where every result
    carries them with non-zero spread -- a parameter held fixed in one run has
    no density to compare. Wrapped parameters need no special handling here:
    both sets are binned on one shared grid, so a posterior split across the
    wrap point is split identically in both.
    """
    empty = [_empty_metrics() for _ in results]
    index = _reference_index(results)
    if index is None or len(results) < 2:
        return empty, index, 0
    if len(results[index].posterior) < JSD_N:
        bilby.core.utils.logger.warning(
            f"reference has {len(results[index].posterior)} samples, fewer than "
            f"JSD_N={JSD_N}; skipping the agreement columns")
        return empty, None, 0

    reference = results[index]
    rng = np.random.default_rng(_JSD_RNG_SEED)
    # A result shorter than JSD_N gets no draw at all: comparing it at its own
    # size would put a number in the table that is not on the same scale as the
    # rest of the column.
    draws = [r.posterior.iloc[rng.permutation(len(r.posterior))[:JSD_N]]
             if len(r.posterior) >= JSD_N else None for r in results]

    keys = [
        k for k in reference.search_parameter_keys
        if all(k in d and np.ptp(d[k]) > 0 for d in draws if d is not None)
    ]
    bounds = {k: _periodic_bounds(reference, k) for k in keys}
    rows = []
    for i, draw in enumerate(draws):
        if i == index or draw is None or not keys:
            rows.append(_empty_metrics())
            continue
        rows.append(_compare_pair(draw, draws[index], keys, bounds))
    return rows, index, JSD_N


def reference_floor(results, reference_index):
    """The reference posterior compared with itself: what finite N costs.

    Two finite draws from the *same* distribution do not give JSD = 0, and that
    offset is comparable with the divergences measured on the examples that
    agree well. Without it a number in the table reads as a measurement when it
    is the estimator's own noise.

    Returns ``({metric: floor}, n_used)``, one floor per :data:`METRICS`. It
    takes two *disjoint* draws and so needs ``2 * JSD_N`` samples, which not
    every reference has -- the gaussian example's dynesty run has 2727. Rather
    than report nothing there, it falls back to the largest disjoint split the
    posterior does support and returns that size. Since the floor falls with N,
    a value measured at ``n_used < JSD_N`` is an upper bound on the floor at
    ``JSD_N``, which is still enough to tell a real difference from noise.

    The estimate is itself noisy -- one split, averaged over however many
    parameters the example has -- so treat it as an order of magnitude.
    """
    nothing = {m: float("nan") for m in METRICS}
    if reference_index is None:
        return nothing, 0
    reference = results[reference_index]
    posterior = reference.posterior
    n_used = min(JSD_N, len(posterior) // 2)
    if n_used < 2:
        return nothing, 0
    rng = np.random.default_rng(_JSD_RNG_SEED)
    index = rng.permutation(len(posterior))
    a, b = posterior.iloc[index[:n_used]], posterior.iloc[index[n_used:2 * n_used]]
    keys = [k for k in reference.search_parameter_keys if np.ptp(posterior[k]) > 0]
    # Recentred exactly as in divergence_from_reference, or the floor would be
    # measured under a different convention from the values it bounds.
    bounds = {k: _periodic_bounds(reference, k) for k in keys}
    summary = _compare_pair(a, b, keys, bounds)
    return {m: summary[m]["mean"] for m in METRICS}, n_used


def _software_versions(result):
    """``{package: version}`` for VERSION_PACKAGES, read from *this result's own*
    ``meta_data["environment_packages"]`` -- the environment that actually
    produced it, not whatever is active in the shell running ``make compare``.

    ``environment_packages`` is bilby's own snapshot of every conda/pip
    package in that environment (parallel ``name``/``version`` columns); this
    just looks up the packages this project cares about within it. A package
    missing from that snapshot (not installed when the result was produced,
    e.g. no reason for a dynesty-only environment to have aspire/minipcn) or
    a result with no ``environment_packages`` at all (an older result
    predating this metadata, or ``BILBY_INCLUDE_GLOBAL_METADATA`` unset) is
    silently omitted rather than reported as an error.

    ``environment_packages`` is a plain dict of parallel lists for a result
    loaded from hdf5, but a ``pandas.DataFrame`` for one loaded from json --
    bilby's json (de)serialisation round-trips a dict-of-lists back into a
    DataFrame. ``len(...) == 0`` (rather than a bare truthiness check) is
    what actually works on both: a non-empty DataFrame's truth value is
    ambiguous and raises.
    """
    env = (result.meta_data or {}).get("environment_packages")
    if env is None or len(env) == 0 or "name" not in env or "version" not in env:
        return {}
    installed = dict(zip(env["name"], env["version"]))
    return {package: installed[package] for package in VERSION_PACKAGES if package in installed}


def comparison_table(rows, reference_label):
    """``(headers, cells, align)`` for the comparison table.

    One definition, rendered twice -- once as plain text to the terminal and
    once as markdown into the README -- so the table a run prints and the table
    it writes cannot drift apart.

    *cells* is a list of lists of already-formatted strings, and *align* is
    ``"<"`` or ``">"`` per column for the fixed-width rendering.
    """
    headers = ["method", "log Z", "±", "Mevals", "effic.", "time"]
    align = ["<", ">", ">", ">", ">", ">"]
    if reference_label:
        headers += ["JSD (mbits)", "JSD worst", "EMD (σ)", "EMD worst"]
        align += [">", "<", ">", "<"]
    headers += ["settings"]
    align += ["<"]

    cells = []
    for row in rows:
        cell = [row["name"],
                _format_metric(row["log_z"]), _format_metric(row["log_z_err"]),
                row["mevals"], row["efficiency"], row["time"]]
        if reference_label:
            for metric in METRICS:
                summary = row["metrics"][metric]
                cell += [_format_metric(summary["mean"]), _format_worst(summary)]
        cell.append(row["settings"] or MISSING)
        cells.append(cell)
    return headers, cells, align


def versions_table(rows):
    """``(headers, cells, align)`` for the per-row software-versions table.

    Same one-definition-rendered-twice pattern as :func:`comparison_table`.
    One row per result, one column per tracked package -- deliberately a
    second table rather than more columns bolted onto the comparison table,
    since VERSION_PACKAGES already has 8 entries and the comparison table is
    wide enough without them.
    """
    headers = ["method"] + list(VERSION_PACKAGES)
    align = ["<"] + [">"] * len(VERSION_PACKAGES)
    cells = [
        [row["name"]] + [row["versions"].get(package, MISSING) for package in VERSION_PACKAGES]
        for row in rows
    ]
    return headers, cells, align


def _render_text_table(headers, cells, align):
    """The table as fixed-width lines, each column as wide as its widest cell."""
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h)
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:{a}{w}}}" for a, w in zip(align, widths))
    # Headers are left-aligned regardless: a right-aligned title over a column
    # of short values ends up detached from them.
    head_fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    rule = "-" * (sum(widths) + 2 * (len(widths) - 1))
    return [head_fmt.format(*headers), rule] + [fmt.format(*c) for c in cells], len(rule)


def _render_markdown_table(headers, cells, code_columns=(0, -1)):
    """The same table as markdown, with *code_columns* cells as code.

    Defaults to the comparison table's convention (method name, settings) --
    the two free-text-ish columns that read better in a monospace font.
    ``versions_table`` passes every column, since a version string is exactly
    the kind of thing code font is for and there is no free-text column to
    single out.
    """
    def row(values):
        return "| " + " | ".join(values) + " |"

    n = len(headers)
    code_columns = {c % n for c in code_columns} if cells else set()

    lines = [row(headers), row(["---"] * n)]
    for c in cells:
        c = list(c)
        for i in code_columns:
            if c[i] != MISSING:
                c[i] = f"`{c[i]}`"
        lines.append(row(c))
    return lines


def _metric_note(reference_label, n_samples, floors, floor_n):
    """The paragraphs explaining the agreement columns and their noise floors."""
    note = [
        f"Both agreement columns compare each 1-D marginal with `{reference_label}` "
        f"at a fixed {n_samples} samples per side (the same count in every example, "
        "so values are comparable across them), and report the mean over sampled "
        "parameters alongside the single parameter that scores worst. The reference "
        "is not ground truth -- a small value means agreement with dynesty, not "
        "correctness.",
        "`JSD` is the Jensen-Shannon divergence in millibits, which is dominated by "
        "width and shape mismatch and saturates once two densities barely overlap. "
        "`EMD` is the earth-mover distance divided by the reference posterior's "
        "standard deviation for that parameter, so it reads as a displacement: a "
        "posterior shifted bodily by one reference sigma scores 1.0, and unlike the "
        "JSD it keeps growing once the overlap is gone.",
    ]
    if not any(np.isfinite(v) for v in floors.values()):
        return note
    bound = "at most " if floor_n < n_samples else ""
    quoted = ", ".join(f"{v:.2f} {METRIC_UNITS[m]}" for m, v in floors.items()
                       if np.isfinite(v))
    floor_note = (f"**Noise floor: {bound}{quoted}.** That is `{reference_label}` "
                  f"against itself, two disjoint {floor_n}-sample draws from the one "
                  "posterior. Two finite samples of the *same* distribution do not "
                  "score zero, so anything at or below this level is consistent with "
                  "perfect agreement, and differences among such values are not "
                  "measurements.")
    if floor_n < n_samples:
        floor_note += (" The split needs twice its size, and this reference has too "
                       f"few samples for {n_samples}; the floor falls with N, so the "
                       f"true figure at N={n_samples} is lower than the one quoted.")
    floor_note += (" It is one split averaged over the sampled parameters, so read it"
                   " as an order of magnitude, not a threshold.")
    return note + [floor_note]


def write_readme(path, rows, reference_label, n_samples, floors=None,
                 floor_n=0, targets=()):
    """Write the example's README.md: what it is, how to run it, and the table.

    Regenerated on every ``make compare``, so it always reflects the results
    actually on disk rather than a hand-copied snapshot that can go stale.

    Each *row* must carry a ``"versions"`` entry (see :func:`_software_versions`)
    -- read from that specific result's own metadata, not the environment
    running ``make compare`` right now, since results can be generated at
    different times or copied in from elsewhere.
    """
    floors = {m: float("nan") for m in METRICS} if floors is None else floors
    name = os.path.basename(os.path.dirname(os.path.abspath(path))) or "example"
    methods = ", ".join(f"`{r['name']}`" for r in rows)
    lines = [
        f"# {name} — sampler comparison", "",
        f"Posteriors from {len(rows)} samplers ({methods}) on the {name} example, "
        + (f"each compared against the `{reference_label}` reference."
           if reference_label else "with no reference sampler among them."),
        "", "## Running it", "", "```",
        "make all       # every sampler in turn, then this comparison",
        "make compare   # rebuild this table and the corner plot from existing results",
        "```", "",
    ]
    if targets:
        lines += ["Individual samplers: " + ", ".join(f"`make {t}`" for t in targets) + ".", ""]
    lines += ["## Comparison", ""]
    if reference_label:
        for paragraph in _metric_note(reference_label, n_samples, floors, floor_n):
            lines += [paragraph, ""]
    lines += [
        "`Mevals` is millions of likelihood evaluations and `settings` names the "
        "few sampler settings that set a run's cost: the SMC cloud size and "
        "mutation length for anything running aspire, the live-point count for "
        "dynesty, and nothing for the methods that draw straight from the Laplace "
        "proposal.",
        "",
    ]
    headers, cells, _ = comparison_table(rows, reference_label)
    lines += _render_markdown_table(headers, cells)
    lines.append("")
    if any(row.get("versions") for row in rows):
        lines += [
            "## Software versions",
            "",
            "Recorded in each result's own metadata at the time it was produced -- "
            "not necessarily what is installed now, and can legitimately differ "
            "row to row if results were generated at different times.",
            "",
        ]
        v_headers, v_cells, _ = versions_table(rows)
        lines += _render_markdown_table(v_headers, v_cells, code_columns=range(len(v_headers)))
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _make_targets(directory):
    """Sampler targets from the example's Makefile .PHONY line, if there is one.

    Read rather than derived from the result labels: a label collapses config
    variants onto their family (``rejection_user`` -> ``rejection``), so the
    labels cannot recover ``make rejection-user``.
    """
    path = os.path.join(directory, "Makefile")
    if not os.path.isfile(path):
        return ()
    for line in open(path):
        if line.startswith(".PHONY:"):
            return tuple(t for t in line.split(":", 1)[1].split()
                         if t not in ("all", "compare"))
    return ()


def compare(pattern, filename, injection_parameters=None, sampler_only_labels=False,
            colour_overrides=None, parameters=None):
    """Load result files matching pattern, print comparison table, and create corner plot.

    Parameters
    ----------
    pattern : str
        Glob pattern for result files (e.g., ``/path/to/*_result.*``).
    filename : str
        Path for output corner plot.
    parameters : list, optional
        Parameters to plot. Defaults to the first result's sampled parameters.
        Pass this to include parameters that were analytically marginalised
        during sampling and reconstructed afterwards -- those are in the
        posterior but not in ``search_parameter_keys``, so they would otherwise
        be silently dropped from the figure.
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
            label = os.path.basename(r.label).capitalize()
            labels.append(label)
            logger.info(f"Loaded {f} ({label})")
        except Exception as exc:
            logger.warning(f"Could not load {f}: {exc}")

    # Optionally shorten the legend to just the sampler name, dropping the shared
    # example prefix (computed after loading so the shared prefix can be
    # detected across all results). Run time is in the README table, not the
    # legend.
    if sampler_only_labels and results:
        labels = sampler_labels(results)

    # Extract injection parameters from first result if not provided
    if injection_parameters is None and results:
        injection_parameters = getattr(results[0], "injection_parameters", None)

    # Divergence from the reference sampler, computed once for both the printed
    # table and the README so the two cannot disagree.
    divergences, reference_index, jsd_n = divergence_from_reference(results)
    reference_label = (os.path.basename(results[reference_index].label)
                       if reference_index is not None else None)
    floors, floor_n = reference_floor(results, reference_index)

    # Comparison table
    rows = []
    for r, _lab in zip(results, labels):
        # Coerce a missing/None evidence to NaN, but preserve a genuine 0.0 --
        # `x or np.nan` would wrongly turn a real 0.0 evidence into NaN.
        log_z = getattr(r, "log_evidence", None)
        log_z = np.nan if log_z is None else log_z
        log_z_err = getattr(r, "log_evidence_err", None)
        log_z_err = np.nan if log_z_err is None else log_z_err
        secs = r.sampling_time.total_seconds()
        run_stats = r.meta_data.get("run_statistics", {})
        # ``run_statistics`` is written by this package's sampler and by bilby's
        # own (dynesty).  Third-party plugins -- aspire's, for one -- populate
        # only bilby's standard ``num_likelihood_evaluations``, so fall back to
        # that rather than printing a dash for a count the result does carry.
        n_like = _as_count(run_stats.get("nlikelihood"))
        if not np.isfinite(n_like):
            n_like = _as_count(getattr(r, "num_likelihood_evaluations", None))
        # The Laplace sampler records its own "efficiency" (final samples per
        # likelihood evaluation).  Other samplers (e.g. dynesty) don't, but
        # bilby stores nlikelihood and neffsamples, so we reconstruct the same
        # quantity: effective independent samples per likelihood evaluation.
        # For the Laplace family the draws are iid, so neff ~= n and the two
        # definitions coincide.
        eff = run_stats.get("efficiency", np.nan)
        if not np.isfinite(eff):
            neff = run_stats.get("neffsamples", np.nan)
            if not np.isfinite(_as_count(neff)):
                # Last resort for a plugin that records neither: the posterior's
                # own length.  Exact only when the sampler returns iid draws
                # (aspire rejection-samples to iid before handing them back);
                # a weighted posterior would make this an over-estimate.
                neff = len(r.posterior) if getattr(r, "posterior", None) is not None else np.nan
            if np.isfinite(neff) and np.isfinite(n_like) and n_like:
                eff = 100.0 * neff / n_like
        rows.append(dict(name=os.path.basename(r.label), log_z=log_z, log_z_err=log_z_err,
                         mevals=_format_mevals(n_like), efficiency=_format_efficiency(eff),
                         time=_format_time(secs), settings=_settings_summary(r),
                         metrics=divergences[len(rows)], versions=_software_versions(r)))

    # Printed from the same cells the README is built from, so the two tables
    # cannot disagree.
    headers, cells, align = comparison_table(rows, reference_label)
    table, width = _render_text_table(headers, cells, align)
    print("\n" + "=" * width)
    print("Comparison")
    print("=" * width)
    print("\n".join(table))
    print("-" * width)
    quoted = ", ".join(f"{v:.2f} {METRIC_UNITS[m]}" for m, v in floors.items()
                       if np.isfinite(v))
    if reference_label and quoted:
        bound = "<=" if floor_n < jsd_n else ""
        print(f"Agreement at N={jsd_n} against {reference_label}; noise floor "
              f"{bound}{quoted} (two {floor_n}-sample halves of the reference). "
              "At or below that = agreement.")
    print("=" * width + "\n")

    if any(row.get("versions") for row in rows):
        v_headers, v_cells, v_align = versions_table(rows)
        v_table, v_width = _render_text_table(v_headers, v_cells, v_align)
        print("=" * v_width)
        print("Software versions (per result's own metadata)")
        print("=" * v_width)
        print("\n".join(v_table))
        print("=" * v_width + "\n")

    # README beside the corner plot, i.e. in the example's own directory.
    directory = os.path.dirname(os.path.abspath(filename))
    readme = write_readme(os.path.join(directory, "README.md"), rows, reference_label,
                          jsd_n, floors=floors, floor_n=floor_n,
                          targets=_make_targets(directory))
    logger.info(f"Comparison README written to {readme}")

    import matplotlib.pyplot as plt

    plot_parameters = parameters or results[0].search_parameter_keys

    # Half the default handle length, so the legend's line samples take up
    # less of its width -- plot_multiple's legend is a bare ax.legend() with
    # no kwargs of its own, so this is the only lever available without
    # reaching into bilby's plotting internals.
    with plt.rc_context({"legend.handlelength": plt.rcParams["legend.handlelength"] / 2}):
        fig = bilby.core.result.plot_multiple(
            results,
            labels=labels,
            colours=colours_for_results(results, overrides=colour_overrides),
            parameters=plot_parameters,
            filename=filename,
            titles=False,
            save=False,
        )

    # Overlay injection truth values if provided
    if injection_parameters:
        overlay_injection_lines(fig, plot_parameters, injection_parameters)

    fig.savefig(filename, dpi=400)
    plt.close(fig)
    logger.info(f"Comparison corner plot saved to {filename}")

    return results, labels
