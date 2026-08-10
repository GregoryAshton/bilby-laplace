"""Shared comparison utilities for Laplace example scripts."""

import glob
import os
import re

import bilby
import numpy as np
from scipy.stats import entropy, gaussian_kde

# JS divergences are a few thousandths of a nat on these problems, which is
# unreadable; millibits is the unit the numbers are quoted in.
MBITS_PER_NAT = 1000.0 / np.log(2)

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
    "laplace": "#009E73",  # bluish green    - raw Laplace/Gaussian approximation
    "inprior": "#CC79A7",  # reddish purple
    "rejection": "#D55E00",  # vermillion
    "importance": "#56B4E9",  # sky blue
    "smc": "#E69F00",  # orange              - the headline method
    "smcdirect": "#785EF0",  # violet        - SMC from the prior, no-Laplace control
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

    Note that ``"smcdirect"`` is deliberately spelled without a separator: the
    tokeniser splits on ``-`` and ``_``, so ``"smc-direct"`` would tokenise to
    ``["smc", "direct"]`` and collide with the ``"smc"`` family.
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


def _reference_index(results):
    """Index of the dynesty result, or None if the set has no reference run."""
    for i, r in enumerate(results):
        if sampler_family(getattr(r, "label", "")) == "dynesty":
            return i
    return None


def divergence_from_reference(results):
    """Mean JS divergence of each result's 1-D marginals from the dynesty run.

    Returns ``(rows, reference_index, n_samples)``, where *rows* is a list
    aligned to *results* of ``{"mean": mbits, "worst": name, "worst_value":
    mbits}`` (values NaN and *worst* None where undefined), and *n_samples* is
    :data:`JSD_N`, the fixed size every divergence is evaluated at.
    ``reference_index`` is None when no dynesty result is present, or when the
    reference is itself too short, in which case every row is empty.

    Parameters are those the reference sampled, kept only where every result
    carries them with non-zero spread -- a parameter held fixed in one run has
    no density to compare. Wrapped parameters need no special handling here:
    both sets are binned on one shared grid, so a posterior split across the
    wrap point is split identically in both.
    """
    empty = [dict(mean=float("nan"), worst=None, worst_value=float("nan")) for _ in results]
    index = _reference_index(results)
    if index is None or len(results) < 2:
        return empty, index, 0
    if len(results[index].posterior) < JSD_N:
        bilby.core.utils.logger.warning(
            f"reference has {len(results[index].posterior)} samples, fewer than "
            f"JSD_N={JSD_N}; skipping the divergence column")
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
            rows.append(dict(mean=float("nan"), worst=None, worst_value=float("nan")))
            continue
        per_key = {}
        for k in keys:
            pair = [draw[k].to_numpy(), draws[index][k].to_numpy()]
            if bounds[k]:
                pair = _recentre_periodic(pair, bounds[k])
            per_key[k] = _jsd(*pair)
        finite = {k: v for k, v in per_key.items() if np.isfinite(v)}
        if not finite:
            rows.append(dict(mean=float("nan"), worst=None, worst_value=float("nan")))
            continue
        worst = max(finite, key=finite.get)
        rows.append(dict(mean=np.mean(list(finite.values())) * MBITS_PER_NAT,
                         worst=worst, worst_value=finite[worst] * MBITS_PER_NAT))
    return rows, index, JSD_N


def reference_floor(results, reference_index):
    """The reference posterior compared with itself: what finite N costs.

    Two finite draws from the *same* distribution do not give JSD = 0, and that
    offset is comparable with the divergences measured on the examples that
    agree well. Without it a number in the table reads as a measurement when it
    is the estimator's own noise.

    Returns ``(floor_mbits, n_used)``. It takes two *disjoint* draws and so
    needs ``2 * JSD_N`` samples, which not every reference has -- the gaussian
    example's dynesty run has 2727. Rather than report nothing there, it falls
    back to the largest disjoint split the posterior does support and returns
    that size. Since the floor falls with N, a value measured at ``n_used <
    JSD_N`` is an upper bound on the floor at ``JSD_N``, which is still enough
    to tell a real difference from noise.

    The estimate is itself noisy -- one split, averaged over however many
    parameters the example has -- so treat it as an order of magnitude.
    """
    if reference_index is None:
        return float("nan"), 0
    reference = results[reference_index]
    posterior = reference.posterior
    n_used = min(JSD_N, len(posterior) // 2)
    if n_used < 2:
        return float("nan"), 0
    rng = np.random.default_rng(_JSD_RNG_SEED)
    index = rng.permutation(len(posterior))
    a, b = posterior.iloc[index[:n_used]], posterior.iloc[index[n_used:2 * n_used]]
    values = []
    for k in reference.search_parameter_keys:
        if np.ptp(posterior[k]) <= 0:
            continue
        # Recentred exactly as in divergence_from_reference, or the floor would
        # be measured under a different convention from the values it bounds.
        pair = [a[k].to_numpy(), b[k].to_numpy()]
        wrapped = _periodic_bounds(reference, k)
        if wrapped:
            pair = _recentre_periodic(pair, wrapped)
        values.append(_jsd(*pair))
    finite = [v for v in values if np.isfinite(v)]
    return (np.mean(finite) * MBITS_PER_NAT if finite else float("nan")), n_used


def write_readme(path, rows, reference_label, n_samples, floor=float("nan"),
                 floor_n=0, targets=()):
    """Write the example's README.md: what it is, how to run it, and the table.

    Regenerated on every ``make compare``, so it always reflects the results
    actually on disk rather than a hand-copied snapshot that can go stale.
    """
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
        note = ("`JSD` is the mean over sampled parameters of the Jensen-Shannon "
                f"divergence of each 1-D marginal from `{reference_label}`, in "
                f"millibits, evaluated at a fixed {n_samples} samples per side "
                "(the same count in every example, so values are comparable "
                "across them). The reference is not ground truth -- a small "
                "value means agreement with dynesty, not correctness.")
        if np.isfinite(floor):
            bound = "at most " if floor_n < n_samples else ""
            note += (f"\n\n**Noise floor: {bound}{floor:.2f} mbits.** That is "
                     f"`{reference_label}` against itself, two disjoint "
                     f"{floor_n}-sample draws from the one posterior. Two "
                     "finite samples of the *same* distribution do not score "
                     "zero, so anything at or below this level is consistent "
                     "with perfect agreement, and differences among such "
                     "values are not measurements.")
            if floor_n < n_samples:
                note += (f" The split needs twice its size, and this reference "
                         f"has too few samples for {n_samples}; the floor falls "
                         f"with N, so the true figure at N={n_samples} is lower "
                         "than the one quoted.")
            note += (" It is one split averaged over the sampled parameters, so"
                     " read it as an order of magnitude, not a threshold.")
        lines += [note, ""]
    header = "| method | log Z | ± | n_like | efficiency | time |"
    divider = "|---|---|---|---|---|---|"
    if reference_label:
        header += " JSD (mbits) | worst parameter |"
        divider += "---|---|"
    lines += [header, divider]
    for row in rows:
        line = (f"| `{row['name']}` | {row['log_z']:.2f} | {row['log_z_err']:.2f} | "
                f"{row['n_like_str'].strip()} | {row['eff_str'].strip()} | {row['time']:.1f}s |")
        if reference_label:
            jsd = row["jsd"]
            worst = row["worst"]
            line += (f" {jsd['mean']:.2f} |" if np.isfinite(jsd["mean"]) else " — |")
            line += (f" {worst} ({jsd['worst_value']:.2f}) |" if worst else " — |")
        lines.append(line)
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

    # Divergence from the reference sampler, computed once for both the printed
    # table and the README so the two cannot disagree.
    divergences, reference_index, jsd_n = divergence_from_reference(results)
    reference_label = (os.path.basename(results[reference_index].label)
                       if reference_index is not None else None)
    jsd_floor, jsd_floor_n = reference_floor(results, reference_index)

    # Comparison table
    rows = []
    W = 75 if reference_index is None else 100
    print("\n" + "=" * W)
    print("Comparison")
    print("=" * W)
    head = f"{'Method':<20} {'log Z':>10} {'± σ':>8} {'n_like':>8} {'effic.':>8} {'time':>10}"
    if reference_index is not None:
        head += f" {'JSD':>8} {'worst':>16}"
    print(head)
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
        name = os.path.basename(r.label)
        n_like_str = f"{int(n_like):>8}" if np.isfinite(n_like) else f"{'—':>8}"
        eff_str = f"{_format_efficiency(eff):>8}" if np.isfinite(eff) else f"{'—':>8}"
        jsd = divergences[len(rows)]
        rows.append(dict(name=name, log_z=log_z, log_z_err=log_z_err, time=secs,
                         n_like_str=n_like_str, eff_str=eff_str,
                         jsd=jsd, worst=jsd["worst"]))
        line = f"{name:<20} {log_z:>10.2f} {log_z_err:>8.2f} {n_like_str} {eff_str} {secs:>9.1f}s"
        if reference_index is not None:
            value = f"{jsd['mean']:.2f}" if np.isfinite(jsd["mean"]) else "—"
            line += f" {value:>8} {(jsd['worst'] or '—'):>16}"
        print(line)
    print("-" * W)
    if reference_index is not None and np.isfinite(jsd_floor):
        bound = "<=" if jsd_floor_n < jsd_n else ""
        print(f"JSD at N={jsd_n}; noise floor {bound}{jsd_floor:.2f} mbits "
              f"({reference_label}, two {jsd_floor_n}-sample halves). "
              "At or below that = agreement.")
    print("=" * W + "\n")

    # README beside the corner plot, i.e. in the example's own directory.
    directory = os.path.dirname(os.path.abspath(filename))
    readme = write_readme(os.path.join(directory, "README.md"), rows, reference_label,
                          jsd_n, floor=jsd_floor, floor_n=jsd_floor_n,
                          targets=_make_targets(directory))
    logger.info(f"Comparison README written to {readme}")

    import matplotlib.pyplot as plt

    plot_parameters = parameters or results[0].search_parameter_keys

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
