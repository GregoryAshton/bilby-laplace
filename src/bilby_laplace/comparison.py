"""Shared comparison utilities for Laplace example scripts."""

import glob
import os

import bilby
import numpy as np


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
                    ax.axvline(truths[col], color="C3", ls="--", lw=1.0)
            elif row > col:
                if truths[col] is not None:
                    ax.axvline(truths[col], color="C3", ls="--", lw=0.8, alpha=0.7)
                if truths[row] is not None:
                    ax.axhline(truths[row], color="C3", ls="--", lw=0.8, alpha=0.7)


def compare(pattern, filename, injection_parameters=None):
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
            secs = r.sampling_time.total_seconds()
            if secs >= 3600:
                label += f" ({secs / 3600:.1f} hr)"
            elif secs >= 60:
                label += f" ({secs / 60:.1f} min)"
            else:
                label += f" ({secs:.0f} s)"
            labels.append(label)
            logger.info(f"Loaded {f} ({label})")
        except Exception as exc:
            logger.warning(f"Could not load {f}: {exc}")

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
        log_z = getattr(r, "log_evidence", np.nan) or np.nan
        log_z_err = getattr(r, "log_evidence_err", np.nan) or np.nan
        secs = r.sampling_time.total_seconds()
        run_stats = r.meta_data.get("run_statistics", {})
        n_like = run_stats.get("nlikelihood", np.nan)
        eff = run_stats.get("efficiency", np.nan)
        name = os.path.basename(r.label)
        n_like_str = f"{int(n_like):>8}" if np.isfinite(n_like) else f"{'—':>8}"
        eff_str = f"{eff:>7.1f}%" if np.isfinite(eff) else f"{'—':>8}"
        print(f"{name:<20} {log_z:>10.2f} {log_z_err:>8.2f} {n_like_str} {eff_str} {secs:>9.1f}s")
    print("=" * W + "\n")

    import matplotlib.pyplot as plt

    fig = bilby.core.result.plot_multiple(
        results,
        labels=labels,
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
