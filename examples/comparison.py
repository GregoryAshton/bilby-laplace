"""Shared comparison utilities for Laplace example scripts."""

import glob
import os

import bilby
import numpy as np


def compare(outdir, base_label, filename=None):
    """Load all result files, print comparison table, and create corner plot.

    Parameters
    ----------
    outdir : str
        Output directory containing result files.
    base_label : str
        Base label for result files (pattern: ``{base_label}_*_result.*``).
    filename : str, optional
        Path for output corner plot. If None, uses ``{base_label}_comparison.png``
        in the examples directory.

    Returns
    -------
    results : list
        List of loaded bilby Result objects.
    labels : list
        List of formatted labels for each result.
    """
    logger = bilby.core.utils.logger

    pattern = os.path.join(outdir, f"{base_label}_*_result.*")
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
            label = r.label.replace(f"{base_label}_", "").capitalize()
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
        name = r.label.replace(f"{base_label}_", "")
        n_like_str = f"{int(n_like):>8}" if np.isfinite(n_like) else f"{'—':>8}"
        eff_str = f"{eff:>7.1f}%" if np.isfinite(eff) else f"{'—':>8}"
        print(f"{name:<20} {log_z:>10.2f} {log_z_err:>8.2f} {n_like_str} {eff_str} {secs:>9.1f}s")
    print("=" * W + "\n")

    if len(results) < 2:
        logger.warning(f"Need at least 2 results for a comparison plot, found {len(results)}")
        return results, labels

    import matplotlib.pyplot as plt

    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{base_label}_comparison.png")

    fig = bilby.core.result.plot_multiple(
        results,
        labels=labels,
        filename=filename,
        titles=False,
        save=False,
    )
    fig.savefig(filename, dpi=400)
    plt.close(fig)
    logger.info(f"Comparison corner plot saved to {filename}")

    return results, labels
