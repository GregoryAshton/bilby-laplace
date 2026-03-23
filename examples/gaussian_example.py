"""
Comparison example: Laplace vs dynesty on a 2D Gaussian likelihood.

Usage
-----
    python examples/gaussian_example.py --sampler laplace rejection smc dynesty
    python examples/gaussian_example.py --sampler smc
    python examples/gaussian_example.py --compare
"""

import argparse
import glob
import os

import numpy as np
import bilby


logger = bilby.core.utils.logger
bilby.core.utils.random.seed(1234)
outdir = "outdir_gaussian_example"
base_label = "gaussian"

# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


class GaussianLikelihood(bilby.core.likelihood.Likelihood):
    """2-D correlated Gaussian likelihood."""

    def __init__(self, mu_x=1.0, mu_y=-0.5, sigma_x=0.3, sigma_y=0.5, rho=0.7):
        super().__init__(parameters={"x": None, "y": None})
        self.mu = np.array([mu_x, mu_y])
        cov = np.array([
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2],
        ])
        self._inv_cov = np.linalg.inv(cov)
        self._log_norm = -0.5 * np.log(np.linalg.det(2 * np.pi * cov))

    def log_likelihood(self, parameters=None):
        d = np.array([parameters["x"], parameters["y"]]) - self.mu
        return -0.5 * d @ self._inv_cov @ d + self._log_norm


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

likelihood = GaussianLikelihood()

priors = bilby.core.prior.PriorDict(
    dict(
        x=bilby.core.prior.Uniform(-5, 5, "x"),
        y=bilby.core.prior.Uniform(-5, 5, "y"),
    )
)

injection_parameters = {"x": 1.0, "y": -0.5}

# ---------------------------------------------------------------------------
# Shared sampler kwargs
# ---------------------------------------------------------------------------
_common_laplace = dict(
    likelihood=likelihood,
    priors=priors,
    injection_parameters=injection_parameters,
    outdir=outdir,
    use_injection_for_map=True,
    clean=True,
    sampler="laplace",
    target_nsamples=5000,
    plot_diagnostic=True,
    save="hdf5",
)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def run_laplace():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_laplace",
        resample="None",
    )


def run_rejection():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_rejection",
        resample="rejection",
    )



def run_smc():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_smc",
        resample="smc",
    )


def run_dynesty():
    return bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=f"{base_label}_dynesty",
        clean=True,
        nlive=1000,
    )


def compare():
    """Load all result files in outdir, make a comparison corner plot,
    and print a comparison table."""
    pattern = os.path.join(outdir, f"{base_label}_*_result.*")
    result_files = sorted([f for f in glob.glob(pattern) if not f.endswith('.old')])
    if not result_files:
        logger.warning(f"No result files found matching {pattern}")
        return

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
    for r, lab in zip(results, labels):
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
        logger.warning(
            f"Need at least 2 results for a comparison plot, "
            f"found {len(results)}"
        )
        return

    import matplotlib.pyplot as plt
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
    logger.info(
        f"Comparison corner plot saved to {filename}"
    )


_run_fns = dict(
    laplace=run_laplace,
    rejection=run_rejection,

    smc=run_smc,
    dynesty=run_dynesty,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Laplace vs dynesty on a 2D Gaussian likelihood"
    )
    parser.add_argument(
        "--sampler",
        nargs="+",
        choices=list(_run_fns),
        metavar="SAMPLER",
        help=f"One or more samplers to run: {', '.join(_run_fns)}",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Load all existing results, print evidence table, and plot",
    )
    args = parser.parse_args()

    if not args.sampler and not args.compare:
        parser.print_help()
    else:
        for name in (args.sampler or []):
            _run_fns[name]()
        if args.compare:
            compare()
