"""
Comparison example: Laplace vs dynesty on a 2D Rosenbrock likelihood.

The Rosenbrock (banana) function is a standard non-Gaussian test: its
narrow curved valley makes it a challenging target for Laplace-based
methods, providing a useful illustration of where the approximation breaks
down compared to SMC or nested sampling.

The log-likelihood is:
    log L(x, y) = -[(1 - x)^2 + 100 (y - x^2)^2] / scale

where ``scale`` controls the width of the distribution.

Usage
-----
    python examples/rosenbrock_example.py --sampler laplace rejection smc dynesty
    python examples/rosenbrock_example.py --sampler smc
    python examples/rosenbrock_example.py --compare
"""

import argparse
import glob
import os

import bilby
import numpy as np

logger = bilby.core.utils.logger
bilby.core.utils.random.seed(1234)
outdir = "outdir_rosenbrock_example"
base_label = "rosenbrock"


def setup():
    """Set up likelihood, priors, and sampler configuration."""

    # Likelihood
    class RosenbrockLikelihood(bilby.core.likelihood.Likelihood):
        """2-D Rosenbrock (banana) likelihood.

        Parameters
        ----------
        scale : float
            Controls the width of the distribution. Larger values give a
            broader, easier posterior. Default is 1.0.
        """

        def __init__(self, scale=1.0):
            super().__init__(parameters={"x": None, "y": None})
            self.scale = scale

        def log_likelihood(self, parameters=None):
            x = parameters["x"]
            y = parameters["y"]
            return -((1 - x) ** 2 + 100 * (y - x**2) ** 2) / self.scale

    likelihood = RosenbrockLikelihood(scale=1.0)

    priors = bilby.core.prior.PriorDict(
        dict(
            x=bilby.core.prior.Uniform(-2, 2, "x"),
            y=bilby.core.prior.Uniform(-1, 3, "y"),
        )
    )

    injection_parameters = {"x": 1.0, "y": 1.0}

    # Shared sampler kwargs
    _common = dict(
        likelihood=likelihood,
        priors=priors,
        injection_parameters=injection_parameters,
        outdir=outdir,
    )

    _common_laplace = dict(
        **_common,
        use_injection_for_map=True,
        clean=True,
        sampler="laplace",
        target_nsamples=5000,
        plot_diagnostic=True,
    )

    return _common, _common_laplace


# Samplers
def run_laplace(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_laplace",
        resample="None",
        cov_scaling=1,
    )


def run_rejection(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_rejection",
        resample="rejection",
        cov_scaling=10,
    )


def run_smc(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_smc",
        resample="smc",
        smc_kwargs=dict(
            sampler="minipcn_smc",
            n_initial_samples=10000,
            n_final_samples=5000,
            target_efficiency=[0.5, 0.8],
            adaptive=True,
            sampler_kwargs=dict(
                n_steps=1000,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
                verbose=True,
            ),
        ),
        cov_scaling=10,
    )


def run_dynesty(_common):
    return bilby.run_sampler(
        **_common,
        sampler="dynesty",
        label=f"{base_label}_dynesty",
        clean=True,
        nlive=1000,
    )


def compare():
    """Load all result files in outdir, make a comparison corner plot,
    and print a comparison table."""
    pattern = os.path.join(outdir, f"{base_label}_*_result.*")
    result_files = sorted([f for f in glob.glob(pattern) if not f.endswith(".old")])
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
        logger.warning(f"Need at least 2 results for a comparison plot, " f"found {len(results)}")
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
    logger.info(f"Comparison corner plot saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laplace vs dynesty on a 2D Rosenbrock likelihood")
    parser.add_argument(
        "--sampler",
        nargs="+",
        choices=["laplace", "rejection", "smc", "dynesty"],
        metavar="SAMPLER",
        help="One or more samplers to run: laplace, rejection, smc, dynesty",
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
        # Only set up likelihood/priors if running samplers
        if args.sampler:
            _common, _common_laplace = setup()

            _run_fns = {
                "laplace": lambda: run_laplace(_common_laplace),
                "rejection": lambda: run_rejection(_common_laplace),
                "smc": lambda: run_smc(_common_laplace),
                "dynesty": lambda: run_dynesty(_common),
            }

            for name in args.sampler:
                _run_fns[name]()

        # Compare only needs to read result files
        if args.compare:
            compare()
