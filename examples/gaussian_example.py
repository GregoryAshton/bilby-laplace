"""
Comparison example: Laplace vs dynesty on a 2D Gaussian likelihood.

Usage
-----
    python examples/gaussian_example.py --rejection
    python examples/gaussian_example.py --smc
    python examples/gaussian_example.py --importance
    python examples/gaussian_example.py --dynesty
    python examples/gaussian_example.py --plot-combined
"""

import argparse
import glob
import os

import numpy as np
import bilby


logger = bilby.core.utils.logger
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
    use_injection_for_maxL=True,
    clean=True,
    sampler="laplace",
    target_nsamples=5000,
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


def run_importance():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_importance",
        resample="importance",
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


def plot_combined():
    """Load all result files in outdir and make a comparison corner plot."""
    pattern = os.path.join(outdir, f"{base_label}_*_result.*")
    result_files = sorted(glob.glob(pattern))
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

    if len(results) < 2:
        logger.warning(
            f"Need at least 2 results for a comparison plot, "
            f"found {len(results)}"
        )
        if len(results) == 1:
            results[0].plot_corner()
        return

    bilby.core.result.plot_multiple(
        results,
        labels=labels,
        filename=os.path.join(outdir, f"{base_label}_comparison_corner.png"),
        titles=False,
    )
    logger.info(
        f"Comparison corner plot saved to "
        f"{outdir}/{base_label}_comparison_corner.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Laplace vs dynesty on a 2D Gaussian likelihood"
    )
    parser.add_argument(
        "--laplace",
        action="store_true",
        help="Run Laplace approximation (no resampling)",
    )
    parser.add_argument(
        "--rejection",
        action="store_true",
        help="Run Laplace with rejection resampling",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="Run Laplace with importance resampling",
    )
    parser.add_argument(
        "--smc",
        action="store_true",
        help="Run Laplace with SMC resampling",
    )
    parser.add_argument(
        "--dynesty",
        action="store_true",
        help="Run dynesty nested sampling",
    )
    parser.add_argument(
        "--plot-combined",
        action="store_true",
        help="Load all existing results and produce a comparison corner plot",
    )
    args = parser.parse_args()

    if args.laplace:
        run_laplace().plot_corner()
    if args.rejection:
        run_rejection().plot_corner()
    if args.importance:
        run_importance().plot_corner()
    if args.smc:
        run_smc().plot_corner()
    if args.dynesty:
        run_dynesty().plot_corner()
    if args.plot_combined:
        plot_combined()

    if not any([args.laplace, args.rejection, args.importance, args.smc,
                args.dynesty, args.plot_combined]):
        parser.print_help()
