"""
Laplace approximation on a 2D correlated Gaussian likelihood.

Usage
-----
    python run.py --sampler laplace rejection smc dynesty
    python run.py --compare
"""

import argparse

import bilby
import numpy as np

from bilby_laplace.comparison import compare

logger = bilby.core.utils.logger
bilby.core.utils.random.seed(1234)
outdir = "outdir_gaussian_example"
base_label = "gaussian"


def setup():
    """Set up likelihood, priors, and sampler configuration."""

    # Likelihood
    class GaussianLikelihood(bilby.core.likelihood.Likelihood):
        """2-D correlated Gaussian likelihood."""

        def __init__(self, mu_x=1.0, mu_y=-0.5, sigma_x=0.3, sigma_y=0.5, rho=0.7):
            super().__init__(parameters={"x": None, "y": None})
            self.mu = np.array([mu_x, mu_y])
            cov = np.array(
                [
                    [sigma_x**2, rho * sigma_x * sigma_y],
                    [rho * sigma_x * sigma_y, sigma_y**2],
                ]
            )
            self._inv_cov = np.linalg.inv(cov)
            self._log_norm = -0.5 * np.log(np.linalg.det(2 * np.pi * cov))

        def log_likelihood(self, parameters=None):
            d = np.array([parameters["x"], parameters["y"]]) - self.mu
            return -0.5 * d @ self._inv_cov @ d + self._log_norm

    likelihood = GaussianLikelihood()

    priors = bilby.core.prior.PriorDict(
        dict(
            x=bilby.core.prior.Uniform(-5, 5, "x"),
            y=bilby.core.prior.Uniform(-5, 5, "y"),
        )
    )

    injection_parameters = {"x": 1.0, "y": -0.5}

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
        save="hdf5",
    )

    return _common, _common_laplace


# Samplers
def run_laplace(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_laplace",
        resample="None",
    )


def run_rejection(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_rejection",
        resample="rejection",
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
                n_steps=5,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
            ),
        ),
        cov_scaling=1,
    )


def run_dynesty(_common):
    return bilby.run_sampler(
        **_common,
        sampler="dynesty",
        label=f"{base_label}_dynesty",
        clean=True,
        nlive=1000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laplace vs dynesty on a 2D Gaussian likelihood")
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
            _results, _labels = compare(outdir, base_label, filename=f"{base_label}_comparison.png")
