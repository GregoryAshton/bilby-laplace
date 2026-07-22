"""
Laplace approximation on a 2D Rosenbrock (banana) likelihood.

Usage
-----
    python run.py --sampler laplace rejection smc dynesty
    python run.py --compare
"""

import argparse

import bilby

from bilby_laplace.comparison import compare

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
            x=bilby.core.prior.Uniform(-2, 2, "$x$"),
            y=bilby.core.prior.Uniform(-1, 3, "$y$"),
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
        label=f"{base_label}_inprior",
        resample="inprior",
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
            n_samples=5000,
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
            pattern = f"{outdir}/{base_label}_*_result.*"
            filename = f"{base_label}_comparison.png"
            _results, _labels = compare(pattern, filename, sampler_only_labels=True)
