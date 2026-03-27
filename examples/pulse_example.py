"""
Comparison example: Laplace vs dynesty on a time-series Gaussian pulse likelihood.

A simple synthetic time series containing a Gaussian pulse embedded in white noise.
The task is to estimate the pulse parameters: amplitude, time-of-arrival (location),
and width (standard deviation). This is a realistic setting for signal detection in
noisy data, relevant to gravitational wave searches and other signal processing tasks.

The log-likelihood (data fit) is:
    log L(A, t0, σ) = -0.5 * sum((data - A * exp(-(t - t0)^2 / (2σ^2)))^2)

Usage
-----
    python examples/pulse_example.py --sampler laplace rejection smc dynesty
    python examples/pulse_example.py --sampler smc
    python examples/pulse_example.py --compare
"""

import argparse

import bilby
import numpy as np
from comparison import compare

logger = bilby.core.utils.logger
bilby.core.utils.random.seed(1234)
outdir = "outdir_pulse_example"
base_label = "pulse"


def setup():
    """Set up likelihood, priors, and sampler configuration."""

    # Likelihood
    class PulseLikelihood(bilby.core.likelihood.Likelihood):
        """Time-series Gaussian pulse likelihood.

        Computes the log-likelihood of observing data containing a Gaussian pulse
        with unknown amplitude, time-of-arrival, and width embedded in white noise.

        Parameters
        ----------
        times : array_like
            Time array (length N).
        data : array_like
            Noisy time-series data (length N).
        noise_sigma : float
            Standard deviation of the white noise. Default is 0.1.
        """

        def __init__(self, times, data, noise_sigma=0.1):
            super().__init__(parameters={"amplitude": None, "t0": None, "sigma": None})
            self.times = times
            self.data = data
            self.noise_sigma = noise_sigma

        def log_likelihood(self, parameters=None):
            A = parameters["amplitude"]
            t0 = parameters["t0"]
            sigma = parameters["sigma"]
            # Gaussian pulse model
            pulse = A * np.exp(-0.5 * ((self.times - t0) / sigma) ** 2)
            # Likelihood: chi-squared misfit
            residuals = self.data - pulse
            return -0.5 * np.sum((residuals / self.noise_sigma) ** 2)

    # Generate synthetic data: true pulse + white noise
    np.random.seed(1234)
    times = np.linspace(0, 10, 100)
    true_A = 100.0
    true_t0 = 5.0
    true_sigma = 0.5
    true_pulse = true_A * np.exp(-0.5 * ((times - true_t0) / true_sigma) ** 2)
    noise = np.random.normal(0, 0.1, len(times))
    data = true_pulse + noise

    likelihood = PulseLikelihood(times=times, data=data, noise_sigma=0.1)

    priors = bilby.core.prior.PriorDict(
        dict(
            amplitude=bilby.core.prior.Uniform(0, 500, "amplitude"),
            t0=bilby.core.prior.Uniform(2, 8, "t0"),
            sigma=bilby.core.prior.Uniform(0.1, 2, "sigma"),
        )
    )

    injection_parameters = {"amplitude": true_A, "t0": true_t0, "sigma": true_sigma}

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
        cov_scaling=5,
    )


def run_smc(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_smc",
        resample="smc",
        smc_kwargs=dict(
            sampler="minipcn_smc",
            n_initial_samples=5000,
            n_final_samples=5000,
            target_efficiency=[0.5, 0.8],
            adaptive=True,
            sampler_kwargs=dict(
                n_steps=100,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
                verbose=True,
            ),
        ),
        cov_scaling=5,
    )


def run_dynesty(_common):
    return bilby.run_sampler(
        **_common,
        label=f"{base_label}_dynesty",
        sampler="dynesty",
        clean=True,
        nlive=1000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laplace vs dynesty on a Gaussian pulse detection problem")
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
            _results, _labels = compare(outdir, base_label)
