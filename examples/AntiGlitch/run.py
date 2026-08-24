#!/usr/bin/env python

"""
Laplace approximation on a simulated anti-glitch in Gaussian noise.

Usage
-----
    python run.py --sampler laplace rejection smc smc-direct dynesty
    python run.py --compare
"""

import argparse

import bilby
import numpy as np
from bilby.core.prior import Uniform

from bilby_laplace.comparison import colours_for_results
from bilby_laplace.comparison import compare as compare_results
from bilby_laplace.comparison import overlay_injection_lines

logger = bilby.core.utils.logger
# The strain/noise realisation. Reseeded here so every sampler in this example
# analyses identical data.
DATA_SEED = 1234
# The sampler's own random stream. Passed to run_sampler as `sampling_seed`,
# which bilby routes to each sampler's own seed argument -- for dynesty that
# builds its `rstate`, which is otherwise drawn from OS entropy and leaves the
# run unreproducible. Note the aspire plugin silently discards it, so the
# smc-direct control is not yet covered.
SAMPLING_SEED = 20260810
bilby.core.utils.random.seed(DATA_SEED)
outdir = "outdir_antiglitch_example"
base_label = "antiglitch"

N_STEPS = 5
N_SAMPLES = 2000


def anti_glitch_model(frequency, psd, tstart, A, f, phi, tc, log_gamma):
    gamma = 10**log_gamma
    htilde = np.exp(-gamma / 2 * (np.log(frequency) - np.log(f)) ** 2)
    N = np.sqrt(np.sum(np.abs(htilde) ** 2 / psd))
    dt = tc - tstart
    return A * np.exp(2j * phi - 2 * np.pi * 1j * frequency * dt) * htilde / N


class AntiGlitchLikelihood(bilby.Likelihood):
    def __init__(self, ifo, priors, time_marginalization=False, phase_marginalization=False):
        super().__init__({key: None for key in priors})
        self.ifo = ifo
        self._prior = priors.copy()
        self.time_marginalization = time_marginalization
        self.phase_marginalization = phase_marginalization
        self._noise_log_likelihood_value = None

        # The internal copy keeps the full prior (time marginalisation needs the
        # tc range); the caller's dict has the marginalised entries replaced by
        # floats so bilby fixes them rather than sampling them.
        if self.time_marginalization:
            self._marginalized_parameters.append("tc")
            self._setup_time_marginalization()
            priors["tc"] = float(self.ifo.start_time)
        if self.phase_marginalization:
            self._marginalized_parameters.append("phi")
            priors["phi"] = float(0)

    @property
    def priors(self):
        return self._prior

    # The model's own parameters, so a caller passing extras (or bilby passing
    # the full sample) does not break the call.
    _MODEL_KEYS = ("A", "f", "phi", "tc", "log_gamma")

    def get_prediction(self, parameters):
        prediction = np.zeros_like(self.ifo.frequency_array, dtype=complex)
        mask = self.ifo.frequency_mask
        prediction[mask] = anti_glitch_model(
            self.ifo.frequency_array[mask],
            self.ifo.power_spectral_density_array[mask],
            self.ifo.start_time,
            **{k: parameters[k] for k in self._MODEL_KEYS},
        )
        return prediction

    def generate_posterior_sample_from_marginalized_likelihood(self, parameters):
        """Draw tc and phi back in, conditioned on the sampled parameters.

        Time and phase are marginalised analytically during sampling, so the
        sampler never visits them. Their conditional posteriors are available in
        closed form given the other parameters, so they are drawn afterwards --
        the posterior is then over all five parameters, and comparing samplers on
        tc and phi tests the reconstruction as well as the sampling.
        """
        parameters = dict(parameters)
        signal = self.get_prediction(parameters)
        if self.time_marginalization:
            parameters["tc"] = self._sample_time(signal, parameters)
            signal = self.get_prediction(parameters)
        if self.phase_marginalization:
            parameters["phi"] = self._sample_phase(signal)
        return parameters

    def _sample_phase(self, signal):
        d_inner_h, h_inner_h, _ = self.calculate_snrs(signal)
        phases = np.linspace(0, 2 * np.pi, 101)
        log_post = d_inner_h * np.exp(-2j * phases) - h_inner_h / 2
        post = np.exp(log_post.real - max(log_post.real))
        return bilby.core.prior.Interped(phases, post).sample()

    def _sample_time(self, signal, parameters):
        # Upsampled relative to the data so the reconstructed time is not
        # quantised to the analysis sampling rate.
        upsampled = 16384
        times = bilby.core.utils.create_time_series(
            sampling_frequency=upsampled,
            starting_time=parameters["tc"] - self.ifo.start_time,
            duration=self.ifo.duration,
        )
        times = (times % self.ifo.duration) + self.ifo.start_time
        prior = self._prior["tc"]
        in_prior = (times >= prior.minimum) & (times < prior.maximum)
        times = times[in_prior]

        n_time_steps = int(self.ifo.duration * upsampled)
        signal_long = np.zeros(n_time_steps, dtype=complex)
        data = np.zeros(n_time_steps, dtype=complex)
        psd = np.ones(n_time_steps)
        ifo_length = len(self.ifo.frequency_domain_strain)
        mask = self.ifo.frequency_mask
        signal_long[:ifo_length] = signal
        data[:ifo_length] = np.conj(self.ifo.frequency_domain_strain)
        psd[:ifo_length][mask] = self.ifo.power_spectral_density_array[mask]

        d_inner_h = np.fft.fft(signal_long * data / psd)[in_prior]
        h_inner_h = self.ifo.optimal_snr_squared(signal=signal).real

        if self.phase_marginalization:
            log_like = bilby.gw.utils.ln_i0(abs(d_inner_h)) - h_inner_h / 2
        else:
            log_like = d_inner_h.real - h_inner_h / 2

        post = np.exp(log_like - max(log_like)) * prior.prob(times)
        keep = post > max(post) / 1000
        if sum(keep) < 3:
            keep[1:-1] = keep[1:-1] | keep[2:] | keep[:-2]
        return bilby.core.prior.Interped(times[keep], post[keep]).sample()

    def calculate_snrs(self, signal):
        ifo = self.ifo
        d_inner_h = ifo.inner_product(signal=signal)
        optimal_snr_squared = ifo.optimal_snr_squared(signal=signal)
        if self.time_marginalization:
            d_inner_h_array = (4 / ifo.duration) * np.fft.fft(
                signal[0:-1] * ifo.frequency_domain_strain.conjugate()[0:-1] / ifo.power_spectral_density_array[0:-1]
            )
        else:
            d_inner_h_array = None
        return d_inner_h, optimal_snr_squared.real, d_inner_h_array

    def phase_marginalized_likelihood(self, d_inner_h, h_inner_h):
        return bilby.gw.utils.ln_i0(abs(d_inner_h)) - h_inner_h / 2

    def _setup_time_marginalization(self):
        self._delta_tc = 2 / self.ifo.sampling_frequency
        self._times = (
            self.ifo.start_time
            + np.linspace(0, self.ifo.duration, int(self.ifo.duration / 2 * self.ifo.sampling_frequency + 1))[1:]
        )
        self.time_mask = (self._times >= self._prior["tc"].minimum) & (self._times <= self._prior["tc"].maximum)
        self.time_prior_array = self._prior["tc"].prob(self._times) * self._delta_tc

    def time_marginalized_likelihood(self, d_inner_h_tc_array, h_inner_h):
        from scipy.special import logsumexp

        d_inner_h_tc_array = d_inner_h_tc_array[self.time_mask]
        time_prior_array = self.time_prior_array[self.time_mask]
        if self.phase_marginalization:
            log_l_tc_array = self.phase_marginalized_likelihood(d_inner_h_tc_array, h_inner_h)
        else:
            log_l_tc_array = np.real(d_inner_h_tc_array) - h_inner_h / 2
        return logsumexp(log_l_tc_array, b=time_prior_array, axis=-1)

    def log_likelihood_ratio(self, parameters=None):
        # bilby's current API passes the sample explicitly; older code sets
        # self.parameters and calls with none. Support both.
        params = self.parameters if parameters is None else {**self.parameters, **parameters}
        d_inner_h, h_inner_h, d_inner_h_array = self.calculate_snrs(self.get_prediction(params))
        if self.time_marginalization:
            log_l = self.time_marginalized_likelihood(d_inner_h_array, h_inner_h)
        elif self.phase_marginalization:
            log_l = self.phase_marginalized_likelihood(d_inner_h, h_inner_h)
        else:
            log_l = np.real(d_inner_h) - h_inner_h / 2
        return float(np.real(log_l))

    def noise_log_likelihood(self):
        if self._noise_log_likelihood_value is None:
            mask = self.ifo.frequency_mask
            self._noise_log_likelihood_value = float(
                np.real(
                    -bilby.gw.utils.noise_weighted_inner_product(
                        self.ifo.frequency_domain_strain[mask],
                        self.ifo.frequency_domain_strain[mask],
                        self.ifo.power_spectral_density_array[mask],
                        self.ifo.duration,
                    )
                    / 2
                )
            )
        return self._noise_log_likelihood_value

    def log_likelihood(self, parameters=None):
        return self.log_likelihood_ratio(parameters) + self.noise_log_likelihood()


def reconstruct_marginalized(samples, likelihood=None, priors=None):
    """bilby conversion_function: add tc and phi back to a posterior.

    Applied by run_sampler to every sampler's output, so all four are compared
    over the same five parameters rather than the three that were sampled.

    bilby calls this twice with different signatures -- once on the posterior
    with (samples, likelihood, priors), and once on the injection parameters
    with just the dict. The injection already has tc and phi, so that call is a
    no-op.
    """
    import pandas as pd

    if likelihood is None or not hasattr(samples, "iterrows"):
        return samples

    fixed = {k: v for k, v in likelihood.parameters.items()}
    rows = []
    for _, row in samples.iterrows():
        parameters = {**fixed, **{k: row[k] for k in samples.columns if k in likelihood.parameters}}
        rows.append(likelihood.generate_posterior_sample_from_marginalized_likelihood(parameters))
    drawn = pd.DataFrame(rows, index=samples.index)
    for key in ("tc", "phi"):
        if key in drawn:
            samples[key] = drawn[key].to_numpy()
    return samples


def setup():
    """Build the interferometer, injected signal, likelihood and shared kwargs."""
    duration = 4
    sampling_frequency = 2048
    start_time = 0.0

    # A is not a strain amplitude: the model is normalised by the PSD-weighted
    # norm of its own envelope, so the optimal SNR is 2A/sqrt(duration) whatever
    # the detector. A=20 at duration=4 gives SNR 20.
    injection_parameters = dict(
        A=20.0,
        f=120.0,
        phi=1.3,
        tc=start_time + duration / 2,
        log_gamma=0.7,
    )

    ifo = bilby.gw.detector.get_empty_interferometer("L1")
    ifo.minimum_frequency = 20
    ifo.maximum_frequency = 512
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
    )

    mask = ifo.frequency_mask
    signal = np.zeros_like(ifo.frequency_array, dtype=complex)
    signal[mask] = anti_glitch_model(
        ifo.frequency_array[mask],
        ifo.power_spectral_density_array[mask],
        ifo.start_time,
        **injection_parameters,
    )
    ifo.set_strain_data_from_frequency_domain_strain(
        ifo.frequency_domain_strain + signal,
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    logger.info(f"Injected optimal SNR: {np.sqrt(ifo.optimal_snr_squared(signal).real):.2f}")

    priors = bilby.core.prior.PriorDict(
        dict(
            A=Uniform(0, 100, name="A", latex_label="$A$"),
            f=Uniform(20, 500, name="f", latex_label="$f$"),
            phi=Uniform(0, 2 * np.pi, name="phi", boundary="periodic", latex_label=r"$\varphi$"),
            tc=Uniform(
                injection_parameters["tc"] - 0.1,
                injection_parameters["tc"] + 0.1,
                name="tc",
                latex_label="$t_c$",
            ),
            log_gamma=Uniform(-2, 2, name="log_gamma", latex_label=r"$\log_{10}\gamma$"),
        )
    )

    likelihood = AntiGlitchLikelihood(ifo, priors=priors, time_marginalization=True, phase_marginalization=True)

    _common = dict(
        sampling_seed=SAMPLING_SEED,
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        injection_parameters=injection_parameters,
        conversion_function=reconstruct_marginalized,
        save="hdf5",
    )

    _common_laplace = dict(
        **_common,
        use_injection_for_map=True,
        plot_diagnostic=True,
        clean=True,
        resume=False,
        sampler="laplace",
        target_nsamples=N_SAMPLES,
        npool=1,
    )

    return _common, _common_laplace


def run_laplace(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_inprior",
        resample="inprior",
    )


def run_rejection(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_rejection",
        resample="rejection",
        max_iterations=10000000,
        batch_nsamples=10000,
    )


def run_smc(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_smc",
        resample="smc",
        smc_kwargs=dict(
            sampler="minipcn_smc",
            n_initial_samples=10000,
            n_samples=N_SAMPLES,
            adaptive=True,
            target_efficiency=(0.5, 0.8),
            target_efficiency_rate=0.5,
            sampler_kwargs=dict(
                n_steps=N_STEPS,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
            ),
        ),
        smc_plot_every=0,
    )


def run_smc_direct(_common):
    return bilby.run_sampler(
        **_common,
        sampler="aspire",
        n_samples=N_SAMPLES,
        n_initial_samples=10000,
        sample_kwargs=dict(
            sampler="minipcn_smc",
            adaptive=True,
            target_efficiency=(0.5, 0.8),
            target_efficiency_rate=0.5,
            sampler_kwargs=dict(
                n_steps=N_STEPS,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
            ),
        ),
        label=f"{base_label}_aspire",
        enable_checkpointing=False,
        npool=1,
    )


def run_dynesty(_common):
    return bilby.run_sampler(
        **_common,
        sampler="dynesty",
        label=f"{base_label}_dynesty",
        nlive=1000,
        sample="acceptance-walk",
        naccept=60,
        maxmcmc=5000,
        npool=1,
        clean=True,
        resume=False,
    )


# tc and phi are marginalised during sampling and drawn back in afterwards, so
# they live in the posterior but not in search_parameter_keys. Naming them here
# is what puts them on the comparison figure.
PLOT_PARAMETERS = ["A", "f", "log_gamma", "tc", "phi"]


def compare():
    """Comparison corner plot and evidence table for all results in outdir."""
    results, labels = compare_results(
        f"{outdir}/{base_label}_*_result.*",
        f"{base_label}_comparison.png",
        sampler_only_labels=True,
        parameters=PLOT_PARAMETERS,
    )
    if len(results) < 2:
        return

    import matplotlib.pyplot as plt

    filename = f"{base_label}_comparison.png"
    inj = getattr(results[0], "injection_parameters", None)
    try:
        fig = bilby.core.result.plot_multiple(
            results,
            labels=labels,
            colours=colours_for_results(results),
            parameters=PLOT_PARAMETERS,
            filename=filename,
            titles=False,
            save=False,
        )
    except Exception as exc:
        logger.warning(f"Could not create comparison plot: {exc}")
        return
    overlay_injection_lines(fig, PLOT_PARAMETERS, inj)
    fig.savefig(filename, dpi=400)
    plt.close(fig)
    logger.info(f"Corner plot saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti-glitch injection in Gaussian noise")
    parser.add_argument(
        "--sampler",
        nargs="+",
        choices=["laplace", "rejection", "smc", "smc-direct", "dynesty"],
        metavar="SAMPLER",
        help="One or more samplers to run: laplace, rejection, smc, smc-direct, dynesty",
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
        if args.sampler:
            _common, _common_laplace = setup()

            _run_fns = {
                "laplace": lambda: run_laplace(_common_laplace),
                "rejection": lambda: run_rejection(_common_laplace),
                "smc": lambda: run_smc(_common_laplace),
                "smc-direct": lambda: run_smc_direct(_common),
                "dynesty": lambda: run_dynesty(_common),
            }

            for name in args.sampler:
                _run_fns[name]()

        if args.compare:
            compare()
