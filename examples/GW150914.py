#!/usr/bin/env python

"""
Parameter estimation on GW150914 using the Laplace sampler.

The Laplace approximation runs in minutes rather than hours. It works well
for the more Gaussian parameters (chirp mass, mass ratio, inclination) but
will be less accurate for parameters with non-Gaussian or multi-modal
posteriors (sky location, distance).

Pass --dynesty to run a full dynesty comparison

Data is fetched from GWOSC via gwpy. See
https://gwpy.github.io/docs/stable/timeseries/remote-access.html
for details on accessing data on the LIGO Data Grid instead.
"""

import argparse
import numpy as np
import bilby
from bilby.core.prior import Constraint, PowerLaw, Sine, Uniform
from bilby.gw.prior import (
    AlignedSpin,
    BBHPriorDict,
    UniformInComponentsChirpMass,
    UniformInComponentsMassRatio,
)
from gwpy.timeseries import TimeSeries

from aspire.utils import configure_logger
configure_logger()

logger = bilby.core.utils.logger
outdir = "outdir"
base_label = "GW150914"

# Get the data
trigger_time = 1126259462.4
detectors = ["H1", "L1"]
maximum_frequency = 512
minimum_frequency = 20
roll_off = 0.4
duration = 4
post_trigger_duration = 2
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 16 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info(f"Downloading analysis data for ifo {det}")
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info(f"Downloading PSD data for ifo {det}")
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, cache=True)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)


# Set up the prior
priors = BBHPriorDict(
    dictionary=dict(
        #chirp_mass=UniformInComponentsChirpMass(
        #name="chirp_mass", minimum=25, maximum=35, unit="$M_{\\odot}$"
        #),
        chirp_mass=31.2,
        #mass_ratio=UniformInComponentsMassRatio(
        #name="mass_ratio", minimum=0.125, maximum=1
        #),
        mass_ratio=1,
        mass_1=Constraint(name="mass_1", minimum=10, maximum=80),
        mass_2=Constraint(name="mass_2", minimum=10, maximum=80),
        chi_1=0, #AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.99)),
        chi_2=0, #AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.99)),
        luminosity_distance=PowerLaw(
            alpha=2,
            name="luminosity_distance",
            minimum=50,
            maximum=2000,
            unit="Mpc",
            latex_label="$d_L$",
        ),
        zenith=Sine(name="zenith"),
        azimuth=Uniform(name="azimuth", minimum=0, maximum=2 * np.pi, boundary="periodic"),
        #theta_jn=Sine(name="theta_jn"),
        theta_jn=1.4,
        #psi=Uniform(name="psi", minimum=0, maximum=np.pi, boundary="periodic"),
        psi = 0.5,
        phase=Uniform(name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"),
        geocent_time=Uniform(
            minimum=trigger_time - 0.1,
            maximum=trigger_time + 0.1,
            name="geocent_time",
            latex_label=r"$t_{\rm geo}$",
            unit="$s$",
        ),
    )
)

# Set up the waveform generator and likelihood
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomPv2",
        "reference_frequency": 50,
    },
)

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    time_marginalization=True,
    phase_marginalization=True,
    distance_marginalization=True,
    jitter_time=False,
    reference_frame="H1L1",
)

# Run the sampler
def run_laplace(args):
    return bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label=f"{base_label}_laplace",
        use_injection_for_maxL=False,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
        plot_diagnostic=True,
        clean=True,
        cov_scaling=3,
        extension="hdf5",
        sampler="laplace",
        resample="smc",
        smc_kwargs=dict(
            backend="minipcn",
            n_samples=1000,
            n_final_samples=5000,
            target_efficiency=[0.5, 0.8],
            adaptive=True,
            sampler_kwargs=dict(
                n_steps=5,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
                verbose=True,
            ),
        ),
)


def run_dynesty(args):
    return bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        outdir=outdir,
        label=f"{base_label}_dynesty",
        nlive=250,
        check_point_delta_t=1800,
        check_point_plot=True,
        npool=1,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
        clean=False,
        resume=True,
        extension="hdf5"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dynesty",
        action="store_true",
        help="Run dynesty and produce a comparison corner plot",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="If true, run clean",
    )
    args = parser.parse_args()

    if args.dynesty:
        result_dynesty = run_dynesty(args)
        result_dynesty.plot_corner()

    result_laplace = run_laplace(args)
    result_laplace.plot_corner()

    if args.dynesty:
        bilby.core.result.plot_multiple(
            [result_laplace, result_dynesty],
            labels=["Laplace", "Dynesty"],
            filename=f"{outdir}/{base_label}_comparison_corner.png",
            titles=False,
        )
