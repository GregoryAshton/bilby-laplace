#!/usr/bin/env python

"""
Laplace approximation on a simulated BNS signal using A1 and CE detectors.

Supports three likelihood types: std (standard), rb (relative binning), mb (multi-banding).

Usage
-----
    python run.py --likelihood rb --sampler laplace rejection smc dynesty
    python run.py --likelihood rb --compare
    python run.py --compare
"""

import argparse

import bilby
import numpy as np
from bilby.core.prior import Constraint, Cosine, Sine, Uniform
from bilby.gw.likelihood import (
    MBGravitationalWaveTransient,
    RelativeBinningGravitationalWaveTransient,
)
from bilby.gw.prior import (
    AlignedSpin,
    BNSPriorDict,
    UniformInComponentsChirpMass,
    UniformInComponentsMassRatio,
)

from bilby_laplace.comparison import colours_for_results
from bilby_laplace.comparison import compare as compare_results
from bilby_laplace.comparison import overlay_injection_lines

logger = bilby.core.utils.logger
bilby.core.utils.random.seed(1234)

# Base name for comparison-plot filenames. Kept separate from the per-run label
# prefix (the likelihood type, e.g. "rb-laplace") so the comparison plots have a
# stable name regardless of which likelihood was run.
base_label = "bns"


def setup(likelihood_type="rb"):
    """Set up detectors, likelihood, priors, and sampler configuration.

    Parameters
    ----------
    likelihood_type : {"std", "rb", "mb"}
        "std" uses the standard GravitationalWaveTransient (full frequency grid).
        "rb" uses RelativeBinningGravitationalWaveTransient (heterodyning).
        "mb" uses MBGravitationalWaveTransient (multi-banding).
    """
    outdir = "outdir_bns_example"
    run_prefix = likelihood_type

    # Injection parameters
    injection_parameters = dict(
        chirp_mass=1.4,
        mass_ratio=1,
        chi_1=0.00,
        chi_2=0.00,
        luminosity_distance=500.0,  # Mpc
        theta_jn=0.5,
        psi=1.3,
        phase=2.1,
        geocent_time=0.0,
        ra=1.2,
        dec=1.17,
        lambda_1=310.0,
        lambda_2=310.0,
    )

    # Detector setup
    duration = 128
    sampling_frequency = 1024
    minimum_frequency = 40

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomD_NRTidalv2",
        reference_frequency=100,
    )

    # Waveform generator for injection (uses standard model)
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments,
    )

    # Einstein Telescope and Cosmic Explorer
    ifo_list = bilby.gw.detector.InterferometerList(["A1", "CE"])

    for ifo in ifo_list:
        ifo.minimum_frequency = minimum_frequency

    ifo_list.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - duration + 2,
    )

    ifo_list.inject_signal(
        parameters=injection_parameters,
        waveform_generator=waveform_generator,
    )

    # Priors for BNS
    priors = BNSPriorDict(
        dictionary=dict(
            chirp_mass=UniformInComponentsChirpMass(
                name="chirp_mass", minimum=1.399, maximum=1.401, unit=r"$M_{\odot}$"
            ),
            mass_ratio=UniformInComponentsMassRatio(name="mass_ratio", minimum=0.2, maximum=1.0),
            mass_1=Constraint(name="mass_1", minimum=1.0, maximum=2.8),
            mass_2=Constraint(name="mass_2", minimum=1.0, maximum=2.8),
            luminosity_distance=bilby.gw.prior.UniformSourceFrame(
                name="luminosity_distance",
                minimum=100,
                maximum=10000,
                unit="Mpc",
                latex_label=r"$d_L$",
            ),
            theta_jn=Sine(name="theta_jn"),
            psi=Uniform(name="psi", minimum=0, maximum=np.pi / 2, boundary="periodic"),
            phase=Uniform(name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"),
            geocent_time=Uniform(
                minimum=injection_parameters["geocent_time"] - 0.01,
                maximum=injection_parameters["geocent_time"] + 0.01,
                name="geocent_time",
                latex_label=r"$t_{\rm geo}$",
                unit="$s$",
            ),
            ra=Uniform(
                name="ra",
                minimum=0,
                maximum=2 * np.pi,
                boundary="periodic",
                latex_label=r"$\alpha$",
            ),
            dec=Cosine(
                name="dec",
                latex_label=r"$\delta$",
            ),
            chi_1=AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.05)),
            chi_2=AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.05)),
            lambda_1=Uniform(name="lambda_1", minimum=0, maximum=5000),
            lambda_2=Uniform(name="lambda_2", minimum=0, maximum=5000),
        )
    )

    # Fixed parameters to simplify the PE
    for key in ["chi_1", "chi_2"]:
        priors[key] = injection_parameters[key]

    if likelihood_type == "std":
        std_waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments,
        )

        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifo_list,
            waveform_generator=std_waveform_generator,
            priors=priors,
            time_marginalization=True,
            phase_marginalization=True,
            distance_marginalization=True,
            jitter_time=False,
        )

    elif likelihood_type == "rb":
        rb_waveform_arguments = waveform_arguments.copy()
        rb_waveform_arguments["frequency_bin_edges"] = np.logspace(
            np.log10(minimum_frequency),
            np.log10(sampling_frequency / 2),
            100,
        )
        rb_waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star_relative_binning,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=rb_waveform_arguments,
        )

        chirp_mass = injection_parameters["chirp_mass"]
        mass_ratio = injection_parameters["mass_ratio"]
        m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
        fiducial_parameters = injection_parameters.copy()
        fiducial_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(m1, m2)
        fiducial_parameters["mass_ratio"] = m2 / m1

        likelihood = RelativeBinningGravitationalWaveTransient(
            ifo_list,
            rb_waveform_generator,
            priors=priors,
            fiducial_parameters=fiducial_parameters,
            time_marginalization=False,
            phase_marginalization=True,
            distance_marginalization=True,
            jitter_time=False,
            epsilon=0.25,
        )

    elif likelihood_type == "mb":
        mb_waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.binary_neutron_star_frequency_sequence,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments,
        )

        likelihood = MBGravitationalWaveTransient(
            interferometers=ifo_list,
            waveform_generator=mb_waveform_generator,
            priors=priors,
            reference_chirp_mass=injection_parameters["chirp_mass"],
            time_marginalization=True,
            phase_marginalization=True,
            distance_marginalization=True,
            accuracy_factor=0.1,
            linear_interpolation=False,
            jitter_time=False,
        )

    else:
        raise ValueError(f"Unknown likelihood_type {likelihood_type!r}; choose 'std', 'rb', or 'mb'")

    # Shared sampler kwargs
    _common = dict(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        injection_parameters=injection_parameters,
        conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
        result_class=bilby.gw.result.CBCResult,
        save="hdf5",
        use_ratio=True,
    )

    _common_laplace = dict(
        **_common,
        use_injection_for_map=True,
        plot_diagnostic=True,
        clean=True,
        sampler="laplace",
        target_nsamples=1000,
        use_unit_cube=True,
    )

    return _common, _common_laplace, outdir, run_prefix


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def run_laplace(_common_laplace, run_prefix):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{run_prefix}-laplace",
        resample="inprior",
        cov_scaling=1,
        jacobian_cap_scale=1,
    )


def run_rejection(_common_laplace, run_prefix):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{run_prefix}-rejection",
        resample="rejection",
        cov_scaling=1,
        jacobian_cap_scale=1,
        max_iterations=10000000,
        batch_nsamples=10000,
        prior_parameters=["lambda_1", "lambda_2", "psi"],
    )


def run_smc(_common_laplace, run_prefix):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{run_prefix}-smc",
        resample="smc",
        smc_kwargs=dict(
            sampler="minipcn_smc",
            n_initial_samples=1000,
            n_final_samples=5000,
            adaptive=True,
            sampler_kwargs=dict(
                n_steps=10,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
            ),
        ),
        cov_scaling=2,
        jacobian_cap_scale=1,
        hessian_kwargs={"initial_step": 0.001, "step_factor": 2, "maxiter": 10},
        prior_parameters=["lambda_1", "lambda_2", "psi"],
    )


def run_dynesty(_common, run_prefix):
    return bilby.run_sampler(
        **_common,
        sampler="dynesty",
        label=f"{run_prefix}-dynesty",
        nlive=1000,
        dlogz=0.1,
        check_point_delta_t=1800,
        check_point_plot=True,
        npool=32,
        clean=False,
        resume=True,
    )


def compare(outdir):
    """Load all result files, make comparison corner plots, and print comparison table."""
    pattern = f"{outdir}/*_result.*"
    full_filename = f"{base_label}_comparison.png"
    results, labels = compare_results(pattern, full_filename, sampler_only_labels=True)

    # Custom plotting for this example
    import matplotlib.pyplot as plt

    intrinsic_params = ["mass_1", "mass_2", "chi_1", "chi_2", "lambda_1", "lambda_2"]
    extrinsic_params = [
        "ra",
        "dec",
        "luminosity_distance",
        "theta_jn",
        "psi",
        "geocent_time",
        "phase",
    ]

    plot_sets = [
        ("comparison_intrinsic", intrinsic_params),
        ("comparison_extrinsic", extrinsic_params),
    ]

    inj = getattr(results[0], "injection_parameters", None)

    for suffix, parameters in plot_sets:
        filename = f"{base_label}_{suffix}.png"
        # Check if the parameters are sampled
        plot_parameters = []
        for p in parameters:
            # Check if the posterior set has a non-zero range
            samples = results[0].posterior.get(p)
            if samples is not None and np.ptp(samples) > 0:
                plot_parameters.append(p)

        if len(plot_parameters) == 0:
            logger.info(f"No sampled parameters found for {suffix} plot; skipping")
            continue

        logger.info(f"Creating {suffix} corner plot for parameters: {plot_parameters}")
        fig = bilby.core.result.plot_multiple(
            results,
            labels=labels,
            colours=colours_for_results(results),
            parameters=plot_parameters,
            filename=filename,
            titles=False,
            save=False,
        )

        overlay_injection_lines(fig, plot_parameters, inj)

        fig.savefig(filename, dpi=400)
        plt.close(fig)
        logger.info(f"Corner plot saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BNS injection with fast likelihood (ET + CE)")
    parser.add_argument(
        "--likelihood",
        choices=["std", "rb", "mb"],
        default=None,
        help=(
            "Likelihood: std (standard), rb (relative binning), or mb"
            " (multi-banding). Required when running samplers."
        ),
    )
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
        _outdir = "outdir_bns_example"

        if args.sampler:
            if args.likelihood is None:
                parser.error("--likelihood is required when running samplers")
            _common, _common_laplace, _outdir, _run_prefix = setup(args.likelihood)

            _run_fns = {
                "laplace": lambda: run_laplace(_common_laplace, _run_prefix),
                "rejection": lambda: run_rejection(_common_laplace, _run_prefix),
                "smc": lambda: run_smc(_common_laplace, _run_prefix),
                "dynesty": lambda: run_dynesty(_common, _run_prefix),
            }

            for name in args.sampler:
                _run_fns[name]()

        if args.compare:
            compare(_outdir)
