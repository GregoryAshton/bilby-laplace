#!/usr/bin/env python

"""
Binary neutron star parameter estimation with three likelihood options.

Uses LIGO India (A1) and Cosmic Explorer (CE) detectors to observe a
simulated binary neutron star merger signal.

  std (standard)
    Full GravitationalWaveTransient likelihood evaluated on the complete
    frequency grid.  Slowest but exact reference.

  rb (relative binning / heterodyning)
    Computes the likelihood in a narrow time-frequency window around the signal
    using a reference waveform, then rapidly evaluates likelihood ratios at
    nearby points.  Typically 10-100x faster than a full evaluation.

  mb (multi-banding, Morisaki 2021, arXiv:2104.07813)
    Divides the frequency domain into sub-bands with progressively coarser
    frequency resolution at high frequencies where the waveform phase evolution
    is slow.  Provides similar speed gains without requiring a fiducial waveform.

Usage
-----
    python examples/bns_heterodyned_example.py --likelihood std --sampler laplace dynesty
    python examples/bns_heterodyned_example.py --likelihood rb --sampler laplace rejection smc dynesty
    python examples/bns_heterodyned_example.py --likelihood mb --sampler laplace
    python examples/bns_heterodyned_example.py --likelihood rb --compare
"""

import argparse
import os

import bilby
import numpy as np
from bilby.core.prior import Constraint, Sine, Uniform
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

from bilby_laplace.comparison import compare as compare_results

logger = bilby.core.utils.logger
bilby.core.utils.random.seed(1234)


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
    base_label = likelihood_type

    # Injection parameters
    injection_parameters = dict(
        chirp_mass=1.4,
        mass_ratio=1,
        chi_1=0.00,
        chi_2=0.00,
        luminosity_distance=1000.0,  # Mpc
        theta_jn=0.5,
        psi=0.3,
        phase=2.1,
        geocent_time=0.0,
        ra=1.0,
        dec=0.5,
        lambda_1=50.0,  # Tidal deformability
        lambda_2=50.0,
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
            luminosity_distance=bilby.core.prior.PowerLaw(
                alpha=2,
                name="luminosity_distance",
                minimum=50,
                maximum=10000,
                unit="Mpc",
                latex_label=r"$d_L$",
            ),
            theta_jn=Sine(name="theta_jn"),
            psi=Uniform(name="psi", minimum=0, maximum=np.pi / 2, boundary="periodic"),
            phase=Uniform(name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"),
            geocent_time=Uniform(
                minimum=injection_parameters["geocent_time"] - 0.05,
                maximum=injection_parameters["geocent_time"] + 0.05,
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
            dec=Uniform(
                name="dec",
                minimum=-np.pi / 2,
                maximum=np.pi / 2,
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
            time_marginalization=True,
            phase_marginalization=True,
            distance_marginalization=True,
            jitter_time=False,
            chi=2,
            epsilon=0.5,
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
        # conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
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

    return _common, _common_laplace, outdir, base_label


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def run_laplace(_common_laplace, base_label):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}-laplace",
        resample="inprior",
        cov_scaling=1,
        jacobian_cap_scale=1,
    )


def run_rejection(_common_laplace, base_label):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}-rejection",
        resample="rejection",
        cov_scaling=2,
        jacobian_cap_scale=1,
        max_iterations=1000000,
        batch_nsamples=10000,
    )


def run_smc(_common_laplace, base_label):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}-smc",
        resample="smc",
        smc_kwargs=dict(
            sampler="minipcn_smc",
            n_initial_samples=1000,
            n_final_samples=5000,
            adaptive=True,
            sampler_kwargs=dict(
                n_steps=5,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
            ),
        ),
        cov_scaling=10,
        jacobian_cap_scale=1,
    )


def run_dynesty(_common, base_label):
    return bilby.run_sampler(
        **_common,
        sampler="dynesty",
        label=f"{base_label}-dynesty",
        nlive=500,
        dlogz=0.5,
        check_point_delta_t=1800,
        check_point_plot=True,
        npool=1,
        clean=False,
        resume=True,
    )


def compare(outdir, base_label):
    """Load all result files, make comparison corner plots, and print comparison table."""
    results, labels = compare_results(outdir, base_label)
    if len(results) < 2:
        return

    import matplotlib.pyplot as plt

    plotdir = os.path.dirname(os.path.abspath(__file__))

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
        ("comparison", None),
        ("comparison_intrinsic", intrinsic_params),
        ("comparison_extrinsic", extrinsic_params),
    ]

    inj = getattr(results[0], "injection_parameters", None)

    for suffix, parameters in plot_sets:
        filename = os.path.join(plotdir, f"{base_label}_{suffix}.png")
        try:
            fig = bilby.core.result.plot_multiple(
                results,
                labels=labels,
                parameters=parameters,
                filename=filename,
                titles=False,
                save=False,
            )
        except Exception as exc:
            logger.warning(f"Could not create {suffix} plot: {exc}")
            continue

        # Overlay injection truth values
        if inj:
            params = parameters if parameters is not None else results[0].search_parameter_keys
            truths = [inj.get(p) for p in params]
            ndim = len(params)
            axes = fig.get_axes()
            if len(axes) == ndim * ndim:
                axes_grid = np.array(axes).reshape(ndim, ndim)
                for row in range(ndim):
                    for col in range(ndim):
                        ax = axes_grid[row, col]
                        if row == col:
                            if truths[col] is not None:
                                ax.axvline(truths[col], color="k", ls="--", lw=1.0)
                        elif row > col:
                            if truths[col] is not None:
                                ax.axvline(truths[col], color="k", ls="--", lw=0.8, alpha=0.7)
                            if truths[row] is not None:
                                ax.axhline(truths[row], color="k", ls="--", lw=0.8, alpha=0.7)

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
        _base_label = args.likelihood  # None means compare all likelihoods

        if args.sampler:
            if args.likelihood is None:
                parser.error("--likelihood is required when running samplers")
            _common, _common_laplace, _outdir, _base_label = setup(args.likelihood)

            _run_fns = {
                "laplace": lambda: run_laplace(_common_laplace, _base_label),
                "rejection": lambda: run_rejection(_common_laplace, _base_label),
                "smc": lambda: run_smc(_common_laplace, _base_label),
                "dynesty": lambda: run_dynesty(_common, _base_label),
            }

            for name in args.sampler:
                _run_fns[name]()

        if args.compare:
            compare(_outdir, _base_label)
