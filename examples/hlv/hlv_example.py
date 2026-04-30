#!/usr/bin/env python

"""
Parameter estimation on a simulated BBH signal injected into Gaussian noise.

Uses a three-detector HLV (Hanford-Livingston-Virgo) network with bilby's
built-in injection infrastructure.  No real data download is needed -- the
interferometers are initialised with their design power spectral densities
and Gaussian noise is generated internally.

Usage
-----
    python examples/hlv_example.py --sampler laplace rejection smc dynesty
    python examples/hlv_example.py --sampler smc
    python examples/hlv_example.py --compare
"""

import argparse
import os

import bilby
import numpy as np
from bilby.core.prior import Constraint, Sine, Uniform
from bilby.gw.prior import (
    AlignedSpin,
    BBHPriorDict,
    UniformInComponentsChirpMass,
    UniformInComponentsMassRatio,
)

from bilby_laplace.comparison import compare as compare_results

logger = bilby.core.utils.logger
bilby.core.utils.random.seed(1234)
outdir = "outdir_hlv_example"
base_label = "hlv"


def setup():
    """Set up detector, likelihood, priors, and sampler configuration."""
    # Injection parameters
    injection_parameters = dict(
        chirp_mass=30.0,
        mass_ratio=0.8,
        chi_1=0.05,
        chi_2=-0.02,
        luminosity_distance=1000.0,
        theta_jn=0.4,
        psi=0.659,
        phase=1.3,
        geocent_time=1126259642.413,
        azimuth=1.375,
        zenith=1.2108,
    )

    # Detector setup
    duration = 4
    sampling_frequency = 2048
    minimum_frequency = 20

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXAS",
        reference_frequency=100,
        minimum_frequency=minimum_frequency,
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    ifo_list = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    ifo_list.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - duration + 2,
    )

    # Convert from zen/az to ra/dec for injection
    injection_parameters_radec = injection_parameters.copy()

    ra, dec = bilby.gw.utils.zenith_azimuth_to_ra_dec(
        injection_parameters["zenith"],
        injection_parameters["azimuth"],
        injection_parameters["geocent_time"],
        ifo_list,
    )

    injection_parameters_radec["ra"] = ra
    injection_parameters_radec["dec"] = dec

    ifo_list.inject_signal(
        parameters=injection_parameters_radec,
        waveform_generator=waveform_generator,
    )

    # Priors
    priors = BBHPriorDict(
        dictionary=dict(
            chirp_mass=UniformInComponentsChirpMass(name="chirp_mass", minimum=25, maximum=35, unit=r"$M_{\odot}$"),
            mass_ratio=UniformInComponentsMassRatio(name="mass_ratio", minimum=0.125, maximum=1),
            mass_1=Constraint(name="mass_1", minimum=10, maximum=80),
            mass_2=Constraint(name="mass_2", minimum=10, maximum=80),
            chi_1=AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.99)),
            chi_2=AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.99)),
            luminosity_distance=bilby.core.prior.PowerLaw(
                alpha=2,
                name="luminosity_distance",
                minimum=100,
                maximum=3000,
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
            zenith=Sine(name="zenith"),
            azimuth=Uniform(
                name="azimuth",
                minimum=0,
                maximum=2 * np.pi,
                boundary="periodic",
            ),
        )
    )

    # Likelihood
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

    # Shared sampler kwargs
    _common = dict(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        injection_parameters=injection_parameters,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
        save="hdf5",
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

    return _common, _common_laplace


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def run_laplace(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_laplace",
        resample="inprior",
        cov_scaling=1,
        jacobian_cap_scale=1,
    )


def run_rejection(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_rejection",
        resample="rejection",
        cov_scaling=2,
        jacobian_cap_scale=1,
        max_iterations=10000,
        batch_nsamples=10000,
    )


def run_smc(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_smc",
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
        cov_scaling=1,
        jacobian_cap_scale=1,
    )


def run_dynesty(_common):
    return bilby.run_sampler(
        **_common,
        sampler="dynesty",
        label=f"{base_label}_dynesty",
        nlive=1000,
        check_point_delta_t=1800,
        check_point_plot=True,
        npool=1,
        clean=False,
        resume=True,
    )


def compare():
    """Load all result files in outdir, make comparison corner plots,
    and print a comparison table. Custom to HLV example for intrinsic/extrinsic plots."""
    results, labels = compare_results(outdir, base_label)
    if len(results) < 2:
        return

    import matplotlib.pyplot as plt

    plotdir = os.path.dirname(os.path.abspath(__file__))

    intrinsic_params = ["mass_1", "mass_2", "chi_1", "chi_2"]
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
    parser = argparse.ArgumentParser(description="BBH injection in Gaussian noise (HLV network)")
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
