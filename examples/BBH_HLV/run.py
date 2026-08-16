#!/usr/bin/env python

"""
Laplace approximation on a simulated BBH signal in an HLV network.

Usage
-----
    python run.py --sampler laplace rejection smc smc-direct dynesty
    python run.py --compare
"""

import argparse

import bilby
import numpy as np
from bilby.core.prior import Constraint, Sine, Uniform
from bilby.gw.prior import (
    BBHPriorDict,
    UniformInComponentsChirpMass,
    UniformInComponentsMassRatio,
)

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
outdir = "outdir_hlv_example"
base_label = "hlv"

N_STEPS = 300
N_SAMPLES = 10000
N_FINAL_SAMPLES = 10000
TARGET_EFFICIENCY = (0.5, 0.8)
TARGET_EFFICIENCY_RATE = 0.5


def setup():
    """Build the detectors, likelihood, priors, and shared sampler kwargs."""
    injection_parameters = dict(
        chirp_mass=30.0,
        mass_ratio=0.8,
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=1000.0,
        theta_jn=0.4,
        psi=0.659,
        phase=1.3,
        geocent_time=1126259642.413,
        azimuth=1.375,
        zenith=1.2108,
    )
    # The injected value in the sampled coordinate, inverting the conversion
    # above. Needed for the truth overlays and because `use_injection_for_map`
    # seeds the optimiser from this dict, which must cover every sampled
    # parameter. The signal itself is still injected with `phase`.
    injection_parameters["delta_phase"] = np.mod(
        injection_parameters["phase"]
        + np.sign(np.cos(injection_parameters["theta_jn"])) * injection_parameters["psi"],
        2 * np.pi,
    )

    duration = 4
    sampling_frequency = 2048
    minimum_frequency = 20

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXP",
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

    priors = BBHPriorDict(
        dictionary=dict(
            chirp_mass=UniformInComponentsChirpMass(
                name="chirp_mass", minimum=25, maximum=35, unit=r"$M_{\odot}$", latex_label=r"$\mathcal{M}$"
            ),
            mass_ratio=UniformInComponentsMassRatio(name="mass_ratio", minimum=0.125, maximum=1, latex_label=r"$q$"),
            mass_1=Constraint(name="mass_1", minimum=10, maximum=80),
            mass_2=Constraint(name="mass_2", minimum=10, maximum=80),
            a_1=Uniform(name="a_1", minimum=0, maximum=0.99, latex_label=r"$a_1$"),
            a_2=Uniform(name="a_2", minimum=0, maximum=0.99, latex_label=r"$a_2$"),
            # cos_tilt_1/cos_tilt_2 were tried here and do not work on this
            # example. The reparameterisation is prior-preserving as a
            # *distribution*, but a density mode is not invariant under a change
            # of coordinates: the Sine prior suppresses tilt -> 0, while a flat
            # prior in the cosine does not, and the MAP duly moves to
            # cos_tilt_1 = +1.000000 exactly (aligned spin). The waveform Fisher
            # then steps outside [-1, 1], arccos returns NaN and the waveform
            # call fails. Even with a boundary-safe derivative, centring a
            # Gaussian proposal on a prior edge is the wrong place to be.
            tilt_1=Sine(name="tilt_1", latex_label=r"$\theta_1$"),
            tilt_2=Sine(name="tilt_2", latex_label=r"$\theta_2$"),
            phi_12=Uniform(
                name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic", latex_label=r"$\Delta\phi$"
            ),
            phi_jl=Uniform(
                name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic", latex_label=r"$\phi_{JL}$"
            ),
            luminosity_distance=bilby.core.prior.PowerLaw(
                alpha=2,
                name="luminosity_distance",
                minimum=100,
                maximum=3000,
                unit="Mpc",
                latex_label=r"$d_L$",
            ),
            theta_jn=Sine(name="theta_jn", latex_label=r"$\theta_{JN}$"),
            psi=Uniform(name="psi", minimum=0, maximum=np.pi / 2, boundary="periodic", latex_label=r"$\psi$"),
            # Sampled in place of `phase`. `convert_to_lal_binary_black_hole_parameters`
            # -- already this example's `parameter_conversion` -- recovers
            # phase = mod(delta_phase - sign(cos(theta_jn)) * psi, 2*pi), so the
            # likelihood is unchanged and `phase` reappears in the posterior via
            # the conversion function.
            #
            # For fixed psi and theta_jn this is a shear on the phase circle:
            # a bijection with unit Jacobian, so a Uniform(0, 2pi) on delta_phase
            # induces exactly the Uniform(0, 2pi) on phase it replaces. The prior
            # and the target are therefore identical -- only the coordinates the
            # sampler moves in change, which keeps the dynesty run a valid
            # reference in either coordinate.
            delta_phase=Uniform(
                name="delta_phase",
                minimum=0,
                maximum=2 * np.pi,
                boundary="periodic",
                latex_label=r"$\delta\phi$",
            ),
            geocent_time=Uniform(
                minimum=injection_parameters["geocent_time"] - 0.05,
                maximum=injection_parameters["geocent_time"] + 0.05,
                name="geocent_time",
                latex_label=r"$t_{\rm geo}$",
                unit="$s$",
            ),
            zenith=Sine(name="zenith", latex_label=r"$\kappa$"),
            azimuth=Uniform(
                name="azimuth",
                minimum=0,
                maximum=2 * np.pi,
                boundary="periodic",
                latex_label=r"$\epsilon$",
            ),
        )
    )

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator,
        priors=priors,
        time_marginalization=True,
        phase_marginalization=False,
        distance_marginalization=True,
        jitter_time=False,
        reference_frame="H1L1",
    )

    _common = dict(
        sampling_seed=SAMPLING_SEED,
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
        resume=False,
        sampler="laplace",
        target_nsamples=5000,
        use_unit_cube=True,
        fisher_method="waveform",
        npool=16,
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
            n_final_samples=N_FINAL_SAMPLES,
            adaptive=True,
            target_efficiency=TARGET_EFFICIENCY,
            target_efficiency_rate=TARGET_EFFICIENCY_RATE,
            sampler_kwargs=dict(
                n_steps=N_STEPS,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
            ),
        ),
        smc_plot_every=0,
        n_modes=3,
        mode_weights="laplace",
        mode_separation_sigma=1,
        mode_search_nsamples=5000,
        mode_search_subspace=["zenith", "azimuth", "delta_phase"],
        # The log-posterior is exactly pi-periodic in delta_phase -- the two
        # lobes are degenerate to machine precision -- so the mirror is written
        # down rather than left to the random multi-start search, which on this
        # problem returned three candidates all in the same lobe and sampled
        # half the posterior.
        mode_symmetries=[("delta_phase", np.pi)],
        # `smc_prior_flow="laplace"` was tried here and is much worse (mean JSD
        # 86 against 8.8): GaussianMixtureFlow is an *unbounded, non-periodic*
        # Gaussian mixture, so the log q it supplies is wrong at the prior
        # edges and across the five periodic coordinates this example declares.
        # The trained flow gets those right via aspire's bounded/angular
        # preconditioning. See the note on the kwarg in sampler.py.
    )


def run_smc_direct(_common):
    return bilby.run_sampler(
        **_common,
        sampler="aspire",
        n_samples=N_SAMPLES,
        n_initial_samples=10000,
        n_final_samples=N_FINAL_SAMPLES,
        sample_kwargs=dict(
            sampler="minipcn_smc",
            adaptive=True,
            target_efficiency=TARGET_EFFICIENCY,
            target_efficiency_rate=TARGET_EFFICIENCY_RATE,
            sampler_kwargs=dict(
                n_steps=N_STEPS,
                target_acceptance_rate=0.234,
                step_fn="tpcn",
            ),
        ),
        label=f"{base_label}_aspire",
        enable_checkpointing=False,
        npool=16,
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
        check_point_delta_t=1800,
        check_point_plot=True,
        npool=16,
        clean=False,
        resume=True,
    )


def compare():
    """Comparison corner plots and evidence table for all results in outdir."""
    pattern = f"{outdir}/{base_label}_*_result.*"
    full_filename = f"{base_label}_comparison.png"
    results, labels = compare_results(
        pattern,
        full_filename,
        sampler_only_labels=True,
    )
    if len(results) < 2:
        return

    import matplotlib.pyplot as plt

    intrinsic_params = ["mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2"]
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
        filename = f"{base_label}_{suffix}.png"
        try:
            fig = bilby.core.result.plot_multiple(
                results,
                labels=labels,
                colours=colours_for_results(results),
                parameters=parameters,
                filename=filename,
                titles=False,
                save=False,
            )
        except Exception as exc:
            logger.warning(f"Could not create {suffix} plot: {exc}")
            continue

        params = parameters if parameters is not None else results[0].search_parameter_keys
        overlay_injection_lines(fig, params, inj)

        fig.savefig(filename, dpi=400)
        plt.close(fig)
        logger.info(f"Corner plot saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BBH injection in Gaussian noise (HLV network)")
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
