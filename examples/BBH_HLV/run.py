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
bilby.core.utils.random.seed(1234)
outdir = "outdir_hlv_example"
base_label = "hlv"

# "smc-direct" is aspire driven straight from the prior, with no Laplace stage
# at all.  It is a configuration of SMC specific to this example rather than a
# method of its own, so it has no colour in the shared palette; without an
# override it would inherit the SMC green and be indistinguishable from the
# Laplace-seeded run it exists to be compared against.  The violet is from the
# IBM colourblind-safe palette and stays separable from that green and the
# dynesty blue.
COLOUR_OVERRIDES = {"smc-direct": "#785EF0"}

# Shared GW settings for the SMC stage.  Kept in step with
# examples/BNS_3G/run.py: the goal is one configuration that holds across GW
# problems rather than per-example tuning, so treat a change here as a change to
# both -- with one deliberate exception, N_MUTATION_STEPS below.  Note what is
# *absent*: no cov_scaling, no jacobian_cap_scale, no prior_parameters, no
# hessian_kwargs.  Those were
# per-example compensations for a prior-precision term that collapsed proposal
# widths at a prior cusp; that is fixed in LaplacePosteriorEstimator, so they
# should no longer be needed.

# MCMC steps per SMC temperature level.  100 was the optimum of a paired sweep;
# the accuracy gain flattens above it while cost stays linear.
#
# The aspire paper (Sec. 5, p16) quotes n_steps=500, reduced to 80 for
# precession, but that is *not* directly comparable: it applies to a small
# tempered cloud (n_samples=1000) expanded once at the end via n_final_samples,
# so 1000 x 80 is 80k evaluations per tempering iteration against 800k for the
# cloud used here.  Their configuration was measured on this example and does
# not transfer -- see N_PARTICLES.
N_MUTATION_STEPS = 100

# Particles carried through every tempering iteration.  Accuracy on the spin
# block tracks this strongly and nothing else: sweeping 1000 / 2000 / 5000 /
# 10000 with every other setting fixed moved mean |dmu| 0.27 -> 0.21 -> 0.17 ->
# 0.12 sigma and the worst width ratio 0.91 -> 0.82 -> 0.87 -> 0.35, with tilt_1
# stuck near 0.55 for every starved cloud before recovering to 0.79 at 10000.
# Shrinking this to the paper's 1000 and recovering the sample count with
# n_final_samples does *not* work: the final expansion is one resample-and-
# mutate at beta=1, so it restores the sample count without restoring the
# exploration the tempering never did.  The no-Laplace control degrades just as
# much on a small cloud, so this is a property of the problem, not the seeding.
N_PARTICLES = 10000

GW_SMC_SETTINGS = dict(
    smc_kwargs=dict(
        sampler="minipcn_smc",
        n_initial_samples=10000,
        n_samples=N_PARTICLES,
        adaptive=True,
        target_efficiency=0.5,
        sampler_kwargs=dict(
            n_steps=N_MUTATION_STEPS,
            target_acceptance_rate=0.234,
            step_fn="tpcn",
        ),
    ),
    smc_plot_every=0,
    n_modes=3,
    mode_weights="laplace",
    mode_separation_sigma=1,
    mode_search_nsamples=5000,
)

# Matching configuration for the no-Laplace control, expressed in the aspire
# plugin's own kwarg layout (n_samples / n_initial_samples at the top level,
# sample_kwargs forwarded to sample_posterior).
GW_SMC_DIRECT_SETTINGS = dict(
    n_samples=N_PARTICLES,
    n_initial_samples=10000,
    sample_kwargs=dict(
        sampler="minipcn_smc",
        adaptive=True,
        sampler_kwargs=dict(
            n_steps=N_MUTATION_STEPS,
            target_acceptance_rate=0.234,
            step_fn="tpcn",
        ),
    ),
)


def setup():
    """Set up detector, likelihood, priors, and sampler configuration."""
    # Injection parameters
    injection_parameters = dict(
        chirp_mass=30.0,
        mass_ratio=0.8,
        # Precessing spins.  Values follow bilby's standard 15-D CBC tutorial,
        # chosen so the spins are large enough and tilted enough that the
        # orbital plane genuinely precesses rather than reducing to the
        # aligned-spin case.
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

    # Detector setup
    duration = 4
    sampling_frequency = 2048
    minimum_frequency = 20

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXP",  # precessing; XAS is aligned-spin only
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
            chirp_mass=UniformInComponentsChirpMass(
                name="chirp_mass", minimum=25, maximum=35, unit=r"$M_{\odot}$", latex_label=r"$\mathcal{M}$"
            ),
            mass_ratio=UniformInComponentsMassRatio(name="mass_ratio", minimum=0.125, maximum=1, latex_label=r"$q$"),
            mass_1=Constraint(name="mass_1", minimum=10, maximum=80),
            mass_2=Constraint(name="mass_2", minimum=10, maximum=80),
            # bilby's standard precessing-spin priors: isotropic spin
            # orientations (Sine tilts, uniform azimuths) with uniform
            # magnitudes.
            a_1=Uniform(name="a_1", minimum=0, maximum=0.99, latex_label=r"$a_1$"),
            a_2=Uniform(name="a_2", minimum=0, maximum=0.99, latex_label=r"$a_2$"),
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
            phase=Uniform(name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic", latex_label=r"$\phi$"),
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

    # Likelihood
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator,
        priors=priors,
        time_marginalization=True,
        # Phase marginalisation assumes the phase enters as a single overall
        # factor, which holds only for a non-precessing, dominant-mode
        # waveform.  With precession the orbital phase and the precession phase
        # are not degenerate in that way, so phase is sampled instead.
        phase_marginalization=False,
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
        resume=False,
        sampler="laplace",
        target_nsamples=5000,
        use_unit_cube=True,
        fisher_method="waveform",
        npool=16,
    )

    return _common, _common_laplace


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def run_laplace(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        # Label by the actual resample method ("inprior"), not the CLI target
        # name, so this method gets the same colour/legend as the equivalent
        # run in the other examples (see bilby_laplace.comparison).
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
    """Run the SMC resampling stage with the shared GW settings.

    ``mode_search_subspace`` is set here rather than in ``GW_SMC_SETTINGS``
    because it names this example's degenerate coordinates, not a tuned value.

    ``zenith``/``azimuth`` are the sky pair (BNS_3G uses ``ra``/``dec``).
    ``phase`` is included because with precession we cannot marginalise it away,
    and the posterior is bimodal with lobes about pi apart.  The search pins
    every coordinate outside the subspace at the primary MAP, so leaving phase
    out made the second lobe structurally undiscoverable: all three modes came
    back within 0.2 rad of the same phase and the initial cloud was seeded
    entirely in one lobe.
    """
    return bilby.run_sampler(
        **_common_laplace,
        **GW_SMC_SETTINGS,
        label=f"{base_label}_smc",
        resample="smc",
        mode_search_subspace=["zenith", "azimuth", "phase"],
    )


def run_smc_direct(_common):
    """Aspire's SMC on its own, with no Laplace stage.

    The control for ``run_smc``: same SMC sampler and the same particle count,
    but seeded from prior draws rather than from the Laplace proposal, and via
    ``aspire_bilby``'s own plugin rather than ours.  What it isolates is what the
    Laplace stage buys -- everything downstream of the initial cloud is held
    fixed.
    """
    return bilby.run_sampler(
        **_common,
        **GW_SMC_DIRECT_SETTINGS,
        sampler="aspire",
        label=f"{base_label}_smc_direct",
        enable_checkpointing=False,
        npool=16,
    )


def run_dynesty(_common):
    """Reference run, using the settings used for production GW parameter
    estimation.  Kept in step with examples/BNS_3G/run.py.

    ``sample="acceptance-walk"`` takes a fixed ``naccept`` accepted MCMC steps
    per point rather than adapting the chain length from the autocorrelation,
    so the cost per point is predictable and the worker pool stays busy instead
    of waiting on stragglers.
    """
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
    """Load all result files in outdir, make comparison corner plots,
    and print a comparison table. Custom to HLV example for intrinsic/extrinsic plots."""
    pattern = f"{outdir}/{base_label}_*_result.*"
    full_filename = f"{base_label}_comparison.png"
    results, labels = compare_results(
        pattern,
        full_filename,
        sampler_only_labels=True,
        colour_overrides=COLOUR_OVERRIDES,
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
                colours=colours_for_results(results, overrides=COLOUR_OVERRIDES),
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
        # Only set up likelihood/priors if running samplers
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

        # Compare only needs to read result files
        if args.compare:
            compare()
