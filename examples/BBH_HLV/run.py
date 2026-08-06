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
    AlignedSpin,
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
# *absent*: no cov_scaling, no prior_parameters, no hessian_kwargs.  Those were
# per-example compensations for a prior-precision term that collapsed proposal
# widths at a prior cusp; that is fixed in LaplacePosteriorEstimator, so they
# should no longer be needed.

# MCMC steps per SMC temperature level.  This is the one setting that does *not*
# transfer between problems, so it is named here rather than buried in the
# shared block.  It sets how far a particle can travel per tempering iteration,
# and the requirement scales with how hard the posterior is to traverse, not
# with anything the shared settings control.
#   BBH_HLV: 20  -- a sweep found 10 unstable (58-nat log Z spread), 20 stable
#                   (0.99), and 40 no better at twice the cost.
#   BNS_3G: 100  -- with two detectors the sky is a timing *ring*; at 20 the
#                   cloud covers one arc of it (ra>4 occupancy 0.006 against
#                   dynesty's 0.575), at 100 it spans the ring (0.447) and the
#                   evidence deficit falls from 1.2 nats to 0.55.
N_MUTATION_STEPS = 20
GW_SMC_SETTINGS = dict(
    smc_kwargs=dict(
        sampler="minipcn_smc",
        n_initial_samples=10000,
        n_samples=5000,
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
    jacobian_cap_scale=1,
)

# Matching configuration for the no-Laplace control, expressed in the aspire
# plugin's own kwarg layout (n_samples / n_initial_samples at the top level,
# sample_kwargs forwarded to sample_posterior).
GW_SMC_DIRECT_SETTINGS = dict(
    n_samples=5000,
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
            chirp_mass=UniformInComponentsChirpMass(
                name="chirp_mass", minimum=25, maximum=35, unit=r"$M_{\odot}$", latex_label=r"$\mathcal{M}$"
            ),
            mass_ratio=UniformInComponentsMassRatio(name="mass_ratio", minimum=0.125, maximum=1, latex_label=r"$q$"),
            mass_1=Constraint(name="mass_1", minimum=10, maximum=80),
            mass_2=Constraint(name="mass_2", minimum=10, maximum=80),
            chi_1=AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.99), latex_label=r"$\chi_1$"),
            chi_2=AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.99), latex_label=r"$\chi_2$"),
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
        cov_scaling=1,
        jacobian_cap_scale=1,
    )


def run_rejection(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_rejection",
        resample="rejection",
        cov_scaling=dict(others=1, chi_1=5, chi_2=5),
        jacobian_cap_scale=1,
        max_iterations=10000000,
        batch_nsamples=10000,
    )


def run_smc(_common_laplace):
    """Run the SMC resampling stage with the shared GW settings.

    ``mode_search_subspace`` is set here rather than in ``GW_SMC_SETTINGS``
    because it names this example's sky coordinates (``zenith``/``azimuth``,
    where BNS_3G uses ``ra``/``dec``).  It identifies *which* coordinates a sky
    degeneracy lives in, not a tuned value.
    """
    return bilby.run_sampler(
        **_common_laplace,
        **GW_SMC_SETTINGS,
        label=f"{base_label}_smc",
        resample="smc",
        mode_search_subspace=["zenith", "azimuth"],
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
    return bilby.run_sampler(
        **_common,
        sampler="dynesty",
        label=f"{base_label}_dynesty",
        nlive=1000,
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
