#!/usr/bin/env python

"""
Laplace approximation on a simulated BNS signal using A1 and CE detectors.

Supports two likelihood types: std (standard) and rb (relative binning).

Usage
-----
    python run.py --likelihood rb --sampler laplace rejection smc smc-direct dynesty
    python run.py --likelihood rb --compare
    python run.py --compare
"""

import argparse
import timeit

import bilby
import numpy as np
from bilby.core.prior import Constraint, Cosine, Sine, Uniform
from bilby.gw.likelihood import RelativeBinningGravitationalWaveTransient
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

# The two detectors whose baseline defines the sampled sky frame.  Used for
# *both* the injection conversion and the likelihood: bilby's
# ``zenith_azimuth_to_ra_dec`` silently takes the first two entries of whatever
# interferometer list it is handed, so passing the full network would refer the
# injection to a different baseline than the likelihood samples in -- placing
# the signal in one frame and searching in another.
REFERENCE_FRAME = ["A1", "CE"]

# "smc-direct" is aspire driven straight from the prior, with no Laplace stage.
# It is a configuration of SMC rather than a method of its own, so it has no
# colour in the shared palette; without an override it would inherit the SMC
# green and be indistinguishable from the Laplace-seeded run it exists to be
# compared against.  The violet is from the IBM colourblind-safe palette and
# stays separable from that green and the dynesty blue.
COLOUR_OVERRIDES = {"smc-direct": "#785EF0"}

# Shared GW settings for the SMC stage.  Kept in step with
# examples/BBH_HLV/run.py: the goal is one configuration that holds across GW
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
N_MUTATION_STEPS = 100
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


def setup(likelihood_type="rb"):
    """Set up detectors, likelihood, priors, and sampler configuration.

    Parameters
    ----------
    likelihood_type : {"std", "rb"}
        "std" uses the standard GravitationalWaveTransient (full frequency grid).
        "rb" uses RelativeBinningGravitationalWaveTransient (heterodyning).
    """
    outdir = "outdir_bns_example"
    run_prefix = likelihood_type

    # Injection parameters
    injection_parameters = dict(
        chirp_mass=1.4,
        mass_ratio=1,
        chi_1=0.00,
        chi_2=0.00,
        luminosity_distance=200.0,  # Mpc
        theta_jn=0.5,
        psi=1.3,
        phase=2.1,
        geocent_time=0.0,
        # Detector-frame sky coordinates, referred to the A1-CE baseline.  With
        # only two detectors the timing constrains the sky to a *ring*: in this
        # frame that ring is zenith = const with azimuth free -- one tight
        # coordinate and one flat one -- whereas in ra/dec it is a curved arc
        # smeared across both, which no Gaussian proposal can cover.  These
        # values reproduce the previous ra=1.2, dec=1.17 to 1e-8 rad, so the
        # injected data (and the dynesty reference) are unchanged.
        zenith=1.8963621973,
        azimuth=2.9762214543,
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
    ifo_list = bilby.gw.detector.InterferometerList(["L1", "A1", "CE"])

    for ifo in ifo_list:
        ifo.minimum_frequency = minimum_frequency

    ifo_list.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - duration + 2,
    )

    # The waveform projection needs ra/dec; convert from the detector frame.
    injection_parameters_radec = injection_parameters.copy()
    reference_ifos = bilby.gw.detector.InterferometerList(
        [ifo for name in REFERENCE_FRAME for ifo in ifo_list if ifo.name == name]
    )
    ra, dec = bilby.gw.utils.zenith_azimuth_to_ra_dec(
        injection_parameters["zenith"],
        injection_parameters["azimuth"],
        injection_parameters["geocent_time"],
        reference_ifos,
    )
    injection_parameters_radec["ra"] = ra
    injection_parameters_radec["dec"] = dec

    ifo_list.inject_signal(
        parameters=injection_parameters_radec,
        waveform_generator=waveform_generator,
    )

    # Priors for BNS
    priors = BNSPriorDict(
        dictionary=dict(
            chirp_mass=UniformInComponentsChirpMass(
                name="chirp_mass", minimum=1.399, maximum=1.401, unit=r"$M_{\odot}$", latex_label=r"$\mathcal{M}$"
            ),
            mass_ratio=UniformInComponentsMassRatio(name="mass_ratio", minimum=0.2, maximum=1.0, latex_label=r"$q$"),
            mass_1=Constraint(name="mass_1", minimum=1.0, maximum=2.8),
            mass_2=Constraint(name="mass_2", minimum=1.0, maximum=2.8),
            luminosity_distance=bilby.gw.prior.UniformSourceFrame(
                name="luminosity_distance",
                minimum=50,
                maximum=10000,
                unit="Mpc",
                latex_label=r"$d_L$",
            ),
            theta_jn=Sine(name="theta_jn", latex_label=r"$\theta_{JN}$"),
            # psi is pi-periodic in the antenna response, so the prior must
            # span the full pi.  A pi/2 range aliases the injected value and
            # drags theta_jn with it -- the two enter F+/Fx together.
            psi=Uniform(name="psi", minimum=0, maximum=np.pi, boundary="periodic", latex_label=r"$\psi$"),
            phase=Uniform(name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic", latex_label=r"$\phi$"),
            geocent_time=Uniform(
                minimum=injection_parameters["geocent_time"] - 0.01,
                maximum=injection_parameters["geocent_time"] + 0.01,
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
            chi_1=AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.05), latex_label=r"$\chi_1$"),
            chi_2=AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.05), latex_label=r"$\chi_2$"),
            lambda_1=Uniform(name="lambda_1", minimum=0, maximum=5000, latex_label=r"$\Lambda_1$"),
            lambda_2=Uniform(name="lambda_2", minimum=0, maximum=5000, latex_label=r"$\Lambda_2$"),
        )
    )

    # chi_1 and chi_2 are sampled with the AlignedSpin priors above.  Note the
    # injection sits at chi = 0, which is where the AlignedSpin density
    # diverges logarithmically -- so the MAP is drawn onto that cusp and this
    # exercises the prior-precision clamp in LaplacePosteriorEstimator directly.

    # Marginalisation, identical for both likelihoods.  Keeping them the same
    # is what makes ``std`` a usable reference for ``rb``: any difference
    # between the two is then the binning approximation, not the marginalisation
    # (see ``validate_likelihood``).  Time marginalisation is off because
    # relative binning's time-marginalised path takes a full-resolution FFT per
    # call -- exactly the cost relative binning exists to avoid -- so
    # ``geocent_time`` is sampled instead.  Its prior spans +/-0.01 s about
    # zero, so there is no GPS-epoch precision problem for the SMC flow.
    marg = dict(
        time_marginalization=False,
        phase_marginalization=True,
        distance_marginalization=True,
        jitter_time=False,
    )

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
            **marg,
            reference_frame=REFERENCE_FRAME,
        )

    elif likelihood_type == "rb":
        # Copy rather than share: when the likelihood sets up its bins it
        # writes `frequencies` into the generator's waveform_arguments, which
        # must not leak into the generator used for the injection.  The bin
        # placement itself is controlled by `chi` and `epsilon` on the
        # likelihood -- there is no `frequency_bin_edges` waveform kwarg, and
        # passing one only produced bilby's "unused waveform kwargs" warning.
        rb_waveform_arguments = waveform_arguments.copy()
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
            **marg,
            epsilon=0.25,
            reference_frame=REFERENCE_FRAME,
        )

    else:
        raise ValueError(f"Unknown likelihood_type {likelihood_type!r}; choose 'std' or 'rb'")

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

    # The waveform Fisher gives a better-conditioned proposal than the scalar
    # Hessian and is supported for the standard and relative-binning likelihoods
    # (relative binning is an approximation to the full-resolution likelihood the
    # Fisher is built on). Multi-banding is not supported, so it falls back to the
    # Hessian. When using the waveform Fisher the estimator works directly in
    # parameter space, so use_unit_cube is disabled.
    use_waveform_fisher = likelihood_type in ("std", "rb")
    fisher_method = "waveform" if use_waveform_fisher else "hessian"

    _common_laplace = dict(
        **_common,
        use_injection_for_map=True,
        plot_diagnostic=True,
        clean=True,
        resume=False,
        sampler="laplace",
        target_nsamples=5000,
        use_unit_cube=not use_waveform_fisher,
        fisher_method=fisher_method,
        npool=16,
    )

    return _common, _common_laplace, outdir, run_prefix


def validate_likelihood(variation=0.001):
    """Build both likelihoods on identical data and compare them.

    Agreement means the fast likelihoods are faithful for this event;
    disagreement is the root cause to chase before trusting any posterior built
    on them.  Ported from ``3G_STM/analysis.py``.

    The evaluation point is deliberately perturbed off the injection: relative
    binning is *exact* at its fiducial parameters (the waveform ratio r(f) is
    identically 1 there), so evaluating at the injection would report zero
    binning error no matter how coarse the bins.  ``setup`` gives both
    likelihoods the same marginalisation, so what this measures is the binning
    approximation and nothing else.
    """
    results = {}
    for name in ("std", "rb"):
        # Reseed so every flavour analyses the same noise realisation; setup()
        # draws the strain, so without this each build would see new data.
        bilby.core.utils.random.seed(1234)
        _common, _, _, _ = setup(name)
        likelihood = _common["likelihood"]
        injection_parameters = _common["injection_parameters"]

        eval_parameters = dict(injection_parameters)
        eval_parameters["chirp_mass"] = injection_parameters["chirp_mass"] * (1.0 + variation)
        likelihood.parameters.update(eval_parameters)

        llr = likelihood.log_likelihood_ratio()
        n_eval = timeit.Timer(likelihood.log_likelihood_ratio).autorange()[0]
        eval_time = timeit.timeit(likelihood.log_likelihood_ratio, number=n_eval) / n_eval
        results[name] = (llr, eval_time)

    std_llr = results.get("std", (float("nan"),))[0]
    w = 14
    header = (
        f"  {'Likelihood':<10}  {'log_L_ratio':>{w}}  {'delta from std':>{w}}  "
        f"{'% from std':>{w}}  {'implied SNR':>{w}}  {'eval time (ms)':>{w}}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print()
    print(f"  Likelihood validation at chirp_mass perturbed by {variation:+.2%}")
    print(sep)
    print(header)
    print(sep)
    for name, (llr, eval_time) in results.items():
        delta = llr - std_llr
        pct = 100.0 * delta / std_llr if std_llr else float("nan")
        pct_str = f"{pct:+.4f}" if np.isfinite(pct) else "-"
        snr = np.sqrt(max(0.0, 2.0 * llr))
        print(
            f"  {name:<10}  {llr:>{w}.4f}  {delta:>+{w}.4f}  "
            f"{pct_str:>{w}}  {snr:>{w}.2f}  {eval_time * 1e3:>{w}.4f}"
        )
    print(sep)
    print()
    return results


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def run_laplace(_common_laplace, run_prefix):
    return bilby.run_sampler(
        **_common_laplace,
        # Label by the actual resample method ("inprior"), not the CLI target
        # name, so this method gets the same colour/legend as the equivalent
        # run in the other examples (see bilby_laplace.comparison).
        label=f"{run_prefix}-inprior",
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
    """Run the SMC resampling stage with the shared GW settings.

    ``mode_search_subspace`` is set here rather than in ``GW_SMC_SETTINGS``
    because it names this example's sky coordinates (``zenith``/``azimuth``, as in
    BBH_HLV).  It identifies *which* coordinates a
    sky degeneracy lives in, not a tuned value.
    """
    return bilby.run_sampler(
        **_common_laplace,
        **GW_SMC_SETTINGS,
        label=f"{run_prefix}-smc",
        resample="smc",
        mode_search_subspace=["zenith", "azimuth"],
    )


def run_smc_direct(_common, run_prefix):
    """Aspire's SMC on its own, with no Laplace stage.

    The control for ``run_smc``: same SMC sampler and the same particle count,
    but seeded from prior draws rather than from the Laplace proposal, and via
    ``aspire_bilby``'s own plugin rather than ours.  What it isolates is what the
    Laplace stage buys -- everything downstream of the initial cloud is held
    fixed.

    ``enable_checkpointing=False`` because the plugin otherwise resumes from
    ``{label}_aspire_checkpoint.h5`` on every rerun, silently reporting a
    10-second run and zero likelihood evaluations.
    """
    return bilby.run_sampler(
        **_common,
        **GW_SMC_DIRECT_SETTINGS,
        sampler="aspire",
        label=f"{run_prefix}-smc-direct",
        enable_checkpointing=False,
        npool=16,
    )


def run_dynesty(_common, run_prefix):
    """Reference run, using the settings used for production GW parameter
    estimation.

    ``sample="acceptance-walk"`` with a fixed ``naccept`` draws a set number of
    accepted MCMC steps per point rather than adapting the chain length, which
    makes the cost per point predictable and parallelises far better than
    ``act-walk``.  That matters here: with the aligned spins free, ``act-walk``
    was taking >12000 likelihood calls per accepted point at 0.1% efficiency and
    was hours from converging.
    """
    return bilby.run_sampler(
        **_common,
        sampler="dynesty",
        label=f"{run_prefix}-dynesty",
        nlive=1000,
        sample="acceptance-walk",
        naccept=60,
        maxmcmc=5000,
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
    results, labels = compare_results(
        pattern,
        full_filename,
        sampler_only_labels=True,
        colour_overrides=COLOUR_OVERRIDES,
    )

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
            colours=colours_for_results(results, overrides=COLOUR_OVERRIDES),
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
        choices=["std", "rb"],
        default=None,
        help=(
            "Likelihood: std (standard) or rb (relative binning)."
            " Required when running samplers."
        ),
    )
    parser.add_argument(
        "--sampler",
        nargs="+",
        choices=["laplace", "rejection", "smc", "smc-direct", "dynesty"],
        metavar="SAMPLER",
        help=(
            "One or more samplers to run: laplace, rejection, smc, smc-direct"
            " (SMC straight from the prior, no Laplace stage), dynesty"
        ),
    )
    parser.add_argument(
        "--validate-likelihood",
        action="store_true",
        help=(
            "Build the std and rb likelihoods on identical data and compare"
            " them at a perturbed point; run before trusting rb results"
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Load all existing results, print evidence table, and plot",
    )
    args = parser.parse_args()

    if not args.sampler and not args.compare and not args.validate_likelihood:
        parser.print_help()
    else:
        _outdir = "outdir_bns_example"

        if args.validate_likelihood:
            validate_likelihood()

        if args.sampler:
            if args.likelihood is None:
                parser.error("--likelihood is required when running samplers")
            _common, _common_laplace, _outdir, _run_prefix = setup(args.likelihood)

            _run_fns = {
                "laplace": lambda: run_laplace(_common_laplace, _run_prefix),
                "rejection": lambda: run_rejection(_common_laplace, _run_prefix),
                "smc": lambda: run_smc(_common_laplace, _run_prefix),
                "smc-direct": lambda: run_smc_direct(_common, _run_prefix),
                "dynesty": lambda: run_dynesty(_common, _run_prefix),
            }

            for name in args.sampler:
                _run_fns[name]()

        if args.compare:
            compare(_outdir)
