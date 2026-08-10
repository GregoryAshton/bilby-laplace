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

base_label = "bns"

REFERENCE_FRAME = ["A1", "CE"]

N_STEPS = 100
N_SAMPLES = 5000
N_FINAL_SAMPLES = 10000
TARGET_EFFICIENCY = (0.5, 0.8)
TARGET_EFFICIENCY_RATE = 0.5


def setup(likelihood_type="rb"):
    """Build the detectors, likelihood, priors, and shared sampler kwargs.

    ``likelihood_type`` is "std" (full frequency grid) or "rb" (relative
    binning).
    """
    outdir = "outdir_bns_example"
    run_prefix = likelihood_type

    injection_parameters = dict(
        chirp_mass=1.4,
        mass_ratio=1,
        chi_1=0.00,
        chi_2=0.00,
        luminosity_distance=200.0,
        theta_jn=0.5,
        psi=1.3,
        phase=2.1,
        geocent_time=0.0,
        zenith=1.8963621973,
        azimuth=2.9762214543,
        lambda_1=310.0,
        lambda_2=310.0,
    )

    duration = 128
    sampling_frequency = 1024
    minimum_frequency = 40

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomD_NRTidalv2",
        reference_frequency=100,
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments,
    )

    ifo_list = bilby.gw.detector.InterferometerList(["L1", "A1", "CE"])

    for ifo in ifo_list:
        ifo.minimum_frequency = minimum_frequency

    ifo_list.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - duration + 2,
    )

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


def run_laplace(_common_laplace, run_prefix):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{run_prefix}-inprior",
        resample="inprior",
    )


def run_rejection(_common_laplace, run_prefix):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{run_prefix}-rejection",
        resample="rejection",
        max_iterations=10000000,
        batch_nsamples=10000,
        prior_parameters=["lambda_1", "lambda_2", "psi"],
    )


def run_smc(_common_laplace, run_prefix):
    return bilby.run_sampler(
        **_common_laplace,
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
        label=f"{run_prefix}-smc",
        resample="smc",
        mode_search_subspace=["zenith", "azimuth"],
    )


def run_smc_direct(_common, run_prefix):
    return bilby.run_sampler(
        **_common,
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
        sampler="aspire",
        label=f"{run_prefix}-smcdirect",
        enable_checkpointing=False,
        npool=16,
    )


def run_dynesty(_common, run_prefix):
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
    )

    import matplotlib.pyplot as plt

    intrinsic_params = [
        "chirp_mass",
        "mass_ratio",
        "mass_1",
        "mass_2",
        "chi_1",
        "chi_2",
        "lambda_1",
        "lambda_2",
    ]
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
        plot_parameters = []
        for p in parameters:
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
