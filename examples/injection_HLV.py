#!/usr/bin/env python

"""
Parameter estimation on a simulated BBH signal injected into Gaussian noise.

Uses a three-detector HLV (Hanford-Livingston-Virgo) network with bilby's
built-in injection infrastructure.  No real data download is needed -- the
interferometers are initialised with their design power spectral densities
and Gaussian noise is generated internally.

Usage
-----
    python examples/injection_HLV.py --rejection
    python examples/injection_HLV.py --smc
    python examples/injection_HLV.py --importance
    python examples/injection_HLV.py --dynesty
    python examples/injection_HLV.py --compare
"""

import argparse
import glob
import os

import numpy as np
import bilby
from bilby.core.prior import Constraint, Sine, Uniform
from bilby.gw.prior import (
    AlignedSpin,
    BBHPriorDict,
    UniformInComponentsChirpMass,
    UniformInComponentsMassRatio,
)

logger = bilby.core.utils.logger
outdir = "outdir_injection_HLV"
base_label = "injection_HLV"

# ---------------------------------------------------------------------------
# Injection parameters
# ---------------------------------------------------------------------------
injection_parameters = dict(
    chirp_mass=28.0,
    mass_ratio=0.8,
    chi_1=0.05,
    chi_2=-0.02,
    luminosity_distance=800.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    zenith=0.0,
    azimuth=0.0,
)

# ---------------------------------------------------------------------------
# Detector setup
# ---------------------------------------------------------------------------
duration = 4
sampling_frequency = 2048
minimum_frequency = 20
maximum_frequency = 1024

waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50,
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
ifo_list.inject_signal(
    parameters=injection_parameters,
    waveform_generator=waveform_generator,
)

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
priors = BBHPriorDict(
    dictionary=dict(
        chirp_mass=UniformInComponentsChirpMass(
            name="chirp_mass", minimum=25, maximum=35, unit=r"$M_{\odot}$"
        ),
        mass_ratio=UniformInComponentsMassRatio(
            name="mass_ratio", minimum=0.125, maximum=1
        ),
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
        psi=Uniform(
            name="psi", minimum=0, maximum=np.pi, boundary="periodic"
        ),
        phase=Uniform(
            name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
        ),
        geocent_time=Uniform(
            minimum=injection_parameters["geocent_time"] - 0.1,
            maximum=injection_parameters["geocent_time"] + 0.1,
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

# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Shared sampler kwargs
# ---------------------------------------------------------------------------
_common_laplace = dict(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    injection_parameters=injection_parameters,
    use_injection_for_maxL=True,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
    plot_diagnostic=True,
    clean=True,
    sampler="laplace",
    target_nsamples=1000,
    cov_scaling=1,
    extension="hdf5",
)

_smc_kwargs = dict(
    backend="minipcn",
    n_samples=5000,
    n_final_samples=1000,
    target_efficiency=[0.5, 0.8],
    adaptive=True,
    sampler_kwargs=dict(
        n_steps=50,
        target_acceptance_rate=0.234,
        step_fn="tpcn",
        verbose=True,
    ),
)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def run_laplace():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_laplace",
        resample="None",
    )


def run_rejection():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_rejection",
        resample="rejection",
    )


def run_importance():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_importance",
        resample="importance",
    )


def run_smc():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_smc",
        resample="smc",
        smc_kwargs=_smc_kwargs,
    )


def run_dynesty():
    return bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        outdir=outdir,
        label=f"{base_label}_dynesty",
        injection_parameters=injection_parameters,
        nlive=500,
        check_point_delta_t=1800,
        check_point_plot=True,
        npool=1,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
        clean=False,
        resume=True,
        extension="hdf5",
    )


def compare():
    """Load all result files in outdir, make a comparison corner plot,
    and print an evidence comparison table."""
    pattern = os.path.join(outdir, f"{base_label}_*_result.*")
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        logger.warning(f"No result files found matching {pattern}")
        return

    results = []
    labels = []
    for f in result_files:
        try:
            r = bilby.core.result.read_in_result(filename=f)
            results.append(r)
            label = r.label.replace(f"{base_label}_", "").capitalize()
            secs = r.sampling_time.total_seconds()
            if secs >= 3600:
                label += f" ({secs / 3600:.1f} hr)"
            elif secs >= 60:
                label += f" ({secs / 60:.1f} min)"
            else:
                label += f" ({secs:.0f} s)"
            labels.append(label)
            logger.info(f"Loaded {f} ({label})")
        except Exception as exc:
            logger.warning(f"Could not load {f}: {exc}")

    # Evidence comparison table
    print("\n" + "=" * 60)
    print("Evidence comparison")
    print("=" * 60)
    print(f"{'Method':<25} {'log Z':>10} {'± σ':>10} {'time':>10}")
    print("-" * 60)
    for r, lab in zip(results, labels):
        log_z = getattr(r, "log_evidence", np.nan)
        log_z_err = getattr(r, "log_evidence_err", np.nan)
        secs = r.sampling_time.total_seconds()
        if log_z is None:
            log_z = np.nan
        if log_z_err is None:
            log_z_err = np.nan
        name = r.label.replace(f"{base_label}_", "")
        print(f"{name:<25} {log_z:>10.2f} {log_z_err:>10.2f} {secs:>9.1f}s")
    print("=" * 60 + "\n")

    if len(results) < 2:
        logger.warning(
            f"Need at least 2 results for a comparison plot, "
            f"found {len(results)}"
        )
        if len(results) == 1:
            results[0].plot_corner()
        return

    bilby.core.result.plot_multiple(
        results,
        labels=labels,
        filename=os.path.join(outdir, f"{base_label}_comparison_corner.png"),
        titles=False,
    )
    logger.info(
        f"Comparison corner plot saved to "
        f"{outdir}/{base_label}_comparison_corner.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BBH injection in Gaussian noise (HLV network)"
    )
    parser.add_argument(
        "--laplace",
        action="store_true",
        help="Run Laplace approximation (no resampling)",
    )
    parser.add_argument(
        "--rejection",
        action="store_true",
        help="Run Laplace with rejection resampling",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="Run Laplace with importance resampling",
    )
    parser.add_argument(
        "--smc",
        action="store_true",
        help="Run Laplace with SMC resampling",
    )
    parser.add_argument(
        "--dynesty",
        action="store_true",
        help="Run dynesty nested sampling",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Load all existing results, print evidence table, and plot",
    )
    args = parser.parse_args()

    if args.laplace:
        run_laplace().plot_corner()
    if args.rejection:
        run_rejection().plot_corner()
    if args.importance:
        run_importance().plot_corner()
    if args.smc:
        run_smc().plot_corner()
    if args.dynesty:
        run_dynesty().plot_corner()
    if args.compare:
        compare()

    if not any([args.laplace, args.rejection, args.importance, args.smc,
                args.dynesty, args.compare]):
        parser.print_help()
