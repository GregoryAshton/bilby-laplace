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
bilby.core.utils.random.seed(1234)
outdir = "outdir_hlv_example"
base_label = "hlv"

# ---------------------------------------------------------------------------
# Injection parameters
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Detector setup
# ---------------------------------------------------------------------------
duration = 4
sampling_frequency = 2048
minimum_frequency = 20
maximum_frequency = 512

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
    injection_parameters['zenith'],
    injection_parameters['azimuth'],
    injection_parameters['geocent_time'],
    ifo_list,
)

injection_parameters_radec['ra'] = ra
injection_parameters_radec['dec'] = dec

ifo_list.inject_signal(
    parameters=injection_parameters_radec,
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
            name="psi", minimum=0, maximum=np.pi / 2, boundary="periodic"
        ),
        phase=Uniform(
            name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
        ),
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
    use_injection_for_map=True,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
    plot_diagnostic=True,
    clean=True,
    sampler="laplace",
    target_nsamples=1000,
    save="hdf5",
    use_unit_cube=True,
)

_smc_kwargs = dict(
    sampler="minipcn_smc",
    n_initial_samples=1000,
    n_final_samples=5000,
    target_efficiency=[0.5, 0.8],
    adaptive=True,
    sampler_kwargs=dict(
        n_steps=5,
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
        cov_scaling=1,
    )


def run_rejection():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_rejection",
        resample="rejection",
        cov_scaling=2,
    )


def run_smc():
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_smc",
        resample="smc",
        smc_kwargs=_smc_kwargs,
        cov_scaling=2,
    )


def run_dynesty():
    return bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        outdir=outdir,
        label=f"{base_label}_dynesty",
        injection_parameters=injection_parameters,
        nlive=250,
        check_point_delta_t=1800,
        check_point_plot=True,
        npool=1,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
        clean=False,
        resume=True,
        save="hdf5",
    )


def compare():
    """Load all result files in outdir, make a comparison corner plot,
    and print a comparison table."""
    pattern = os.path.join(outdir, f"{base_label}_*_result.*")
    result_files = sorted([f for f in glob.glob(pattern) if not f.endswith('.old')])
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

    # Comparison table
    W = 75
    print("\n" + "=" * W)
    print("Comparison")
    print("=" * W)
    print(f"{'Method':<20} {'log Z':>10} {'± σ':>8} {'n_like':>8} {'effic.':>8} {'time':>10}")
    print("-" * W)
    for r, lab in zip(results, labels):
        log_z = getattr(r, "log_evidence", np.nan) or np.nan
        log_z_err = getattr(r, "log_evidence_err", np.nan) or np.nan
        secs = r.sampling_time.total_seconds()
        run_stats = r.meta_data.get("run_statistics", {})
        n_like = run_stats.get("nlikelihood", np.nan)
        eff = run_stats.get("efficiency", np.nan)
        name = r.label.replace(f"{base_label}_", "")
        n_like_str = f"{int(n_like):>8}" if np.isfinite(n_like) else f"{'—':>8}"
        eff_str = f"{eff:>7.1f}%" if np.isfinite(eff) else f"{'—':>8}"
        print(f"{name:<20} {log_z:>10.2f} {log_z_err:>8.2f} {n_like_str} {eff_str} {secs:>9.1f}s")
    print("=" * W + "\n")

    if len(results) < 2:
        logger.warning(
            f"Need at least 2 results for a comparison plot, "
            f"found {len(results)}"
        )
        return

    import matplotlib.pyplot as plt
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{base_label}_comparison.png")
    fig = bilby.core.result.plot_multiple(
        results,
        labels=labels,
        filename=filename,
        titles=False,
        save=False,
    )

    # Overlay injection truth values
    inj = getattr(results[0], "injection_parameters", None)
    if inj:
        params = results[0].search_parameter_keys
        truths = [inj.get(p) for p in params]
        ndim = len(params)
        axes_grid = np.array(fig.get_axes()).reshape(ndim, ndim)
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
    logger.info(
        f"Comparison corner plot saved to {filename}"
    )


_run_fns = dict(
    laplace=run_laplace,
    rejection=run_rejection,

    smc=run_smc,
    dynesty=run_dynesty,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BBH injection in Gaussian noise (HLV network)"
    )
    parser.add_argument(
        "--sampler",
        nargs="+",
        choices=list(_run_fns),
        metavar="SAMPLER",
        help=f"One or more samplers to run: {', '.join(_run_fns)}",
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
        for name in (args.sampler or []):
            _run_fns[name]()
        if args.compare:
            compare()
