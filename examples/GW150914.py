#!/usr/bin/env python

"""
Parameter estimation on GW150914 using the Laplace sampler.

The Laplace approximation runs in minutes rather than hours. It works well
for the more Gaussian parameters (chirp mass, mass ratio, inclination) but
will be less accurate for parameters with non-Gaussian or multi-modal
posteriors (sky location, distance).

Data is fetched from GWOSC via gwpy. See
https://gwpy.github.io/docs/stable/timeseries/remote-access.html
for details on accessing data on the LIGO Data Grid instead.

Usage
-----
    python examples/GW150914.py --sampler laplace rejection smc dynesty
    python examples/GW150914.py --sampler smc
    python examples/GW150914.py --compare
"""

import argparse
import glob
import os

import numpy as np
import bilby
from bilby.core.prior import Constraint, PowerLaw, Sine, Uniform
from bilby.gw.prior import (
    AlignedSpin,
    BBHPriorDict,
    UniformInComponentsChirpMass,
    UniformInComponentsMassRatio,
)
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
outdir = "outdir_GW150914"
base_label = "GW150914"


def setup():
    """Download data, set up priors, and create sampler configuration."""
    # Data
    trigger_time = 1126259462.4
    detectors = ["H1", "L1"]
    maximum_frequency = 512
    minimum_frequency = 20
    roll_off = 0.4
    duration = 4
    post_trigger_duration = 2
    end_time = trigger_time + post_trigger_duration
    start_time = end_time - duration

    psd_duration = 16 * duration
    psd_start_time = start_time - psd_duration
    psd_end_time = start_time

    ifo_list = bilby.gw.detector.InterferometerList([])
    for det in detectors:
        logger.info(f"Downloading analysis data for ifo {det}")
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        data = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        ifo.strain_data.set_from_gwpy_timeseries(data)

        logger.info(f"Downloading PSD data for ifo {det}")
        psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, cache=True)
        psd_alpha = 2 * roll_off / duration
        psd = psd_data.psd(
            fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
        )
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value
        )
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        ifo_list.append(ifo)

    # Priors
    priors = BBHPriorDict(
        dictionary=dict(
            chirp_mass=31.2,
            mass_ratio=1,
            mass_1=Constraint(name="mass_1", minimum=10, maximum=80),
            mass_2=Constraint(name="mass_2", minimum=10, maximum=80),
            chi_1=0,
            chi_2=0,
            luminosity_distance=PowerLaw(
                alpha=2,
                name="luminosity_distance",
                minimum=50,
                maximum=2000,
                unit="Mpc",
                latex_label="$d_L$",
            ),
            zenith=Sine(name="zenith"),
            azimuth=Uniform(name="azimuth", minimum=0, maximum=2 * np.pi, boundary="periodic"),
            theta_jn=1.4,
            psi=0.5,
            phase=Uniform(name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"),
            geocent_time=Uniform(
                minimum=trigger_time - 0.1,
                maximum=trigger_time + 0.1,
                name="geocent_time",
                latex_label=r"$t_{\rm geo}$",
                unit="$s$",
            ),
        )
    )

    # Waveform generator and likelihood
    waveform_generator = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2",
            "reference_frequency": 50,
        },
    )

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
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
        save="hdf5",
    )

    _common_laplace = dict(
        **_common,
        use_injection_for_map=False,
        plot_diagnostic=True,
        clean=True,
        cov_scaling=1,
        sampler="laplace",
    )

    _smc_kwargs = dict(
        sampler="minipcn_smc",
        n_initial_samples=5000,
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

    return _common, _common_laplace, _smc_kwargs


# Samplers
def run_laplace(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_laplace",
        resample="None",
    )


def run_rejection(_common_laplace):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_rejection",
        resample="rejection",
    )


def run_smc(_common_laplace, _smc_kwargs):
    return bilby.run_sampler(
        **_common_laplace,
        label=f"{base_label}_smc",
        resample="smc",
        n_modes=2,
        smc_kwargs=_smc_kwargs,
    )


def run_dynesty(_common):
    return bilby.run_sampler(
        **_common,
        sampler="dynesty",
        label=f"{base_label}_dynesty",
        nlive=250,
        check_point_delta_t=1800,
        check_point_plot=True,
        npool=1,
        clean=False,
        resume=True,
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
    filename = os.path.join(outdir, f"{base_label}_comparison_corner.png")
    fig = bilby.core.result.plot_multiple(
        results,
        labels=labels,
        filename=filename,
        titles=False,
        save=False,
    )
    fig.savefig(filename, dpi=400)
    plt.close(fig)
    logger.info(
        f"Comparison corner plot saved to "
        f"{outdir}/{base_label}_comparison_corner.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameter estimation on GW150914"
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
        # Only download data and set up if running samplers
        if args.sampler:
            _common, _common_laplace, _smc_kwargs = setup()

            _run_fns = {
                "laplace": lambda: run_laplace(_common_laplace),
                "rejection": lambda: run_rejection(_common_laplace),
                "smc": lambda: run_smc(_common_laplace, _smc_kwargs),
                "dynesty": lambda: run_dynesty(_common),
            }

            for name in args.sampler:
                _run_fns[name]()

        # Compare only needs to read result files
        if args.compare:
            compare()
