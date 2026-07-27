"""Fisher matrix from gravitational-wave waveform derivatives.

For a bilby ``GravitationalWaveTransient``-like likelihood the Fisher matrix is

    F_ij = sum_detectors Re (d_i h | d_j h)

where ``(a|b)`` is the noise-weighted inner product and ``d_i h`` is the
derivative of the projected detector strain with respect to parameter ``i``.

Unlike the numerical Hessian of the scalar log-likelihood, this is the genuine
Fisher matrix: it is positive semi-definite by construction, it needs only
*first* derivatives of the waveform (far better behaved under finite
differencing than a scalar second derivative), and it drops the noisy,
realization-dependent residual-times-curvature term that makes the observed
information indefinite.

Relative-binning likelihoods are supported: they are an approximation to the
full-resolution likelihood, so the full-resolution Fisher is exactly the correct
proposal for them. Their waveform generator returns strain only at the coarse
frequency bin edges during sampling, so ``_full_resolution_waveforms`` flips it
to full-grid evaluation for the derivative computation. ROQ and multi-banded
likelihoods remain unsupported (their inner product genuinely differs from the
full-resolution one used here).
"""

from contextlib import contextmanager

import numpy as np
from bilby.core.utils import logger

DEFAULT_EPS = 1e-6
DEFAULT_EPS_MASS = 1e-8
DEFAULT_EPS_TIME = 1e-5

# Mass parameters: waveforms are extremely sensitive to these, so a much finer
# finite-difference step is used (following GWFish).
MASS_PARAMETERS = {
    "chirp_mass",
    "chirp_mass_source",
    "mass_1",
    "mass_2",
    "mass_1_source",
    "mass_2_source",
    "total_mass",
}

# Time parameters carry a GPS epoch (~1e9 s), so a *relative* step would be
# hundreds of seconds -- enough to completely alias the exp(-2*pi*i*f*dt)
# detector response and make the time derivative meaningless. They need an
# absolute step in seconds instead.
TIME_PARAMETERS = {"geocent_time"}


def _is_time_parameter(name):
    """True for GPS-epoch time parameters (``geocent_time`` and per-detector
    ``{ifo}_time`` reference times), which require an absolute step."""
    return name in TIME_PARAMETERS or name.endswith("_time")


# Reduced-order likelihoods evaluate a different (approximate) inner product
# than the full-resolution one used here, so the waveform Fisher would be
# inconsistent with the likelihood the user is actually sampling.
#
# Relative binning is deliberately *not* in this list: it is an approximation to
# the full-resolution likelihood, so the full-resolution Fisher is exactly the
# correct (and better-conditioned) proposal for it. Its waveform generator can
# already produce full-grid strain -- ``_full_resolution_waveforms`` below flips
# it into that mode for the duration of the derivative computation.
_REDUCED_ORDER_MARKERS = ("ROQ", "MBGravitationalWaveTransient")


@contextmanager
def _full_resolution_waveforms(likelihood):
    """Temporarily switch a relative-binning likelihood's waveform generator to
    full-grid ("fiducial") evaluation.

    A ``RelativeBinningGravitationalWaveTransient`` samples with its waveform
    generator returning strain only at the ~100 frequency bin edges, which is
    incompatible with the full-resolution inner product (PSD / frequency mask)
    used by :func:`waveform_fisher_matrix`. bilby exposes ``_set_fiducial`` /
    ``_unset_fiducial`` to flip the generator between full-grid (``fiducial=1``)
    and bin-edge (``fiducial=0``) evaluation, invalidating the waveform cache in
    both directions. We reuse those so the Fisher sees full-grid waveforms and
    the likelihood is restored to its sampling state (``fiducial=0``) on exit,
    even if an exception is raised. A no-op for non-RB likelihoods.
    """
    set_fiducial = getattr(likelihood, "_set_fiducial", None)
    unset_fiducial = getattr(likelihood, "_unset_fiducial", None)
    if not callable(set_fiducial) or not callable(unset_fiducial):
        yield
        return
    set_fiducial()
    try:
        yield
    finally:
        unset_fiducial()


def is_gw_waveform_likelihood(likelihood):
    """True if the likelihood exposes interferometers and a waveform generator."""
    return hasattr(likelihood, "interferometers") and hasattr(likelihood, "waveform_generator")


def validate_waveform_likelihood(likelihood):
    """Raise ``ValueError`` if the likelihood is unsupported by this path."""
    if not is_gw_waveform_likelihood(likelihood):
        raise ValueError(
            "fisher_method='waveform' requires a GravitationalWaveTransient-like "
            "likelihood with `interferometers` and `waveform_generator` attributes; "
            f"got {type(likelihood).__name__}. Use fisher_method='hessian'."
        )

    # Phase / time / distance marginalisation are supported: those parameters
    # are reinstated in the Fisher and then marginalised out via the Schur
    # complement (see LaplacePosteriorEstimator._calculate_precision_waveform).
    # Calibration marginalisation is not: it integrates a *discrete* index over
    # calibration curves, which is not a continuous Fisher direction.
    if getattr(likelihood, "calibration_marginalization", False):
        raise ValueError(
            "fisher_method='waveform' does not support calibration_marginalization: "
            "it marginalises a discrete calibration index, not a continuous "
            "parameter. Use fisher_method='hessian'."
        )

    name = type(likelihood).__name__
    if any(marker in name for marker in _REDUCED_ORDER_MARKERS):
        raise ValueError(
            f"fisher_method='waveform' does not support reduced-order likelihood "
            f"{name!r}, whose inner product differs from the full-resolution one "
            "used here. Use fisher_method='hessian'."
        )


def _step(name, value, eps, eps_mass, eps_time):
    """Central-difference step.

    Masses use a fine relative step (waveforms are extremely sensitive to them);
    time parameters use an absolute step in seconds (a relative step off a GPS
    epoch would be hundreds of seconds and alias the waveform); everything else
    uses a relative step with an absolute floor.
    """
    if name in MASS_PARAMETERS:
        return eps_mass * max(abs(value), 1.0)
    if _is_time_parameter(name):
        return eps_time
    return max(eps, eps * abs(value))


def waveform_fisher_matrix(
    likelihood,
    parameter_names,
    base_parameters,
    eps=DEFAULT_EPS,
    eps_mass=DEFAULT_EPS_MASS,
    eps_time=DEFAULT_EPS_TIME,
):
    """Fisher matrix ``F_ij = sum_det Re (d_i h | d_j h)`` via central differences.

    Parameters
    ----------
    likelihood : GravitationalWaveTransient-like
        Must expose ``interferometers`` and ``waveform_generator``.
    parameter_names : list of str
        Parameters to differentiate, defining the basis of the matrix.
    base_parameters : dict
        Full parameter dictionary at the evaluation point (the MAP), including
        any fixed/derived parameters the waveform and projection require.
    eps, eps_mass : float
        Relative finite-difference steps (mass parameters use ``eps_mass``).
    eps_time : float
        Absolute finite-difference step, in seconds, for time parameters
        (``geocent_time`` and per-detector ``{ifo}_time``). A relative step is
        unusable here because the value is a GPS epoch (~1e9 s).

    Returns
    -------
    np.ndarray
        The ``(N, N)`` Fisher matrix in the order of ``parameter_names``.
    """
    from bilby.gw.utils import noise_weighted_inner_product

    wg = likelihood.waveform_generator
    ifos = likelihood.interferometers
    names = list(parameter_names)
    n = len(names)

    # ``get_detector_response`` needs ra/dec/geocent_time. When the likelihood
    # uses a detector-based reference frame it samples zenith/azimuth (and,
    # optionally, a detector time) instead, converting to ra/dec internally.
    # Reproduce that conversion here so derivatives w.r.t. zenith/azimuth (and the
    # time-dependence of the sky location) flow through it.  No-op for the "sky"
    # frame, where ra/dec are already the sampled parameters.
    reference_frame = getattr(likelihood, "reference_frame", "sky")
    needs_sky_conversion = reference_frame not in (None, "sky") and hasattr(likelihood, "get_sky_frame_parameters")

    def response_params(params):
        if not needs_sky_conversion:
            return params
        converted = dict(params)
        converted.update(likelihood.get_sky_frame_parameters(params))
        return converted

    # First derivatives of the projected detector strain, per detector. For a
    # relative-binning likelihood, evaluate on the full frequency grid (see
    # _full_resolution_waveforms) so the strain is compatible with the
    # full-resolution inner product below; a no-op for other likelihoods.
    derivs = {ifo.name: [] for ifo in ifos}
    with _full_resolution_waveforms(likelihood):
        for name in names:
            value = float(base_parameters[name])
            dp = _step(name, value, eps, eps_mass, eps_time)
            plus = dict(base_parameters)
            plus[name] = value + 0.5 * dp
            minus = dict(base_parameters)
            minus[name] = value - 0.5 * dp
            pol_plus = wg.frequency_domain_strain(plus)
            pol_minus = wg.frequency_domain_strain(minus)
            plus_response = response_params(plus)
            minus_response = response_params(minus)
            for ifo in ifos:
                h_plus = ifo.get_detector_response(pol_plus, plus_response)
                h_minus = ifo.get_detector_response(pol_minus, minus_response)
                derivs[ifo.name].append((h_plus - h_minus) / dp)

    fisher = np.zeros((n, n))
    for ifo in ifos:
        mask = ifo.frequency_mask
        psd = ifo.power_spectral_density_array[mask]
        duration = ifo.duration
        d = [deriv[mask] for deriv in derivs[ifo.name]]
        for i in range(n):
            for j in range(i, n):
                value = float(np.real(noise_weighted_inner_product(d[i], d[j], psd, duration)))
                fisher[i, j] += value
                fisher[j, i] = fisher[i, j]

    logger.info(f"Computed waveform Fisher matrix over {n} parameter(s) and {len(ifos)} detector(s)")
    return fisher
