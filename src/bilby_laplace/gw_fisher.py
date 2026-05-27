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
"""

import numpy as np
from bilby.core.utils import logger

DEFAULT_EPS = 1e-6
DEFAULT_EPS_MASS = 1e-8

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

# Reduced-order likelihoods evaluate a different (approximate) inner product
# than the full-resolution one used here, so the waveform Fisher would be
# inconsistent with the likelihood the user is actually sampling.
_REDUCED_ORDER_MARKERS = ("ROQ", "RelativeBinning", "MBGravitationalWaveTransient")


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

    for flag in (
        "phase_marginalization",
        "time_marginalization",
        "distance_marginalization",
        "calibration_marginalization",
    ):
        if getattr(likelihood, flag, False):
            raise ValueError(
                f"fisher_method='waveform' does not support {flag}: the waveform "
                "Fisher cannot account for marginalised parameters. Disable "
                "marginalisation or use fisher_method='hessian'."
            )

    name = type(likelihood).__name__
    if any(marker in name for marker in _REDUCED_ORDER_MARKERS):
        raise ValueError(
            f"fisher_method='waveform' does not support reduced-order likelihood "
            f"{name!r}, whose inner product differs from the full-resolution one "
            "used here. Use fisher_method='hessian'."
        )


def _step(name, value, eps, eps_mass):
    """Central-difference step: relative with an absolute floor; finer for masses."""
    if name in MASS_PARAMETERS:
        return eps_mass * max(abs(value), 1.0)
    return max(eps, eps * abs(value))


def waveform_fisher_matrix(likelihood, parameter_names, base_parameters, eps=DEFAULT_EPS, eps_mass=DEFAULT_EPS_MASS):
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

    # First derivatives of the projected detector strain, per detector.
    derivs = {ifo.name: [] for ifo in ifos}
    for name in names:
        value = float(base_parameters[name])
        dp = _step(name, value, eps, eps_mass)
        plus = dict(base_parameters)
        plus[name] = value + 0.5 * dp
        minus = dict(base_parameters)
        minus[name] = value - 0.5 * dp
        pol_plus = wg.frequency_domain_strain(plus)
        pol_minus = wg.frequency_domain_strain(minus)
        for ifo in ifos:
            h_plus = ifo.get_detector_response(pol_plus, plus)
            h_minus = ifo.get_detector_response(pol_minus, minus)
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
