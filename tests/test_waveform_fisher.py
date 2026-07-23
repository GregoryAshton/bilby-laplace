"""Tests for the waveform-Fisher path, focusing on marginalised likelihoods.

When the likelihood analytically marginalises over phase/time/distance, the
Fisher is built over the augmented parameter set and the marginalised block is
removed via its Schur complement. The key correctness property is that this
*marginalises* over those parameters (the sampled-parameter sub-block of the
full covariance) rather than *conditioning* on them (fixing them). These tests
use a stub likelihood and a monkeypatched Fisher so they are fast, deterministic,
and need no GW waveform stack.
"""

import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform

import bilby_laplace.gw_fisher as gw_fisher
from bilby_laplace.laplace import LaplacePosteriorEstimator


class _FakeGWLikelihood:
    """Minimal object satisfying the waveform-Fisher path's requirements."""

    def __init__(self, marginalized, marg_priors, calibration=False):
        self.interferometers = []
        self.waveform_generator = None
        self.calibration_marginalization = calibration
        self.marginalized_parameters = list(marginalized)
        self.priors = dict(marg_priors)
        self.parameters = {}

    def log_likelihood(self, parameters=None):
        return 0.0


def _sampled_priors():
    return PriorDict(dict(x=Uniform(-5, 5, "x"), y=Uniform(-5, 5, "y")))


def test_step_uses_absolute_step_for_time_parameters():
    """Time parameters carry a GPS epoch (~1e9 s); a relative step would be
    hundreds of seconds and alias the waveform. They must use the absolute
    eps_time step regardless of the (huge) value."""
    gps = 1.126259642e9
    step = gw_fisher._step("geocent_time", gps, eps=1e-6, eps_mass=1e-8, eps_time=1e-5)
    assert step == 1e-5
    # Per-detector reference times ({ifo}_time) get the same treatment.
    assert gw_fisher._step("H1_time", gps, eps=1e-6, eps_mass=1e-8, eps_time=1e-5) == 1e-5


def test_step_relative_and_mass_steps_unchanged():
    """Non-time parameters keep the relative step; masses use the fine step."""
    assert gw_fisher._step("luminosity_distance", 1000.0, eps=1e-6, eps_mass=1e-8, eps_time=1e-5) == pytest.approx(1e-3)
    assert gw_fisher._step("chirp_mass", 30.0, eps=1e-6, eps_mass=1e-8, eps_time=1e-5) == pytest.approx(3e-7)
    # Relative step has an absolute floor of eps for small/zero values.
    assert gw_fisher._step("theta_jn", 0.0, eps=1e-6, eps_mass=1e-8, eps_time=1e-5) == pytest.approx(1e-6)


def test_waveform_allows_phase_time_distance_marginalization():
    """Marginalising phase/time/distance is now supported (no error)."""
    like = _FakeGWLikelihood(
        ["geocent_time", "phase", "luminosity_distance"],
        {
            "geocent_time": Uniform(-0.1, 0.1, "geocent_time"),
            "phase": Uniform(0, 2 * np.pi, "phase"),
            "luminosity_distance": Uniform(100, 2000, "luminosity_distance"),
        },
    )
    est = LaplacePosteriorEstimator(like, _sampled_priors(), fisher_method="waveform")
    assert est._supported_marginalized_names() == [
        "geocent_time",
        "phase",
        "luminosity_distance",
    ]


def test_waveform_refuses_calibration_marginalization():
    like = _FakeGWLikelihood(
        ["recalib_index"],
        {},
        calibration=True,
    )
    with pytest.raises(ValueError, match="calibration"):
        LaplacePosteriorEstimator(like, _sampled_priors(), fisher_method="waveform")


def test_marginalization_is_schur_complement_not_conditioning(monkeypatch):
    """The reduced precision must be the Schur complement of the marginalised
    block -- equivalently, its inverse is the sampled sub-block of the *full*
    covariance (marginalising), not the inverse of the sampled sub-block of the
    precision (conditioning)."""
    # Full Fisher over [x, y, luminosity_distance]; correlations couple the
    # sampled parameters to the marginalised one so marginal != conditional.
    fisher_full = np.array(
        [
            [4.0, 1.0, 1.5],
            [1.0, 3.0, 0.8],
            [1.5, 0.8, 2.0],
        ]
    )
    monkeypatch.setattr(gw_fisher, "waveform_fisher_matrix", lambda *a, **k: fisher_full.copy())

    like = _FakeGWLikelihood(
        ["luminosity_distance"],
        {"luminosity_distance": Uniform(100, 2000, "luminosity_distance")},  # flat -> zero prior precision
    )
    est = LaplacePosteriorEstimator(
        like,
        _sampled_priors(),
        fisher_method="waveform",
        marginalized_reference={"luminosity_distance": 800.0},
    )
    reduced = est.calculate_posterior_precision({"x": 0.0, "y": 0.0})

    # Analytic Schur complement of the distance block.
    f_rr, f_rm, f_mm, f_mr = (
        fisher_full[:2, :2],
        fisher_full[:2, 2:],
        fisher_full[2:, 2:],
        fisher_full[2:, :2],
    )
    schur = f_rr - f_rm @ np.linalg.inv(f_mm) @ f_mr
    np.testing.assert_allclose(reduced, schur, atol=1e-10)

    # The marginal property: inverse of the reduced precision equals the
    # sampled sub-block of the full covariance...
    np.testing.assert_allclose(np.linalg.inv(reduced), np.linalg.inv(fisher_full)[:2, :2], atol=1e-10)
    # ...and is strictly wider than the conditioning result (fixing distance).
    assert np.linalg.inv(reduced)[0, 0] > np.linalg.inv(fisher_full[:2, :2])[0, 0]


def test_waveform_precision_bounded_by_prior(monkeypatch):
    """An unconstrained direction (flat prior, ~zero Fisher) must fall back to
    the prior variance rather than inverting to a runaway covariance -- the psi
    case under phase marginalisation."""
    # x well-constrained, y almost unconstrained, luminosity_distance marginalised.
    fisher_full = np.diag([100.0, 1e-8, 50.0])
    monkeypatch.setattr(gw_fisher, "waveform_fisher_matrix", lambda *a, **k: fisher_full.copy())

    like = _FakeGWLikelihood(
        ["luminosity_distance"],
        {"luminosity_distance": Uniform(100, 2000, "luminosity_distance")},
    )
    est = LaplacePosteriorEstimator(
        like,
        _sampled_priors(),  # x, y ~ Uniform(-5, 5): prior variance = 100 / 12
        fisher_method="waveform",
        marginalized_reference={"luminosity_distance": 800.0},
    )
    cov = est.calculate_posterior_covariance({"x": 0.0, "y": 0.0})
    prior_var = (10.0**2) / 12.0

    assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert np.all(np.diag(cov) <= prior_var * (1 + 1e-6))
    # y saturates the prior bound; x stays tight.
    assert np.diag(cov)[1] == pytest.approx(prior_var, rel=1e-6)
    assert np.diag(cov)[0] < 0.1


class _RecordingIfo:
    name = "H1"
    frequency_mask = np.array([True, True, True])
    power_spectral_density_array = np.ones(3)
    duration = 4.0

    def __init__(self):
        self.seen = []

    def get_detector_response(self, waveform_polarizations, parameters):
        self.seen.append(dict(parameters))
        # Strain depends on ra so a zenith perturbation (which maps to ra) has
        # a non-zero derivative once the frame conversion is applied.
        return np.array(
            [parameters["ra"], 2 * parameters["ra"], 3 * parameters["ra"]],
            dtype=complex,
        )


class _FakeWaveformGenerator:
    def frequency_domain_strain(self, parameters):
        return {"plus": np.ones(3, dtype=complex)}


def test_reference_frame_conversion_applied_before_response():
    """For a detector-based frame the zenith/azimuth are converted to ra/dec
    (via the likelihood's own conversion) before the detector response, so the
    derivatives flow through the conversion instead of vanishing."""
    ifo = _RecordingIfo()

    class _Like:
        interferometers = [ifo]
        waveform_generator = _FakeWaveformGenerator()
        reference_frame = "H1L1"

        def get_sky_frame_parameters(self, parameters):
            return {
                "ra": 2.0 * parameters["zenith"],
                "dec": 0.5 * parameters["azimuth"],
                "geocent_time": parameters["geocent_time"],
            }

    base = {"zenith": 0.5, "azimuth": 1.0, "geocent_time": 0.0}
    fisher = gw_fisher.waveform_fisher_matrix(_Like(), ["zenith", "azimuth"], base)

    # Every response call received converted ra/dec (not the raw zenith/azimuth).
    assert ifo.seen and all("ra" in p and "dec" in p for p in ifo.seen)
    # The conversion is parameter-dependent: perturbing zenith changed ra.
    assert len({round(p["ra"], 9) for p in ifo.seen}) > 1
    # zenith is therefore constrained (non-zero Fisher), not a null direction.
    assert fisher[0, 0] > 0


def test_sky_frame_skips_conversion():
    """For the plain sky frame ra/dec are used directly; no conversion is done."""
    ifo = _RecordingIfo()

    class _Like:
        interferometers = [ifo]
        waveform_generator = _FakeWaveformGenerator()
        reference_frame = "sky"

        def get_sky_frame_parameters(self, parameters):
            raise AssertionError("conversion must not be attempted for the sky frame")

    base = {"ra": 1.1, "dec": -0.3, "geocent_time": 0.0}
    fisher = gw_fisher.waveform_fisher_matrix(_Like(), ["ra"], base)
    assert fisher[0, 0] > 0


def test_marginalized_reference_used_over_reconstruction(monkeypatch):
    """A finite reference value is used directly; reconstruction is not called."""
    monkeypatch.setattr(gw_fisher, "waveform_fisher_matrix", lambda *a, **k: np.eye(3))

    def _boom(*a, **k):
        raise AssertionError("reconstruction should not be called when a reference is given")

    like = _FakeGWLikelihood(
        ["luminosity_distance"],
        {"luminosity_distance": Uniform(100, 2000, "luminosity_distance")},
    )
    like.generate_posterior_sample_from_marginalized_likelihood = _boom
    est = LaplacePosteriorEstimator(
        like,
        _sampled_priors(),
        fisher_method="waveform",
        marginalized_reference={"luminosity_distance": 800.0},
    )
    values = est._resolve_marginalized_values({"x": 0.0, "y": 0.0}, ["luminosity_distance"])
    assert values == {"luminosity_distance": 800.0}


def test_reconstruction_fallback_when_no_reference(monkeypatch):
    """With no reference value, the parameter is reconstructed from the likelihood."""
    monkeypatch.setattr(gw_fisher, "waveform_fisher_matrix", lambda *a, **k: np.eye(3))

    like = _FakeGWLikelihood(
        ["luminosity_distance"],
        {"luminosity_distance": Uniform(100, 2000, "luminosity_distance")},
    )
    like.generate_posterior_sample_from_marginalized_likelihood = lambda params: {
        **params,
        "luminosity_distance": 950.0,
    }
    est = LaplacePosteriorEstimator(like, _sampled_priors(), fisher_method="waveform")
    values = est._resolve_marginalized_values({"x": 0.0, "y": 0.0}, ["luminosity_distance"])
    assert values == {"luminosity_distance": 950.0}
