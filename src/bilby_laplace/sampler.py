import datetime
import sys

import numpy as np
import pandas as pd
import tqdm
from bilby.core.sampler.base_sampler import Sampler, signal_wrapper
from bilby.core.utils import logger, random
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, truncnorm

from .laplace import LaplacePosteriorEstimator

try:
    from bilby.core.sampler.base_sampler import SamplerError
except ImportError:
    SamplerError = RuntimeError

try:
    from bilby.core.utils import safe_save_figure
except ImportError:

    def safe_save_figure(fig, filename, **kwargs):
        fig.savefig(filename, **kwargs)


class TruncatedMVNProposal:
    """Per-marginal independent truncated Gaussian proposal.

    Each parameter is sampled independently from a truncated normal
    distribution whose scale is the marginal standard deviation
    ``sqrt(cov[i, i])`` and whose support is clipped to the prior bounds.
    The log-pdf is the sum of the per-marginal truncated normal log-pdfs.

    Sampling is always efficient regardless of how much wider the Gaussian
    is than the prior: every draw lands within the prior support.  Off-
    diagonal covariance elements are not used in sampling or the log-pdf;
    inter-parameter correlations are recovered through the likelihood during
    the acceptance step.
    """

    def __init__(self, mean, cov, lower, upper):
        self.mean = mean
        self._sigma = np.sqrt(np.diag(cov))
        self._ndim = len(mean)
        # Standardised bounds for truncnorm: a = (lo - μ)/σ, b = (hi - μ)/σ
        self._a = (lower - mean) / self._sigma
        self._b = (upper - mean) / self._sigma
        self._dists = [
            truncnorm(a=self._a[i], b=self._b[i], loc=mean[i], scale=self._sigma[i]) for i in range(self._ndim)
        ]

    def sample(self, n):
        return np.column_stack([d.rvs(n, random_state=random.rng) for d in self._dists])

    def logpdf(self, x):
        x = np.atleast_2d(x)
        return np.sum(
            [d.logpdf(x[:, i]) for i, d in enumerate(self._dists)],
            axis=0,
        )


class GaussianFlow:
    """Minimal aspire-compatible Flow wrapping a multivariate Gaussian.

    Implements the ``log_prob`` and ``sample_and_log_prob`` interface required
    by aspire's SMC sampler as the ``prior_flow`` argument.
    """

    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov
        self._dist = multivariate_normal(mean=mean, cov=cov)

    def log_prob(self, x):
        return self._dist.logpdf(np.asarray(x))

    def sample_and_log_prob(self, n_samples):
        x = random.rng.multivariate_normal(self._mean, self._cov, n_samples)
        return x, self._dist.logpdf(x)


class GaussianMixtureFlow:
    """Aspire-compatible Flow wrapping an equal-weight Gaussian mixture.

    Used as ``prior_flow`` when multiple MAP estimates are available, so the
    SMC annealing path starts from a mixture that covers all discovered modes.
    """

    def __init__(self, means, covs):
        self._dists = [multivariate_normal(mean=m, cov=c) for m, c in zip(means, covs)]
        self._k = len(self._dists)
        self._log_w = -np.log(self._k)  # equal weights in log space

    def log_prob(self, x):
        x = np.asarray(x)
        # log_probs: shape (K, N) or (K,) for a single point
        log_probs = np.array([d.logpdf(x) for d in self._dists])
        return logsumexp(log_probs + self._log_w, axis=0)

    def sample_and_log_prob(self, n_samples):
        idx = random.rng.integers(0, self._k, n_samples)
        x = np.array([random.rng.multivariate_normal(self._dists[i].mean, self._dists[i].cov) for i in idx])
        return x, self.log_prob(x)


def kish_log_effective_sample_size(ln_weights):
    """Kish effective sample size from log unnormalised weights.

    Returns log(ESS) where ESS = (sum w)^2 / sum(w^2).
    """
    ln_weights = np.asarray(ln_weights, dtype=float)
    # Remove -inf entries for numerical stability
    finite = np.isfinite(ln_weights)
    if not np.any(finite):
        return -np.inf
    ln_w = ln_weights[finite]
    log_ess = 2.0 * logsumexp(ln_w) - logsumexp(2.0 * ln_w)
    return log_ess


class Laplace(Sampler):
    """Bilby sampler implementing the Laplace approximation.

    Finds the MAP (maximum a posteriori) with scipy optimisation, computes the
    inverse Hessian of the log-posterior as a Gaussian proposal covariance,
    then draws posterior samples via a resample method (if requested).

    Parameters
    ----------
    likelihood : bilby.core.likelihood.Likelihood
    priors : bilby.core.prior.PriorDict or dict
    outdir : str
    label : str
    resample : str or None
        Resampling method: ``'rejection'`` (default), ``'importance'``, ``'smc'``, or
        ``None`` / ``'None'`` to skip resampling and return raw Laplace-approximation
        samples.
    target_nsamples : int
        Target number of posterior samples.
    batch_nsamples : int
        For methods that draw samples in batches, this specifies the number of
        samples drawn per batch from the proposal distribution.
    prior_nsamples : int
        Number of prior draws used in the MAP search (multi-start mode).
    minimization_method : str
        Optimization method. Default is ``'differential_evolution'`` (global
        optimizer; recommended for real data). Set to ``'Nelder-Mead'`` to use
        the legacy multi-start local optimizer.
    plot_diagnostic : bool
        If True, produce a corner diagnostic plot after resampling.
    cov_scaling : float
        Multiplicative scale applied to the Laplace covariance.
    sampling_cov : pd.DataFrame, tuple, or None
        Pre-computed covariance to use in place of the Laplace-estimated
        covariance. Must be one of:

        - A pandas DataFrame whose row index and column index are both the
          parameter names. Any ordering is accepted; the matrix is reordered
          internally.
        - A two-element tuple ``(parameter_names, cov)`` where
          ``parameter_names`` is a sequence of parameter-name strings and
          ``cov`` is an ``(N, N)`` array-like covariance in that order.
        - ``None`` (default) to compute the posterior covariance from the
          Hessian of the log-posterior at the MAP.

        The MAP search still runs; only the covariance step is replaced.
        ``cov_scaling`` and the eigenvalue-based covariance validation are
        still applied.  Not compatible with ``n_modes > 1``.

        Examples
        --------
        Given an ``(N, N)`` array ``C`` and a list ``parameter_names``::

            import pandas as pd
            cov_df = pd.DataFrame(C, index=parameter_names, columns=parameter_names)
            result = bilby.run_sampler(..., sampler="laplace", sampling_cov=cov_df)

        Or pass the tuple form directly::

            result = bilby.run_sampler(
                ..., sampler="laplace", sampling_cov=(parameter_names, C)
            )
    fisher_method : str
        How to estimate the posterior precision. ``'hessian'`` (default)
        finite-differences the scalar log-posterior. ``'waveform'`` builds the
        genuine Fisher matrix from gravitational-wave waveform derivatives
        (positive semi-definite by construction). Requires a
        ``GravitationalWaveTransient``-like likelihood without marginalisation
        and not a reduced-order (ROQ / relative-binning / multi-band) variant;
        it works directly in parameter space, so ``use_unit_cube`` and
        ``jacobian_cap_scale`` are ignored.
    fisher_kwargs : dict or None
        Keyword arguments forwarded to the waveform-Fisher computation when
        ``fisher_method='waveform'`` (recognised keys: ``eps``, ``eps_mass``).
    use_injection_for_map : bool
        If True and injection_parameters are set, use them as the starting
        point for the MAP search.
    fail_on_error : bool
        If True, raise SamplerError when sampling fails; otherwise just log.
    n_modes : int
        Number of distinct posterior modes to search for when
        ``resample='smc'``.  When ``n_modes > 1`` the optimiser is restarted
        from multiple prior draws and distinct MAP estimates are combined into
        an equal-weight Gaussian mixture proposal for the SMC.  Modes are
        deduplicated by requiring a normalised separation of at least 3-sigma
        in any parameter.  Default is 1 (single Gaussian, original behaviour).
    mode_search_nsamples : int
        Number of prior draws used when searching for secondary modes
        (``n_modes > 1``).  Higher values make mode discovery more
        reliable in high-dimensional spaces, at the cost of more
        likelihood evaluations.  Uses Latin hypercube sampling for
        even coverage.  Default is 500.
    smc_kwargs : dict or None
        Configuration for SMC sampling (only used when ``resample='smc'``).
        Recognised keys:

        ``sampler`` : str
            Aspire posterior sampler: ``'importance'`` (default), ``'smc'``,
            or any other strategy accepted by
            ``aspire.Aspire.sample_posterior``.
        ``n_initial_samples`` : int
            Number of initial samples drawn from the Laplace proposal and
            passed to ``aspire.fit()`` (default 1000).
        ``n_final_samples`` : int
            Number of output samples requested from
            ``aspire.sample_posterior()``.  Defaults to ``target_nsamples``
            if not set.

        Any other keys are forwarded directly to
        ``aspire.Aspire.sample_posterior()``, so all aspire parameters are
        accessible this way.
    prior_parameters : list or None
        List of parameter names for which initial proposal samples should be
        replaced with independent draws from the prior. Use this for parameters
        with wide posteriors consistent with their prior, where the Hessian
        poorly constrains the proposal covariance. Default is None (no
        replacement).
    """

    sampler_name = "laplace"
    sampling_seed_key = "seed"
    default_kwargs = dict(
        resample="rejection",
        target_nsamples=10000,
        batch_nsamples=1000,
        prior_nsamples=100,
        minimization_method="differential_evolution",
        plot_diagnostic=False,
        cov_scaling=1,
        sampling_cov=None,
        use_injection_for_map=True,
        fail_on_error=True,
        use_unit_cube=True,
        jacobian_cap_scale=1.0,
        hessian_kwargs=None,
        fisher_method="hessian",
        fisher_kwargs=None,
        n_modes=1,
        mode_search_nsamples=500,
        smc_kwargs=None,
        max_iterations=1e6,
        prior_parameters=None,
    )

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        plot=False,
        exit_code=77,
        skip_import_verification=True,
        **kwargs,
    ):
        super().__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            exit_code=exit_code,
            **kwargs,
        )

    def _resolve_sampling_cov(self, sampling_cov, parameter_names):
        """Normalize a user-provided sampling covariance to an ndarray.

        Accepts either a pandas DataFrame with parameter-named index/columns,
        a ``(names, cov)`` tuple, or ``None``.  Returns an ``(N, N)`` ndarray
        ordered by ``parameter_names``, or ``None`` when the input is ``None``.

        All mismatches (unknown names, missing names, duplicates, wrong shape,
        non-symmetry, non-PSD) raise ``ValueError`` up-front.
        """
        if sampling_cov is None:
            return None

        if isinstance(sampling_cov, pd.DataFrame):
            cov_df = sampling_cov
        elif isinstance(sampling_cov, tuple) and len(sampling_cov) == 2:
            names, cov = sampling_cov
            names = list(names)
            cov = np.asarray(cov, dtype=float)
            if cov.shape != (len(names), len(names)):
                raise ValueError(
                    f"sampling_cov array has shape {cov.shape}; expected "
                    f"({len(names)}, {len(names)}) for parameter_names of "
                    f"length {len(names)}."
                )
            cov_df = pd.DataFrame(cov, index=names, columns=names)
        else:
            raise TypeError(
                "sampling_cov must be a pandas DataFrame with parameter-named "
                "index/columns, a (parameter_names, cov) tuple, or None. "
                f"Got {type(sampling_cov).__name__}."
            )

        row_names = list(cov_df.index)
        col_names = list(cov_df.columns)

        if len(set(row_names)) != len(row_names):
            dups = sorted({n for n in row_names if row_names.count(n) > 1})
            raise ValueError(f"sampling_cov parameter names contain duplicates: {dups}")

        if set(row_names) != set(col_names):
            row_only = sorted(set(row_names) - set(col_names))
            col_only = sorted(set(col_names) - set(row_names))
            raise ValueError(
                f"sampling_cov DataFrame row index and column index must "
                f"contain the same parameter names. Row-only: {row_only}. "
                f"Column-only: {col_only}."
            )

        expected = set(parameter_names)
        got = set(row_names)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            if missing:
                raise ValueError(f"sampling_cov is missing covariance entries for parameter(s): {missing}")
            if extra:
                raise ValueError(f"sampling_cov contains unknown parameter(s) not in the model: {extra}")

        cov_df = cov_df.loc[list(parameter_names), list(parameter_names)]
        C = np.asarray(cov_df.values, dtype=float)

        asym = float(np.max(np.abs(C - C.T)))
        scale = max(float(np.max(np.abs(C))), 1.0)
        if asym > 1e-8 * scale:
            raise ValueError(
                f"sampling_cov is not symmetric: max asymmetry {asym:.3g} " f"exceeds tolerance {1e-8 * scale:.3g}."
            )
        C = 0.5 * (C + C.T)

        eigvals = np.linalg.eigvalsh(C)
        max_eig = float(eigvals.max())
        tol = 1e-8 * max(max_eig, 1.0)
        if eigvals.min() < -tol:
            raise ValueError(
                f"sampling_cov is not positive semi-definite: minimum "
                f"eigenvalue {eigvals.min():.3g} is below tolerance {-tol:.3g}."
            )

        return C

    def _replace_with_prior_samples(self, samples, parameter_names):
        """Replace specified parameters with draws from the prior.

        For parameters in self.kwargs['prior_parameters'], replace values
        in the samples DataFrame with independent draws from their priors.
        This is useful for parameters with wide posteriors consistent with
        their prior, where the Hessian poorly constrains the proposal covariance.

        Parameters
        ----------
        samples : pd.DataFrame
            Samples with columns for each parameter.
        parameter_names : list
            Names of all parameters in order.

        Returns
        -------
        samples : pd.DataFrame
            Modified samples with specified parameters replaced by prior draws.
        """
        prior_params = self.kwargs.get("prior_parameters") or []
        if not prior_params:
            return samples

        bad_params = [p for p in prior_params if p in parameter_names]
        unknown_params = [p for p in prior_params if p not in parameter_names]

        if unknown_params:
            logger.warning(f"prior_parameters contains unknown parameters: {unknown_params}")

        for param in bad_params:
            prior_draws = self.priors[param].sample(len(samples))
            samples[param] = prior_draws
            logger.debug(f"Replaced initial {param} samples with {len(samples)} prior draws")

        return samples

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Return expected output files/dirs (used by bilby_pipe / HTCondor)."""
        return [], []

    @signal_wrapper
    def run_sampler(self):
        self.start_time = datetime.datetime.now()
        cov_scaling = self.kwargs["cov_scaling"]

        estimator = LaplacePosteriorEstimator(
            likelihood=self.likelihood,
            priors=self.priors,
            minimization_method=self.kwargs["minimization_method"],
            n_prior_samples=self.kwargs["prior_nsamples"],
            use_unit_cube=self.kwargs["use_unit_cube"],
            jacobian_cap_scale=self.kwargs["jacobian_cap_scale"],
            hessian_kwargs=self.kwargs["hessian_kwargs"],
            fisher_method=self.kwargs["fisher_method"],
            fisher_kwargs=self.kwargs["fisher_kwargs"],
        )

        # Validate any user-provided sampling covariance up-front (before the
        # MAP search) so naming/shape errors surface immediately.
        user_cov = self._resolve_sampling_cov(self.kwargs["sampling_cov"], estimator.parameter_names)
        if user_cov is not None and self.kwargs["n_modes"] > 1:
            raise SamplerError(
                "sampling_cov cannot be combined with n_modes > 1; "
                "multi-mode search builds an independent covariance per mode."
            )

        # Choose starting point for MAP search
        if self.injection_parameters and self.kwargs["use_injection_for_map"]:
            fallback = self.priors.sample_subset(estimator.parameter_names)
            missing = [k for k in estimator.parameter_names if k not in self.injection_parameters]
            if missing:
                logger.warning(
                    f"use_injection_for_map=True but the following parameters are not in "
                    f"injection_parameters (using prior samples as fallback): {missing}"
                )
            initial_sample = {
                key: self.injection_parameters.get(key, fallback[key]) for key in estimator.parameter_names
            }
        else:
            initial_sample = None

        map_sample_dict = estimator.get_MAP_sample(initial_sample)
        mean = np.array(list(map_sample_dict.values()))
        if user_cov is not None:
            logger.info("Using user-provided sampling covariance (skipping Laplace estimate)")
            covariance = user_cov
        else:
            covariance = estimator.calculate_posterior_covariance(map_sample_dict)
        cov = cov_scaling * covariance
        cov = self._validate_covariance(estimator, mean, cov)

        msg = "Gaussian proposal (MAP +/- 1-sigma):\n " + "\n ".join(
            f"{key}: {val:.5f} +/- {np.sqrt(var):.5f}" for (key, val), var in zip(map_sample_dict.items(), np.diag(cov))
        )
        logger.info(msg)

        # Build the proposal distribution.  For rejection and importance
        # sampling we use a truncated Gaussian (per-marginal independent
        # truncated normals) so that every draw lands within the prior support.
        # This eliminates wasted likelihood evaluations on out-of-bounds
        # samples and is especially important when the Laplace-derived sigma is
        # much larger than the prior width.
        proposal = TruncatedMVNProposal(
            mean,
            cov,
            lower=estimator.prior_bounds_min,
            upper=estimator.prior_bounds_max,
        )

        if self.kwargs["plot_diagnostic"]:
            init_samples = self._draw_inprior_samples(proposal, 5000, estimator.parameter_names)
            self.create_proposal_diagnostic(mean, cov, estimator.parameter_names, init_samples)

        # Laplace evidence (always available)
        log_evidence_laplace = estimator.log_evidence_laplace(map_sample_dict, covariance)
        log_evidence = log_evidence_laplace
        log_evidence_err = np.nan

        target_nsamples = self.kwargs["target_nsamples"]
        resample = self.kwargs["resample"]
        if resample == "None":
            resample = None

        if resample is None:
            samples, logl, g_samples, efficiency = self._sample_laplace(mean, cov, estimator, target_nsamples)
        elif resample == "smc":
            samples, logl, g_samples, efficiency, smc_log_z, smc_log_z_err = self._run_smc(
                mean, cov, proposal, estimator, cov_scaling
            )
            if smc_log_z is not None:
                log_evidence = float(smc_log_z)
                log_evidence_err = float(smc_log_z_err)
        elif resample == "rejection":
            samples, logl, g_samples, efficiency, log_evidence, log_evidence_err = self._run_rejection_sampling(
                proposal, estimator, map_sample_dict
            )
        elif resample == "inprior":
            samples, logl, g_samples, efficiency = self._run_inprior(proposal, estimator)
            log_evidence = np.nan
            log_evidence_err = np.nan
        elif resample == "importance":
            samples, logl, g_samples, efficiency, log_evidence, log_evidence_err = self._run_importance_sampling(
                proposal, estimator, map_sample_dict
            )
        else:
            raise ValueError(
                f"Unknown resample method {resample!r}. "
                f"Expected one of: None, 'rejection', 'importance', 'inprior', 'smc'."
            )

        end_time = datetime.datetime.now()
        self.sampling_time = end_time - self.start_time

        if self.use_ratio:
            logl -= self.likelihood.noise_log_likelihood()

        logger.info(
            f"Log-evidence summary: "
            f"Laplace={log_evidence_laplace:.2f}, "
            f"final={log_evidence:.2f} "
            f"+/- {log_evidence_err:.2f}"
        )

        self._generate_result(
            samples,
            logl,
            log_evidence=log_evidence,
            log_evidence_err=log_evidence_err,
            log_evidence_laplace=log_evidence_laplace,
            efficiency=efficiency,
            nlikelihood=len(g_samples),
        )

        return self.result

    def _generate_result(
        self,
        samples,
        log_likelihood_evaluations,
        log_evidence=np.nan,
        log_evidence_err=np.nan,
        **run_stats,
    ):
        self.result.samples = samples[self.search_parameter_keys].values
        self.result.log_likelihood_evaluations = log_likelihood_evaluations
        self.result.log_evidence = log_evidence
        self.result.log_evidence_err = log_evidence_err
        run_stats["sampling_time_s"] = self.sampling_time.total_seconds()
        self.result.meta_data["run_statistics"] = run_stats

    def _sample_laplace(self, mean, cov, estimator, target_nsamples):
        """Draw samples directly from the Gaussian approximation without resampling."""
        logger.info(f"Drawing {target_nsamples} samples from " f"Gaussian approximation (no resampling)")
        samples_array = random.rng.multivariate_normal(mean, cov, target_nsamples)
        samples = pd.DataFrame(samples_array, columns=estimator.parameter_names)
        samples = self._replace_with_prior_samples(samples, estimator.parameter_names)
        logl = np.full(target_nsamples, np.nan)
        return samples, logl, samples, 100.0

    def _draw_inprior_samples(self, proposal, n, parameter_names):
        """Draw *n* samples from *proposal* filtered to the prior support.

        Draws in batches, discarding any sample where the full prior
        log-probability is ``-inf``.  No likelihood evaluations are performed.
        Returns a ``(n, ndim)`` float array.
        """
        batch_nsamples = self.kwargs["batch_nsamples"]
        collected = []
        n_collected = 0
        total_drawn = 0

        while n_collected < n:
            if self._check_iteration_limit("In-prior sampling", total_drawn, n_collected):
                break
            x_batch = proposal.sample(batch_nsamples)
            g_batch = pd.DataFrame(x_batch, columns=parameter_names)
            logpi = np.real(np.array(self.priors.ln_prob(g_batch, axis=0)))
            in_prior = ~np.isinf(logpi)
            total_drawn += batch_nsamples
            if in_prior.any():
                collected.append(x_batch[in_prior])
                n_collected += int(in_prior.sum())

        if not collected:
            return np.empty((0, len(parameter_names)))
        x_out = np.vstack(collected)[:n]
        logger.debug(
            f"Drew {len(x_out)} in-prior samples from {total_drawn} proposals "
            f"({100.0 * len(x_out) / total_drawn:.1f}% efficiency)"
        )
        df_out = pd.DataFrame(x_out, columns=parameter_names)
        df_out = self._replace_with_prior_samples(df_out, parameter_names)
        x_out = df_out.values
        return x_out

    def _check_iteration_limit(self, method_name, n_proposed, n_accepted):
        """Check if iteration limit has been hit and acceptance rate is below 1%.

        Raises SamplerError or logs error depending on fail_on_error setting.
        Returns True if limit was hit, False otherwise.
        """
        max_iterations = int(self.kwargs["max_iterations"])
        if n_proposed < max_iterations:
            return False

        acceptance_rate = 100.0 * n_accepted / n_proposed
        if acceptance_rate >= 1.0:
            return False

        msg = (
            f"{method_name} hit iteration limit "
            f"({max_iterations:.0e} samples) with only "
            f"{n_accepted} accepted samples "
            f"({acceptance_rate:.2f}% < 1%). "
            f"The proposal is poorly matched to the likelihood. "
            f"Consider increasing cov_scaling."
        )
        if self.kwargs["fail_on_error"]:
            raise SamplerError(msg)
        else:
            logger.error(msg)
        return True

    def _run_inprior(self, proposal, estimator):
        """Draw and filter samples to those within prior bounds.

        Draws from the proposal and keeps only samples within the prior support,
        discarding out-of-bounds samples. Returns in-prior samples with their
        likelihoods evaluated.

        Returns ``(samples, logl, g_samples, efficiency)``.
        """
        target_nsamples = self.kwargs["target_nsamples"]
        batch_nsamples = self.kwargs["batch_nsamples"]

        logger.info(f"Drawing samples from proposal and filtering to prior support " f"(target: {target_nsamples})")

        samples_list = []
        logl_list = []
        total_drawn = 0
        n_accepted = 0

        pbar = tqdm.tqdm(
            total=target_nsamples,
            desc="Filtering to prior",
            unit="sample",
            dynamic_ncols=True,
        )

        while n_accepted < target_nsamples:
            if self._check_iteration_limit("In-prior sampling", total_drawn, n_accepted):
                pbar.close()
                break
            x_batch = proposal.sample(batch_nsamples)
            g_batch = pd.DataFrame(x_batch, columns=estimator.parameter_names)

            # Apply prior replacement BEFORE checking if in prior
            g_batch = self._replace_with_prior_samples(g_batch, estimator.parameter_names)
            x_batch = g_batch.values

            logpi = np.real(np.array(self.priors.ln_prob(g_batch, axis=0)))
            in_prior = ~np.isinf(logpi)

            total_drawn += batch_nsamples
            batch_accepted = int(np.sum(in_prior))

            if in_prior.any():
                x_in = x_batch[in_prior]
                logl_in = estimator.log_likelihood_from_array(x_in.T)

                samples_list.append(x_in)
                logl_list.append(logl_in)
                n_accepted += batch_accepted

                samples_to_show = min(batch_accepted, max(0, target_nsamples - pbar.n))
                pbar.update(samples_to_show)
                pbar.set_postfix(
                    {
                        "drawn": total_drawn,
                        "eff": f"{100.0 * n_accepted / total_drawn:.1f}%",
                    }
                )

        pbar.close()

        if not samples_list:
            empty = pd.DataFrame(columns=estimator.parameter_names)
            return empty, np.array([]), empty, 0.0

        x_out = np.vstack(samples_list)[:target_nsamples]
        logl_out = np.hstack(logl_list)[:target_nsamples]
        samples = pd.DataFrame(x_out, columns=estimator.parameter_names)

        efficiency = 100.0 * n_accepted / total_drawn if total_drawn else 0.0
        logger.info(f"Filtering complete: kept {len(x_out)} of {total_drawn} samples ({efficiency:.1f}% efficiency)")

        return samples, logl_out, samples, efficiency

    def _run_smc(self, mean, cov, proposal, estimator, cov_scaling):
        """Build the Laplace proposal flow and run SMC sampling.

        Handles multi-mode discovery when ``n_modes > 1``, then delegates to
        ``_smc_sample``.  Returns ``(samples, logl, g_samples, efficiency,
        smc_log_z, smc_log_z_err, nlikelihood)`` where ``smc_log_z`` is
        ``None`` if the aspire result did not carry a log-evidence attribute.
        """
        n_modes = self.kwargs["n_modes"]
        if n_modes > 1:
            map_estimates = self._find_multiple_maps(estimator, n_modes, cov_scaling)
            proposal_flow = GaussianMixtureFlow(
                [m for m, _ in map_estimates],
                [c for _, c in map_estimates],
            )
        else:
            proposal_flow = GaussianFlow(mean, cov)

        samples, logl, smc_log_z, smc_log_z_err = self._smc_sample(proposal_flow, proposal, estimator)

        if self.kwargs["plot_diagnostic"]:
            self.create_smc_diagnostic(samples, proposal_flow)

        return samples, logl, samples, 100.0, smc_log_z, smc_log_z_err

    def _compute_ln_ratios(self, x, g_df, proposal, estimator):
        """Compute log[L(x)π(x)/g(x)] for a batch of proposal samples.

        Returns ``(ln_r, logl)`` arrays of length ``len(g_df)``.
        Samples outside the prior get ``ln_r = -inf``.
        """
        logpi = np.real(np.array(self.priors.ln_prob(g_df, axis=0)))
        in_prior = ~np.isinf(logpi)
        logl = np.full(len(g_df), -np.inf)
        if in_prior.any():
            logl[in_prior] = estimator.log_likelihood_from_array(x[in_prior].T)
        else:
            msg = "All proposal samples fell outside the prior"
            if self.kwargs["fail_on_error"]:
                raise SamplerError(msg)
            logger.debug(msg)
        log_g = proposal.logpdf(x)
        ln_r = logl + logpi - log_g
        return ln_r, logl

    def _run_rejection_sampling(self, proposal, estimator, map_sample_dict):
        """Draw samples from the proposal using rejection sampling.

        The acceptance probability for a proposal θ is L(θ)π(θ) / (M·g(θ)),
        where M is an upper bound on L(θ)π(θ)/g(θ).  The bound is established
        by a pre-scan batch before any samples are accepted, so that a
        mid-run bound update (which would invalidate earlier acceptances)
        cannot occur.

        Returns ``(samples, logl, g_samples, efficiency, log_evidence, log_evidence_err)``.
        """
        target_nsamples = self.kwargs["target_nsamples"]
        batch_nsamples = self.kwargs["batch_nsamples"]
        mean = proposal.mean

        # --- Establish the rejection bound ln_M ---
        # Start from the analytic value at the MAP.
        ln_M = (
            float(estimator.log_likelihood_from_array(mean))
            + sum(
                np.log(max(self.priors[k].prob(float(map_sample_dict[k])), 1e-300)) for k in estimator.parameter_names
            )
            - float(proposal.logpdf(mean.reshape(1, -1))[0])
        )

        # Pre-scan: draw in-prior calibration samples to find the empirical
        # maximum of L(x)π(x)/g(x).  Using in-prior samples avoids wasting
        # calibration evaluations on out-of-bounds points and gives a tighter
        # bound estimate.  These samples are discarded (not accepted/rejected)
        # so the bound is fixed before the main loop begins.
        x_cal = self._draw_inprior_samples(proposal, batch_nsamples, estimator.parameter_names)
        g_cal = pd.DataFrame(x_cal, columns=estimator.parameter_names)
        ln_r_cal, _ = self._compute_ln_ratios(x_cal, g_cal, proposal, estimator)
        finite_cal = np.isfinite(ln_r_cal)
        if finite_cal.any():
            empirical_max = float(np.max(ln_r_cal[finite_cal]))
            if empirical_max > ln_M:
                logger.info(f"Pre-scan raised rejection bound " f"{ln_M:.2f} → {empirical_max:.2f}")
                ln_M = empirical_max

        logger.info(
            f"Drawing {target_nsamples} samples using rejection sampling "
            f"(batch size {batch_nsamples}, ln_M = {ln_M:.2f})"
        )

        # --- Main rejection loop ---
        all_samples, all_logl, all_g_samples, all_ln_r = [], [], [], []
        n_accepted = n_proposed = 0
        n_bound_violations = 0

        pbar = tqdm.tqdm(total=target_nsamples, desc="Rejection sampling", file=sys.stdout)

        while n_accepted < target_nsamples:
            if self._check_iteration_limit("Rejection sampling", n_proposed, n_accepted):
                pbar.close()
                break

            x = proposal.sample(batch_nsamples)
            g_df = pd.DataFrame(x, columns=estimator.parameter_names)

            # Apply prior replacement BEFORE computing likelihoods
            g_df = self._replace_with_prior_samples(g_df, estimator.parameter_names)
            x = g_df.values

            ln_r, logl = self._compute_ln_ratios(x, g_df, proposal, estimator)
            finite = np.isfinite(ln_r)

            # Count samples that exceed the bound (accepted with prob 1,
            # which may introduce minor bias).
            if finite.any():
                n_over = int(np.sum(ln_r[finite] > ln_M))
                n_bound_violations += n_over

            # Accept if log(U) < ln_r - ln_M  ⟺  U < L(x)π(x) / (M·g(x))
            log_u = np.log(random.rng.uniform(size=batch_nsamples))
            accepted = finite & (log_u < ln_r - ln_M)

            n_proposed += batch_nsamples
            n_accepted += int(accepted.sum())
            efficiency = 100.0 * n_accepted / n_proposed
            pbar.set_postfix({"acceptance": f"{efficiency:.3f}%"}, refresh=False)
            pbar.update(int(accepted.sum()))

            g_accepted = g_df[accepted].reset_index(drop=True)
            all_samples.append(g_accepted)
            all_logl.append(logl[accepted])
            all_g_samples.append(g_df)
            all_ln_r.append(ln_r)

        pbar.close()

        if not all_samples:
            empty = pd.DataFrame(columns=estimator.parameter_names)
            return empty, np.array([]), empty, 0.0, np.nan, np.nan

        if n_bound_violations > 0:
            logger.warning(
                f"{n_bound_violations} of {n_proposed} proposal samples "
                f"({100.0 * n_bound_violations / n_proposed:.1f}%) exceeded "
                f"the rejection bound and were accepted with probability 1. "
                f"The posterior may be slightly biased — consider increasing "
                f"cov_scaling to widen the proposal."
            )

        samples = pd.concat(all_samples, ignore_index=True)
        logl = np.concatenate(all_logl)
        g_samples = pd.concat(all_g_samples, ignore_index=True)
        ln_r_all = np.concatenate(all_ln_r)
        efficiency = 100.0 * len(samples) / len(g_samples)

        log_evidence, log_evidence_err = self._compute_is_evidence(ln_r_all)

        logger.info(
            f"Rejection sampling complete: {len(samples)} accepted from "
            f"{len(g_samples)} proposals ({efficiency:.1f}%)"
        )

        if self.kwargs["plot_diagnostic"]:
            weights = np.exp(ln_r_all - ln_M)
            try:
                self.create_resample_diagnostic(samples, g_samples, mean, weights, method="rejection")
            except Exception as e:
                logger.warning(f"Failed to create rejection sampling diagnostic plot: {e}")

        return samples, logl, g_samples, efficiency, log_evidence, log_evidence_err

    def _run_importance_sampling(self, proposal, estimator, map_sample_dict):
        """Draw samples from the proposal using importance resampling.

        Accumulates batches until ``target_nsamples`` are collected by
        resampling proportionally to the IS weights L(θ)π(θ)/g(θ).

        Returns ``(samples, logl, g_samples, efficiency, log_evidence, log_evidence_err)``.
        """
        target_nsamples = self.kwargs["target_nsamples"]
        batch_nsamples = self.kwargs["batch_nsamples"]

        all_samples, all_logl, all_g_samples, all_ln_r = [], [], [], []
        n_accepted = n_proposed = 0

        logger.info(f"Drawing {target_nsamples} samples using importance resampling " f"(batch size {batch_nsamples})")
        pbar = tqdm.tqdm(total=target_nsamples, desc="Importance sampling", file=sys.stdout)

        while n_accepted < target_nsamples:
            if self._check_iteration_limit("Importance sampling", n_proposed, n_accepted):
                pbar.close()
                break

            x = proposal.sample(batch_nsamples)
            g_df = pd.DataFrame(x, columns=estimator.parameter_names)

            # Apply prior replacement BEFORE computing likelihoods
            g_df = self._replace_with_prior_samples(g_df, estimator.parameter_names)
            x = g_df.values

            logpi = np.real(np.array(self.priors.ln_prob(g_df, axis=0)))
            in_prior = ~np.isinf(logpi)
            logl = np.full(batch_nsamples, -np.inf)
            if in_prior.any():
                logl[in_prior] = estimator.log_likelihood_from_array(x[in_prior].T)
            else:
                msg = "All proposal samples fell outside the prior"
                if self.kwargs["fail_on_error"]:
                    raise SamplerError(msg)
                logger.debug(msg)

            log_g = proposal.logpdf(x)
            ln_r = logl + logpi - log_g

            finite = np.isfinite(ln_r)
            if not finite.any():
                continue

            # Normalised weights for this batch
            ln_r_finite = ln_r[finite]
            ln_w = ln_r_finite - np.max(ln_r_finite)
            w = np.exp(ln_w)
            w /= w.sum()

            ess = int(np.floor(np.exp(kish_log_effective_sample_size(ln_r_finite))))
            n_draw = min(ess, len(g_df))

            if n_draw == 0:
                pbar.update(0)
                continue

            finite_idx = np.where(finite)[0]
            chosen = random.rng.choice(len(finite_idx), size=n_draw, p=w)
            idx = finite_idx[chosen]

            n_proposed += batch_nsamples
            n_accepted += n_draw
            efficiency = 100.0 * n_accepted / n_proposed
            pbar.set_postfix({"acceptance": f"{efficiency:.3f}%"}, refresh=False)
            pbar.update(n_draw)

            g_selected = g_df.iloc[idx].reset_index(drop=True)
            all_samples.append(g_selected)
            all_logl.append(logl[idx])
            all_g_samples.append(g_df)
            all_ln_r.append(ln_r)

        pbar.close()

        if not all_samples:
            empty = pd.DataFrame(columns=estimator.parameter_names)
            return empty, np.array([]), empty, 0.0, np.nan, np.nan

        samples = pd.concat(all_samples, ignore_index=True)
        logl = np.concatenate(all_logl)
        g_samples = pd.concat(all_g_samples, ignore_index=True)
        ln_r_all = np.concatenate(all_ln_r)
        efficiency = 100.0 * len(samples) / len(g_samples)

        log_evidence, log_evidence_err = self._compute_is_evidence(ln_r_all)

        logger.info(
            f"Importance sampling complete: {len(samples)} drawn from "
            f"{len(g_samples)} proposals ({efficiency:.1f}% effective)"
        )

        if self.kwargs["plot_diagnostic"]:
            ln_r_shifted = ln_r_all - np.nanmax(ln_r_all)
            weights = np.where(np.isfinite(ln_r_shifted), np.exp(ln_r_shifted), 0.0)
            self.create_resample_diagnostic(samples, g_samples, proposal.mean, weights, method="importance")

        return samples, logl, g_samples, efficiency, log_evidence, log_evidence_err

    def _compute_is_evidence(self, ln_r):
        """Estimate log Z and its uncertainty from raw IS log-weights.

        Uses ``log Z = logsumexp(ln_r) - log(N)`` with a delta-method variance.
        Returns ``(log_evidence, log_evidence_err)``.
        """
        finite = np.isfinite(ln_r)
        if not np.any(finite):
            return np.nan, np.nan

        ln_w = ln_r[finite]
        n_total = len(ln_r)
        log_z = logsumexp(ln_w) - np.log(n_total)
        log_z2 = logsumexp(2 * ln_w) - 2 * np.log(n_total)
        var_ratio = np.exp(log_z2 - 2 * log_z) - 1.0 / n_total
        log_z_err = np.sqrt(var_ratio) if var_ratio > 0 else 0.0

        logger.info(f"IS log-evidence: {log_z:.2f} +/- {log_z_err:.2f}")
        return log_z, log_z_err

    def _smc_sample(self, proposal_flow, proposal, estimator):
        """Run posterior sampling via aspire, starting from the Laplace proposal.

        Initial samples are drawn from *proposal* (a ``TruncatedMVNProposal``)
        filtered to the prior support, matching the inprior/rejection sampling
        approach.  ``aspire.sample_posterior()`` then refines these toward the
        true posterior.
        """
        from aspire import Aspire
        from aspire.samples import Samples
        from aspire_bilby.utils import get_aspire_functions

        parameter_names = estimator.parameter_names

        prior_bounds = {key: (self.priors[key].minimum, self.priors[key].maximum) for key in parameter_names}

        functions = get_aspire_functions(self.likelihood, self.priors, parameter_names)

        aspire_sampler = Aspire(
            log_likelihood=functions.log_likelihood,
            log_prior=functions.log_prior,
            dims=len(parameter_names),
            parameters=parameter_names,
            prior_bounds=prior_bounds,
        )

        # Copy so we can pop without mutating the user's dict
        smc_kw = dict(self.kwargs.get("smc_kwargs") or {})
        sampler_type = smc_kw.pop("sampler", "importance")
        n_initial = smc_kw.pop("n_initial_samples", 1000)
        n_final = smc_kw.pop("n_final_samples", self.kwargs["target_nsamples"])

        # Draw initial samples filtered to the prior support, consistent with
        # the inprior/rejection sampling paths.
        initial_theta = self._draw_inprior_samples(proposal, n_initial, parameter_names)
        initial_samples = Samples(initial_theta, parameters=parameter_names)
        aspire_sampler.fit(initial_samples)

        logger.info(f"Starting Aspire sampling (sampler: {sampler_type})")
        result, self._smc_history = aspire_sampler.sample_posterior(
            n_final, sampler=sampler_type, return_history=True, **smc_kw
        )

        x_out = np.asarray(result.x)
        samples = pd.DataFrame(x_out, columns=parameter_names)
        logl = estimator.log_likelihood_from_array(x_out.T)

        smc_log_z = getattr(result, "log_evidence", None)
        smc_log_z_err = getattr(result, "log_evidence_error", np.nan)
        if smc_log_z is not None:
            logger.info(f"Aspire log-evidence: {smc_log_z:.2f} " f"+/- {smc_log_z_err:.2f}")

        return samples, logl, smc_log_z, smc_log_z_err

    def _validate_covariance(self, estimator, mean, cov):
        """Validate the covariance by checking likelihood along
        each principal axis.

        At 1-sigma from the MAP along each eigenvector, the
        log-likelihood should drop by 0.5 for a Gaussian.  If the
        actual drop is significantly less (posterior wider than
        the Gaussian predicts), that eigenvalue is inflated to
        match.  Directions are never shrunk.
        """
        eigvals, eigvecs = np.linalg.eigh(cov)
        logl_peak = float(estimator.log_likelihood_from_array(mean))

        any_inflated = False
        for i in range(len(eigvals)):
            sigma_i = np.sqrt(max(eigvals[i], 1e-30))
            direction = eigvecs[:, i]

            # Evaluate at +/- 1 sigma
            logl_plus = float(estimator.log_likelihood_from_array(mean + sigma_i * direction))
            logl_minus = float(estimator.log_likelihood_from_array(mean - sigma_i * direction))

            # Collect finite drops only (skip out-of-bounds)
            drops = []
            if np.isfinite(logl_plus):
                drops.append(logl_peak - logl_plus)
            if np.isfinite(logl_minus):
                drops.append(logl_peak - logl_minus)
            if not drops:
                continue

            # Use the smaller drop (the wider side)
            actual_drop = min(drops)
            expected_drop = 0.5

            # Only inflate when posterior is notably wider
            if 0 < actual_drop < expected_drop * 0.5:
                inflation = expected_drop / actual_drop
                eigvals[i] *= inflation
                any_inflated = True
                logger.info(
                    f"Widening proposal along axis {i}: "
                    f"posterior is {inflation:.1f}x wider "
                    f"than Gaussian approximation"
                )
            elif actual_drop <= 0:
                # Likelihood flat or rising — inflate
                eigvals[i] *= 4.0
                any_inflated = True
                logger.info(f"Widening proposal along axis {i}: " f"likelihood is flat at 1-sigma, " f"expanding by 4x")

        if any_inflated:
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            cov = 0.5 * (cov + cov.T)

        # Guarantee strict positive definiteness for scipy's _PSD check.
        # scipy.stats.multivariate_normal (allow_singular=False) requires all
        # eigenvalues to exceed eps = 1e6 * machine_eps * max_eigval ≈ 2.22e-10 * max.
        # Use 1e-9 * max as the floor to stay comfortably above that threshold.
        eigvals_out = np.linalg.eigvalsh(cov)
        min_floor = max(1e-9 * eigvals_out.max(), 1e-30)
        if eigvals_out.min() < min_floor:
            cov = cov + (min_floor - eigvals_out.min()) * np.eye(len(cov))

        return cov

    def _latin_hypercube_prior(self, parameter_names, n_samples):
        """Draw *n_samples* from the prior using Latin hypercube
        sampling for even coverage.

        Generates a stratified grid in [0,1]^D, shuffles each
        column independently, then maps through each prior's
        inverse CDF (``rescale``).
        """
        ndim = len(parameter_names)

        # Stratified uniform in each dimension
        intervals = np.arange(n_samples, dtype=float)
        lhs_unit = np.column_stack([(intervals + random.rng.uniform(size=n_samples)) / n_samples for _ in range(ndim)])
        # Shuffle each column independently
        for j in range(ndim):
            random.rng.shuffle(lhs_unit[:, j])

        # Map [0,1] -> prior support via inverse CDF
        samples = np.empty_like(lhs_unit)
        for j, key in enumerate(parameter_names):
            samples[:, j] = self.priors[key].rescale(lhs_unit[:, j])

        return samples

    def _find_multiple_maps(self, estimator, n_modes, cov_scaling):
        """Find up to *n_modes* distinct MAP estimates and their
        covariances.

        Uses differential evolution (DE) for the primary mode, then
        multi-start local optimization from prior samples to discover
        secondary modes.  Each candidate is polished, deduplicated
        by 3-sigma separation, and has its covariance validated.

        Returns a list of ``(mean_array, cov_array)`` pairs sorted by
        descending log-posterior.
        """
        parameter_names = estimator.parameter_names
        logger.info(f"Searching for up to {n_modes} posterior " f"mode(s)")

        # --- 1. Find primary mode with DE ---
        result = estimator._maximize_posterior_differential_evolution()
        best_mean = np.array(result.x)
        best_logp = -result.fun
        best_dict = dict(zip(parameter_names, best_mean))
        logger.info(f"Primary mode found: " f"log-posterior = {best_logp:.2f}")

        try:
            covariance = estimator.calculate_posterior_covariance(best_dict)
            cov = cov_scaling * covariance
            cov = self._validate_covariance(estimator, best_mean, cov)
            std_scale = np.sqrt(np.diag(cov))
        except Exception as exc:
            raise SamplerError(f"Covariance estimation failed for primary mode: {exc}")

        found_modes = [(best_mean, cov, best_logp)]

        if n_modes <= 1:
            self._log_mode_summary(found_modes, parameter_names)
            return [(m, c) for m, c, _ in found_modes]

        # --- 2. Multi-start search for secondary modes ---
        n_starts = self.kwargs["mode_search_nsamples"]
        logger.info(f"Evaluating {n_starts} prior samples " f"(Latin hypercube) to search for " f"secondary modes")

        # Latin hypercube in [0,1]^D, then map to prior
        prior_x = self._latin_hypercube_prior(parameter_names, n_starts)
        prior_logp = np.array([float(estimator.log_posterior_from_array(x)) for x in prior_x])

        # Sort descending by posterior
        order = np.argsort(prior_logp)[::-1]

        # Polish top candidates far from existing modes
        max_polish = 10 * n_modes
        n_polished = 0

        for idx in order:
            if len(found_modes) >= n_modes:
                break
            if n_polished >= max_polish:
                break

            x = prior_x[idx]

            # Skip if near an existing mode
            near_existing = any(np.max(np.abs(x - m) / std_scale) < 3.0 for m, _, _ in found_modes)
            if near_existing:
                continue

            # Polish with local optimizer
            n_polished += 1
            sample_dict = dict(zip(parameter_names, x))
            polished = estimator._maximize_posterior_from_initial_sample(sample_dict)
            p_mean = np.array(polished.x)
            p_logp = -polished.fun
            logger.debug(f"Candidate {n_polished}: " f"log-posterior = {p_logp:.2f} " f"after local optimisation")

            # Re-check after polishing
            is_dup = any(np.max(np.abs(p_mean - m) / std_scale) < 3.0 for m, _, _ in found_modes)
            if is_dup:
                logger.debug(f"Candidate {n_polished} converged to " f"a known mode; skipping")
                continue

            try:
                p_dict = dict(zip(parameter_names, p_mean))
                p_covariance = estimator.calculate_posterior_covariance(p_dict)
                p_cov = cov_scaling * p_covariance
                p_cov = self._validate_covariance(estimator, p_mean, p_cov)
                found_modes.append((p_mean, p_cov, p_logp))
                logger.info(f"Secondary mode {len(found_modes) - 1} " f"found: log-posterior = {p_logp:.2f}")
            except Exception as exc:
                logger.warning(f"Could not compute covariance for " f"candidate {n_polished}: {exc}")

        # --- 3. Sort and summarise ---
        found_modes.sort(key=lambda r: r[2], reverse=True)
        self._log_mode_summary(found_modes, parameter_names)
        return [(m, c) for m, c, _ in found_modes]

    @staticmethod
    def _log_mode_summary(found_modes, parameter_names):
        """Log a table summarising the discovered modes."""
        header = f"{'Mode':<6} {'log-posterior':>14}  " + "  ".join(f"{p:>12}" for p in parameter_names)
        rows = []
        for i, (mean, _cov, logp) in enumerate(found_modes):
            vals = "  ".join(f"{v:>+12.4f}" for v in mean)
            rows.append(f"  {i:<4d} {logp:>14.2f}  {vals}")
        logger.info(f"Summary of {len(found_modes)} mode(s):\n" f"  {header}\n" + "\n".join(rows))

    def create_proposal_diagnostic(self, mean, cov, parameter_names, init_samples=None):
        """Corner plot comparing the Gaussian proposal, initialization samples, and injection parameters.

        Shows:
        - Initialization samples (actual samples drawn from proposal with prior replacement applied)
          as histograms on diagonal and scatter on off-diagonal
        - Gaussian proposal as 1-D curves on diagonal and 2-D contours on off-diagonal
        - MAP estimate and injection parameters (if available) as markers

        Parameters
        ----------
        mean : array
            MAP estimate (mean of proposal)
        cov : array
            Covariance matrix of proposal
        parameter_names : list
            Names of parameters
        init_samples : array, optional
            Initial samples drawn from proposal (shape: N_samples x N_params).
            If None, prior samples are shown instead.
        """
        import corner
        import matplotlib.lines as mpllines
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        labels = [k.replace("_", " ") for k in parameter_names]
        ndim = len(parameter_names)

        proposal_sigmas = np.sqrt(np.diag(cov))

        # Use initialization samples if provided, otherwise use prior samples
        if init_samples is not None:
            display_samples = init_samples
            sample_color, sample_ls = "C0", "-"
            sample_label = "Initial samples"
        else:
            n_samples = 10000
            display_samples = np.column_stack([self.priors[k].sample(n_samples) for k in parameter_names])
            sample_color, sample_ls = "C0", "-"
            sample_label = "Prior samples"

        ranges = [(self.priors[k].minimum, self.priors[k].maximum) for k in parameter_names]

        g_color, g_ls = "k", "--"

        panel_size = 2.5
        fig_size = panel_size * ndim
        fig = corner.corner(
            display_samples,
            color=sample_color,
            contour_kwargs={"linestyles": sample_ls, "alpha": 0.8},
            hist_kwargs={"density": True, "ls": sample_ls, "alpha": 0.8},
            no_fill_contours=True,
            plot_density=False,
            plot_datapoints=True,
            plot_contours=False,
            fill_contours=True,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
            bins=50,
            smooth=0.7,
            max_n_ticks=5,
            labels=labels,
            truths=mean,
            truth_color="C1",
            fig=plt.figure(figsize=(fig_size, fig_size)),
            range=ranges,
        )

        axes_grid = np.array(fig.get_axes()).reshape(ndim, ndim)

        # 1-D analytic Gaussian marginals on diagonal panels
        for i in range(ndim):
            ax = axes_grid[i, i]
            lo, hi = ranges[i]
            xs = np.linspace(lo, hi, 300)
            ys = norm.pdf(xs, loc=mean[i], scale=proposal_sigmas[i])
            ax.plot(xs, ys, color=g_color, lw=1.5, ls=g_ls)
            # Extend y-axis to accommodate full Gaussian peak
            y_max = ax.get_ylim()[1]
            ax.set_ylim(top=max(y_max, ys.max() * 1.1))

        # 2-D samples on off-diagonal panels with low alpha,
        # overlaid with Gaussian proposal contours.
        for row in range(ndim):
            for col in range(row):
                ax = axes_grid[row, col]
                ax.scatter(
                    display_samples[:, col],
                    display_samples[:, row],
                    s=1,
                    alpha=0.01,
                    color=sample_color,
                )

        # 2-D analytic Gaussian contours on off-diagonal panels.
        # Contour levels exp(-0.5) and exp(-2) relative to the peak match the
        # 1-sigma and 2-sigma enclosed-mass fractions for a 2-D Gaussian.
        for row in range(ndim):
            for col in range(row):
                ax = axes_grid[row, col]
                x_lo, x_hi = ranges[col]
                y_lo, y_hi = ranges[row]
                xs = np.linspace(x_lo, x_hi, 60)
                ys = np.linspace(y_lo, y_hi, 60)
                X, Y = np.meshgrid(xs, ys)
                pos = np.dstack([X, Y])
                sub_mean = mean[[col, row]]
                sub_cov = cov[np.ix_([col, row], [col, row])]
                try:
                    Z = multivariate_normal(mean=sub_mean, cov=sub_cov).pdf(pos)
                    Z_max = Z.max()
                    if Z_max > 0:
                        ax.contour(
                            X,
                            Y,
                            Z,
                            levels=[Z_max * np.exp(-2), Z_max * np.exp(-0.5)],
                            colors=g_color,
                            linestyles=[g_ls],
                            alpha=0.8,
                        )
                except Exception:
                    pass

        legend_handles = [
            mpllines.Line2D([0], [0], color=sample_color, linestyle=sample_ls),
            mpllines.Line2D([0], [0], color=g_color, linestyle=g_ls),
            mpllines.Line2D(
                [0],
                [0],
                color="C1",
                linestyle=":",
                marker="s",
                markersize=8,
                linewidth=1.5,
            ),
        ]
        legend_labels = [sample_label, "Gaussian proposal", "MAP"]

        # Add injection parameters if available
        if self.injection_parameters:
            legend_handles.append(
                mpllines.Line2D(
                    [0],
                    [0],
                    color="C2",
                    linestyle=":",
                    marker="x",
                    markersize=8,
                    linewidth=1.5,
                )
            )
            legend_labels.append("Injection")

            # Overlay injection parameters on axes
            injection_array = np.array([self.injection_parameters.get(k, np.nan) for k in parameter_names])
            for i in range(ndim):
                ax = axes_grid[i, i]
                if np.isfinite(injection_array[i]):
                    ax.axvline(injection_array[i], color="C2", lw=1.5, ls=":")

            for row in range(ndim):
                for col in range(row):
                    ax = axes_grid[row, col]
                    if np.isfinite(injection_array[col]) and np.isfinite(injection_array[row]):
                        ax.scatter(
                            [injection_array[col]],
                            [injection_array[row]],
                            color="C2",
                            marker="x",
                            s=80,
                            zorder=5,
                            linewidths=1.5,
                        )

        axes_grid[0, 0].legend(
            legend_handles,
            legend_labels,
            fontsize="small",
        )
        fig.suptitle("Gaussian proposal: initial samples, injection, and MAP")

        filename = f"{self.outdir}/{self.label}_diagnostic_proposal.png"
        safe_save_figure(fig=fig, filename=filename, dpi=150)
        plt.close(fig)
        return fig

    def create_resample_diagnostic(self, samples, raw_samples, mean, weights, method):
        """Produce a corner plot comparing the proposal and resampled posteriors."""
        import corner
        import matplotlib.lines as mpllines
        import matplotlib.pyplot as plt

        labels = [k.replace("_", " ") for k in self.search_parameter_keys]
        labels.append("weights")

        corner_kwargs = dict(
            bins=50,
            smooth=0.7,
            max_n_ticks=5,
            truths=np.concatenate((mean, [1])),
            truth_color="C3",
            labels=labels,
        )

        xs = samples[self.search_parameter_keys].values
        xs = np.concatenate((xs, np.random.uniform(0, 1, len(xs)).reshape(-1, 1)), axis=1)
        rxs = raw_samples[self.search_parameter_keys].values
        rxs = np.concatenate((rxs, weights.reshape(-1, 1)), axis=1)

        # Sort by weight for cleaner scatter colouring
        idxs = np.argsort(weights)
        rxs = rxs[idxs]

        g_color, g_ls = "k", "--"
        f_color, f_ls = "C0", "-"

        panel_size = 2.5
        fig_size = panel_size * len(labels)
        lines = []
        fig = corner.corner(
            rxs,
            color=g_color,
            contour_kwargs={"linestyles": g_ls, "alpha": 0.8},
            hist_kwargs={"density": True, "ls": g_ls, "alpha": 0.8},
            data_kwargs={"alpha": 1},
            no_fill_contours=True,
            alpha=0.8,
            plot_density=False,
            plot_datapoints=False,
            fill_contours=False,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
            fig=plt.figure(figsize=(fig_size, fig_size)),
            **corner_kwargs,
        )
        lines.append(mpllines.Line2D([0], [0], color=g_color, linestyle=g_ls))

        if len(xs) > len(samples.keys()):
            fig = corner.corner(
                xs,
                color=f_color,
                contour_kwargs={"linestyles": f_ls, "alpha": 0.8},
                contourf_kwargs={"alpha": 0.8},
                hist_kwargs={"density": True, "ls": f_ls, "alpha": 0.8},
                no_fill_contours=True,
                fig=fig,
                alpha=0.1,
                plot_density=True,
                plot_datapoints=False,
                fill_contours=False,
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
                range=[1] * self.ndim + [(0, 1)],
                **corner_kwargs,
            )
            lines.append(mpllines.Line2D([0], [0], color=f_color, linestyle=f_ls))

        axes = np.array(fig.get_axes())
        labels = ["$g(x)$"] + (["$f(x)$"] if len(lines) > 1 else [])
        axes[0].legend(lines, labels)
        fig.suptitle(f"Resampling method: {method}")

        filename = f"{self.outdir}/{self.label}_diagnostic_resample_{method}.png"
        safe_save_figure(fig=fig, filename=filename, dpi=150)
        plt.close(fig)
        return fig

    def create_smc_diagnostic(self, samples, proposal_flow):
        """Produce a corner plot comparing the Laplace proposal and SMC output.

        Mode locations (MAP estimates) are overlaid as vertical lines on the
        diagonal panels and as scatter points on the off-diagonal panels.
        """
        import corner
        import matplotlib.lines as mpllines
        import matplotlib.pyplot as plt

        labels = [k.replace("_", " ") for k in self.search_parameter_keys]
        corner_kwargs = dict(
            bins=50,
            smooth=0.7,
            max_n_ticks=5,
            labels=labels,
        )

        # Extract mode means from the flow
        if isinstance(proposal_flow, GaussianMixtureFlow):
            mode_means = [d.mean for d in proposal_flow._dists]
        else:
            mode_means = [proposal_flow._mean]

        # Draw reference samples from the proposal (Gaussian or mixture)
        n = len(samples)
        laplace_samples, _ = proposal_flow.sample_and_log_prob(n)

        g_color, g_ls = "k", "--"
        f_color, f_ls = "C0", "-"
        mode_colors = [f"C{i + 1}" for i in range(len(mode_means))]

        panel_size = 2.5
        fig_size = panel_size * self.ndim
        fig = corner.corner(
            laplace_samples,
            color=g_color,
            contour_kwargs={"linestyles": g_ls, "alpha": 0.8},
            hist_kwargs={"density": True, "ls": g_ls, "alpha": 0.8},
            no_fill_contours=True,
            plot_density=False,
            plot_datapoints=False,
            fill_contours=False,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
            fig=plt.figure(figsize=(fig_size, fig_size)),
            **corner_kwargs,
        )
        fig = corner.corner(
            samples[self.search_parameter_keys].values,
            color=f_color,
            contour_kwargs={"linestyles": f_ls, "alpha": 0.8},
            contourf_kwargs={"alpha": 0.8},
            hist_kwargs={"density": True, "ls": f_ls, "alpha": 0.8},
            no_fill_contours=True,
            fig=fig,
            plot_density=True,
            plot_datapoints=False,
            fill_contours=False,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
            range=[1] * self.ndim,
            **corner_kwargs,
        )

        # Overlay mode locations on the axes grid
        ndim = self.ndim
        axes_grid = np.array(fig.get_axes()).reshape(ndim, ndim)
        for mode_mean, mc in zip(mode_means, mode_colors):
            for row in range(ndim):
                for col in range(ndim):
                    ax = axes_grid[row, col]
                    if row == col:
                        ax.axvline(mode_mean[col], color=mc, lw=1.5, ls=":")
                    elif row > col:
                        ax.scatter(
                            [mode_mean[col]],
                            [mode_mean[row]],
                            color=mc,
                            marker="+",
                            s=80,
                            zorder=5,
                            linewidths=1.5,
                        )

        # Build legend
        legend_handles = [
            mpllines.Line2D([0], [0], color=g_color, linestyle=g_ls),
            mpllines.Line2D([0], [0], color=f_color, linestyle=f_ls),
        ]
        legend_labels = ["Initial (Laplace)", "Final (SMC)"]
        for i, mc in enumerate(mode_colors):
            legend_handles.append(
                mpllines.Line2D(
                    [0],
                    [0],
                    color=mc,
                    linestyle=":",
                    marker="+",
                    markersize=8,
                    linewidth=1.5,
                )
            )
            legend_labels.append(f"Mode {i}")

        axes_grid[0, 0].legend(legend_handles, legend_labels, fontsize="small")
        fig.suptitle("Resampling method: SMC")

        filename = f"{self.outdir}/{self.label}_diagnostic_smc_samples.png"
        safe_save_figure(fig=fig, filename=filename, dpi=150)
        plt.close(fig)

        # History diagnostics (beta schedule, ESS, acceptance, etc.)
        history = getattr(self, "_smc_history", None)
        if history is not None and history.beta:
            fig_stats, _ = plt.subplots(6, 1, sharex=True, figsize=(8, 14))
            fig_stats = history.plot(fig=fig_stats)
            fig_stats.suptitle("SMC diagnostics")
            fig_stats.tight_layout()
            safe_save_figure(
                fig=fig_stats,
                filename=f"{self.outdir}/{self.label}_diagnostic_smc_stats.png",
                dpi=150,
            )
            plt.close(fig_stats)

            if history.sample_history:
                n_params = len(self.search_parameter_keys)
                fig_bands, axs = plt.subplots(
                    n_params,
                    1,
                    sharex=True,
                    figsize=(8, 2.5 * n_params),
                )
                history.plot_quantile_bands(
                    parameters=self.search_parameter_keys,
                    ax=axs,
                )
                fig_bands.suptitle("SMC parameter evolution")
                fig_bands.tight_layout()
                safe_save_figure(
                    fig=fig_bands,
                    filename=f"{self.outdir}/{self.label}_diagnostic_smc_evolution.png",
                    dpi=150,
                )
                plt.close(fig_bands)

        return fig
