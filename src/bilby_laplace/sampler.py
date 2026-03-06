import datetime
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import tqdm

from bilby.core.sampler.base_sampler import Sampler, signal_wrapper
from bilby.core.utils import logger, random

from .matrix import FisherMatrixPosteriorEstimator

try:
    from bilby.core.sampler.base_sampler import SamplerError
except ImportError:
    SamplerError = RuntimeError

try:
    from bilby.core.utils import safe_save_figure
except ImportError:
    def safe_save_figure(fig, filename, **kwargs):
        fig.savefig(filename, **kwargs)


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

    Estimates the maximum likelihood with scipy optimisation, computes the
    inverse Fisher Information Matrix (iFIM) as a Gaussian proposal covariance,
    then draws posterior samples via rejection or importance resampling.

    Parameters
    ----------
    likelihood : bilby.core.likelihood.Likelihood
    priors : bilby.core.prior.PriorDict or dict
    outdir : str
    label : str
    resample : str or None
        Resampling method: ``'rejection'`` (default), ``'importance'``, or
        ``None`` / ``'None'`` to skip resampling entirely and return raw
        Laplace-approximation samples.
    target_nsamples : int
        Target number of posterior samples.
    batch_nsamples : int
        Samples drawn per batch from the proposal distribution.
    prior_nsamples : int
        Number of prior draws used in the maximum-likelihood search.
    minimization_method : str
        Optimization method. Default is ``'differential_evolution'`` (global
        optimizer; recommended for real data). Set to ``'Nelder-Mead'`` to use
        the legacy multi-start local optimizer.
    fd_eps : float
        Finite-difference step size relative to prior width.
    plot_diagnostic : bool
        If True, produce a corner diagnostic plot after resampling.
    cov_scaling : float
        Multiplicative scale applied to the iFIM covariance.
    use_injection_for_maxL : bool
        If True and injection_parameters are set, use them as the starting
        point for the max-likelihood search.
    fail_on_error : bool
        If True, raise SamplerError when sampling fails; otherwise just log.
    smc_kwargs : dict or None
        Configuration for SMC sampling (only used when ``resample='smc'``).
        Recognised keys:

        ``backend`` : str
            Aspire SMC backend: ``'emcee'`` (default), ``'minipcn'``, or
            ``'blackjax'``.
        ``n_samples`` : int
            Number of SMC particles (default 1000).
        ``n_final_samples`` : int
            Number of output samples after final resampling.  Defaults to
            ``target_nsamples`` if not set.
        ``target_efficiency`` : float
            Target ESS/N ratio for the adaptive β schedule (default 0.5).
        ``sampler_kwargs`` : dict
            Passed verbatim to the MCMC mutation kernel, e.g.
            ``{'nsteps': 20}`` for the emcee backend.

        Any other keys are forwarded directly to ``aspire_sampler.sample()``,
        so all aspire parameters (``min_beta_step``, ``max_beta_step``,
        ``max_n_steps``, ``store_sample_history``, ``beta_tolerance``, …)
        are accessible this way.
    """

    sampler_name = "laplace"
    sampling_seed_key = "seed"
    default_kwargs = dict(
        resample="rejection",
        target_nsamples=10000,
        batch_nsamples=1000,
        prior_nsamples=100,
        minimization_method="differential_evolution",
        fd_eps=1e-6,
        plot_diagnostic=False,
        cov_scaling=1,
        use_injection_for_maxL=True,
        fail_on_error=False,
        use_unit_cube=True,
        smc_kwargs=None,
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

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Return expected output files/dirs (used by bilby_pipe / HTCondor)."""
        return [], []

    @signal_wrapper
    def run_sampler(self):
        self.start_time = datetime.datetime.now()
        cov_scaling = self.kwargs["cov_scaling"]

        fisher_mpe = FisherMatrixPosteriorEstimator(
            likelihood=self.likelihood,
            priors=self.priors,
            minimization_method=self.kwargs["minimization_method"],
            n_prior_samples=self.kwargs["prior_nsamples"],
            fd_eps=self.kwargs["fd_eps"],
            use_unit_cube=self.kwargs["use_unit_cube"],
        )

        # Choose starting point for max-likelihood search
        if self.injection_parameters and self.kwargs["use_injection_for_maxL"]:
            initial_sample = {
                key: self.injection_parameters[key]
                for key in fisher_mpe.parameter_names
                if key in self.injection_parameters
            }
        else:
            initial_sample = None

        maxL_sample_dict = fisher_mpe.get_maximum_likelihood_sample(initial_sample)
        mean = np.array(list(maxL_sample_dict.values()))
        iFIM = fisher_mpe.calculate_iFIM(maxL_sample_dict)
        cov = cov_scaling * iFIM

        msg = "Generation distribution:\n " + "\n ".join(
            f"{key}: {val:.5f} +/- {np.sqrt(var):.5f}"
            for (key, val), var in zip(maxL_sample_dict.items(), np.diag(cov))
        )
        logger.info(msg)

        target_nsamples = self.kwargs["target_nsamples"]
        batch_nsamples = self.kwargs["batch_nsamples"]
        resample = self.kwargs["resample"]
        if resample == "None":
            resample = None

        if resample is None:
            logger.info(
                f"Pure Laplace approximation: drawing {target_nsamples} samples directly"
            )
            samples_array = random.rng.multivariate_normal(mean, cov, target_nsamples)
            samples = pd.DataFrame(samples_array, columns=fisher_mpe.parameter_names)
            logl = np.full(target_nsamples, np.nan)
            g_samples = samples
            efficiency = 100.0
        elif resample == "smc":
            samples, logl = self._smc_sample(mean, cov, fisher_mpe)
            g_samples = samples
            efficiency = 100.0
            if self.kwargs["plot_diagnostic"]:
                self.create_smc_diagnostic(samples, mean, cov)
        else:
            nsamples = 0
            all_g_samples = []
            all_samples = []
            all_logl = []
            all_weights = []
            efficiency = 0.0

            logger.info(
                f"Starting sampling in batches of {batch_nsamples} "
                f"to produce {target_nsamples} samples"
            )
            pbar = tqdm.tqdm(
                total=target_nsamples,
                desc=f"{resample.capitalize()} sampling",
                file=sys.stdout,
                initial=0,
            )

            _resample_methods = dict(
                rejection=self._rejection_sample,
                importance=self._importance_sample,
            )

            while nsamples < target_nsamples:
                g_samples, g_logl, g_logpi, discard_inef = (
                    self._draw_samples_from_generating_distribution(
                        mean, cov, fisher_mpe, batch_nsamples
                    )
                )

                if resample in _resample_methods:
                    weights = self._calculate_weights(g_samples, g_logl, g_logpi, mean, cov)
                    samples, logl = _resample_methods[resample](g_samples, g_logl, weights)
                    efficiency = 100.0 * len(samples) / len(g_samples)
                else:
                    logger.info("No resampling applied")
                    samples = g_samples
                    logl = g_logl
                    weights = np.ones_like(g_logl)
                    efficiency = 100.0

                nsamples += len(samples)
                pbar.set_postfix(
                    {
                        "eff": f"{efficiency:.3f}%",
                        "de": f"{discard_inef:.1f}%",
                        "cs": f"{cov_scaling:.2f}",
                    },
                    refresh=False,
                )
                if len(samples) > 0:
                    pbar.update(len(samples))
                    all_g_samples.append(g_samples)
                    all_samples.append(samples)
                    all_logl.append(logl)
                    all_weights.append(weights)
                else:
                    pbar.update(0)

            pbar.close()

            g_samples = pd.concat(all_g_samples, ignore_index=True)
            samples = pd.concat(all_samples, ignore_index=True)
            logl = np.concatenate(all_logl)
            weights = np.concatenate(all_weights)
            efficiency = 100.0 * len(samples) / len(g_samples)

            logger.info(f"Finished sampling: total efficiency is {efficiency:.3f}%")

            if self.kwargs["plot_diagnostic"]:
                self.create_resample_diagnostic(
                    samples, g_samples, mean, weights, method=resample
                )

        end_time = datetime.datetime.now()
        self.sampling_time = end_time - self.start_time

        if self.use_ratio:
            logl -= self.likelihood.noise_log_likelihood()

        self._generate_result(
            samples, logl, efficiency=efficiency, nlikelihood=len(g_samples)
        )

        return self.result

    def _generate_result(self, samples, log_likelihood_evaluations, **run_stats):
        posterior = samples[self.search_parameter_keys].copy()
        posterior["log_likelihood"] = log_likelihood_evaluations
        self.result.posterior = posterior
        self.result.log_likelihood_evaluations = log_likelihood_evaluations
        run_stats["sampling_time_s"] = self.sampling_time.total_seconds()
        self.result.meta_data["run_statistics"] = run_stats

    def _draw_samples_from_generating_distribution(
        self, mean, cov, fisher_mpe, nsamples
    ):
        samples_array = random.rng.multivariate_normal(mean, cov, nsamples)
        samples = pd.DataFrame(samples_array, columns=fisher_mpe.parameter_names)

        logpi = self.priors.ln_prob(samples, axis=0)
        logl = np.full(len(samples), -np.inf)

        in_prior = ~np.isinf(logpi)
        outside_prior_count = int(np.sum(~in_prior))
        discard_inef = 100.0 * outside_prior_count / len(samples)

        if outside_prior_count < len(samples):
            logl[in_prior] = fisher_mpe.log_likelihood_from_array(
                samples.values[in_prior].T
            )
        else:
            msg = "Sampling has failed: no viable samples left"
            if self.kwargs["fail_on_error"]:
                raise SamplerError(msg)
            else:
                logger.debug(msg)

        logpi = np.real(np.array(logpi))
        return samples, logl, logpi, discard_inef

    def _calculate_weights(self, g_samples, g_logl, g_logpi, mean, cov):
        g_logl_norm = multivariate_normal.logpdf(g_samples, mean=mean, cov=cov)

        ln_weights = g_logl + g_logpi - g_logl_norm

        # Remove impossible samples for ESS calculation
        finite_mask = np.isfinite(ln_weights)
        ln_weights_viable = ln_weights[finite_mask]

        # Scale so max weight is 1 (avoids overflow in exp)
        if len(ln_weights_viable) > 0:
            ln_weights -= np.max(ln_weights_viable)

        self.ess = int(np.floor(np.exp(kish_log_effective_sample_size(ln_weights_viable))))
        logger.debug(f"Calculated weights; effective sample size = {self.ess}")

        return np.exp(ln_weights)

    def _rejection_sample(self, g_samples, g_logl, weights):
        logger.debug(f"Rejection sampling from {len(g_samples)} proposal samples")

        w_max = np.max(weights)
        uniform = np.random.uniform(0, w_max, len(g_samples))
        accepted = uniform < weights

        samples = g_samples[accepted].reset_index(drop=True)
        logl = g_logl[accepted]

        if len(samples) < self.ndim:
            msg = "Number of accepted samples less than ndim: sampling may have failed"
            if self.kwargs["fail_on_error"]:
                raise SamplerError(msg)
            else:
                logger.debug(msg)

        return samples, logl

    def _importance_sample(self, g_samples, g_logl, weights):
        logger.debug(f"Importance sampling from {len(g_samples)} proposal samples")

        normalized_weights = weights / np.sum(weights)
        idxs = np.random.choice(len(g_samples), size=self.ess, p=normalized_weights)
        samples = g_samples.iloc[idxs].reset_index(drop=True)
        logl = g_logl[idxs]

        if self.ess < self.ndim:
            msg = "Effective sample size less than ndim: sampling has failed"
            if self.kwargs["fail_on_error"]:
                raise SamplerError(msg)
            else:
                logger.debug(msg)

        return samples, logl

    def _smc_sample(self, mean, cov, fisher_mpe):
        """Run SMC via aspire, annealing from the Laplace proposal to the
        true posterior.

        The SMC path is::

            log π_β(x) = log N(x|MAP, iFIM⁻¹) + β · log_lik_correction(x)

        where ``log_lik_correction = log L_bilby + log π_bilby − log N``,
        so that at β=1 we recover the true posterior.
        """
        import importlib

        from scipy.stats import multivariate_normal as mvn

        parameter_names = fisher_mpe.parameter_names
        proposal = mvn(mean=mean, cov=cov)

        def log_prior_aspire(samples):
            return proposal.logpdf(np.asarray(samples.x))

        def log_lik_aspire(samples):
            x = np.asarray(samples.x)  # (N, D)
            df = pd.DataFrame(x, columns=parameter_names)
            log_pi = np.real(
                np.array(self.priors.ln_prob(df, axis=0))
            )
            log_l = np.full(len(x), -np.inf)
            in_prior = np.isfinite(log_pi)
            if np.any(in_prior):
                log_l[in_prior] = (
                    fisher_mpe.log_likelihood_from_array(x[in_prior].T)
                )
            log_q = proposal.logpdf(x)
            return log_l + log_pi - log_q

        prior_flow = GaussianFlow(mean, cov)

        _backends = {
            "emcee": "aspire.samplers.smc.emcee.EmceeSMC",
            "minipcn": "aspire.samplers.smc.minipcn.MiniPCNSMC",
            "blackjax": "aspire.samplers.smc.blackjax.BlackJAXSMC",
        }

        # Copy so we can pop without mutating the user's dict
        smc_kw = dict(self.kwargs.get("smc_kwargs") or {})
        backend = smc_kw.pop("backend", "emcee")
        if backend not in _backends:
            raise ValueError(
                f"Unknown SMC backend {backend!r}. "
                f"Choose from {list(_backends)}"
            )
        module_path, class_name = _backends[backend].rsplit(".", 1)
        SMCClass = getattr(importlib.import_module(module_path), class_name)

        logger.info(f"Initialising aspire {backend} SMC sampler")
        sampler = SMCClass(
            log_likelihood=log_lik_aspire,
            log_prior=log_prior_aspire,
            dims=len(parameter_names),
            prior_flow=prior_flow,
            xp=np,
            parameters=parameter_names,
        )

        # Apply defaults for keys not set by the user
        smc_kw.setdefault("n_samples", 1000)
        smc_kw.setdefault("n_final_samples", self.kwargs["target_nsamples"])
        smc_kw.setdefault("adaptive", True)
        smc_kw.setdefault("target_efficiency", 0.5)
        result = sampler.sample(**smc_kw)

        x_out = np.asarray(result.x)
        samples = pd.DataFrame(x_out, columns=parameter_names)
        # Recompute the true bilby log-likelihood on the SMC output samples
        # (result.log_likelihood holds log_lik_aspire, the correction term)
        logl = fisher_mpe.log_likelihood_from_array(x_out.T)
        return samples, logl

    def create_resample_diagnostic(self, samples, raw_samples, mean, weights, method):
        """Produce a corner plot comparing the proposal and resampled posteriors."""
        import corner
        import matplotlib.pyplot as plt
        import matplotlib.lines as mpllines

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
        weights_sorted = weights[idxs]

        g_color, g_ls = "k", "--"
        f_color, f_ls = "C0", "-"

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

        axes = np.array(fig.get_axes())
        axes[0].legend(lines, ["$g(x)$", "$f(x)$"])
        fig.suptitle(f"Resampling method: {method}")

        filename = f"{self.outdir}/{self.label}_resample_{method}.png"
        safe_save_figure(fig=fig, filename=filename, dpi=150)
        plt.close(fig)
        return fig

    def create_smc_diagnostic(self, samples, mean, cov):
        """Produce a corner plot comparing the Laplace proposal and SMC output."""
        import corner
        import matplotlib.pyplot as plt
        import matplotlib.lines as mpllines

        labels = [k.replace("_", " ") for k in self.search_parameter_keys]
        corner_kwargs = dict(
            bins=50,
            smooth=0.7,
            max_n_ticks=5,
            truths=mean,
            truth_color="C3",
            labels=labels,
        )

        # Draw reference samples from the Laplace proposal
        n = len(samples)
        laplace_samples = random.rng.multivariate_normal(mean, cov, n)

        g_color, g_ls = "k", "--"
        f_color, f_ls = "C0", "-"

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

        axes = np.array(fig.get_axes())
        axes[0].legend(
            [
                mpllines.Line2D([0], [0], color=g_color, linestyle=g_ls),
                mpllines.Line2D([0], [0], color=f_color, linestyle=f_ls),
            ],
            ["Initial (Laplace)", "Final (SMC)"],
        )
        fig.suptitle("Resampling method: SMC")

        filename = f"{self.outdir}/{self.label}_resample_smc.png"
        safe_save_figure(fig=fig, filename=filename, dpi=150)
        plt.close(fig)
        return fig
