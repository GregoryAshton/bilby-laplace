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
    resample : str
        Resampling method: ``'rejection'`` (default) or ``'importance'``.
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

        nsamples = 0
        target_nsamples = self.kwargs["target_nsamples"]
        batch_nsamples = self.kwargs["batch_nsamples"]
        resample = self.kwargs["resample"]
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
            if len(samples) > self.ndim:
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
