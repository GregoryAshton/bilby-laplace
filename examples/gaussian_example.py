"""
Minimal example: Fisher sampler on a 2D Gaussian likelihood.

Usage
-----
    python examples/gaussian_example.py
    python examples/gaussian_example.py --also-dynesty   # comparison run
"""

import argparse
import numpy as np
import bilby

# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

class GaussianLikelihood(bilby.core.likelihood.Likelihood):
    """2-D uncorrelated Gaussian likelihood centred at (mu_x, mu_y)."""

    def __init__(self, mu_x=1.0, mu_y=-0.5, sigma_x=0.3, sigma_y=0.5):
        super().__init__(parameters={"x": None, "y": None})
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def log_likelihood(self):
        x = self.parameters["x"]
        y = self.parameters["y"]
        return (
            -0.5 * ((x - self.mu_x) / self.sigma_x) ** 2
            - 0.5 * ((y - self.mu_y) / self.sigma_y) ** 2
            - np.log(2 * np.pi * self.sigma_x * self.sigma_y)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(also_dynesty: bool = False):
    likelihood = GaussianLikelihood()

    priors = bilby.core.prior.PriorDict(
        dict(
            x=bilby.core.prior.Uniform(-5, 5, "x"),
            y=bilby.core.prior.Uniform(-5, 5, "y"),
        )
    )

    injection_parameters = {"x": 1.0, "y": -0.5}

    # --- Fisher run ---
    result_fisher = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="fisher",
        injection_parameters=injection_parameters,
        outdir="outdir",
        label="gaussian_fisher",
        clean=True,
        # Fisher-specific kwargs:
        resample="rejection",
        target_nsamples=5000,
        batch_nsamples=500,
        prior_nsamples=50,
        use_injection_for_maxL=True,
    )

    print("\n=== Fisher result ===")
    print(f"Posterior shape : {result_fisher.posterior.shape}")
    print(result_fisher.posterior[["x", "y"]].describe())
    print("run_statistics  :", result_fisher.meta_data.get("run_statistics"))

    result_fisher.plot_corner(
        truths=injection_parameters,
        filename="outdir/gaussian_fisher_corner.png",
    )

    # --- Optional dynesty comparison ---
    if also_dynesty:
        result_dynesty = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            injection_parameters=injection_parameters,
            outdir="outdir",
            label="gaussian_dynesty",
            nlive=200,
            clean=True,
        )
        result_dynesty.plot_corner(
            truths=injection_parameters,
            filename="outdir/gaussian_dynesty_corner.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--also-dynesty",
        action="store_true",
        help="Also run dynesty for comparison",
    )
    args = parser.parse_args()
    main(also_dynesty=args.also_dynesty)
