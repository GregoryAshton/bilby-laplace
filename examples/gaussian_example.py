"""
Comparison example: Laplace vs dynesty on a 2D Gaussian likelihood.

Usage
-----
    python examples/gaussian_example.py
"""

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

def main():
    likelihood = GaussianLikelihood()

    priors = bilby.core.prior.PriorDict(
        dict(
            x=bilby.core.prior.Uniform(-5, 5, "x"),
            y=bilby.core.prior.Uniform(-5, 5, "y"),
        )
    )

    injection_parameters = {"x": 1.0, "y": -0.5}

    # --- Laplace run ---
    result_laplace = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="laplace",
        injection_parameters=injection_parameters,
        outdir="outdir",
        label="gaussian_laplace",
        clean=True,
        resample="rejection",
        target_nsamples=5000,
        batch_nsamples=500,
        prior_nsamples=50,
        use_injection_for_maxL=True,
    )

    # --- Dynesty run ---
    result_dynesty = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        injection_parameters=injection_parameters,
        outdir="outdir",
        label="gaussian_dynesty",
        clean=True,
        nlive=500,
    )

    # --- Comparison plot ---
    bilby.core.result.plot_multiple(
        [result_laplace, result_dynesty],
        labels=["Laplace", "Dynesty"],
        filename="outdir/gaussian_comparison_corner.png",
        titles=False,
    )

    # --- Summary ---
    for label, result in [("Laplace", result_laplace), ("Dynesty", result_dynesty)]:
        print(f"\n=== {label} ===")
        print(result.posterior[["x", "y"]].describe())
        if "run_statistics" in result.meta_data:
            print("run_statistics:", result.meta_data["run_statistics"])


if __name__ == "__main__":
    main()
