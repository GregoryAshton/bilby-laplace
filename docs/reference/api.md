# Configuration & API reference

All sampler options are passed through `bilby.run_sampler(..., sampler="laplace", **kwargs)`.
The reference below is generated directly from the source docstrings, so it always
matches the installed version.

## The `Laplace` sampler

Every keyword argument and its default is documented here.

::: bilby_laplace.sampler.Laplace
    options:
      members: false
      show_root_heading: true
      show_root_full_path: false

## `LaplacePosteriorEstimator`

The engine behind the sampler: MAP finding, posterior precision/covariance
estimation, the Laplace evidence, and Gaussian sampling. Useful directly if you want
the covariance without running the full sampler.

::: bilby_laplace.laplace.LaplacePosteriorEstimator
    options:
      members:
        - get_MAP_sample
        - calculate_posterior_precision
        - calculate_posterior_covariance
        - log_evidence_laplace
        - sample_dataframe
        - log_likelihood_from_array
      show_root_heading: true
      show_root_full_path: false

## Waveform Fisher matrix

The gravitational-wave Fisher route (`fisher_method="waveform"`).

::: bilby_laplace.gw_fisher
    options:
      members:
        - waveform_fisher_matrix
        - is_gw_waveform_likelihood
        - validate_waveform_likelihood
      show_root_heading: false
