# Examples

Runnable example scripts live in the [`examples/`](https://github.com/GregoryAshton/bilby-laplace/tree/main/examples)
directory. Each script supports multiple samplers and a `--compare` mode that prints
a summary table and a corner plot overlaying the methods.

Each example directory has a `Makefile`; run targets from within it:

```bash
cd examples/gaussian
make laplace          # raw Laplace (no resampling)
make rejection        # rejection resampling
make rejection-user   # rejection with a user-supplied sampling_cov
make smc              # SMC resampling
make dynesty          # nested-sampling reference
make compare          # load all results, print evidence table, plot
```

## Available examples

| Example | What it demonstrates |
|---|---|
| `examples/gaussian` | 2-D correlated Gaussian — the Laplace approximation is exact here, so it is the cleanest correctness check. Includes a `rejection-user` variant that feeds the known covariance via `sampling_cov`. |
| `examples/rosenbrock` | A strongly non-Gaussian, curved target — a stress test where `resample="smc"` matters. |
| `examples/BBH_HLV` | A simulated binary-black-hole injection in an H–L–V network (no data download needed). A realistic high-dimensional GW problem. |
| `examples/BNS_3G` | A binary-neutron-star injection in a 3rd-generation network. |

## A minimal script

```python
import bilby

likelihood = ...   # any bilby Likelihood
priors = ...       # a bilby PriorDict

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="laplace",
    resample="rejection",
    target_nsamples=5000,
    plot_diagnostic=True,
    outdir="outdir",
    label="example",
)
result.plot_corner()
```

See the [gaussian example](https://github.com/GregoryAshton/bilby-laplace/blob/main/examples/gaussian/run.py)
for a complete, self-contained script including the likelihood definition.
