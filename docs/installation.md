# Installation

## From PyPI

```bash
pip install bilby-laplace
```

## From source

```bash
git clone https://github.com/GregoryAshton/bilby-laplace
cd bilby-laplace
pip install -e .
```

Installing in editable mode registers the `bilby.laplace` entry-point so that
`bilby.run_sampler(..., sampler="laplace")` works automatically — Bilby discovers
the sampler through its plugin system with no further configuration.

## Optional extras

The waveform-Fisher route (`fisher_method="waveform"`) and the SMC resampling
route (`resample="smc"`) rely on additional packages:

- **SMC** requires [aspire](https://github.com/bilby-dev/aspire) and `aspire_bilby`.
- **Waveform Fisher** requires a gravitational-wave likelihood, i.e. the `bilby.gw`
  stack (LAL waveforms, etc.).

To build the documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Requirements

- Python ≥ 3.9
- `scipy ≥ 1.15` (the Hessian is computed with `scipy.differentiate.hessian`)
- `bilby ≥ 2.7`

Versions are derived from annotated git tags via `setuptools-scm`; there is no
hardcoded version in `pyproject.toml`.
