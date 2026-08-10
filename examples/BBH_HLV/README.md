# BBH_HLV — sampler comparison

Posteriors from 4 samplers (`hlv_dynesty`, `hlv_inprior`, `hlv_smc`, `hlv_smcdirect`) on the BBH_HLV example, each compared against the `hlv_dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make laplace`, `make rejection`, `make smc`, `make smc-direct`, `make dynesty`.

## Comparison

`JSD` is the mean over sampled parameters of the Jensen-Shannon divergence of each 1-D marginal from `hlv_dynesty`, in millibits, evaluated at a fixed 2000 samples per side (the same count in every example, so values are comparable across them). The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

**Noise floor: 1.65 mbits.** That is `hlv_dynesty` against itself, two disjoint 2000-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

| method | log Z | ± | n_like | efficiency | time | JSD (mbits) | worst parameter |
|---|---|---|---|---|---|---|---|
| `hlv_dynesty` | -12118.09 | 0.19 | 43244293 | 0.016% | 12821.1s | — | — |
| `hlv_inprior` | nan | nan | 5000 | 99.9% | 12.9s | 71.65 | phase (246.84) |
| `hlv_smc` | -12118.58 | 0.03 | 10862648 | 0.092% | 2322.5s | 15.10 | a_1 (48.63) |
| `hlv_smcdirect` | -12118.67 | 0.03 | 16330000 | 0.061% | 3080.0s | 7.82 | phase (25.33) |

