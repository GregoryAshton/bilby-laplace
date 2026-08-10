# BNS_3G — sampler comparison

Posteriors from 3 samplers (`rb-dynesty`, `rb-smc`, `rb-smcdirect`) on the BNS_3G example, each compared against the `rb-dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make std-all`, `make rb-all`, `make std-compare`, `make rb-compare`, `make validate`.

## Comparison

`JSD` is the mean over sampled parameters of the Jensen-Shannon divergence of each 1-D marginal from `rb-dynesty`, in millibits, evaluated at a fixed 2000 samples per side (the same count in every example, so values are comparable across them). The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

**Noise floor: 0.99 mbits.** That is `rb-dynesty` against itself, two disjoint 2000-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

| method | log Z | ± | n_like | efficiency | time | JSD (mbits) | worst parameter |
|---|---|---|---|---|---|---|---|
| `rb-dynesty` | -181179.85 | 0.21 | 41464605 | 0.016% | 2094.2s | — | — |
| `rb-smc` | -181180.37 | 0.05 | 7425642 | 0.1% | 252.9s | 1.68 | zenith (3.40) |
| `rb-smcdirect` | -181180.77 | 0.06 | 11225000 | 0.089% | 309.8s | 7.52 | mass_ratio (27.06) |

