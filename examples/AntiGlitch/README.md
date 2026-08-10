# AntiGlitch — sampler comparison

Posteriors from 4 samplers (`antiglitch_dynesty`, `antiglitch_inprior`, `antiglitch_rejection`, `antiglitch_smc`) on the AntiGlitch example, each compared against the `antiglitch_dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make laplace`, `make rejection`, `make smc`, `make dynesty`.

## Comparison

`JSD` is the mean over sampled parameters of the Jensen-Shannon divergence of each 1-D marginal from `antiglitch_dynesty`, in millibits, evaluated at a fixed 2000 samples per side (the same count in every example, so values are comparable across them). The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

**Noise floor: at most 1.30 mbits.** That is `antiglitch_dynesty` against itself, two disjoint 1619-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. The split needs twice its size, and this reference has too few samples for 2000; the floor falls with N, so the true figure at N=2000 is lower than the one quoted. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

| method | log Z | ± | n_like | efficiency | time | JSD (mbits) | worst parameter |
|---|---|---|---|---|---|---|---|
| `antiglitch_dynesty` | -1989.62 | 0.12 | 1638270 | 0.3% | 65.3s | — | — |
| `antiglitch_inprior` | nan | nan | 5000 | 100.0% | 2.4s | 1.79 | A (2.56) |
| `antiglitch_rejection` | -1989.59 | 0.00 | 180000 | 2.9% | 6.4s | 1.56 | f (1.89) |
| `antiglitch_smc` | -1989.58 | 0.01 | 520000 | 1.0% | 23.9s | 1.99 | f (3.24) |

