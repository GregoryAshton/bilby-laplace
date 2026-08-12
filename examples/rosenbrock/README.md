# rosenbrock — sampler comparison

Posteriors from 4 samplers (`rosenbrock_dynesty`, `rosenbrock_inprior`, `rosenbrock_rejection`, `rosenbrock_smc`) on the rosenbrock example, each compared against the `rosenbrock_dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make laplace`, `make rejection`, `make smc`, `make dynesty`.

## Comparison

`JSD` is the mean over sampled parameters of the Jensen-Shannon divergence of each 1-D marginal from `rosenbrock_dynesty`, in millibits, evaluated at a fixed 2000 samples per side (the same count in every example, so values are comparable across them). The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

**Noise floor: at most 0.73 mbits.** That is `rosenbrock_dynesty` against itself, two disjoint 1201-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. The split needs twice its size, and this reference has too few samples for 2000; the floor falls with N, so the true figure at N=2000 is lower than the one quoted. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

| method | log Z | ± | n_like | efficiency | time | JSD (mbits) | worst parameter |
|---|---|---|---|---|---|---|---|
| `rosenbrock_dynesty` | -4.13 | 0.07 | 212241 | 1.7% | 12.7s | — | — |
| `rosenbrock_inprior` | nan | nan | 5000 | 100.0% | 0.7s | 44.59 | y (60.75) |
| `rosenbrock_rejection` | -4.09 | 0.01 | 202000 | 2.5% | 3.0s | 1.50 | x (1.69) |
| `rosenbrock_smc` | -4.16 | 0.03 | 18288897 | 0.027% | 143.2s | 1.13 | y (1.18) |

