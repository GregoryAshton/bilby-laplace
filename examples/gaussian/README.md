# gaussian — sampler comparison

Posteriors from 5 samplers (`dynesty`, `laplace`, `rejection`, `rejection_user`, `smc`) on the gaussian example, each compared against the `dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make laplace`, `make rejection`, `make rejection-user`, `make smc`, `make dynesty`.

## Comparison

`JSD` is the mean over sampled parameters of the Jensen-Shannon divergence of each 1-D marginal from `dynesty`, in millibits, evaluated at a fixed 2000 samples per side (the same count in every example, so values are comparable across them). The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

**Noise floor: at most 2.17 mbits.** That is `dynesty` against itself, two disjoint 1363-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. The split needs twice its size, and this reference has too few samples for 2000; the floor falls with N, so the true figure at N=2000 is lower than the one quoted. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

| method | log Z | ± | n_like | efficiency | time | JSD (mbits) | worst parameter |
|---|---|---|---|---|---|---|---|
| `dynesty` | -4.45 | 0.08 | 113387 | 3.4% | 8.7s | — | — |
| `laplace` | -4.61 | nan | 5000 | 100.0% | 2.4s | 0.99 | y (1.09) |
| `rejection` | -4.60 | 0.00 | 61000 | 8.3% | 2.6s | 1.70 | x (2.14) |
| `rejection_user` | -4.60 | 0.00 | 61000 | 8.3% | 1.9s | 1.58 | x (2.25) |
| `smc` | -4.61 | 0.01 | 45000 | 11.1% | 17.6s | 1.41 | x (1.47) |

