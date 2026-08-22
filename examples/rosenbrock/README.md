# rosenbrock — sampler comparison

Posteriors from 6 samplers (`rosenbrock_aspire`, `rosenbrock_dynesty`, `rosenbrock_emcee`, `rosenbrock_inprior`, `rosenbrock_rejection`, `rosenbrock_smc`) on the rosenbrock example, each compared against the `rosenbrock_dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make laplace`, `make rejection`, `make smc`, `make aspire`, `make emcee`, `make dynesty`.

## Comparison

Both agreement columns compare each 1-D marginal with `rosenbrock_dynesty` at a fixed 2000 samples per side (the same count in every example, so values are comparable across them), and report the mean over sampled parameters alongside the single parameter that scores worst. The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

`JSD` is the Jensen-Shannon divergence in millibits, which is dominated by width and shape mismatch and saturates once two densities barely overlap. `EMD` is the earth-mover distance divided by the reference posterior's standard deviation for that parameter, so it reads as a displacement: a posterior shifted bodily by one reference sigma scores 1.0, and unlike the JSD it keeps growing once the overlap is gone.

**Noise floor: at most 0.73 mbits, 0.03 sigma.** That is `rosenbrock_dynesty` against itself, two disjoint 1201-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. The split needs twice its size, and this reference has too few samples for 2000; the floor falls with N, so the true figure at N=2000 is lower than the one quoted. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

`Mevals` is millions of likelihood evaluations and `settings` names the few sampler settings that set a run's cost: the SMC cloud size and mutation length for anything running aspire, the live-point count for dynesty, and nothing for the methods that draw straight from the Laplace proposal.

| method | log Z | ± | Mevals | effic. | time | JSD (mbits) | JSD worst | EMD (σ) | EMD worst | settings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rosenbrock_aspire` | -4.06 | 0.03 | 20 | 0.025% | 1.2m | 2.26 | x (2.91) | 0.07 | x (0.07) | `nsamples=5000, nsteps=1000` |
| `rosenbrock_dynesty` | -4.13 | 0.07 | 0.212 | 1.7% | 25.1s | — | — | — | — | `nlive=1000` |
| `rosenbrock_emcee` | — | — | 0.76 | 0.3% | 1.0m | 2.32 | x (2.68) | 0.07 | y (0.08) | — |
| `rosenbrock_inprior` | — | — | 0.005 | 100.0% | 3.4s | 43.15 | y (59.23) | 0.28 | x (0.32) | — |
| `rosenbrock_rejection` | -4.09 | 0.01 | 0.202 | 2.5% | 4.7s | 1.49 | x (2.05) | 0.05 | x (0.06) | — |
| `rosenbrock_smc` | -4.12 | 0.03 | 18.3 | 0.027% | 1.4m | 1.43 | x (1.73) | 0.04 | y (0.04) | `nsamples=5000, nsteps=1000` |

