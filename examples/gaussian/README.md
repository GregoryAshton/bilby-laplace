# gaussian — sampler comparison

Posteriors from 6 samplers (`aspire`, `dynesty`, `laplace`, `rejection`, `rejection_user`, `smc`) on the gaussian example, each compared against the `dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make laplace`, `make rejection`, `make rejection-user`, `make smc`, `make aspire`, `make dynesty`.

## Comparison

Both agreement columns compare each 1-D marginal with `dynesty` at a fixed 2000 samples per side (the same count in every example, so values are comparable across them), and report the mean over sampled parameters alongside the single parameter that scores worst. The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

`JSD` is the Jensen-Shannon divergence in millibits, which is dominated by width and shape mismatch and saturates once two densities barely overlap. `EMD` is the earth-mover distance divided by the reference posterior's standard deviation for that parameter, so it reads as a displacement: a posterior shifted bodily by one reference sigma scores 1.0, and unlike the JSD it keeps growing once the overlap is gone.

**Noise floor: at most 2.82 mbits, 0.07 sigma.** That is `dynesty` against itself, two disjoint 1319-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. The split needs twice its size, and this reference has too few samples for 2000; the floor falls with N, so the true figure at N=2000 is lower than the one quoted. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

`Mevals` is millions of likelihood evaluations and `settings` names the few sampler settings that set a run's cost: the SMC cloud size and mutation length for anything running aspire, the live-point count for dynesty, and nothing for the methods that draw straight from the Laplace proposal.

| method | log Z | ± | Mevals | effic. | time | JSD (mbits) | JSD worst | EMD (σ) | EMD worst | settings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `aspire` | -4.62 | 0.03 | 0.18 | 2.8% | 9.2s | 2.07 | y (2.34) | 0.04 | y (0.05) | `nsamples=5000, nsteps=5` |
| `dynesty` | -4.61 | 0.08 | 0.114 | 3.4% | 7.8s | — | — | — | — | `nlive=1000` |
| `laplace` | -4.61 | — | 0.005 | 100.0% | 1.0s | 0.92 | x (1.07) | 0.03 | x (0.03) | — |
| `rejection` | -4.61 | 0.01 | 0.027 | 18.8% | 1.3s | 2.63 | x (3.16) | 0.07 | x (0.07) | — |
| `rejection_user` | -4.61 | 0.01 | 0.027 | 18.8% | 1.1s | 2.47 | y (2.50) | 0.05 | y (0.06) | — |
| `smc` | -4.61 | 0.01 | 0.045 | 11.1% | 11.2s | 0.68 | y (0.71) | 0.03 | y (0.03) | `nsamples=5000, nsteps=5` |

