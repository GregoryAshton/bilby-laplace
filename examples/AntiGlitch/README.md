# AntiGlitch — sampler comparison

Posteriors from 5 samplers (`antiglitch_aspire`, `antiglitch_dynesty`, `antiglitch_inprior`, `antiglitch_rejection`, `antiglitch_smc`) on the AntiGlitch example, each compared against the `antiglitch_dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make laplace`, `make rejection`, `make smc`, `make aspire`, `make dynesty`.

## Comparison

Both agreement columns compare each 1-D marginal with `antiglitch_dynesty` at a fixed 2000 samples per side (the same count in every example, so values are comparable across them), and report the mean over sampled parameters alongside the single parameter that scores worst. The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

`JSD` is the Jensen-Shannon divergence in millibits, which is dominated by width and shape mismatch and saturates once two densities barely overlap. `EMD` is the earth-mover distance divided by the reference posterior's standard deviation for that parameter, so it reads as a displacement: a posterior shifted bodily by one reference sigma scores 1.0, and unlike the JSD it keeps growing once the overlap is gone.

**Noise floor: at most 1.39 mbits, 0.04 sigma.** That is `antiglitch_dynesty` against itself, two disjoint 1563-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. The split needs twice its size, and this reference has too few samples for 2000; the floor falls with N, so the true figure at N=2000 is lower than the one quoted. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

`Mevals` is millions of likelihood evaluations and `settings` names the few sampler settings that set a run's cost: the SMC cloud size and mutation length for anything running aspire, the live-point count for dynesty, and nothing for the methods that draw straight from the Laplace proposal.

| method | log Z | ± | Mevals | effic. | time | JSD (mbits) | JSD worst | EMD (σ) | EMD worst | settings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `antiglitch_aspire` | -1989.50 | 0.06 | 0.128 | 1.6% | 36.8s | 1.88 | f (2.46) | 0.05 | f (0.06) | `nsamples=2000, nsteps=5` |
| `antiglitch_dynesty` | -1989.68 | 0.12 | 1.63 | 0.3% | 7.8m | — | — | — | — | `nlive=1000` |
| `antiglitch_inprior` | — | — | 0.002 | 100.0% | 3.3s | 1.11 | A (1.29) | 0.03 | log_gamma (0.04) | — |
| `antiglitch_rejection` | -1989.59 | 0.00 | 0.05 | 4.5% | 16.8s | 1.20 | A (1.35) | 0.04 | log_gamma (0.06) | — |
| `antiglitch_smc` | -1989.55 | 0.02 | 0.018 | 11.1% | 16.6s | 1.07 | A (1.39) | 0.04 | A (0.06) | `nsamples=2000, nsteps=5` |

