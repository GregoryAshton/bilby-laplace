# BNS_3G — sampler comparison

Posteriors from 4 samplers (`rb-aspire`, `rb-dynesty`, `rb-inprior`, `rb-smc`) on the BNS_3G example, each compared against the `rb-dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make std-all`, `make rb-all`, `make std-compare`, `make rb-compare`, `make validate`.

## Comparison

Both agreement columns compare each 1-D marginal with `rb-dynesty` at a fixed 2000 samples per side (the same count in every example, so values are comparable across them), and report the mean over sampled parameters alongside the single parameter that scores worst. The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

`JSD` is the Jensen-Shannon divergence in millibits, which is dominated by width and shape mismatch and saturates once two densities barely overlap. `EMD` is the earth-mover distance divided by the reference posterior's standard deviation for that parameter, so it reads as a displacement: a posterior shifted bodily by one reference sigma scores 1.0, and unlike the JSD it keeps growing once the overlap is gone.

**Noise floor: 1.35 mbits, 0.04 sigma.** That is `rb-dynesty` against itself, two disjoint 2000-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

`Mevals` is millions of likelihood evaluations and `settings` names the few sampler settings that set a run's cost: the SMC cloud size and mutation length for anything running aspire, the live-point count for dynesty, and nothing for the methods that draw straight from the Laplace proposal.

| method | log Z | ± | Mevals | effic. | time | JSD (mbits) | JSD worst | EMD (σ) | EMD worst | settings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rb-aspire` | -181182.38 | 0.08 | 18.9 | 0.053% | 5.7m | 1.68 | chi_1 (2.63) | 0.05 | mass_ratio (0.08) | `nsamples=2500, nsteps=300` |
| `rb-dynesty` | -181182.19 | 0.22 | 42.9 | 0.016% | 25.8m | — | — | — | — | `nlive=1000` |
| `rb-inprior` | — | — | 0.005 | 99.4% | 23.2s | 128.95 | lambda_1 (375.03) | 1.04 | lambda_1 (3.66) | — |
| `rb-smc` | -181182.70 | 0.06 | 10.5 | 0.095% | 4.2m | 1.73 | mass_ratio (2.95) | 0.04 | psi (0.06) | `nsamples=2500, nsteps=300` |

