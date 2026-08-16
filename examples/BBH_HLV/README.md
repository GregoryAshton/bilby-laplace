# BBH_HLV — sampler comparison

Posteriors from 4 samplers (`hlv_aspire`, `hlv_dynesty`, `hlv_inprior`, `hlv_smc`) on the BBH_HLV example, each compared against the `hlv_dynesty` reference.

## Running it

```
make all       # every sampler in turn, then this comparison
make compare   # rebuild this table and the corner plot from existing results
```

Individual samplers: `make laplace`, `make rejection`, `make smc`, `make smc-direct`, `make dynesty`.

## Comparison

Both agreement columns compare each 1-D marginal with `hlv_dynesty` at a fixed 2000 samples per side (the same count in every example, so values are comparable across them), and report the mean over sampled parameters alongside the single parameter that scores worst. The reference is not ground truth -- a small value means agreement with dynesty, not correctness.

`JSD` is the Jensen-Shannon divergence in millibits, which is dominated by width and shape mismatch and saturates once two densities barely overlap. `EMD` is the earth-mover distance divided by the reference posterior's standard deviation for that parameter, so it reads as a displacement: a posterior shifted bodily by one reference sigma scores 1.0, and unlike the JSD it keeps growing once the overlap is gone.

**Noise floor: 0.96 mbits, 0.04 sigma.** That is `hlv_dynesty` against itself, two disjoint 2000-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

`Mevals` is millions of likelihood evaluations and `settings` names the few sampler settings that set a run's cost: the SMC cloud size and mutation length for anything running aspire, the live-point count for dynesty, and nothing for the methods that draw straight from the Laplace proposal.

| method | log Z | ± | Mevals | effic. | time | JSD (mbits) | JSD worst | EMD (σ) | EMD worst | settings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hlv_aspire` | -12118.66 | 0.03 | 48.3 | 0.021% | 3.5h | 5.20 | tilt_1 (15.81) | 0.11 | tilt_1 (0.29) | `nsamples=10000, nsteps=300` |
| `hlv_dynesty` | -12118.15 | 0.19 | 42.5 | 0.017% | 10.4h | — | — | — | — | `nlive=1000` |
| `hlv_inprior` | — | — | 0.005 | 100.0% | 10.1s | 97.20 | delta_phase (479.52) | 0.39 | phi_12 (0.70) | — |
| `hlv_smc` | -12118.41 | 0.03 | 29.2 | 0.034% | 1.9h | 3.51 | tilt_1 (12.30) | 0.07 | a_1 (0.15) | `nsamples=10000, nsteps=300` |

