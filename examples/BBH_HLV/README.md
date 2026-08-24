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

**Noise floor: 0.93 mbits, 0.03 sigma.** That is `hlv_dynesty` against itself, two disjoint 2000-sample draws from the one posterior. Two finite samples of the *same* distribution do not score zero, so anything at or below this level is consistent with perfect agreement, and differences among such values are not measurements. It is one split averaged over the sampled parameters, so read it as an order of magnitude, not a threshold.

`Mevals` is millions of likelihood evaluations and `settings` names the few sampler settings that set a run's cost: the SMC cloud size and mutation length for anything running aspire, the live-point count for dynesty, and nothing for the methods that draw straight from the Laplace proposal.

| method | log Z | ± | Mevals | effic. | time | JSD (mbits) | JSD worst | EMD (σ) | EMD worst | settings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hlv_aspire` | -12118.58 | 0.03 | 48.3 | 0.021% | 1.2h | 2.98 | tilt_1 (7.42) | 0.08 | tilt_1 (0.17) | `nsamples=10000, nsteps=300` |
| `hlv_dynesty` | -12118.37 | 0.19 | 44.6 | 0.016% | 1.6h | — | — | — | — | `nlive=1000` |
| `hlv_inprior` | — | — | 0.005 | 100.0% | 21.1s | 93.86 | delta_phase (416.48) | 0.42 | mass_ratio (0.81) | — |
| `hlv_smc` | -12118.43 | 0.03 | 29.2 | 0.034% | 52.0m | 4.59 | chirp_mass (12.65) | 0.10 | a_1 (0.19) | `nsamples=10000, nsteps=300` |
