# OOS Rolling Results: Stable vs Baseline — Full Analysis

_Auto-generated: 2026-05-03 10:38 — re-run `python Code/gen_oos_analysis.py` to refresh._

## 1. Summary Table (5Y train / 6M test, 3500 epochs)

| Model | Epochs | N windows | Mean OOS (bps) | Mean IS (bps) | Max OOS (bps) |
|-------|--------|-----------|----------------|---------------|---------------|
| dim2 baseline | 3500 | 18 | **19.0** | 10.5 | 77.0 |
| dim2 stable | 3500 | 18 | 127.6 | 138.7 | **1811.7** |
| dim3 baseline | 3500 | 18 | 66.5 | 11.6 | **632.6** |
| dim3 stable | 3500 | 18 | **21.6** | 10.5 | 145.7 |
| dim4 baseline | 3500 | 18 | 100.5 | 9.3 | **691.8** |
| dim4 stable | 3500 | 18 | **12.1** | 8.1 | 24.2 |

**Bottom line:** Stable wins at dim3, dim4 (mean OOS 12.1 bps for dim4 stable). Baseline edges out stable at dim2 — largely driven by ODE crash(es) inflating the stable mean.

---

## 2. NaN Currency RMSE — What Is It?

The `NaN` entries in per-currency RMSE columns (GBP from 2022-07-01, USD from 2023-07-01 onward) are **not model failures**. They simply mean that currency has **no observations in that test window** (data ends earlier in the dataset). The `avg_rmse_bps` is computed as `nanmean` across available currencies and is unaffected.

`n_test_bad` is the more important metric — it counts rows where the ODE integration produced NaN/Inf and those rows were excluded from RMSE.

---

## 3. Big Errors — Two Distinct Mechanisms

### Mechanism A: ODE Numerical Divergence (`n_test_bad > 0`)
The ODE integrator itself crashes for certain test observations, producing NaN/Inf outputs which are excluded. This inflates `avg_rmse_bps` even with fewer 'good' rows.

| Window | Model | `n_test_bad` | avg OOS | Worst currencies |
|--------|-------|--------------|---------|-----------------|
| 2016-07-01 | dim2_stable | **8** | 1812 bps | EUR=5096, JPY=2845, SEK=2650 |
| 2016-01-01 | dim3_baseline | **5** | 633 bps | JPY=3791, EUR=1063, SEK=790 |
| 2022-07-01 | dim4_baseline | **10** | 395 bps | NOK=1485, AUD=1086, USD=502 |
| 2016-01-01 | dim4_baseline | **4** | 373 bps | JPY=2813, SEK=426, EUR=65 |

### Mechanism B: Finite but Wildly Wrong (`n_test_bad = 0`, huge RMSE)
The ODE completes but predicts completely wrong values.

| Window | Model | λ_max | avg OOS | Worst currencies |
|--------|-------|-------|---------|-----------------|
| 2022-01-01 | dim4_baseline | **+1.40** | 692 bps | CAD=2851, AUD=2339, USD=221 |
| 2022-07-01 | dim3_baseline | **+1.09** | 296 bps | USD=1194, CAD=1045, NOK=35 |
| 2022-07-01 | dim2_stable | −0.14 (stable) | 266 bps | USD=1648, NOK=166, CAD=164 |
| 2019-07-01 | dim4_baseline | **+1.39** | 182 bps | EUR=1456, DKK=105, JPY=16 |
| 2022-07-01 | dim3_stable | −0.18 (stable) | 146 bps | USD=572, AUD=280, CAD=199 |
| 2015-07-01 | dim2_baseline | **+2.13** | 77 bps | SEK=623, USD=12, GBP=11 |

**Baseline explosions** are driven by positive eigenvalues (λ_max > 0) — the drift matrix M is explosive, which generalises catastrophically OOS when latent `z` extrapolates outside the training manifold.

**Stable model failures** (all-negative eigenvalues) are pure **regime mismatch** — the model has too few factors to extrapolate through extreme market regimes (e.g. the 2022 rate-hike cycle).

---

## 4. Eigenvalue Patterns

### Baseline models — eigenvalues **unconstrained**:

- **dim2_baseline**: λ range [-2.86, 5.31]; 18/18 windows have at least one positive eigenvalue (explosive manifold)
- **dim3_baseline**: λ range [-4.76, 3.30]; 13/18 windows have at least one positive eigenvalue (explosive manifold)
- **dim4_baseline**: λ range [-3.58, 2.68]; 18/18 windows have at least one positive eigenvalue (explosive manifold)

### Stable models — eigenvalues all ≤ 0 by construction:

- **dim2_stable**: λ range [-14.039, -0.001]; 0/18 windows with λ₁ > −0.05 (near unit-root); failure windows: 2016-07-01, 2022-07-01
- **dim3_stable**: λ range [-8.647, -0.004]; 0/18 windows with λ₁ > −0.05 (near unit-root); failure windows: 2022-07-01
- **dim4_stable**: λ range [-17.753, -0.002]; 0/18 windows with λ₁ > −0.05 (near unit-root); no failure windows

**Near-zero eigenvalue issue**: When λ ≈ 0, the process is essentially a random walk, making it sensitive to out-of-distribution latent `z` values. This can cause ODE solver failures (dim2_stable 2016-07-01) or inflated errors. Higher-dimensional stable models appear more robust because the additional fast-mean-reverting factors stabilise the ODE numerically.

---

## 5. All Windows — Per-Model RMSE (bps)

| Window | dim2 baseline | dim2 stable | dim3 baseline | dim3 stable | dim4 baseline | dim4 stable |
|--------|-------|-------|-------|-------|-------|-------|
| 2015-01-01 | 10.2 | 19.5 | 9.8 | 11.8 | 12.7 | 12.4 |
| 2015-07-01 | 77.0 | 10.5 | 18.0 | 12.5 | 12.7 | 8.2 |
| 2016-01-01 | 21.9 | 18.4 | **633** ⚠️5 | 26.0 | **373** ⚠️4 | 24.2 |
| 2016-07-01 | 14.1 | **1812** ⚠️8 | 16.4 | 16.3 | 12.2 | 11.0 |
| 2017-01-01 | 8.6 | 7.6 | 10.6 | 10.7 | 7.3 | 7.5 |
| 2017-07-01 | 8.7 | 7.0 | 11.3 | 10.8 | 8.6 | 7.1 |
| 2018-01-01 | 9.0 | 28.3 | 18.7 | 13.6 | 12.6 | 9.9 |
| 2018-07-01 | 8.8 | 8.1 | 18.6 | 13.1 | 9.8 | 8.0 |
| 2019-01-01 | 11.8 | 11.3 | 17.9 | 11.9 | 10.3 | 7.0 |
| 2019-07-01 | 24.8 | 11.7 | 21.2 | 15.9 | **182** | 9.0 |
| 2020-01-01 | 13.4 | 12.4 | 13.3 | 12.7 | 9.9 | 11.1 |
| 2020-07-01 | 12.6 | 8.6 | 10.6 | 11.6 | 17.5 | 9.0 |
| 2021-01-01 | 17.0 | 9.3 | 20.9 | 16.2 | 14.2 | 9.0 |
| 2021-07-01 | 17.4 | 14.0 | 14.8 | 15.0 | 13.0 | 13.8 |
| 2022-01-01 | 18.6 | 19.5 | 22.0 | 19.8 | **692** | 17.6 |
| 2022-07-01 | 34.9 | **266** | **296** | **146** | **395** ⚠️10 | 23.7 |
| 2023-01-01 | 15.2 | 16.5 | 18.3 | 13.8 | 15.9 | 16.9 |
| 2023-07-01 | 18.1 | 15.9 | 26.9 | 12.0 | 10.9 | 13.1 |

Values > 100 bps shown in **bold**. ⚠️N means N ODE-crash rows excluded.

---

## 6. Key Takeaways

1. **dim4 stable is the best model** at 3500 epochs — mean OOS 12.1 bps, max 24.2 bps, n_bad = 0.
2. **dim2 baseline** is the runner-up — mean OOS 19.0 bps, max 77.0 bps.
3. **dim2**: baseline wins by 108.6 bps mean OOS (127.6 vs 19.0).
4. **dim3**: stable wins by 44.9 bps mean OOS (21.6 vs 66.5).
5. **dim4**: stable wins by 88.4 bps mean OOS (12.1 vs 100.5).
6. **Baseline positive eigenvalues** are the root cause of all Mechanism-B explosions.
7. **2022-H2 rate-hike cycle** is the hardest period — baseline models explode (300–700 bps), dim3_stable shows regime mismatch (~146 bps), dim4_stable handles it cleanly (~24 bps).
8. **Per-currency NaN values** (GBP from 2022-07, USD from 2023-07) are data availability, not model failures.

