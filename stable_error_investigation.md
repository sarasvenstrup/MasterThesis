# Investigation: Why Do dim2/dim3 Stable Models Have Big Errors?

## TL;DR

There are **two completely different failure modes** — one is an ODE solver crash, the other is regime mismatch. Both are caused by OOS latent vectors `z` extrapolating outside the training manifold, but the _consequences_ differ.

---

## Failure 1 — dim2_stable 2016-07-01 (ODE crash + 1812 bps)

### What happened
- `n_test_bad = 8` — 8 observations produced NaN/Inf from the ODE solver and were **excluded**
- The remaining 46 "good" rows still had **mean RMSE of 1475 bps** and **max RMSE of 6159 bps**
- So the ODE didn't crash completely, it solved but gave absurd results

### What the predictions look like
| Date | CCY | actual `r₀` | fitted `r₀` | row RMSE |
|------|-----|------------|------------|----------|
| 2016-11-30 | EUR | −0.0020 (−0.2%) | **−0.929 (−93%!!)** | 6159 bps |
| 2016-12-30 | EUR | −0.0020 | −0.910 | 5995 bps |
| 2016-10-31 | JPY | −0.0009 | **−0.910 (−91%)** | 5221 bps |

The model predicts short rates of −90% when actual rates are near 0%. The ODE is computing physically impossible swap rates.

### Root cause
Training z range: `z_1 ∈ [−0.073, +0.001]`, `z_2 ∈ [−0.026, +0.210]`

The 2016-H2 test data encodes to `z` values **outside** this range (BoJ NIRP Feb 2016 created unusual rate dynamics). These OOS `z` values feed into sigma/r_tilde decoder layers that output extreme parameter values (e.g. σ pushing toward 0 or blowing up), which make the Riccati/affine ODE either:
- **Numerically stiff** → solver gives up → NaN (the 8 bad rows)
- **Numerically "solved" but divergent** → rates of −90% (the 46 "good" rows)

**The key enabler**: dim2_stable has λ₁ ≈ −0.001 to −0.049 (near-zero). With almost no mean-reversion, the ODE integrand accumulates over `[0, T]` without damping, and any extreme parameter values get fully amplified. A larger mean-reversion speed (more negative λ) would damp this.

---

## Failure 2 — dim2_stable 2022-07-01 (266 bps) and dim3_stable 2022-07-01 (146 bps)

### What happened
- `n_test_bad = 0` — ODE solved successfully for all 48 test rows
- But predictions are wildly wrong for USD, AUD, CAD

### What the predictions look like

| Model | CCY | actual `r₀` range (2022-H2) | fitted `r₀` range | RMSE |
|-------|-----|-------|------|------|
| dim2_stable | USD | 3.4%–5.1% | **32%–57.7%** | 1648 bps |
| dim2_stable | CAD | 3.7%–4.9% | 3.2%–8.7% | 164 bps |
| dim3_stable | USD | 3.4%–5.1% | **−22% to +3.5%** | 572 bps |
| dim3_stable | AUD | 3.1%–4.1% | **−7.5% to +3.1%** | 280 bps |
| **dim4_stable** | **USD** | **3.4%–5.1%** | **3.0%–4.6%** | **44 bps** ✓ |
| **dim4_stable** | **CAD** | **3.7%–4.9%** | **3.1%–4.2%** | **41 bps** ✓ |

Strikingly, dim2 and dim3 go **in opposite directions** — dim2_stable over-predicts USD rates (57.7%!), while dim3_stable under-predicts/inverts them (−22%). Both wrong, different manifestations of the same problem.

### Root cause: z extrapolation + decoder mapping

The 2022-H2 rate hiking cycle was the fastest in 40 years. The test-period `z` encodings lie **far outside the training manifold** (training was 2017–2022H1, a low-rate environment).

Looking at the end-of-training parameters (2022-06-30):
- **dim2_stable**: `σ₁` for AUD had fallen to ~0.107 (from ~0.42 in 2017), `σ₂` collapsed to ~0.002. The decoder has learned extremely small sigmas by end of training as rates started rising in H1 2022. When the OOS encoder sees the even larger H2 rate shock, it extrapolates to `z` values that produce impossible sigma/r_tilde combos (e.g. σ blowing up → r_tilde of 57%).
- **dim3_stable**: End-of-training parameters show `σ₁ ≈ 0.71`, `σ₂ ≈ 0.66`, `σ₃ ≈ 1.74`. The 3-factor model has very large sigma values and the z-to-parameter mapping extrapolates differently — OOS `z` drives the fitted rate **negative** instead of positive.

**dim4_stable succeeds** because the 4th degree of freedom provides a direction in latent space that can absorb the 2022 rate shift gracefully. The r_tilde predictions stay in `[3.0%, 4.6%]` for USD — close to the actual `[3.4%, 5.1%]`. The extra factor acts as a "regime absorption" mechanism that 2- and 3-factor models simply lack.

---

## Summary

| Failure | Window | Root Cause | Mechanism |
|---------|--------|-----------|-----------|
| ODE crash | dim2_stable 2016-07 | Near-zero λ₁ + OOS z extremes | Numerically stiff/divergent ODE |
| Regime mismatch (over) | dim2_stable 2022-07 | 2 factors insufficient + σ collapse | z extrapolation → r_tilde blows up |
| Regime mismatch (under) | dim3_stable 2022-07 | 3 factors insufficient | z extrapolation → rates go negative |
| **No failure** | **dim4_stable 2022-07** | 4 factors sufficient | 4th factor absorbs rate-hike regime |

**The fix**: Force a minimum eigenvalue gap (λ ≤ −0.1) to avoid near-unit-root dynamics. This would help Failure 1 but not Failure 2 — the only real fix for 2022 is more factors (dim4).

