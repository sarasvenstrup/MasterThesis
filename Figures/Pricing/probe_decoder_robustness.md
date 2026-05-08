# Decoder Robustness Probe
**Date:** 2026-05-07  
**Script:** `probe_decoder_robustness.py`

## Setup

For each of the 6 ep5000 checkpoints, 200 EUR curves are encoded to get `z_0 = encoder(curve)`.
Each `z_0` is then perturbed as `z = z_0 + ε · η` where `η ~ N(0, I_d)`, with 50 independent
draws per curve (10 000 total test points per model per ε). The fraction of decoded discount
curves `P_full = decoder(z)` that are **fully finite** (no NaN or Inf in any entry) is recorded.

This probe is independent of SDE drift and diffusion — it measures only the decoder's ability
to handle inputs that lie off the encoded training manifold.

---

## Results: Fraction of Finite Decoded Curves

| Model | ε = 0.00 | ε = 0.05 | ε = 0.10 | ε = 0.20 | ε = 0.50 | ε = 1.00 | ε = 2.00 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| dim2_baseline | 1.000 | 0.794 | 0.574 | 0.401 | 0.310 | 0.220 | 0.132 |
| dim3_baseline | 1.000 | 0.661 | 0.440 | 0.267 | 0.096 | 0.038 | 0.015 |
| dim4_baseline | 1.000 | 0.946 | 0.809 | 0.538 | 0.180 | 0.076 | 0.043 |
| **dim2_stable** | 1.000 | **0.995** | **0.990** | **0.972** | **0.812** | **0.646** | **0.430** |
| dim3_stable | 1.000 | 0.852 | 0.618 | 0.394 | 0.265 | 0.203 | 0.134 |
| **dim4_stable** | 1.000 | **0.952** | **0.913** | **0.765** | **0.484** | **0.310** | **0.201** |

_All models decode on-manifold points perfectly (ε = 0.00). Bold = best in column._

---

## Key Observations

### 1. Stable decoders are more robust, not less

At every perturbation scale, the stable models decode a **higher fraction** of off-manifold
points successfully than their baseline counterparts of the same dimension:

| Dim | Baseline @ ε=0.10 | Stable @ ε=0.10 | Stable advantage |
|-----|:-----------------:|:---------------:|:----------------:|
| 2 | 57.4% | **99.0%** | +41.6 pp |
| 3 | 44.0% | **61.8%** | +17.8 pp |
| 4 | 80.9% | **91.3%** | +10.4 pp |

The hypothesis that stable's tighter encoded manifold makes its decoder more brittle is
**refuted by this probe**. Stable decoders generalise better to off-manifold inputs at
every scale tested.

### 2. All decoders fail at large displacements

No model maintains reliable decoding beyond ε ≈ 0.5. At ε = 2.0, even the most robust
model (dim2_stable) only decodes 43% of points successfully. This is the root cause of
100% `nan_P_T` in swaption pricing: Euler-Maruyama simulation over 5Y–10Y horizons
routinely produces `‖z_T − z_0‖ ≫ 2.0`, far beyond any recon-only decoder's range.

### 3. dim2_stable is an outlier in robustness

dim2_stable maintains 99% finite decoding at ε=0.05 and 97% at ε=0.20 — far better than
any other model. This likely reflects the 2D latent space being simple enough that the
decoder learns a smoother, more globally valid mapping. The trade-off is coarser
reconstruction (9.2 bps avg vs 5.3 bps for dim4_stable).

### 4. Baseline survival in pricing is not due to decoder quality

dim4_baseline prices ~70% of swaptions yet has a worse decoder than dim4_stable at every ε.
The explanation is that the baseline SDE produces smaller `‖z_T − z_0‖` displacements in
practice — not that its decoder is better. The baseline SDE is not theoretically stable;
it happens to be empirically less explosive on the 2010–2024 EUR test dates.

---

## Implication

The pricing failure of recon-only stable models is **not a decoder quality problem**.
The stable decoder is superior. The failure is a displacement magnitude problem:
real SDE simulation generates latent displacements that exceed the robustness range
of any recon-only decoder. Joint training resolves this by backpropagating pricing errors
through `decoder(z_T)` — directly placing simulated latent states into the decoder's
training distribution.

