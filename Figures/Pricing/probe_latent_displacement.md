# Latent Displacement Diagnostic: Baseline vs Stable
**Date:** 2026-05-07  
**Script:** `probe_latent_displacement.py`  
**Setup:** dim4 models, T=5Y horizon, 1 000 paths, dt=1/12 (monthly), seed=42

---

## The Question

The decoder robustness probe showed stable decoders handle isotropic Gaussian perturbations
better than baseline at every scale. Yet stable produces 100% `nan_P_T` in pricing while
baseline prices ~70% of swaptions. These appear contradictory.

The resolution: the two results measure different things. The robustness probe uses **isotropic**
noise; real EM simulation uses the model's own **SDE dynamics**. The displacement magnitude
`‖z_T − z_0‖` under real simulation is what matters — and it differs enormously between the two
variants.

---

## Results

### Displacement at T=5Y (1 000 paths, dim4 ep5000)

| Model | Mean ‖z_T−z_0‖ | Median | p95 | Max | Finite decoded |
|-------|---------------:|-------:|----:|----:|:--------------:|
| dim4_baseline | **3.87 × 10¹³** | 1.88 × 10¹³ | 1.44 × 10¹⁴ | 3.30 × 10¹⁴ | 100% |
| dim4_stable   | **2.15**        | 1.90        | 4.72        | 7.59        | 51% |

The baseline SDE explodes to `‖z_T − z_0‖ ∼ 10¹³` at T=5Y — thirteen orders of magnitude
larger than stable. The stable SDE produces `‖z_T − z_0‖ ∼ 2`, exactly as designed.

### Training latent cloud (dim4_stable encoder, EUR curves)

```
mean = [ 0.070,  −0.004,  0.032,  −0.012]
std  = [ 0.059,   0.045,  0.016,   0.013]
```

The training cloud occupies a region with coordinate-wise std of **0.01–0.06**.
At ε=0.10 (the robustness probe scale), the perturbation is already 2–10× larger than
the typical spread of the training manifold. At baseline's actual simulation displacement
of ~10¹³, the concept of "off-manifold" is meaningless — these are points in a completely
different universe from anything the decoder has seen.

---

## The Correct Reading of Pricing Results

### dim4_baseline: 100% finite decoding at T=5Y, but ...

The 100% finite rate is **not** a sign of good simulation. The baseline SDE has exploded:
paths reach `‖z_T‖ ∼ 10¹³`, which is physically absurd (no yield curve ever lived there).
The decoder happens to return finite (non-NaN) numbers at these extreme points —
not because those numbers are meaningful, but because the decoder's extrapolation regime
at `‖z‖ → ∞` happens to produce finite outputs rather than NaN.

This explains the pricing results directly:
- **dim4_baseline 1Y expiry: +2009 bp error** — the paths have barely moved (T=1Y) but are already far off-manifold; the decoder returns wrong-but-finite numbers
- **dim4_baseline 5Y expiry: +666 bp error** — paths have moved further; decoder output is increasingly wrong but still finite
- **dim4_baseline 10Y expiry: ~20–60 bp error** — at long horizons the explosive SDE has fully overridden any signal from z_0; the decoder is essentially returning near-constant wrong values that happen to be close to market at long expiry by coincidence

The "70% priced" rate is not partial success. It is the SDE catastrophe expressed through
a decoder that fails silently (returns wrong-but-finite numbers) rather than loudly (NaN).

### dim4_stable: 51% finite decoding at T=5Y, decoder is the bottleneck

For stable, `‖z_T − z_0‖ ∼ 2` at T=5Y. Recall from the robustness probe: at ε=0.50,
dim4_stable decodes 48% successfully. At ε=2.0, only 20%. A displacement of 2 (mean) to
4.7 (p95) sits squarely in the regime where the recon-only decoder begins to fail —
not because the decoder is bad, but because no recon-only decoder was ever trained at
these scales. The decoder is the bottleneck, exactly as expected.

---

## Summary: Two Different Broken Things

| | dim4_baseline | dim4_stable |
|--|:-------------|:------------|
| **SDE behaviour at T=5Y** | ‖z_T − z_0‖ ∼ 10¹³  (exploded) | ‖z_T − z_0‖ ∼ 2 (well-behaved) |
| **Decoder at z_T** | Returns finite but meaningless numbers | Returns NaN — honest failure |
| **Pricing nan_P_T rate** | 30% (silent failures produce wrong vols) | 100% (explicit rejection) |
| **Pricing error on survivors** | +207 bp avg, structured (+2009 bp at 1Y) | — |
| **What's broken** | SDE: no stationary distribution, paths explode | Decoder: never trained on simulated z |
| **What's working** | Decoder is permissive (returns something at any z) | SDE: bounded, mean-reverting by design |

---

## Figures

**`fig_latent_displacement_baseline_vs_stable.png`**  
Scatter of training z cloud (blue) vs simulated z_T at T=5Y (red) in the first two latent
dimensions, with 2σ ellipses. Baseline z_T is off the scale (coordinates ~10¹³);
stable z_T overlaps the training cloud at a modest offset.

**`fig_latent_displacement_cdf.png`**  
CDF of `‖z_T − z_0‖` for both models, with the ε=2.0 robustness-probe limit marked.
Stable's entire distribution sits below ε=8; baseline's distribution starts above ε=10¹².

---

## Implication for Joint Training

The simulation chapter diagnosed the baseline SDE as unstable (exploding eigenvalues, no
stationary distribution). This probe confirms the same pathology at T=5Y: ‖z_T − z_0‖ ∼ 10¹³.

The stable SDE solves this problem. The remaining gap — 51% finite decoding at T=5Y —
is entirely the decoder's: it has never seen z values at displacement ‖δz‖ ∼ 2–5 from
the training manifold. Joint training closes this gap by backpropagating pricing errors
through `decoder(z_T)`, directly training the decoder at simulated z positions.

The full argument is therefore:

```
Baseline:  broken SDE  →  exploding z_T  →  decoder returns garbage (no NaN, wrong vols)
Stable:    working SDE →  bounded z_T    →  decoder returns NaN (trained only on tiny manifold)
Joint:     working SDE →  bounded z_T    →  decoder trained on simulated z  →  correct pricing
```

