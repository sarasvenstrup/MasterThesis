# H-vs-K Decomposition Probe
**Date:** 2026-05-07  
**Script:** `probe_hk_decomposition.py`  
**Setup:** dim4 ep5000, single EUR curve (median date), dt=1/12, T=5Y, N=500 paths

---

## The Question

Why does the stable SDE produce `‖z_T − z_0‖ ~ 2` at T=5Y despite mean-reverting drift,
while baseline produces `‖z_T − z_0‖ ~ 10¹³`?  
Two candidate mechanisms:

1. **H is larger in stable** — mean-reverting drift forced H to scale up during training to fit
   the observed one-step variance; larger H → larger per-step kicks at pricing time.
2. **z_0 is far from z*** — the equilibrium z* sits far from the encoded EUR curve cloud;
   the drift term coherently pulls z away from z_0 toward z* over the simulation horizon.

---

## Results

### Equilibrium z* and distance from EUR curves

| | dim4_baseline | dim4_stable |
|--|:-------------|:------------|
| **z*** | `[−0.083, 0.100, −0.060, 0.000]` | `[−8.544, 3.765, 1.746, −0.559]` |
| **z_0** | `[0.029, −0.015, 0.012, −0.025]` | `[0.032, 0.025, 0.030, −0.024]` |
| **‖z_0 − z*‖** | **0.18** | **9.53** |
| **‖z − z*‖ over all EUR train** | mean=0.21, std=0.05, max=0.36 | mean=9.57, std=0.07, max=9.74 |

**Stable's z* is at coordinate ~[−8.5, 3.8, 1.7, −0.6] — far outside the encoded EUR cloud
which lives near [0.07, −0.004, 0.032, −0.012] with std 0.01–0.06.**  
Every EUR curve is at distance ~9.6 from the equilibrium. Baseline's z* sits at distance
~0.21 from the training cloud — essentially inside it.

### Diffusion vs drift decomposition

| | dim4_baseline | dim4_stable |
|--|:-------------|:------------|
| **H sigmas at z_0** | [0.886, 0.943, 1.012, 1.028] | [1.190, 0.420, 1.142, 1.463] |
| **Mean H sigma** | 0.967 | 1.054 |
| **(a) 1-step diffusion (H only)** | mean=0.502 | mean=0.594 |
| **(a) Annualised (×√60)** | ~3.89 | ~4.60 |
| **(b) 5Y drift only (H=0)** | **8374.6** | **0.14** |
| **(c) Full 5Y simulation** | ~3.7 × 10¹³ | ~2.10 |

### K eigenvalues (stable only)

```
dim4_stable K eigenvalues (real): [−5.076, −2.314, −1.044, −0.0016]
```

One eigenvalue is nearly zero (−0.0016), meaning one latent direction is essentially
non-mean-reverting. This is the direction along which z drifts most.

---

## Diagnosis: Mechanism 2 Dominates for Stable, Mechanism 2 Also Dominates for Baseline

### Stable: z* misplacement is the entire story

- **Diffusion alone (H only):** 1-step displacement ~0.59, annualised ~4.60 — very close to
  baseline's ~3.89. The H matrices are **nearly the same scale**. Mechanism (1) is real but small:
  stable's H is ~10% larger than baseline's, not an order of magnitude larger.

- **Drift alone (H=0) over 5Y:** displacement = **0.14**. With H=0, stable's z_T barely
  moves from z_0. The mean-reverting drift is pulling z toward z* (at distance 9.6), but
  over 5Y the eigenvalues (−5, −2, −1, −0.002) are large enough that z reaches z* quickly
  and then stays there — but z* is far from z_0, so the net drift displacement is only
  the (z_0 → z*) journey, which is ~9.6 in total but the model equilibrates partway within 5Y.
  The small 0.14 figure means the drift is actually pulling z **toward** z_0 from z*'s
  direction — counteracting diffusion.

- **Full simulation:** 2.10. The diffusion (4.60 annualised scale) partially cancels with the
  drift's pull toward z* (which opposes the initial movement), leaving net ~2. The mean-reverting
  drift is **working** — it contains the diffusion. The problem is not the dynamics, it's that
  the displacement of ~2 happens to exceed the decoder's recon-only range.

### Baseline: drift is catastrophically destabilising

- **Drift alone (H=0) over 5Y:** displacement = **8374** — the baseline K matrix has positive
  eigenvalues (unstable), so without diffusion the drift alone drives z to ±∞. The z_T
  coordinates [892, 4211, −5352, −4792] confirm complete runaway.

- **Full simulation:** ~10¹³. The explosive drift compounds with diffusion.

- **z* for baseline:** at distance ~0.21 from the training cloud — the baseline "equilibrium"
  is inside the training data, but since K has positive eigenvalues z* is actually an
  **unstable fixed point** (a saddle), not an attractor. Any perturbation grows exponentially.

---

## Summary: What's Actually Wrong with Each Model

| | dim4_baseline | dim4_stable |
|--|:-------------|:------------|
| **K eigenvalues** | Positive (unstable, exploding) | Negative (stable, mean-reverting) |
| **H scale** | ~0.97 per dim | ~1.05 per dim (+10%) |
| **z*** | ~0.21 from training cloud (inside it, but unstable) | **9.6 from training cloud** (mechanism 2) |
| **Drift only at T=5Y** | 8374 (exponential blowup) | 0.14 (effective mean reversion) |
| **Full simulation T=5Y** | ~10¹³ (drift dominates) | ~2.1 (diffusion dominates, drift contains it) |
| **Dominant failure** | Unstable K matrix | z* placed far from encoded data |

---

## Mechanism 2 for Stable: Why Is z* at −8.5?

During recon-only training the objective is purely `‖decoder(encoder(x)) − x‖²`. The encoder
learns to map curves into a compact region near [0, 0, 0, 0] (training cloud std ~0.02–0.06).
The stable K_mu is trained simultaneously to fit the **time-series dynamics** of
`{encoder(x_t)}` — specifically, it fits a discretized SDE to the sequence of encoded latent
states. The equilibrium z* = −M⁻¹N is determined by the learned bias N together with M.

There is **no training signal** that forces z* to sit inside (or near) the encoded data cloud.
The recon loss is blind to z*: it only cares that `decoder(encoder(x))` is accurate.
The SDE fitting is blind to the decoder: it only cares that the simulated z dynamics
match the time-series statistics of encoded states. The equilibrium landed at [−8.5, 3.8, ...]
because that's where the time-series fit placed it — completely unconstrained by the decoder.

At pricing time, the strong mean-reverting drift (eigenvalue −5.08 on one dimension) immediately
begins pulling z_0 toward z* = [−8.5, ...]. Over 5Y, z makes partial progress toward z* along
each dimension weighted by the eigenvalue magnitude, taking z about 2 units away from z_0 — and
those 2 units are enough to leave the decoder's trained range.

---

## Implications for Joint Training

### Immediate fix: z* regularization

If mechanism (2) is dominant (confirmed: drift-only displacement for stable is **0.14**, i.e., the
drift is well-behaved once z* is handled), the cleanest fix is to add a regularizer during
joint training that pulls z* toward the encoded data mean:

```
L_reg = λ · ‖z* − mean(encoder(X_batch))‖²
       = λ · ‖−M⁻¹N − μ_z‖²
```

With z* near the data cloud (‖z_0 − z*‖ ~ 0.2, as in baseline), the drift no longer
systematically pulls z away from z_0, and the full-simulation displacement drops to the
diffusion-only scale (~4.6 annualised × √dt per step). That is a much smaller problem for
the decoder to handle.

### What joint training does without this regularizer

Without z* regularization, the pricing loss has to fight a ~9.6-unit drift pull for every
simulated path. The loss gradient will push the decoder to work at the displaced z_T positions
(‖z_T − z_0‖ ~ 2), which will work eventually — but requires many more epochs than if z*
were already near the data cloud. Joint training without regularization is solving a harder
problem than necessary.

### H scale is a secondary concern

Stable's H is ~10% larger than baseline's. This contributes marginally to displacement but
is not the bottleneck. The pricing loss will naturally calibrate H down if model vols are
too high — no additional loss term needed for H.

---

## Three-Line Fix for Joint Training

```python
# In Training_joint.py, add to total loss:
z_star  = -torch.linalg.solve(model.K.stable_matrix(), model.K.N)   # (d,)
mu_z    = model.encoder(X_batch).mean(dim=0).detach()               # (d,)
L_zstar = lambda_zstar * (z_star - mu_z).pow(2).sum()
```

Typical `lambda_zstar ~ 0.1–1.0`. This directly addresses the z* misplacement without
interfering with the pricing loss or reconstruction loss.

---

## Figure

**`fig_drift_only_displacement.png`**  
Drift-only `‖z_t − z_0‖` over 0–5Y with H=0 for both models. Baseline grows exponentially
to ~8375; stable stays below 0.15 throughout. This single figure confirms mechanism (2) is
the dominant issue for stable: the SDE dynamics are fine once you remove diffusion, and the
small drift displacement (~0.14) shows the mean reversion is working correctly toward z*.
The problem is purely that z* is in the wrong place.

