# Code/Pricing — Swaption Pricing Pipeline

This folder implements the full swaption pricing pipeline for the pricing chapter
of the thesis. The chapter has three sections and the scripts map directly onto them:

| Chapter section | Scripts |
|-----------------|---------|
| §1 Methodology | *(no script — LaTeX only)* |
| §2 Diagnostics (decoder coverage + forward bias) | `pretraining_diagnostics.py`, `calibrate_H_scale.py`, `calibrate_expiry_scale.py`, `delta_diagnostic.py` |
| §3 The Lambda MPR Fix | `Training_lambda_mpr.py`, `eval_lambda_mpr.py` |

Supporting libraries used by all scripts: `pricing.py`, `load_swapvol_ois.py`.

---

## Script reference

### `pricing.py`
Core pricing library. **Not run directly — imported by all other scripts.**

- `bachelier_price_torch(F, K, sigma, T, N, A)` — Bachelier payer/receiver price (batch, differentiable)
- `swap_rate_torch(P_full, tenor)` — pathwise par swap rate and annuity from decoded bond grid
- `implied_vol_atm(price, A0, T)` — ATM implied normal vol from a Monte Carlo price

---

### `load_swapvol_ois.py`
Data loader for the EUR ATM swaption vol surface. **Not run directly — imported by all other scripts.**

- Reads `SwapData/SwapVol.xlsx`
- Returns a tidy DataFrame with columns `as_of_date`, `option_maturity`, `swap_tenor`, `vol` (in bp)

---

### `pretraining_diagnostics.py`
Produces the §2.1 (decoder coverage) figures and tables.

**Run first.** No dependencies on other pricing scripts.

```
python Code/Pricing/pretraining_diagnostics.py
```

**Outputs** → `Figures/Pricing/`:

| File | Content |
|------|---------|
| `fig_decoder_robustness.png` | Finite-decode fraction vs isotropic perturbation ε |
| `tab_decoder_robustness.tex` | Tabular summary of the robustness probe |
| `fig_latent_displacement_cdf.png` | CDF of ‖z_T − z_0‖ at T=5Y (baseline vs stable) |
| `tab_displacement_summary.tex` | Displacement percentiles |
| `tab_hk_decomposition.tex` | Drift-only vs full displacement decomposition |
| `tab_eigenvector_alignment.tex` | Alignment of z* − z_0 with slowest eigenvector |

---

### `calibrate_H_scale.py`
Prices all swaptions at the unscaled diffusion (`s=1`) and caches the result.
This is an **upstream dependency** for `calibrate_expiry_scale.py` and `delta_diagnostic.py`.
Run it once before any calibration step.

```
python Code/Pricing/calibrate_H_scale.py
```

**Outputs** → `Figures/TrainingResults/dim4_stable_hscale/`:

| File | Content |
|------|---------|
| `baseline_vols_s1.csv` | **Key shared cache.** Model implied vols at s=1 per (date, expiry, tenor). Read by `calibrate_expiry_scale.py`. |
| `calibration_summary.json` | Global s* = 0.151, MAE before/after |
| `vol_mae_sweep.png` | MAE vs diffusion-scale sweep |

> **Note:** The global scale s*=0.151 reduces average vol level but leaves the forward bias
> intact. It is documented here for completeness but not used in the chapter.

---

### `calibrate_expiry_scale.py`
Fits one diffusion scale per expiry row (3 parameters: s_1Y, s_5Y, s_10Y) using
pooled OLS on the training split.

```
python Code/Pricing/calibrate_expiry_scale.py
```

**Requires:** `baseline_vols_s1.csv` from `calibrate_H_scale.py`.

**Outputs** → `Figures/TrainingResults/dim4_stable_hscale/expiry_scale/`:

| File | Content |
|------|---------|
| `expiry_scales.json` | **Key output.** `{"1": 0.129, "5": 0.133, "10": 0.141}` — read at runtime by `delta_diagnostic.py` |
| `expiry_results.csv` | Per-(date, cell) vol errors at the expiry-level scales |
| `expiry_summary.json` | Aggregate MAE / RMSE |
| `tab_expiry_comparison.tex` | LaTeX comparison table |

> **How the s\* values reach `Training_lambda_mpr.py`:**
>
> `calibrate_expiry_scale.py` writes the three scale factors to `expiry_scales.json`.
> `delta_diagnostic.py` reads this JSON automatically at runtime (dynamic link).
>
> `Training_lambda_mpr.py` and `eval_lambda_mpr.py` have the values **hardcoded**:
> ```python
> EXPIRY_SCALES = {1: 0.129, 5: 0.133, 10: 0.141}
> DEFAULT_SCALE = 0.135
> ```
> These were read from `expiry_scales.json` and manually copied into the training
> script. If you re-run `calibrate_expiry_scale.py` and the values change, you must
> update `EXPIRY_SCALES` in both `Training_lambda_mpr.py` and `eval_lambda_mpr.py`
> before retraining.

---

### `delta_diagnostic.py`
Produces the §2.2 (forward bias) tables and figures.

```
python Code/Pricing/delta_diagnostic.py
```

**Requires:** `expiry_scales.json` from `calibrate_expiry_scale.py`.

Prices every EUR (date, expiry, tenor) triple at the calibrated expiry-level scales,
using both payer and receiver to extract the forward bias from put–call parity.

**Outputs** → `Figures/TrainingResults/dim4_stable_hscale/delta_diagnostic/`:

| File | Content |
|------|---------|
| `delta_results.csv` | Per-(date, expiry, tenor): sigma_pay, sigma_rec, sigma_str, forward_bias_bp, d_eff, p_eff |
| `tab_delta_diagnostic.tex` | LaTeX table of per-cell forward bias and p_eff (included in §2.2 via `\input`) |
| `fig_forward_bias.png` | Heatmap of mean forward bias per cell |
| `fig_exercise_prob.png` | Heatmap of p_eff per cell (included in §2.2 via `\includegraphics`) |

---

### `Training_lambda_mpr.py`
Trains the 16-parameter Lambda market-price-of-risk matrix for §3.

```
python Code/Pricing/Training_lambda_mpr.py
```

**Requires:**
- Pretrained stable checkpoint: `Figures/TrainingResults/dim4_stable/ep5000/checkpoint_dim4_ep5000.pt`
- `expiry_scales.json` values hardcoded as `EXPIRY_SCALES` (see note under `calibrate_expiry_scale.py`)

**What it trains:** A single `4×4` matrix `Λ` such that
```
K^Q(z) = K^P(z) - L(z) @ Lambda @ z
```
where `K^P` (model.K), `H` (model.H), and `G` (decoder) are all **frozen**.
`Lambda` is the only trainable object (16 parameters).

The diffusion vol is controlled by the pre-calibrated s* values applied as
`eps_scaled = eps * s*(expiry)` at each simulation step.

**Outputs** → `Figures/TrainingResults/dim4_stable_lambda_mpr/ep1000/`:

| File | Content |
|------|---------|
| `train_lambda_mpr_log_dim4_ep1000.csv` | Per-epoch: loss_price, loss_eig, loss_l2, path_finite_frac, recon_rmse_bps, lambda_min_KQ, Lambda_norm_fro |
| `checkpoint_lambda_ep200.pt` | Checkpoint at epoch 200 (also 400, 600, 800) |
| `checkpoint_lambda_ep1000.pt` | Final checkpoint. Contains `Lambda_matrix` (4×4 tensor) |
| `lambda_mpr_loss_dim4_ep1000.png` | Training loss curve |
| `run_status.json` | Compact summary of key metrics + Lambda matrix at final epoch |

---

### `eval_lambda_mpr.py`
Post-training evaluation of the Lambda MPR model. Produces the §3 results figures and tables.

```
python Code/Pricing/eval_lambda_mpr.py
```

**Requires:**
- Pretrained stable checkpoint (same as above)
- Lambda checkpoints at `dim4_stable_lambda_mpr/ep1000/checkpoint_lambda_ep*.pt`
- `expiry_scales.json` values hardcoded as `EXPIRY_SCALES` (same values as training)

Prices the full EUR test set with the final (ep999) Lambda checkpoint and
reports per-cell ATM vol errors across historical calendar dates.

**Outputs** → `Figures/TrainingResults/dim4_stable_lambda_mpr/ep1000/eval/`:

| File | Content | Used in thesis |
|------|---------|----------------|
| `tab_lambda_per_cell.tex` | Per-cell vol MAE/RMSE/bias | §3 via `\input` |
| `fig_vol_surface.png` | 3×3 scatter: model vs market vol per cell | §3 via `\includegraphics` |
| `fig_vol_error_timeseries.png` | 3×3 time-series: vol error (model−market) over calendar dates | §3 via `\includegraphics` |
| `fig_forward_bias_timeseries.png` | Forward bias over calendar time per cell | §3 via `\includegraphics` |
| `fig_vol_heatmap.png` | Heatmap of final vol MAE per cell | §3 (optional) |
| `per_cell_final.csv` | Raw per-(date, expiry, tenor) results | — |

---

## Full pipeline run order

```
# §2.1 — Decoder coverage diagnostics
python Code/Pricing/pretraining_diagnostics.py

# §2.2 — Forward bias (run steps 2a and 2b first to build the cache)
python Code/Pricing/calibrate_H_scale.py       # 2a: build baseline_vols_s1.csv cache
python Code/Pricing/calibrate_expiry_scale.py  # 2b: fit s*(1Y,5Y,10Y) → expiry_scales.json
python Code/Pricing/delta_diagnostic.py        # 2c: forward bias analysis

# §3 — Lambda MPR training and evaluation
python Code/Pricing/Training_lambda_mpr.py     # ~4 hours on CPU; saves checkpoints every 200 epochs
python Code/Pricing/eval_lambda_mpr.py         # ~1-2 hours on CPU; produces all §3 figures/tables
```

> **If calibrate_expiry_scale.py produces different s\* values than {0.129, 0.133, 0.141},**
> update the `EXPIRY_SCALES` dict in both `Training_lambda_mpr.py` and
> `eval_lambda_mpr.py` before running the Lambda training.

---

## Dependency graph

```
pretraining_diagnostics.py ─────────────────────────────► Figures/Pricing/
                                                           (decoder, displacement, decomp)

calibrate_H_scale.py ──► baseline_vols_s1.csv
                                │
                                ▼
         calibrate_expiry_scale.py ──► expiry_scales.json ──► delta_diagnostic.py
                                │                               │
                                │ (values manually              ▼
                                │  hardcoded ↓)       Figures/.../delta_diagnostic/
                                ▼                      (forward bias tables + figures)
             Training_lambda_mpr.py ──► checkpoint_lambda_ep*.pt
                                               │
                                               ▼
                               eval_lambda_mpr.py ──► Figures/.../eval/
                                                       (per-cell tables + figures)
```

---

## Archive

The `archive/` subfolder contains scripts from earlier experiments that are
**no longer needed for the current chapter** but are preserved for reference:

| File | What it was for |
|------|-----------------|
| `Training_joint.py` | First joint training attempt (shared K in ODE + simulation). Failed because K appears in the no-arbitrage bond-pricing ODE, causing pricing gradients to corrupt reconstruction (5 bp → 2 200 bp). |
| `Training_joint_kq.py` | Separate K^Q experiment (K^P frozen, K^Q + H trained). Failed because H also enters the ODE through the diffusion covariance, degrading reconstruction to 66 bp. |
| `calibrate_per_cell.py` | Per-cell OLS calibration (9 scale parameters). OOS MAE 73 bp — no improvement over expiry-level (72 bp) despite 3× more parameters. |
| `calibrate_improved.py` | Ridge regression + adaptive z_t-conditioned scale. Ridge selected λ=0; adaptive scale OOS 102 bp. Both worse than fixed expiry-level. |
| `calibrate_rolling.py` | Rolling-window OLS: refits per-cell scales using most recent W months. OOS ~98 bp. Confirms the floor is not a vol-regime problem. |
| `calibrate_straddle.py` | Precursor to `delta_diagnostic.py`. Straddle-based forward-bias test using a single pricing function (not the full payer+receiver split). Superseded by `delta_diagnostic.py`. |
| `calibrate_forward_adjusted.py` | Attempted to estimate and remove the forward bias analytically before pricing. Forward bias is sign-inconsistent across cells, making this infeasible. |
| `make_greeks_table.py` | Analytic ATM Bachelier Greeks table (Δ, vega, Γ, Θ). Not included in the current chapter. |
| `make_vol_comparison.py` | Vol comparison figures for baseline vs stable vs calibrated models. From the removed §4 (diffusion-scale calibration section). |
