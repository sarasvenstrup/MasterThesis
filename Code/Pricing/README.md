# Code/Pricing — Swaption Pricing Scripts

All scripts price EUR ATM swaptions using the trained Neural SDE encoder-decoder
checkpoint (`checkpoint_dim4_ep5000.pt`).  Outputs go to
`Figures/Pricing/` and `Figures/TrainingResults/dim4_stable_hscale/`.

---

## Overview of files

| File | Purpose |
|------|---------|
| `pricing.py` | Core pricing library (Bachelier formula, MC swaption pricer, implied-vol inversion). Imported by all calibration scripts — do not run directly. |
| `load_swapvol_ois.py` | Data loader for the EUR ATM vol surface (`SwapData/SwapVol.xlsx`). Imported by all calibration scripts — do not run directly. |
| `pretraining_diagnostics.py` | Produces all figures/tables for the "Why the stable model cannot price" chapter section: decoder robustness probe, latent displacement CDF, H-vs-K decomposition, eigenvector alignment. |
| `make_greeks_table.py` | Generates the analytic ATM Bachelier Greeks table (`Figures/Pricing/tab_atm_greeks.tex`). |
| `calibrate_H_scale.py` | **Global OLS calibration.** Finds a single diffusion scale `s*` minimising vol-space RMSE across all 1 246 swaptions. Entry point for the whole calibration workflow — run this first. |
| `make_vol_comparison.py` | Produces vol heatmaps, scatter plots, and summary table comparing three models (baseline, stable, stable-calibrated). Run after `calibrate_H_scale.py`. |
| `calibrate_per_cell.py` | **Per-cell OLS calibration.** Fits one scale per (expiry, tenor) cell (9 parameters) with a 70/30 train/OOS time split. Reuses `baseline_vols_s1.csv` cached by `calibrate_H_scale.py`. |
| `calibrate_expiry_scale.py` | **Expiry-level OLS calibration (recommended).** Pools all tenors within each expiry row → 3 parameters (`s_1Y`, `s_5Y`, `s_10Y`). Physically motivated (scale affects SDE horizon, not payoff tenor) and more robust than per-cell. OOS MAE 72 bp. |
| `calibrate_rolling.py` | Rolling-window OLS: refits per-cell scales using the most recent W months of history at each OOS date. Cross-validates window length. Result: worse than fixed expiry-level (OOS ~98 bp); included as a diagnostic. |
| `calibrate_improved.py` | Experiments with ridge regularisation (LOO-year CV) and an adaptive scale conditioned on the latent state `z_t`. Result: ridge selects λ=0; adaptive scale OOS ~102 bp. Confirms the residual error is structural. |
| `calibrate_straddle.py` | Straddle diagnostic: prices payer + receiver at `s=1`, computes straddle vol, fits OLS scale on straddle. Tests whether a uniform forward-rate bias from `K` is responsible for the residual error. Result: OOS MAE jumps 75→287 bp (bias is sign-inconsistent across cells). |

---

## Recommended run order

### Step 1 — Diagnostics: why the model cannot price without calibration

```
python Code/Pricing/pretraining_diagnostics.py
```

Outputs to `Figures/Pricing/`:
- `fig_decoder_robustness.png`, `tab_decoder_robustness.tex`
- `fig_latent_displacement_cdf.png`, `tab_displacement_summary.tex`
- `tab_hk_decomposition.tex`, `tab_eigenvector_alignment.tex`

---

### Step 2 — ATM Greeks table

```
python Code/Pricing/make_greeks_table.py
```

Output: `Figures/Pricing/tab_atm_greeks.tex`

---

### Step 3 — Global calibration  *(run before any per-cell scripts)*

```
python Code/Pricing/calibrate_H_scale.py
```

This prices all swaptions at `s=1` and caches the result as
`Figures/TrainingResults/dim4_stable_hscale/baseline_vols_s1.csv`.
All downstream calibration scripts read this cache instead of re-pricing.

Key outputs:
- `baseline_vols_s1.csv` — model vols at `s=1` (shared cache)
- `calibration_summary.json` — global `s*`, MAE before/after
- `vol_mae_sweep.png` — MAE vs diffusion scale sweep
- `checkpoint_dim4_hscale_0.1509.pt` — calibrated checkpoint (for inspection only; use `diffusion_scale=s*` with the original checkpoint for pricing)

---

### Step 4 — Vol comparison figures

```
python Code/Pricing/make_vol_comparison.py
```

Requires `baseline_vols_s1.csv` from Step 3.
Outputs to `Figures/Pricing/vol_comparison/`:
- `fig_vol_heatmap_stable.png`, `fig_vol_heatmap_stable_cal.png`
- `fig_vol_scatter_stable_cal.png`
- `vol_summary_table.tex`

---

### Step 5 — Per-cell calibration and OOS validation

```
python Code/Pricing/calibrate_per_cell.py
```

Requires `baseline_vols_s1.csv` from Step 3.
Outputs to `Figures/TrainingResults/dim4_stable_hscale/per_cell/`:
- `tab_per_cell_scales.tex`, `tab_per_cell_mae.tex`
- `fig_mae_train.png`, `fig_mae_oos.png`, `fig_scatter.png`

---

### Step 6 — Expiry-level calibration  *(recommended calibration)*

```
python Code/Pricing/calibrate_expiry_scale.py
```

Requires `baseline_vols_s1.csv` from Step 3.
Outputs to `Figures/TrainingResults/dim4_stable_hscale/expiry_scale/`:
- `tab_expiry_comparison.tex`
- `expiry_scales.json` — the three recommended scale factors

---

### Step 7 — Diagnostic experiments (optional)

These confirm that the ~72–75 bp OOS floor is structural and cannot be closed
by more sophisticated calibration:

```
python Code/Pricing/calibrate_rolling.py     # rolling-window OLS
python Code/Pricing/calibrate_improved.py    # ridge CV + adaptive z_t scale
python Code/Pricing/calibrate_straddle.py    # straddle forward-bias test
```

All three require `baseline_vols_s1.csv` from Step 3.
`calibrate_rolling.py` also reads `per_cell/per_cell_scales.json` from Step 5.

---

## Key result summary

| Calibration | Params | OOS MAE (bp) |
|-------------|--------|-------------|
| Global OLS | 1 | 91 |
| Expiry-level OLS | 3 | 72 |
| Per-cell OLS | 9 | 73 |
| Daily recalibration (lower bound) | — | 74 |
| Rolling window (12 M) | — | 98 |
| Adaptive z_t scale | 45 | 102 |

The 72–74 bp floor is structural: `K` was identified from yield-curve
reconstruction, not from the martingale condition E^{Q_A}[S_T] = F_0.
The straddle experiment confirms the forward bias is sign-inconsistent
across cells (+239 bp for 5Y×1Y, −401 bp for 1Y×5Y), ruling out a simple
directional drift correction.
