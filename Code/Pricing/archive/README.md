# Archived Pricing Models

Pricing-layer variants that were trained and evaluated during the thesis
work but are **not** the headline model reported in the final write-up.
They are kept here for reproducibility and to document the negative
results referenced in the thesis.

## Headline model (kept in `Code/Pricing/`)

| Model | Files |
|---|---|
| Base (stable, no pricing layer) | `eval_base.py` |
| **Constant MPR** | `Training_constant_mpr.py`, `eval_constant_mpr.py`, `Training_constant_mpr_regimes.py` |

The Constant MPR is the only pricing layer featured in the final thesis.
The base evaluation (`eval_base.py`) is kept for the 415 bp diagnostic.

## Archived models (this folder)

| Model | Files | Status |
|---|---|---|
| Regime MPR | `Training_regime_mpr.py`, `eval_regime_mpr.py` | Abandoned — `A` matrix corrupted `lambda_0` |
| Expiry MPR | `Training_expiry_mpr.py`, `eval_expiry_mpr.py` | Overfits — single-split MAE 34 / 27 train / 54 test |
| Expiry-Tenor Vol MPR | `Training_expiry_tenor_vol_mpr.py`, `eval_expiry_tenor_vol_mpr.py` | Overfits — single-split MAE 34 / 27 train / 53 test |
| State-Conditioned Vol MPR | `Training_state_vol_mpr.py`, `eval_state_vol_mpr.py` | Overfits — saddle-point fix applied, still overfits |
| Stochastic Vol Pricing (CIR) | `Training_sv_pricing.py`, `eval_sv_pricing.py` | v_0(z_0) collapsed; no improvement over State-Cond |
| Daily Vol-Level MPR | `Training_daily_vol_mpr.py`, `eval_daily_vol_mpr.py` | Only marginal improvement on test over Constant MPR |
| Surface Vol MPR (LOO features) | `Training_surface_vol_mpr.py`, `eval_surface_vol_mpr.py`, `Training_surface_vol_mpr_regimes.py` | Uses observed vol-surface features. Reached test MAE around 32 bp, but the use of contemporaneous option-market information was judged out of scope for the final thesis and the model is not reported there. |

## Diagnostics

| File | Purpose |
|---|---|
| `eval_mc_noise_check.py` | One-off check that the 41 bp Constant MPR MAE is structural, not Monte Carlo noise (5,000 paths × 3 seeds: 40.3 ± 0.3 bp). |

## SV-only modules (only imported by archived scripts)

Located in `_modules/`:

| File | Used by |
|---|---|
| `CIR_vol_pricing.py` | `Training_sv_pricing.py`, `eval_sv_pricing.py` |
| `simulate_sv_pricing.py` | `Training_sv_pricing.py`, `eval_sv_pricing.py` |

These were originally in `Code/model/` and `Code/Simulation/` respectively;
moved here because no live script depends on them.

## Where the results went

The per-cell CSVs, training logs, and figures for these models are in:

- `Figures/pricing/archive/` — eval outputs (per-cell results, heatmaps, time-series figures)
- `Figures/TrainingResults/archive/` — training logs, checkpoints, intermediate diagnostics

## Re-running an archived script

The scripts use the project-root path conventions of the time they were
written. If re-running, copy the script back into `Code/Pricing/`
(or fix the relative path imports) before launching, and likewise move
the `_modules/` files back into `Code/model/` and `Code/Simulation/` for
the SV scripts.
