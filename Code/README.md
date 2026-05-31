# Code — Master's Thesis: Neural SDE Interest Rate Model

## Project Overview
This codebase implements a neural SDE-based interest rate model. A neural autoencoder
maps observed swap rates to a low-dimensional latent state z, whose risk-neutral
dynamics are governed by a learnable SDE. Bond prices are recovered by solving a
Riccati-type ODE system, and par swap rates are reconstructed from the resulting
discount curve. The model is trained to minimise the reconstruction error between
observed and model-implied swap rates.

---

## Pipeline Overview
There are two completely separate model pipelines:

| Pipeline | Status | Model file | Training script |
|---|---|---|---|
| **Baseline** | Frozen — do not modify | `model/full_model_baseline.py` | `Training_baseline.py` |
| **Stable** | Active development | `model/full_model.py` | `Training.py` |

**Why separated?** Any change to `full_model.py` or `config.py` during stable development
must never affect baseline results. The baseline pipeline has no dependency on `config.py`
and imports only from `full_model_baseline.py`. Changing stable files cannot touch it.

---

## File Map

### Entry points
| Script | Purpose |
|---|---|
| `run_all_dims.py` | Runs `Training_baseline.py` for all latent dims (3, 2, 4, 1) sequentially |
| `Training_baseline.py` | Trains baseline model for one latent dim, saves checkpoint + CSV logs |
| `Training.py` | Same as above for stable variant |
| `OutOfSampleRoll.py` | Rolling-window out-of-sample evaluation (baseline) |
| `OutOfSampleSplit.py` | Train/test split out-of-sample evaluation, N seeds (baseline) |
| `ResultsGeneratorBaseline.py` | Generates all thesis figures and tables from baseline checkpoints |
| `ResultsGeneratorStable.py` | Same for stable variant |

### Model files (`model/`)
| File | Purpose |
|---|---|
| `full_model_baseline.py` | **Frozen** baseline model — no config dependency, baseline K/H hardcoded |
| `full_model.py` | Stable model — reads `config.VARIANT` to switch between baseline/stable K/H |
| `Encoder.py` | Shared encoder (baseline and stable) |
| `DecoderG.py` | Shared decoder G(z, τ) |
| `K_mu.py` | Baseline drift network K |
| `H_sigma.py` | Baseline diffusion network H |
| `K_mu_stable.py` | Stable drift network K |
| `H_sigma_stable.py` | Stable diffusion network H |
| `R_short.py` | Short rate network R (shared) |

### Config
| File | Purpose |
|---|---|
| `config.py` | **Stable pipeline only.** Sets `VARIANT = "baseline"` or `"stable"`. Not read by any baseline script. |

---

## How to Run

**Full baseline training (all dims):**
```bash
python Code/run_all_dims.py
```
This patches `LATENT_DIM` and `EPOCHS` in `Training_baseline.py` before each run and
restores the file afterwards. Runs dims in order: 3 → 2 → 4 → 1.

**Single dim manually:**
```bash
python Code/Training_baseline.py   # set LATENT_DIM manually in the script first
```

**Out-of-sample evaluation:**
```bash
python Code/OutOfSampleRoll.py
python Code/OutOfSampleSplit.py
```

**Generate thesis figures:**
```bash
python Code/ResultsGeneratorBaseline.py
```

---

## Key Settings

| Setting | Where | What it controls |
|---|---|---|
| `LATENT_DIM` | `Training_baseline.py` (top) | Number of latent factors (1–4) |
| `EPOCHS` | `Training_baseline.py` (top) | Training epochs (patched by `run_all_dims.py`) |
| `SEED` | `Training_baseline.py` (top) | Random seed for weight initialisation |
| `SHOW_PLOTS` | `Training_baseline.py` (top) | Set `False` when running headless/subprocess |
| `VARIANT` | `config.py` | Stable pipeline only — `"baseline"` or `"stable"` |

---

## Output Structure

```
Figures/
  TrainingResults/
    dim{N}_baseline/
      ep{EPOCHS}/
        train_rmse_log_bbg_dim{N}_ep{EPOCHS}.csv   ← per-epoch RMSE log
        run_config.json                             ← seed, lr, env info
        checkpoint_dim{N}_ep{EPOCHS}.pt            ← model state dict
        latent_z_bbg_dim{N}_ep{EPOCHS}.csv         ← latent coordinates
        lr_schedule_*.png
        avg_rmse_bps_convergence_*.png

  OOSResults/
    Roll/OOS_roll_dim{N}_baseline/...
    Split/OOS_split_dim{N}_baseline/...

checkpoints/
  fullmodel_bbg_dim{N}_ep{EPOCHS}.pt               ← full checkpoint (state + config)
```
