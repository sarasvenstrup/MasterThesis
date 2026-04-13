"""
run_pricing_pipeline.py
=======================
Full before/after second-stage calibration pipeline.

Steps
-----
  1. Pre-stage-2 comparison
     Run compare_swapvol.comparison_table() with the Stage-1 (ep3500) stable
     checkpoint.  Output -> swapvol_results/swaption_vol_comparison_pre_stage2.xlsx

  2. Second-stage calibration
     Run Pricing_Training.calibrate_second_stage() to fine-tune H (+ optionally G
     and K.N) against market swaption vols.
     Saves calibrated checkpoint -> TrainingResults/dim2_stable/ep3500/checkpoint_stage2.pt

  3. Post-stage-2 comparison
     Repeat the comparison with the new checkpoint.
     Output -> swapvol_results/swaption_vol_comparison_post_stage2.xlsx

All Excel files are written to   Code/Pricing/swapvol_results/
so the two runs can be opened side-by-side.
"""

import os
import sys

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THESIS_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for p in [SCRIPT_DIR, PROJECT_ROOT, THESIS_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.Pricing.compare_swapvol   import comparison_table
from Code.Pricing.Pricing_Training  import calibrate_second_stage

# =============================================================================
# USER SETTINGS  — adjust as needed
# =============================================================================

# Stage-1 checkpoint (the ep3500 stable model trained in Stage 1)
STAGE1_CKPT = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis"
    r"\Figures\TrainingResults\dim2_stable\ep3500\checkpoint_dim2_ep3500.pt"
)

# Stage-2 checkpoint will be written here
STAGE2_CKPT = (
    r"C:\Users\Bruger\PycharmProjects\MasterThesis"
    r"\Figures\TrainingResults\dim2_stable\ep3500\checkpoint_stage2.pt"
)

# Output directory for all Excel comparison files
OUT_DIR = os.path.join(SCRIPT_DIR, "swapvol_results")

# Pricing comparison settings
PRICING_CFG = dict(
    ccy      = "EUR",
    n_paths  = 2000,
    n_steps  = 120,       # 10 yr at monthly dt
    dt       = 1 / 12,
    payer    = True,
    accrual  = 1.0,
    notional = 1.0,
    max_rows = 50,        # limit rows for speed; set None for full dataset
)

# Second-stage training settings
TRAIN_CFG = dict(
    ccy        = "EUR",
    n_paths    = 512,
    dt         = 1 / 12,
    lr         = 1e-4,
    n_epochs   = 300,
    batch_size = 3,
    log_every  = 25,
    train_G    = True,
    train_K_N  = True,
    lambda_G   = 1.0,
    lambda_K   = 0.01,
    save_path  = STAGE2_CKPT,
)

# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Step 1: Pre-stage-2 pricing ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1 — Pre-stage-2 pricing  (Stage-1 ep3500 stable checkpoint)")
    print("=" * 70)

    df_pre = comparison_table(
        checkpoint_path = STAGE1_CKPT,
        label           = "pre_stage2",
        out_dir         = OUT_DIR,
        **PRICING_CFG,
    )

    # ── Step 2: Second-stage calibration ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — Second-stage calibration  (fine-tuning H, G, K.N)")
    print("=" * 70)

    model, loss_history, grad_history, df_vol = calibrate_second_stage(
        checkpoint_path = STAGE1_CKPT,
        **TRAIN_CFG,
    )

    print(f"\nStage-2 checkpoint written → {STAGE2_CKPT}")

    # ── Step 3: Post-stage-2 pricing ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — Post-stage-2 pricing  (Stage-2 calibrated checkpoint)")
    print("=" * 70)

    df_post = comparison_table(
        checkpoint_path = STAGE2_CKPT,
        label           = "post_stage2",
        out_dir         = OUT_DIR,
        **PRICING_CFG,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — improvement summary")
    print("=" * 70)

    for stage, df in [("pre_stage2", df_pre), ("post_stage2", df_post)]:
        valid = df.dropna(subset=["model_vol_bp"])
        if valid.empty:
            print(f"  [{stage}]  no valid rows")
            continue
        mae  = valid["abs_vol_error_bp"].mean()
        rmse = (valid["vol_error_bp"] ** 2).mean() ** 0.5
        print(f"  [{stage}]  MAE={mae:6.1f} bp   RMSE={rmse:6.1f} bp   N={len(valid)}")

    print(f"\nOutputs written to: {OUT_DIR}")
    print("  swaption_vol_comparison_pre_stage2.xlsx")
    print("  swaption_vol_comparison_post_stage2.xlsx")

    return df_pre, df_post


if __name__ == "__main__":
    run_pipeline()


