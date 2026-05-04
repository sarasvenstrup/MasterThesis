"""
Diagnostic script: investigate why dim2_stable and dim3_stable have big errors.
Compares latent z distributions (train vs test), parameters, and predictions.

Run: python Code/investigate_stable_errors.py
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "Figures", "OOSResults", "Roll")

CASES = [
    ("OOS_roll_dim2_stable", "roll04_2016-07-01", "dim2_stable 2016-07-01 [ODE crash, n_bad=8, 1812 bps]"),
    ("OOS_roll_dim2_stable", "roll16_2022-07-01", "dim2_stable 2022-07-01 [regime mismatch, 266 bps]"),
    ("OOS_roll_dim3_stable", "roll16_2022-07-01", "dim3_stable 2022-07-01 [regime mismatch, 146 bps]"),
    ("OOS_roll_dim4_stable", "roll16_2022-07-01", "dim4_stable 2022-07-01 [GOOD: 24 bps — reference]"),
]


def roll_path(model, roll, fname):
    return os.path.join(BASE, model, "train5Y_test6M_step6M", "ep3500", "rolls", roll, fname)


def load(model, roll, fname):
    p = roll_path(model, roll, fname)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ── 1. latent z: train-period stats  (latent_z.csv = training period only) ──

section("1. LATENT Z  —  TRAINING PERIOD STATISTICS")
print("(latent_z.csv records the encoder output during training)")

for model, roll, label in CASES:
    z = load(model, roll, "latent_z.csv")
    if z is None:
        print(f"\n{label}: FILE MISSING"); continue
    zcols = [c for c in z.columns if c.startswith("z")]
    dates = pd.to_datetime(z["as_of_date"])
    print(f"\n--- {label} ---")
    print(f"  Dates : {dates.min().date()} → {dates.max().date()}  ({len(z)} rows)")
    for c in zcols:
        v = z[c]
        print(f"  {c}: mean={v.mean():.4f}  std={v.std():.4f}  "
              f"min={v.min():.4f}  max={v.max():.4f}")


# ── 2. test predictions: actual vs fitted, identify extreme rows ─────────────

section("2. TEST PREDICTIONS  —  WORST ERRORS")
TENORS = [f"tenor_{i}" for i in range(8)]

for model, roll, label in CASES:
    pred = load(model, roll, "predictions_test.csv")
    if pred is None:
        print(f"\n{label}: FILE MISSING"); continue

    dates = pd.to_datetime(pred["as_of_date"])
    ccys  = pred["ccy"].unique()

    # compute per-row RMSE across all tenors
    actual_cols = [c for c in pred.columns if c.startswith("actual_")]
    fitted_cols = [c for c in pred.columns if c.startswith("fitted_")]
    actual = pred[actual_cols].values
    fitted = pred[fitted_cols].values
    row_rmse = np.sqrt(np.nanmean((actual - fitted) ** 2, axis=1)) * 10000  # bps

    pred2 = pred.copy()
    pred2["row_rmse_bps"] = row_rmse
    top5 = pred2.nlargest(5, "row_rmse_bps")[["as_of_date", "ccy", "row_rmse_bps"] + actual_cols[:4] + fitted_cols[:4]]

    print(f"\n--- {label} ---")
    print(f"  Test rows: {len(pred)}  "
          f"Mean RMSE: {row_rmse.mean():.1f} bps  "
          f"Max RMSE: {row_rmse.max():.1f} bps")
    print(f"  Top-5 worst rows:")
    for _, r in top5.iterrows():
        print(f"    {r['as_of_date']}  {r['ccy']:4s}  rmse={r['row_rmse_bps']:.0f} bps  "
              f"act[0]={r[actual_cols[0]]:.4f}  fit[0]={r[fitted_cols[0]]:.4f}")


# ── 3. parameters: check M eigenvalues and sigma scale ───────────────────────

section("3. MODEL PARAMETERS  —  EIGENVALUES & SIGMA SCALE")

for model, roll, label in CASES:
    params_dir = roll_path(model, roll, "parameters")
    if not os.path.isdir(params_dir):
        print(f"\n{label}: parameters/ dir missing"); continue

    param_files = [f for f in os.listdir(params_dir)]
    print(f"\n--- {label} ---")
    print(f"  Parameter files: {param_files}")

    # Try to read parameters.csv in the roll root
    pcsv = load(model, roll, "parameters.csv")
    if pcsv is not None:
        print(f"  parameters.csv ({len(pcsv)} rows x {len(pcsv.columns)} cols):")
        print("  " + pcsv.to_string(index=False).replace("\n", "\n  "))
    else:
        # peek at individual param files
        for fname in sorted(param_files)[:6]:
            fp = os.path.join(params_dir, fname)
            if os.path.isfile(fp):
                try:
                    arr = np.load(fp) if fname.endswith(".npy") else None
                    if arr is not None:
                        print(f"    {fname}: shape={arr.shape}  min={arr.min():.4f}  max={arr.max():.4f}")
                except Exception:
                    pass


# ── 4. train vs test z: use predictions to infer test z range ────────────────

section("4. ACTUAL vs FITTED  —  SHORT-RATE / USD / CAD / AUD breakdown")
CURRENCIES_OF_INTEREST = ["USD", "AUD", "CAD", "JPY"]

for model, roll, label in CASES:
    pred = load(model, roll, "predictions_test.csv")
    if pred is None:
        continue
    actual_cols = [c for c in pred.columns if c.startswith("actual_")]
    fitted_cols = [c for c in pred.columns if c.startswith("fitted_")]

    print(f"\n--- {label} ---")
    for ccy in CURRENCIES_OF_INTEREST:
        sub = pred[pred["ccy"] == ccy]
        if len(sub) == 0:
            continue
        actual = sub[actual_cols].values
        fitted = sub[fitted_cols].values
        rmse_bps = np.sqrt(np.nanmean((actual - fitted) ** 2)) * 10000
        # Show the first tenor (short rate proxy) actual vs fitted
        a0 = sub[actual_cols[0]].values
        f0 = sub[fitted_cols[0]].values
        print(f"  {ccy}: RMSE={rmse_bps:.0f} bps  "
              f"actual_t0 [{a0.min():.4f}, {a0.max():.4f}]  "
              f"fitted_t0 [{f0.min():.4f}, {f0.max():.4f}]")


print()
print("Done.")

