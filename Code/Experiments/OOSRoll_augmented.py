# =============================================================================
# OOSRoll_augmented.py
#
# Rolling out-of-sample experiment for the augmented-input autoencoder.
# The encoder receives an 11-dimensional input (8 swap rates + 3 derived
# shape features), while the decoder outputs the original 8 swap rates.
# The training loss is MSE over all 11 values; RMSE evaluation uses only
# the 8-dim reconstruction against the original swap rates.
#
# Outputs go to:
#   Figures/OOSResults/Roll/OOS_roll_dim{N}_augmented_input/...
# =============================================================================

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# ── environment setup ─────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    REPO_ROOT = os.getcwd()

for _p in [REPO_ROOT, os.path.dirname(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Code.utils import helpers as H
from Code.load_swapdata import my_data, TARGET_TENORS
from Code.model.full_model import FullModel

VARIANT = "augmented_input"

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.mkldnn.enabled = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# ── settings ──────────────────────────────────────────────────────────────────
USE        = "bbg"
LATENT_DIM = 4

INPUT_DIM_ORIG = 8
INPUT_DIM_AUG  = 11  # 8 rates + 3 derived features

TRAIN_YEARS   = 5
TEST_MONTHS   = 6
STEP_MONTHS   = 6
MIN_TRAIN_OBS = 200

EPOCHS           = 3500
BATCH_SIZE       = 32
EVAL_BATCH_SIZE  = 256
TARGET_MSE       = -1
LOG_EVERY        = 100

max_lr           = 1e-3
final_div_factor = 3000.0

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

FIGURES_DIR = os.path.join(
    REPO_ROOT, "Figures", "OOSResults", "Roll",
    f"OOS_roll_dim{LATENT_DIM}_{VARIANT}",
    f"train{TRAIN_YEARS}Y_test{TEST_MONTHS}M_step{STEP_MONTHS}M",
    f"ep{EPOCHS}"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

oos_csv_path  = os.path.join(
    FIGURES_DIR,
    f"oos_rolling_{USE}_dim{LATENT_DIM}_train{TRAIN_YEARS}Y_test{TEST_MONTHS}M_step{STEP_MONTHS}M.csv"
)
PRED_TEST_CSV  = os.path.join(FIGURES_DIR, "predictions_test_all.csv")
PRED_TRAIN_CSV = os.path.join(FIGURES_DIR, "predictions_train_all.csv")
LATENT_Z_CSV   = os.path.join(FIGURES_DIR, "latent_z_all.csv")
PARAMETERS_CSV = os.path.join(FIGURES_DIR, "parameters_all.csv")

# ── augmentation ──────────────────────────────────────────────────────────────
def augment(x: torch.Tensor) -> torch.Tensor:
    """Append 3 derived shape features to the 8-dim swap-rate vector.

    Features (tenor order [1,2,3,5,10,15,20,30]):
      f1 = 10Y − 1Y   (slope)
      f2 = 30Y − 10Y  (long-end slope)
      f3 = 2×10Y − 1Y − 30Y  (curvature)
    """
    f1 = x[:, 4] - x[:, 0]
    f2 = x[:, 7] - x[:, 4]
    f3 = 2.0 * x[:, 4] - x[:, 0] - x[:, 7]
    return torch.cat([x, f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)

def compute_feats(x: torch.Tensor) -> torch.Tensor:
    """Return the 3 derived shape features for a batch of swap-rate vectors (B, 3)."""
    f1 = x[:, 4] - x[:, 0]
    f2 = x[:, 7] - x[:, 4]
    f3 = 2.0 * x[:, 4] - x[:, 0] - x[:, 7]
    return torch.stack([f1, f2, f3], dim=1)

# ── load data ─────────────────────────────────────────────────────────────────
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor_full.float()

meta = meta_full.copy()
meta["as_of_date"] = pd.to_datetime(meta["as_of_date"])
meta = meta.reset_index(drop=True)

assert len(meta) == X_tensor.shape[0], "meta and X_tensor length mismatch"

# ── helpers ───────────────────────────────────────────────────────────────────
def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask that is True for rows with all finite values."""
    return torch.isfinite(t).all(dim=1)

@torch.no_grad()
def predict_S_hat(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Augment the input, run the model, and return the 8-dim reconstruction on CPU."""
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb     = X[i:i + batch_size].to(device)
        xb_aug = augment(xb)
        outs.append(model(xb_aug).detach().cpu())
    return torch.cat(outs, dim=0)

def rmse_bps_on_subset(model: nn.Module, X_sub: torch.Tensor, meta_sub: pd.DataFrame):
    """Compute per-currency and average RMSE (bps) for a given data subset."""
    S_hat = predict_S_hat(model, X_sub, batch_size=EVAL_BATCH_SIZE)
    mask   = row_finite_mask(X_sub) & row_finite_mask(S_hat)
    n_bad  = int((~mask).sum().item())
    n_good = int(mask.sum().item())
    X_eval    = X_sub[mask]
    S_eval    = S_hat[mask]
    meta_eval = meta_sub.loc[mask.numpy()].reset_index(drop=True)
    rmse_per_ccy = H.rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)
    avg_rmse_bps = float(rmse_per_ccy.mean())
    return rmse_per_ccy, avg_rmse_bps, n_good, n_bad

def make_loader(X_sub: torch.Tensor, batch_size: int):
    """Build a DataLoader yielding (augmented input, original swap rates) pairs."""
    X_aug = augment(X_sub)
    ds    = TensorDataset(X_aug, X_sub)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

WINDOW_SEED = 0

def train_one_window(X_train: torch.Tensor):
    """Train a fresh augmented-input model on one rolling window. Returns (model, mse_history, lr_history)."""
    torch.manual_seed(WINDOW_SEED)
    np.random.seed(WINDOW_SEED)

    model  = FullModel(input_dim=INPUT_DIM_AUG, latent_dim=LATENT_DIM).to(device)
    model.train()

    optim   = torch.optim.Adam(model.parameters(), lr=max_lr)
    loader  = make_loader(X_train, BATCH_SIZE)

    scheduler = OneCycleLR(
        optim,
        max_lr=max_lr,
        steps_per_epoch=len(loader),
        epochs=EPOCHS,
        pct_start=0.3,
        div_factor=1.0,
        final_div_factor=final_div_factor,
    )

    loss_fn         = nn.MSELoss()
    train_mse_hist  = []
    lr_hist         = []
    nan_batches_total = 0

    for epoch in range(EPOCHS):
        running     = 0.0
        n_obs       = 0
        nan_batches = 0

        for batch_idx, (xb_aug, xb) in enumerate(loader):
            xb_aug = xb_aug.to(device)
            xb     = xb.to(device)

            optim.zero_grad(set_to_none=True)

            try:
                S_hat = model(xb_aug)
            except Exception as e:
                nan_batches += 1
                print(f"      [Forward error epoch={epoch} batch={batch_idx}]: {str(e)[:200]}")
                continue

            if not torch.isfinite(S_hat).all():
                nan_batches += 1
                print(f"      [S_hat NaN/Inf epoch={epoch} batch={batch_idx}]")
                continue

            # Loss: MSE over all 11 values (8 swap rates + 3 derived features)
            feat_true = compute_feats(xb)
            feat_hat  = compute_feats(S_hat)
            loss = loss_fn(
                torch.cat([S_hat, feat_hat], dim=1),
                torch.cat([xb,   feat_true], dim=1),
            )

            if not torch.isfinite(loss):
                nan_batches += 1
                print(f"      [Loss NaN/Inf epoch={epoch} batch={batch_idx}]")
                continue

            loss.backward()

            has_nan_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for _, p in model.named_parameters()
            )
            if has_nan_grad:
                nan_batches += 1
                print(f"      [NaN grad epoch={epoch} batch={batch_idx}]")
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            running += float(loss.detach().cpu()) * xb.shape[0]
            n_obs   += xb.shape[0]

        nan_batches_total += nan_batches
        epoch_mse = running / max(n_obs, 1)
        train_mse_hist.append(epoch_mse)
        lr_hist.append(optim.param_groups[0]["lr"])

        if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE:
            break

        if (epoch == 0) or ((epoch + 1) % LOG_EVERY == 0) or (epoch == EPOCHS - 1):
            print(
                f"  epoch={epoch:4d} train_rmse={(epoch_mse**0.5):.6e} "
                f"lr={optim.param_groups[0]['lr']:.2e} used_obs={n_obs} "
                f"nan_batches={nan_batches} total_nan_batches={nan_batches_total}"
            )

    return model, np.array(train_mse_hist), np.array(lr_hist)

# ── per-roll saving helpers ───────────────────────────────────────────────────
@torch.no_grad()
def get_latent(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Return latent vectors for all rows of X."""
    model.eval()
    zs = []
    for i in range(0, len(X), batch_size):
        xb     = X[i:i + batch_size].to(device)
        xb_aug = augment(xb)
        zs.append(model.encoder(xb_aug).cpu())
    return torch.cat(zs, dim=0)

@torch.no_grad()
def extract_parameters(model: nn.Module, X: torch.Tensor, meta_sub: pd.DataFrame) -> pd.DataFrame:
    """Extract per-curve model parameters (mu, sigma, rho, r_tilde) as a DataFrame."""
    model.eval()
    mask   = row_finite_mask(X)
    X_m    = X[mask].to(device)
    X_m_aug = augment(X_m)
    z      = model.encoder(X_m_aug)
    mu     = model.K(z)
    sigmas, rhos = model.H(z)
    r_til  = model.R(z).squeeze(-1)
    d      = model.latent_dim
    rec    = meta_sub.loc[mask.numpy()].copy().reset_index(drop=True)
    rec["as_of_date"] = pd.to_datetime(rec["as_of_date"])
    for k in range(d):
        rec[f"mu_{k+1}"]    = mu[:, k].cpu().numpy()
        rec[f"sigma_{k+1}"] = sigmas[:, k].cpu().numpy()
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            rec[f"rho_{i+1}{j+1}"] = rhos[:, idx].cpu().numpy()
            idx += 1
    rec["r_tilde"] = r_til.cpu().numpy()
    return rec

def _append_csv(df: pd.DataFrame, path: str):
    """Append a DataFrame to a CSV file, writing the header only if the file is new."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode="a", header=write_header, index=False)

def save_roll_outputs(model: nn.Module, X_train: torch.Tensor, meta_train_sub: pd.DataFrame,
                      test_start, X_test: torch.Tensor = None, meta_test: pd.DataFrame = None):
    """Append latent vectors, model parameters, and predictions for one window to the combined CSVs."""
    ts = test_start.date().isoformat()

    # Latent vectors
    Z    = get_latent(model, X_train)
    df_z = meta_train_sub.copy().reset_index(drop=True)
    df_z.insert(0, "test_start", ts)
    for ki in range(LATENT_DIM):
        df_z[f"z_{ki+1}"] = Z[:, ki].numpy()
    _append_csv(df_z, LATENT_Z_CSV)

    # Parameters
    df_p = extract_parameters(model, X_train, meta_train_sub)
    df_p.insert(0, "test_start", ts)
    _append_csv(df_p, PARAMETERS_CSV)

    # Predictions — actual and fitted columns use the original 8-dim swap rates
    tenor_cols = [f"tenor_{t}" for t in TARGET_TENORS]

    def _save_predictions(X: torch.Tensor, meta_sub: pd.DataFrame, path: str):
        S_hat = predict_S_hat(model, X)
        mask  = row_finite_mask(X) & row_finite_mask(S_hat)
        X_np  = X[mask].numpy()
        S_np  = S_hat[mask].numpy()
        m     = meta_sub.loc[mask.numpy()].reset_index(drop=True)
        df_pred = m[["as_of_date", "ccy"]].copy()
        df_pred.insert(0, "test_start", ts)
        for i, t in enumerate(tenor_cols):
            df_pred[f"actual_{t}"] = X_np[:, i]
            df_pred[f"fitted_{t}"] = S_np[:, i]
        _append_csv(df_pred, path)

    _save_predictions(X_train, meta_train_sub, PRED_TRAIN_CSV)
    if X_test is not None and meta_test is not None:
        _save_predictions(X_test, meta_test, PRED_TEST_CSV)

# ── build rolling schedule ────────────────────────────────────────────────────
date_min = max(meta["as_of_date"].min(), pd.Timestamp("2010-01-01"))
date_max = meta["as_of_date"].max()

start = date_min + pd.DateOffset(years=TRAIN_YEARS)
end   = date_max - pd.DateOffset(months=TEST_MONTHS)

roll_starts = []
d = start
while d <= end:
    roll_starts.append(d)
    d = d + pd.DateOffset(months=STEP_MONTHS)

if len(roll_starts) == 0:
    raise RuntimeError("No rolling windows possible.")

print(f"Rolling windows: {len(roll_starts)} from {roll_starts[0].date()} to {roll_starts[-1].date()}")

# ── CSV header ────────────────────────────────────────────────────────────────
cols = (
    ["roll_start", "train_start", "train_end", "test_start", "test_end",
     "n_train", "n_test", "n_test_good", "n_test_bad",
     "n_train_good", "n_train_bad",
     "time_train_sec", "time_test_sec",
     "avg_rmse_bps", "avg_in_rmse_bps"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
    + [f"in_rmse_bps_{ccy}" for ccy in ccy_order]
    + [f"eig_real_{i+1}" for i in range(LATENT_DIM)]
)
pd.DataFrame(columns=cols).to_csv(oos_csv_path, index=False)
print("OOS CSV:", oos_csv_path)

# ── manifest ──────────────────────────────────────────────────────────────────
manifest = {
    "window_seed": WINDOW_SEED, "latent_dim": LATENT_DIM,
    "input_dim": INPUT_DIM_AUG, "variant": VARIANT,
    "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "max_lr": max_lr, "pct_start": 0.3,
    "final_div_factor": final_div_factor,
    "train_years": TRAIN_YEARS, "test_months": TEST_MONTHS,
    "step_months": STEP_MONTHS, "n_windows": len(roll_starts),
    "torch_version": torch.__version__,
    "python_version": sys.version.split()[0],
    "numpy_version": np.__version__,
    "run_started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "window_results": {},
}
manifest_path = os.path.join(FIGURES_DIR, "run_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest initialised: {manifest_path}")

# ── rolling loop ──────────────────────────────────────────────────────────────
avg_rmse_curve    = []
avg_in_rmse_curve = []

for k, test_start in enumerate(roll_starts):
    train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
    train_end   = test_start - pd.DateOffset(days=1)
    test_end    = test_start + pd.DateOffset(months=TEST_MONTHS) - pd.DateOffset(days=1)

    m_train = (meta["as_of_date"] >= train_start) & (meta["as_of_date"] <= train_end)
    m_test  = (meta["as_of_date"] >= test_start)  & (meta["as_of_date"] <= test_end)

    n_train = int(m_train.sum())
    n_test  = int(m_test.sum())

    print(f"\n[{k+1:02d}/{len(roll_starts)}] "
          f"train {train_start.date()}..{train_end.date()} (n={n_train}) | "
          f"test {test_start.date()}..{test_end.date()} (n={n_test})")

    if n_train < MIN_TRAIN_OBS or n_test == 0:
        print("  Skipping (too few observations).")
        continue

    X_train_w      = X_tensor[m_train.values]
    X_test_w       = X_tensor[m_test.values]
    meta_train_sub = meta.loc[m_train.values].reset_index(drop=True)
    meta_test      = meta.loc[m_test.values].reset_index(drop=True)

    # Train
    t0 = time.perf_counter()
    model, train_mse_hist, lr_hist = train_one_window(X_train_w)
    t1 = time.perf_counter()

    # Eigenvalues of drift matrix
    with torch.no_grad():
        if hasattr(model.K, "lin"):
            M = model.K.lin.weight.cpu()
        else:
            M = model.K.stable_matrix().cpu()
        eig_reals = torch.linalg.eigvals(M).real.numpy()
        eig_reals = np.sort(eig_reals)[::-1]

    # IS RMSE
    in_rmse_per_ccy, avg_in_rmse_bps, n_train_good, n_train_bad = rmse_bps_on_subset(
        model, X_train_w, meta_train_sub)

    # OOS RMSE
    t2 = time.perf_counter()
    rmse_per_ccy, avg_rmse_bps, n_good, n_bad = rmse_bps_on_subset(model, X_test_w, meta_test)
    t3 = time.perf_counter()

    avg_rmse_curve.append((test_start, avg_rmse_bps))
    avg_in_rmse_curve.append((test_start, avg_in_rmse_bps))

    row = {
        "roll_start":   test_start.date().isoformat(),
        "train_start":  train_start.date().isoformat(),
        "train_end":    train_end.date().isoformat(),
        "test_start":   test_start.date().isoformat(),
        "test_end":     test_end.date().isoformat(),
        "n_train":      n_train, "n_test": n_test,
        "n_test_good":  n_good,  "n_test_bad": n_bad,
        "n_train_good": n_train_good, "n_train_bad": n_train_bad,
        "time_train_sec": t1 - t0, "time_test_sec": t3 - t2,
        "avg_rmse_bps": avg_rmse_bps, "avg_in_rmse_bps": avg_in_rmse_bps,
    }
    for ccy in ccy_order:
        row[f"rmse_bps_{ccy}"]    = float(rmse_per_ccy.get(ccy, np.nan))
        row[f"in_rmse_bps_{ccy}"] = float(in_rmse_per_ccy.get(ccy, np.nan))
    for i, ev in enumerate(eig_reals):
        row[f"eig_real_{i+1}"] = round(float(ev), 6)

    pd.DataFrame([row], columns=cols).to_csv(oos_csv_path, mode="a", header=False, index=False)

    manifest["window_results"][test_start.date().isoformat()] = {
        "avg_rmse_bps":    round(avg_rmse_bps, 4),
        "avg_in_rmse_bps": round(avg_in_rmse_bps, 4),
        "per_ccy_bps":     {ccy: round(float(rmse_per_ccy.get(ccy, np.nan)), 4) for ccy in ccy_order},
        "in_per_ccy_bps":  {ccy: round(float(in_rmse_per_ccy.get(ccy, np.nan)), 4) for ccy in ccy_order},
        "train_minutes":   round((t1 - t0) / 60, 2),
        "n_train":         n_train, "n_test": n_test,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  IS avg_in_rmse_bps={avg_in_rmse_bps:.3f} | OOS avg_rmse_bps={avg_rmse_bps:.3f} "
          f"| time_train={(t1-t0)/60:.1f}min")

    save_roll_outputs(model, X_train_w, meta_train_sub, test_start,
                      X_test=X_test_w, meta_test=meta_test)
    print(f"  Appended window {k+1:02d} to combined CSVs")

manifest["run_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest finalised: {manifest_path}")
print("\nRolling OOS done.")

# ── plot in-sample vs out-of-sample RMSE over time ───────────────────────────
if avg_rmse_curve:
    dates    = [d for d, _ in avg_rmse_curve]
    oos_vals = [v for _, v in avg_rmse_curve]
    in_vals  = [v for _, v in avg_in_rmse_curve]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(dates, in_vals,  marker="o", linewidth=1.0, label="IS RMSE (bps)")
    ax.plot(dates, oos_vals, marker="o", linewidth=1.0, label="OOS RMSE (bps)")
    ax.set_xlabel("Test window start date")
    ax.set_ylabel("Average RMSE (bps)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "oos_avg_rmse_bps_curve.png"), dpi=300)
    plt.close(fig)
    print("Saved IS vs OOS avg RMSE curve.")
else:
    print("No OOS curve points to plot (all windows skipped).")
