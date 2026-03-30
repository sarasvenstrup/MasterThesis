# ============================= Import Packages ===============================
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

# ============================= Environment Setup & Imports ===============================
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.utils import helpers as H
from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS
from Code.model.full_model import FullModel
from Code.config import VARIANT, confirm_variant
confirm_variant()

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.mkldnn.enabled = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# ============================= Config ===============================
USE = "bbg"
LATENT_DIM = 2

# Recommended rolling window setup (baseline OOS)
TRAIN_YEARS = 5
TEST_MONTHS = 6
STEP_MONTHS = 6
MIN_TRAIN_OBS = 200

# Training setup per window
EPOCHS = 2500
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
TARGET_MSE = -1          # set >0 if you want early stop
LOG_EVERY = 100          # training printouts inside each window

max_lr = 3e-3
final_div_factor = 3000.0

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

# Output folders
FIGURES_DIR = os.path.join(
    REPO_ROOT,
    "Figures",
    "OOSResults",
    "Roll",
    f"OOS_roll_dim{LATENT_DIM}_{VARIANT}",
    f"train{TRAIN_YEARS}Y_test{TEST_MONTHS}M_step{STEP_MONTHS}M",
    f"ep{EPOCHS}"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

oos_csv_path = os.path.join(
    FIGURES_DIR,
    f"oos_rolling_{USE}_dim{LATENT_DIM}_train{TRAIN_YEARS}Y_test{TEST_MONTHS}M_step{STEP_MONTHS}M.csv"
)

SAVE_PER_ROLL_PLOTS = True  # set False if you only want the overall curve plot

# ============================= Load Data ===============================
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor_full.float()

meta = meta_full.copy()
meta["as_of_date"] = pd.to_datetime(meta["as_of_date"])
meta = meta.reset_index(drop=True)

assert len(meta) == X_tensor.shape[0], "meta and X_tensor length mismatch"

# ============================= Helpers ===============================
def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)

@torch.no_grad()
def predict_S_hat(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    outs = []
    N = X.shape[0]
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size].to(device)
        out = model(xb)
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)

def rmse_bps_on_subset(model: nn.Module, X_sub: torch.Tensor, meta_sub: pd.DataFrame):
    S_hat = predict_S_hat(model, X_sub, batch_size=EVAL_BATCH_SIZE)

    mask = row_finite_mask(X_sub) & row_finite_mask(S_hat)
    n_bad = int((~mask).sum().item())
    n_good = int(mask.sum().item())

    X_eval = X_sub[mask]
    S_eval = S_hat[mask]
    meta_eval = meta_sub.loc[mask.numpy()].reset_index(drop=True)

    rmse_per_ccy = H.rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)
    avg_rmse_bps = float(rmse_per_ccy.mean())  # unweighted mean across currencies (recommended)

    return rmse_per_ccy, avg_rmse_bps, n_good, n_bad

def make_loader(X_sub: torch.Tensor, batch_size: int):
    ds = TensorDataset(X_sub)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

WINDOW_SEED = 0  # fixed seed for every rolling window — change to reproduce

def train_one_window(X_train: torch.Tensor):
    torch.manual_seed(WINDOW_SEED)
    np.random.seed(WINDOW_SEED)

    model = FullModel(latent_dim=LATENT_DIM).to(device)
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=max_lr)
    loader = make_loader(X_train, BATCH_SIZE)

    scheduler = OneCycleLR(
        optim,
        max_lr=max_lr,
        steps_per_epoch=len(loader),
        epochs=EPOCHS,
        pct_start=0.3,
        div_factor=1.0,
        final_div_factor=final_div_factor,
    )

    loss_fn = nn.MSELoss()
    train_mse_hist = []
    lr_hist = []
    nan_batches_total = 0

    for epoch in range(EPOCHS):
        running = 0.0
        n_obs = 0
        nan_batches = 0

        for (xb_cpu,) in loader:
            xb = xb_cpu.to(device)

            optim.zero_grad(set_to_none=True)
            S_hat = model(xb)

            loss = loss_fn(S_hat, xb)
            if not torch.isfinite(loss):
                nan_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            running += float(loss.detach().cpu()) * xb.shape[0]
            n_obs += xb.shape[0]

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

# ============================= Build rolling schedule ===============================
date_min = max(meta["as_of_date"].min(), pd.Timestamp("2010-01-01"))
date_max = meta["as_of_date"].max()

start = date_min + pd.DateOffset(years=TRAIN_YEARS)
end = date_max - pd.DateOffset(months=TEST_MONTHS)

roll_starts = []
d = start
while d <= end:
    roll_starts.append(d)
    d = d + pd.DateOffset(months=STEP_MONTHS)

if len(roll_starts) == 0:
    raise RuntimeError("No rolling windows possible. Dataset may be too short for chosen TRAIN_YEARS/TEST_MONTHS.")

print(f"Rolling windows: {len(roll_starts)} from {roll_starts[0].date()} to {roll_starts[-1].date()}")

# ============================= CSV header ===============================
cols = (
    ["roll_start", "train_start", "train_end", "test_start", "test_end",
     "n_train", "n_test", "n_test_good", "n_test_bad",
     "time_train_sec", "time_test_sec",
     "avg_rmse_bps"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
)
pd.DataFrame(columns=cols).to_csv(oos_csv_path, index=False)
print("OOS CSV:", oos_csv_path)

# ============================= Run manifest ===============================
manifest = {
    "window_seed": WINDOW_SEED,
    "latent_dim": LATENT_DIM,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_lr": max_lr,
    "final_div_factor": final_div_factor,
    "train_years": TRAIN_YEARS,
    "test_months": TEST_MONTHS,
    "step_months": STEP_MONTHS,
    "n_windows": len(roll_starts),
    "torch_version": torch.__version__,
    "numpy_version": np.__version__,
    "run_started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "window_results": {},
}
manifest_path = os.path.join(FIGURES_DIR, "run_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest initialised: {manifest_path}")

# ============================= Rolling loop ===============================
avg_rmse_curve = []  # list of (test_start_date, avg_rmse_bps)

for k, test_start in enumerate(roll_starts):
    train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
    train_end = test_start - pd.DateOffset(days=1)

    test_end = test_start + pd.DateOffset(months=TEST_MONTHS) - pd.DateOffset(days=1)

    m_train = (meta["as_of_date"] >= train_start) & (meta["as_of_date"] <= train_end)
    m_test = (meta["as_of_date"] >= test_start) & (meta["as_of_date"] <= test_end)

    n_train = int(m_train.sum())
    n_test = int(m_test.sum())

    print(f"\n[{k+1:02d}/{len(roll_starts)}] "
          f"train {train_start.date()}..{train_end.date()} (n={n_train}) | "
          f"test {test_start.date()}..{test_end.date()} (n={n_test})")

    if n_train < MIN_TRAIN_OBS or n_test == 0:
        print("  Skipping (too few observations).")
        continue

    X_train = X_tensor[m_train.values]
    X_test = X_tensor[m_test.values]
    meta_test = meta.loc[m_test.values].reset_index(drop=True)

    # Train
    t0 = time.perf_counter()
    model, train_mse_hist, lr_hist = train_one_window(X_train)
    t1 = time.perf_counter()

    # Test
    t2 = time.perf_counter()
    rmse_per_ccy, avg_rmse_bps, n_good, n_bad = rmse_bps_on_subset(model, X_test, meta_test)
    t3 = time.perf_counter()

    time_train = t1 - t0
    time_test = t3 - t2

    avg_rmse_curve.append((test_start, avg_rmse_bps))

    # Row for CSV
    row = {
        "roll_start": test_start.date().isoformat(),
        "train_start": train_start.date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": test_end.date().isoformat(),
        "n_train": n_train,
        "n_test": n_test,
        "n_test_good": n_good,
        "n_test_bad": n_bad,
        "time_train_sec": time_train,
        "time_test_sec": time_test,
        "avg_rmse_bps": avg_rmse_bps,
    }
    for ccy in ccy_order:
        row[f"rmse_bps_{ccy}"] = float(rmse_per_ccy.get(ccy, np.nan))

    pd.DataFrame([row], columns=cols).to_csv(oos_csv_path, mode="a", header=False, index=False)

    # update manifest after every window (crash-safe)
    manifest["window_results"][test_start.date().isoformat()] = {
        "avg_rmse_bps":     round(avg_rmse_bps, 4),
        "per_ccy_bps":      {ccy: round(float(rmse_per_ccy.get(ccy, np.nan)), 4) for ccy in ccy_order},
        "train_minutes":    round(time_train / 60, 2),
        "n_train":          n_train,
        "n_test":           n_test,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  OOS avg_rmse_bps={avg_rmse_bps:.3f} | time_train={time_train/60:.1f}min time_test={time_test:.1f}s")

    if SAVE_PER_ROLL_PLOTS:
        # Per-roll LR plot
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
        ax.plot(np.arange(len(lr_hist)), lr_hist, linewidth=1.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"LR — roll {k+1:02d} test_start={test_start.date()}")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"lr_roll{k+1:02d}_{test_start.date()}.png"), dpi=300)
        plt.close(fig)

        # Per-roll train RMSE plot
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
        ax.plot(np.arange(len(train_mse_hist)), np.sqrt(train_mse_hist), linewidth=1.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train RMSE")
        ax.set_title(f"Train RMSE — roll {k+1:02d} test_start={test_start.date()}")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"train_rmse_roll{k+1:02d}_{test_start.date()}.png"), dpi=300)
        plt.close(fig)

manifest["run_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest finalised: {manifest_path}")
print("\nRolling OOS done.")

# ============================= Plot overall OOS avg RMSE curve ===============================
if len(avg_rmse_curve) > 0:
    dates = [d for d, v in avg_rmse_curve]
    vals = [v for d, v in avg_rmse_curve]

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=160)
    ax.plot(dates, vals, marker="o", linewidth=1.0)
    ax.set_xlabel("Test window start date")
    ax.set_ylabel("Average OOS RMSE (bps)")
    ax.set_title(f"OOS rolling avg RMSE (bps) — train={TRAIN_YEARS}Y test={TEST_MONTHS}M step={STEP_MONTHS}M")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "oos_avg_rmse_bps_curve.png"), dpi=300)
    plt.close(fig)

    print("Saved OOS avg RMSE curve plot:", os.path.join(FIGURES_DIR, "oos_avg_rmse_bps_curve.png"))
else:
    print("No OOS curve points to plot (all windows skipped).")