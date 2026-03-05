# ============================= Import Packages ===============================
import time
import numpy as np
import os
import sys
import torch
import torch.nn as nn
torch.set_num_threads(4) # --- Torch thread settings MUST be first Torch-related thing ---
torch.set_num_interop_threads(2)
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from torch.optim.lr_scheduler import OneCycleLR

# ============================= Environment Setup & Imports ===============================

# First we set out working directory, in order for all our outputs to be saved in the same folder.

try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# We now import the needed components, like objects, models, helper functions and data in order to train the model.

from Code.utils import helpers as H
from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS
from Code.model.full_model import FullModel

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# The following line accelerates deep learning operations on CPU, helping us to improve performance when training.
torch.backends.mkldnn.enabled = True

# The following line sets all .grad attributes to None instead of zero in order to lessen memory traffic.
USE_SET_TO_NONE = True
print("CPU threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())


# ==========================================================
# Settings
# ==========================================================
LATENT_DIM = 2
EPOCHS = 5000                      # <-- YOU WANTED 5000
LOG_EVERY = 500                    # <-- save metrics every 500 epochs
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256

TARGET_MSE = 1e-6  # keep if you still want early-stop; otherwise set to -1 or remove

FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", f"dim{LATENT_DIM}", f"ep{EPOCHS}")
os.makedirs(FIGURES_DIR, exist_ok=True)

USE = "bbg"
meta, X_tensor, tenors, df_wide, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

torch.manual_seed(0)
model = FullModel(latent_dim=LATENT_DIM).to(device)
model.train()

max_lr = 3e-3
optim = torch.optim.Adam(model.parameters(), lr=max_lr)

scheduler = OneCycleLR(
    optim,
    max_lr=max_lr,
    steps_per_epoch=len(loader),
    epochs=EPOCHS,
    pct_start=0.3,
    div_factor=1.0,
    final_div_factor=3000.0
)

loss_fn = nn.MSELoss()

# ==========================================================
# Helpers: inference + RMSE logging
# ==========================================================
def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)

@torch.no_grad()
def predict_S_hat(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    outs = []
    N = X.shape[0]
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size].to(device)
        out = model(xb)
        S_hat = out[0]
        outs.append(S_hat.detach().cpu())
    return torch.cat(outs, dim=0)

def eval_rmse_bps(model: nn.Module, X_full: torch.Tensor, meta_full: pd.DataFrame, batch_size: int = 256):
    """
    Returns:
      rmse_per_ccy (pd.Series): index=ccy, values=RMSE in bps
      avg_rmse_bps (float): mean across currencies in rmse_per_ccy
      n_bad (int): filtered rows (non-finite)
      n_good (int): kept rows
    """
    S_hat_all = predict_S_hat(model, X_full, batch_size=batch_size)

    mask = row_finite_mask(X_full) & row_finite_mask(S_hat_all)
    n_bad = int((~mask).sum().item())
    n_good = int(mask.sum().item())

    X_eval = X_full[mask]
    S_eval = S_hat_all[mask]
    meta_eval = meta_full.loc[mask.numpy()].reset_index(drop=True)

    # Your helper already returns RMSE in bps per currency
    rmse_per_ccy = H.rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)  # pd.Series
    avg_rmse_bps = float(rmse_per_ccy.mean())

    return rmse_per_ccy, avg_rmse_bps, n_bad, n_good

# ==========================================================
# CSV logger setup
# ==========================================================
ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
csv_path = os.path.join(FIGURES_DIR, f"train_rmse_log_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.csv")

# Prepare header
csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec", "train_mse", "train_rmse", "avg_rmse_bps", "n_good", "n_bad"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
)

# write header once
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

# ==========================================================
# Train (with timing + periodic eval)
# ==========================================================
train_losses = []
lrs_per_step = []          # per-batch LR (best for OneCycle)
lrs_per_epoch = []         # optional per-epoch snapshot
avg_rmse_bps_hist = []     # (epoch, avg_rmse_bps)
nan_batches_total = 0

t0 = time.perf_counter()
t_last_log = t0

global_step = 0

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    n_obs = 0
    nan_batches = 0

    for (xb_cpu,) in loader:
        xb = xb_cpu.to(device)

        optim.zero_grad(set_to_none=True)
        out = model(xb)
        S_hat = out[0]

        loss = loss_fn(S_hat, xb)
        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

        # record LR per batch (so you can really see OneCycle shape)
        lrs_per_step.append(optim.param_groups[0]["lr"])
        global_step += 1

        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

    nan_batches_total += nan_batches
    epoch_mse = running / max(n_obs, 1)
    train_losses.append(epoch_mse)
    lrs_per_epoch.append(optim.param_groups[0]["lr"])

    epoch_rmse = epoch_mse ** 0.5

    # Optional early stop
    if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE:
        print(f"[STOP] epoch={epoch} train_rmse={epoch_rmse:.6e} lr={optim.param_groups[0]['lr']:.2e}")
        # still log metrics at stop
        do_log = True
    else:
        do_log = ((epoch + 1) % LOG_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)

    if do_log:
        t_now = time.perf_counter()
        time_total = t_now - t0
        time_interval = t_now - t_last_log
        t_last_log = t_now

        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = eval_rmse_bps(
            model, X_tensor, meta, batch_size=EVAL_BATCH_SIZE
        )
        avg_rmse_bps_hist.append((epoch, avg_rmse_bps))

        # Build one row for CSV (ensure every ccy column exists)
        row = {
            "epoch": epoch,
            "time_total_sec": time_total,
            "time_interval_sec": time_interval,
            "train_mse": epoch_mse,
            "train_rmse": epoch_rmse,
            "avg_rmse_bps": avg_rmse_bps,
            "n_good": n_good,
            "n_bad": n_bad,
        }
        for ccy in ccy_order:
            row[f"rmse_bps_{ccy}"] = float(rmse_per_ccy.get(ccy, np.nan))

        pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)

        print(
            f"epoch={epoch:4d} train_rmse={epoch_rmse:.6e} "
            f"avg_rmse_bps={avg_rmse_bps:.3f} lr={optim.param_groups[0]['lr']:.2e} "
            f"used_obs={n_obs} nan_batches={nan_batches} total_nan_batches={nan_batches_total} "
            f"time_total={time_total/60:.1f}min interval={time_interval/60:.1f}min"
        )

    if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE:
        break

print("Training done.")

# ==========================================================
# Plots
# ==========================================================
# 1) Learning rate plot (per step)
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
ax.plot(np.arange(len(lrs_per_step)), lrs_per_step, linewidth=1.0)
ax.set_xlabel("Training step (batch)")
ax.set_ylabel("Learning rate")
ax.set_title(f"Learning rate schedule — OneCycleLR (dim={LATENT_DIM}, ep={EPOCHS})")
fig.tight_layout()
lr_fig_path = os.path.join(FIGURES_DIR, f"lr_schedule_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.png")
fig.savefig(lr_fig_path, dpi=300)
plt.close(fig)
print("Saved LR plot:", lr_fig_path)

# 2) Average RMSE (bps) convergence plot (logged every 500 epochs)
if len(avg_rmse_bps_hist) > 0:
    epochs_logged = [e for e, v in avg_rmse_bps_hist]
    avg_logged = [v for e, v in avg_rmse_bps_hist]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.plot(epochs_logged, avg_logged, marker="o", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average RMSE (bps)")
    ax.set_title(f"Average RMSE across currencies (bps) — convergence (dim={LATENT_DIM})")
    fig.tight_layout()
    rmse_fig_path = os.path.join(FIGURES_DIR, f"avg_rmse_bps_convergence_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.png")
    fig.savefig(rmse_fig_path, dpi=300)
    plt.close(fig)
    print("Saved avg RMSE plot:", rmse_fig_path)
else:
    print("No RMSE history to plot (avg_rmse_bps_hist empty).")