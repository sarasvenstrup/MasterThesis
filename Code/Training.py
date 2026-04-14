# ============================= Import Packages ===============================
import time
import numpy as np
import os
import sys
import torch
import torch.nn as nn

torch.set_num_threads(4)  # --- Torch thread settings MUST be first Torch-related thing ---
torch.set_num_interop_threads(2)

import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader

# ============================= Environment Setup & Imports ===============================
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code import config
config.confirm_variant()
from Code.utils import helpers as H
from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS
from Code.model.full_model import FullModel

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("Active model variant from config.py:", config.VARIANT)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

torch.backends.mkldnn.enabled = True
USE_SET_TO_NONE = True
print("CPU threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())

# ==========================================================
# Settings
# ==========================================================

# --- User option: show plots interactively? ---
SHOW_PLOTS = True  # Set to False to only save plots

LATENT_DIM = 3
EPOCHS = 5000
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256

EVAL_EVERY = 1
LOG_EVERY = 500
TARGET_MSE = 1e-8

FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", "TrainingResults", f"dim{LATENT_DIM}_{config.VARIANT}", f"ep{EPOCHS}")
os.makedirs(FIGURES_DIR, exist_ok=True)

USE = "bbg"
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

torch.manual_seed(0)
model = FullModel(latent_dim=LATENT_DIM).to(device)
model.train()

max_lr = 1e-3
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
    was_training = model.training
    model.eval()

    outs = []
    N = X.shape[0]
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size].to(device)
        S_hat = model(xb)
        outs.append(S_hat.detach().cpu())

    if was_training:
        model.train()

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

    rmse_per_ccy = H.rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)
    avg_rmse_bps = float(rmse_per_ccy.mean())
    return rmse_per_ccy, avg_rmse_bps, n_bad, n_good

# ==========================================================
# CSV logger setup
# ==========================================================
ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
csv_path = os.path.join(FIGURES_DIR, f"train_rmse_log_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.csv")

csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec", "train_mse", "train_rmse",
     "avg_rmse_bps", "n_good", "n_bad"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
)

pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

# ==========================================================
# Train (with timing + eval every epoch + CSV every LOG_EVERY)
# ==========================================================
train_losses = []
lrs_per_step = []
lrs_per_epoch = []
avg_rmse_bps_hist = []  # (epoch, avg_rmse_bps) for EVERY epoch
nan_batches_total = 0

t0 = time.perf_counter()
t_last_log = t0

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    n_obs = 0
    nan_batches = 0

    for (xb_cpu,) in loader:
        xb = xb_cpu.to(device)

        optim.zero_grad(set_to_none=USE_SET_TO_NONE)
        S_hat = model(xb)

        loss = loss_fn(S_hat, xb)
        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

        lrs_per_step.append(optim.param_groups[0]["lr"])
        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

    nan_batches_total += nan_batches
    epoch_mse = running / max(n_obs, 1)
    train_losses.append(epoch_mse)
    lrs_per_epoch.append(optim.param_groups[0]["lr"])
    epoch_rmse = epoch_mse ** 0.5

    # ---- EVAL EVERY EPOCH (for plotting convergence) ----
    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    do_log = ((epoch + 1) % LOG_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)

    if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE and n_obs > 0:
        do_eval = True
        do_log = True

    if do_eval:
        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = eval_rmse_bps(
            model, X_tensor, meta, batch_size=EVAL_BATCH_SIZE
        )
        avg_rmse_bps_hist.append((epoch, avg_rmse_bps))
    else:
        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = (None, np.nan, np.nan, np.nan)

    # ---- LOG/CSV ONLY EVERY LOG_EVERY ----
    if do_log:
        t_now = time.perf_counter()
        time_total = t_now - t0
        time_interval = t_now - t_last_log
        t_last_log = t_now

        row = {
            "epoch": epoch,
            "time_total_sec": time_total,
            "time_interval_sec": time_interval,
            "train_mse": epoch_mse,
            "train_rmse": epoch_rmse,
            "avg_rmse_bps": float(avg_rmse_bps),
            "n_good": int(n_good) if np.isfinite(n_good) else np.nan,
            "n_bad": int(n_bad) if np.isfinite(n_bad) else np.nan,
        }

        for ccy in ccy_order:
            row[f"rmse_bps_{ccy}"] = float(rmse_per_ccy.get(ccy, np.nan)) if rmse_per_ccy is not None else np.nan

        pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)

        print(
            f"epoch={epoch:4d} train_rmse={epoch_rmse:.6e} "
            f"avg_rmse_bps={avg_rmse_bps:.3f} lr={optim.param_groups[0]['lr']:.2e} "
            f"used_obs={n_obs} nan_batches={nan_batches} total_nan_batches={nan_batches_total} "
            f"time_total={time_total/60:.1f}min interval={time_interval/60:.1f}min"
        )

    if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE and n_obs > 0:
        print(f"[STOP] epoch={epoch} train_rmse={epoch_rmse:.6e} lr={optim.param_groups[0]['lr']:.2e}")
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
print("Saved LR plot:", lr_fig_path)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

# 2) Average RMSE (bps) convergence plot
if len(avg_rmse_bps_hist) > 0:
    epochs_logged = [e for e, v in avg_rmse_bps_hist]
    avg_logged = [v for e, v in avg_rmse_bps_hist]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.plot(epochs_logged, avg_logged, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average RMSE (bps)")
    ax.set_title(f"Average RMSE across currencies (bps) — convergence (dim={LATENT_DIM})")
    fig.tight_layout()
    rmse_fig_path = os.path.join(FIGURES_DIR, f"avg_rmse_bps_convergence_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.png")
    fig.savefig(rmse_fig_path, dpi=300)
    print("Saved avg RMSE plot:", rmse_fig_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
else:
    print("No RMSE history to plot (avg_rmse_bps_hist empty).")

# ==========================================================
# Save latent coordinates (z_1, z_2) for scatter plotting
# ==========================================================

@torch.no_grad()
def get_latent(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    was_training = model.training
    model.eval()
    zs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        z = model.encoder(xb)
        zs.append(z.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(zs, dim=0)

Z = get_latent(model, X_tensor, batch_size=EVAL_BATCH_SIZE)  # (N, LATENT_DIM)

df_latent = meta.copy().reset_index(drop=True)
for k in range(LATENT_DIM):
    df_latent[f"z_{k+1}"] = Z[:, k].numpy()

latent_csv_path = os.path.join(FIGURES_DIR, f"latent_z_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.csv")
df_latent.to_csv(latent_csv_path, index=False)
print("Saved latent CSV:", latent_csv_path)

# ==========================================================
# Save trained model checkpoint
# ==========================================================
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    f"fullmodel_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.pt"
)

model_config = {
    "latent_dim": LATENT_DIM,
}

torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": model_config,
    "latent_dim": LATENT_DIM,
    "epochs": EPOCHS,
    "use_data": USE,
    "variant": config.VARIANT,
}, checkpoint_path)

print("Saved checkpoint:", checkpoint_path)

# Also save a plain state_dict alongside the training logs so ResultsGenerator
# can load the ep{EPOCHS} model directly from the Figures/dim{N}/ep{EPOCHS}/ folder.
figures_ckpt_path = os.path.join(FIGURES_DIR, f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt")
torch.save(model.state_dict(), figures_ckpt_path)
print("Saved figures checkpoint:", figures_ckpt_path)