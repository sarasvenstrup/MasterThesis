# ============================= Import Packages ===============================
import time
import numpy as np
import os
import sys
import torch
import torch.nn as nn

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader

# ============================= Environment Setup & Imports ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))          # .../MasterThesis/Code
THESIS_ROOT = os.path.abspath(os.path.join(CODE_ROOT, ".."))         # .../MasterThesis

if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from Code import config
config.confirm_variant()
from Code.utils import helpers as H
from Code.load_swapdata import my_data
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
SHOW_PLOTS = True

LATENT_DIM = 2
EPOCHS = 100
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256

EVAL_EVERY = 1
LOG_EVERY = 1
TARGET_MSE = 1e-8

USE = "bbg"

# ---------- Base run to continue from ----------
BASE_EPOCHS = 200
BASE_RUN_DIR = os.path.join(
    THESIS_ROOT,
    "Figures",
    "TrainingResults",
    f"dim{LATENT_DIM}_{config.VARIANT}",
    f"ep{BASE_EPOCHS}",
)
BASE_BEST_CHECKPOINT_PATH = os.path.join(
    BASE_RUN_DIR,
    f"best_checkpoint_dim{LATENT_DIM}.pt",
)

# ---------- Continuation run output ----------
CONT_RUN_NAME = f"cont_from_ep{BASE_EPOCHS}_fixedcenter"
FIGURES_DIR = os.path.join(
    THESIS_ROOT,
    "Figures",
    "TrainingResults",
    f"dim{LATENT_DIM}_{config.VARIANT}",
    CONT_RUN_NAME,
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------- Model hyperparameters ----------
SIGMA_INIT = 0.0075
K_DRIFT_SCALE_INIT = 0.25
MAX_LR = 5e-5

# ---------- Fixed-center continuation ----------
USE_FIXED_CENTER = True
WARMSTART_FROM_PREVIOUS = True

# Fallback center if checkpoint is missing
MANUAL_CENTER = np.array([-0.04631, 0.04223], dtype=np.float32)

# ==========================================================
# Data
# ==========================================================
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

torch.manual_seed(0)

# ==========================================================
# Helpers
# ==========================================================
@torch.no_grad()
def estimate_latent_center(model: nn.Module, X: torch.Tensor, batch_size: int = 256):
    was_training = model.training
    model.eval()

    zs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        _, aux = model(xb, return_aux=True)
        zs.append(aux["z"].detach().cpu())

    z_all = torch.cat(zs, dim=0)
    z_mean = z_all.mean(dim=0)
    z_std = z_all.std(dim=0)

    if was_training:
        model.train()

    return z_mean, z_std

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

@torch.no_grad()
def print_stable_diagnostics(model: nn.Module, X: torch.Tensor, batch_size: int = 128):
    was_training = model.training
    model.eval()

    xb = X[:batch_size].to(device)
    _, aux = model(xb, return_aux=True)

    z = aux["z"]
    sigma = aux["sigma"]
    diag = torch.diagonal(sigma, dim1=1, dim2=2)

    print("  z mean:", z.mean(dim=0).cpu().numpy())
    print("  z std :", z.std(dim=0).cpu().numpy())

    if hasattr(model.K, "kappa"):
        print("  kappa:", model.K.kappa().detach().cpu().numpy())

    if hasattr(model.K, "theta"):
        print("  theta:", model.K.theta.detach().cpu().numpy())

    print("  sigma diag mean:", diag.mean(dim=0).cpu().numpy())

    if was_training:
        model.train()

@torch.no_grad()
def print_arb_diagnostics(model: nn.Module, X: torch.Tensor, batch_size: int = 16):
    was_training = model.training
    model.eval()

    xb = X[:batch_size].to(device)
    _, aux = model(xb, return_aux=True, do_arb_checks=True)
    arb = aux["arb"]

    if arb is not None:
        print("  max_abs_R mean :", arb["max_abs_R"].mean().item())
        print("  max_abs_SR mean:", arb["max_abs_SR_1to30"].mean().item())

    if was_training:
        model.train()

# ==========================================================
# Build center config + warm-start state
# ==========================================================
k_z_center_init = None
k_learn_center = True
center_source_used = None
warmstart_state_dict = None

if USE_FIXED_CENTER and os.path.exists(BASE_BEST_CHECKPOINT_PATH):
    print("Loading checkpoint for latent-center estimation:")
    print(BASE_BEST_CHECKPOINT_PATH)

    center_model = FullModel(
        latent_dim=LATENT_DIM,
        sigma_init=SIGMA_INIT,
        k_drift_scale_init=K_DRIFT_SCALE_INIT,
        k_learn_center=True,
    ).to(device)

    warmstart_state_dict = torch.load(BASE_BEST_CHECKPOINT_PATH, map_location=device)
    center_model.load_state_dict(warmstart_state_dict)

    z_center_mean, z_center_std = estimate_latent_center(
        center_model, X_tensor, batch_size=EVAL_BATCH_SIZE
    )

    k_z_center_init = z_center_mean.numpy().astype(np.float32)
    k_learn_center = False
    center_source_used = BASE_BEST_CHECKPOINT_PATH

    print("Estimated latent center from checkpoint:", k_z_center_init)
    print("Estimated latent std from checkpoint   :", z_center_std.numpy())

    del center_model

elif USE_FIXED_CENTER:
    print("Base checkpoint not found.")
    print("Falling back to MANUAL_CENTER:", MANUAL_CENTER)

    k_z_center_init = MANUAL_CENTER.copy()
    k_learn_center = False
    center_source_used = "MANUAL_CENTER"

print("Final K center init:", k_z_center_init)
print("Final K learn_center:", k_learn_center)

# ==========================================================
# Create continuation model
# ==========================================================
model = FullModel(
    latent_dim=LATENT_DIM,
    sigma_init=SIGMA_INIT,
    k_drift_scale_init=K_DRIFT_SCALE_INIT,
    k_z_center_init=k_z_center_init,
    k_learn_center=k_learn_center,
).to(device)

if WARMSTART_FROM_PREVIOUS and warmstart_state_dict is not None:
    missing, unexpected = model.load_state_dict(warmstart_state_dict, strict=False)
    print("Warm-start loaded.")
    print("Missing keys   :", missing)
    print("Unexpected keys:", unexpected)

    # Overwrite theta with the fixed center
    if hasattr(model.K, "theta") and k_z_center_init is not None:
        with torch.no_grad():
            model.K.theta.copy_(
                torch.as_tensor(k_z_center_init, device=device, dtype=model.K.theta.dtype)
            )
        print("Overwrote K.theta with fixed center:", model.K.theta.detach().cpu().numpy())

optim = torch.optim.Adam(model.parameters(), lr=MAX_LR)

scheduler = OneCycleLR(
    optim,
    max_lr=MAX_LR,
    steps_per_epoch=len(loader),
    epochs=EPOCHS,
    pct_start=0.3,
    div_factor=10.0,
    final_div_factor=1000.0,
)

loss_fn = nn.MSELoss()

# ==========================================================
# Initial sanity check
# ==========================================================
with torch.no_grad():
    xb0 = X_tensor[:8].to(device)
    S0, aux0 = model(xb0, return_aux=True)

    print("Initial forward OK")
    print("Initial S_hat finite:", torch.isfinite(S0).all().item())
    print("Initial z mean:", aux0["z"].mean(dim=0).cpu().numpy())
    print("Initial z std :", aux0["z"].std(dim=0).cpu().numpy())

    if hasattr(model.K, "kappa"):
        print("Initial kappa:", model.K.kappa().detach().cpu().numpy())
    if hasattr(model.K, "theta"):
        print("Initial theta:", model.K.theta.detach().cpu().numpy())

    diag0 = torch.diagonal(aux0["sigma"], dim1=1, dim2=2)
    print("Initial sigma diag mean:", diag0.mean(dim=0).cpu().numpy())

# ==========================================================
# CSV logger setup
# ==========================================================
ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
csv_path = os.path.join(FIGURES_DIR, f"train_rmse_log_{USE}_dim{LATENT_DIM}_{CONT_RUN_NAME}.csv")

csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec", "train_mse", "train_rmse",
     "avg_rmse_bps", "n_good", "n_bad"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
)

pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

# ==========================================================
# Continuation training
# ==========================================================
best_avg_rmse_bps = float("inf")
best_checkpoint_path = os.path.join(FIGURES_DIR, f"best_checkpoint_dim{LATENT_DIM}.pt")

train_losses = []
lrs_per_step = []
lrs_per_epoch = []
avg_rmse_bps_hist = []
nan_batches_total = 0

t0 = time.perf_counter()
t_last_log = t0

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    n_obs = 0
    nan_batches = 0
    grad_norm_epoch = []

    for (xb_cpu,) in loader:
        xb = xb_cpu.to(device)

        optim.zero_grad(set_to_none=USE_SET_TO_NONE)

        S_hat = model(xb)
        if not torch.isfinite(S_hat).all():
            nan_batches += 1
            continue

        loss = loss_fn(S_hat, xb)
        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        loss.backward()

        all_grads_finite = True
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                all_grads_finite = False
                break

        if not all_grads_finite:
            nan_batches += 1
            optim.zero_grad(set_to_none=USE_SET_TO_NONE)
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_epoch.append(float(grad_norm))

        optim.step()
        scheduler.step()

        lrs_per_step.append(optim.param_groups[0]["lr"])
        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

    if n_obs == 0:
        print("[ABORT] No valid batches were processed this epoch. Stopping.")
        break

    nan_batches_total += nan_batches
    epoch_mse = running / max(n_obs, 1)
    train_losses.append(epoch_mse)
    lrs_per_epoch.append(optim.param_groups[0]["lr"])
    epoch_rmse = epoch_mse ** 0.5
    mean_grad_norm = float(np.mean(grad_norm_epoch)) if len(grad_norm_epoch) > 0 else np.nan

    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    do_log = ((epoch + 1) % LOG_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)

    if TARGET_MSE > 0 and n_obs > 0 and epoch_mse <= TARGET_MSE:
        do_eval = True
        do_log = True

    if do_eval:
        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = eval_rmse_bps(
            model, X_tensor, meta, batch_size=EVAL_BATCH_SIZE
        )
        avg_rmse_bps_hist.append((epoch, avg_rmse_bps))

        if np.isfinite(avg_rmse_bps) and avg_rmse_bps < best_avg_rmse_bps:
            best_avg_rmse_bps = avg_rmse_bps
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"[BEST] epoch={epoch} avg_rmse_bps={avg_rmse_bps:.3f} saved to {best_checkpoint_path}")
    else:
        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = (None, np.nan, np.nan, np.nan)

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
            f"time_total={time_total/60:.1f}min interval={time_interval/60:.1f}min "
            f"mean_grad_norm={mean_grad_norm:.3e}"
        )

        print_stable_diagnostics(model, X_tensor)
        print_arb_diagnostics(model, X_tensor)

    if TARGET_MSE > 0 and n_obs > 0 and epoch_mse <= TARGET_MSE:
        print(f"[STOP] epoch={epoch} train_rmse={epoch_rmse:.6e} lr={optim.param_groups[0]['lr']:.2e}")
        break

print("Continuation training done.")

# ==========================================================
# Plots
# ==========================================================
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
ax.plot(np.arange(len(lrs_per_step)), lrs_per_step, linewidth=1.0)
ax.set_xlabel("Training step (batch)")
ax.set_ylabel("Learning rate")
ax.set_title(f"Continuation LR schedule — OneCycleLR (dim={LATENT_DIM}, ep={EPOCHS})")
fig.tight_layout()
lr_fig_path = os.path.join(FIGURES_DIR, f"lr_schedule_{USE}_dim{LATENT_DIM}_{CONT_RUN_NAME}.png")
fig.savefig(lr_fig_path, dpi=300)
print("Saved LR plot:", lr_fig_path)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

if len(avg_rmse_bps_hist) > 0:
    epochs_logged = [e for e, v in avg_rmse_bps_hist]
    avg_logged = [v for e, v in avg_rmse_bps_hist]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.plot(epochs_logged, avg_logged, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average RMSE (bps)")
    ax.set_title(f"Continuation average RMSE across currencies (bps) — dim={LATENT_DIM}")
    fig.tight_layout()
    rmse_fig_path = os.path.join(FIGURES_DIR, f"avg_rmse_bps_convergence_{USE}_dim{LATENT_DIM}_{CONT_RUN_NAME}.png")
    fig.savefig(rmse_fig_path, dpi=300)
    print("Saved avg RMSE plot:", rmse_fig_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
else:
    print("No RMSE history to plot (avg_rmse_bps_hist empty).")

# ==========================================================
# Save trained model checkpoint
# ==========================================================
CHECKPOINT_DIR = os.path.join(THESIS_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    f"fullmodel_{USE}_dim{LATENT_DIM}_{CONT_RUN_NAME}.pt"
)

model_config = {
    "latent_dim": LATENT_DIM,
    "sigma_init": SIGMA_INIT,
    "k_drift_scale_init": K_DRIFT_SCALE_INIT,
    "k_z_center_init": k_z_center_init.tolist() if k_z_center_init is not None else None,
    "k_learn_center": k_learn_center,
    "max_lr": MAX_LR,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "variant": config.VARIANT,
    "center_source_used": center_source_used,
    "warmstart_from_previous": WARMSTART_FROM_PREVIOUS,
    "base_best_checkpoint_path": BASE_BEST_CHECKPOINT_PATH,
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

figures_ckpt_path = os.path.join(FIGURES_DIR, f"checkpoint_dim{LATENT_DIM}_{CONT_RUN_NAME}.pt")
torch.save(model.state_dict(), figures_ckpt_path)
print("Saved figures checkpoint:", figures_ckpt_path)