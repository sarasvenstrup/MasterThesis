# ============================= Import Packages ===============================
import time
import numpy as np
import os
import sys
import torch
import torch.nn as nn

torch.set_num_threads(4)  # --- Torch thread settings MUST be first Torch-related thing ---
torch.set_num_interop_threads(2)

import json
import pandas as pd
import matplotlib.pyplot as plt
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

# Baseline pipeline: imports from full_model — no config, no stable imports
from Code.utils import helpers as H
from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS
from Code.model.full_model import FullModel

VARIANT = "baseline"  # frozen — hardcoded, never reads config.py

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("Active model variant: baseline (frozen) — FIXED LR experiment")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

torch.backends.mkldnn.enabled = True
USE_SET_TO_NONE = True
print("CPU threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())
print("MKLDNN enabled:", torch.backends.mkldnn.enabled)

# ==========================================================
# Settings
# ==========================================================

SHOW_PLOTS = False

LATENT_DIM = 3
EPOCHS = 5000
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256

EVAL_EVERY = 1
LOG_EVERY = 100
TARGET_MSE = 1e-8

FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", "TrainingResults", f"dim{LATENT_DIM}_baseline_fixed_lr", f"ep{EPOCHS}")
os.makedirs(FIGURES_DIR, exist_ok=True)

USE = "bbg"
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

SEED = 0
FIXED_LR = 10e-3  # constant learning rate — no scheduler

torch.manual_seed(SEED)
model = FullModel(latent_dim=LATENT_DIM).to(device)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=FIXED_LR)
# No scheduler — learning rate stays constant throughout training

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

torch_version = torch.__version__
python_version = sys.version.split()[0]
numpy_version = np.__version__

csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec", "train_mse", "train_rmse",
     "avg_rmse_bps", "n_good", "n_bad"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
    + ["torch_version", "python_version", "numpy_version"]
)

pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "seed": SEED,
    "latent_dim": LATENT_DIM,
    "variant": VARIANT,
    "scheduler": "none (fixed lr)",
    "fixed_lr": FIXED_LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "mkldnn_enabled": torch.backends.mkldnn.enabled,
    "torch_version": torch_version,
    "python_version": python_version,
    "numpy_version": numpy_version,
}
config_path = os.path.join(FIGURES_DIR, "run_config.json")
with open(config_path, "w") as f:
    json.dump(run_config, f, indent=2)
print("Saved run config:", config_path)

# ==========================================================
# Train
# ==========================================================
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

    for batch_idx, (xb_cpu,) in enumerate(loader):
        xb = xb_cpu.to(device)

        optim.zero_grad(set_to_none=USE_SET_TO_NONE)

        try:
            S_hat = model(xb)
        except Exception as e:
            nan_batches += 1
            print(f"    [Forward error at epoch {epoch}, batch {batch_idx}]: {str(e)[:200]}")
            print(f"      Input range: [{xb.min():.3e}, {xb.max():.3e}], shape: {xb.shape}")
            continue

        if not torch.isfinite(S_hat).all():
            nan_batches += 1
            nan_count = (~torch.isfinite(S_hat)).sum().item()
            print(f"    [S_hat has NaN/Inf at epoch {epoch}, batch {batch_idx}]")
            print(f"      S_hat contains {nan_count} NaN/Inf values out of {S_hat.numel()}")
            finite_vals = S_hat[torch.isfinite(S_hat)]
            if finite_vals.numel() > 0:
                print(f"      S_hat range: [{finite_vals.min():.3e}, {finite_vals.max():.3e}]")
            else:
                print(f"      S_hat range: all values are NaN/Inf")
            print(f"      Input range: [{xb.min():.3e}, {xb.max():.3e}]")
            continue

        loss = loss_fn(S_hat, xb)

        if not torch.isfinite(loss):
            nan_batches += 1
            print(f"    [Loss is NaN/Inf at epoch {epoch}, batch {batch_idx}]")
            continue

        loss.backward()

        has_nan_grad = False
        nan_grad_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    has_nan_grad = True
                    nan_count = (~torch.isfinite(param.grad)).sum().item()
                    grad_range = param.grad[torch.isfinite(param.grad)]
                    if len(grad_range) > 0:
                        grad_min, grad_max = grad_range.min().item(), grad_range.max().item()
                    else:
                        grad_min, grad_max = float('nan'), float('nan')
                    nan_grad_params.append({
                        'name': name,
                        'nan_count': nan_count,
                        'total': param.grad.numel(),
                        'grad_min': grad_min,
                        'grad_max': grad_max,
                    })

        if has_nan_grad:
            nan_batches += 1
            print(f"    [Gradients contain NaN/Inf at epoch {epoch}, batch {batch_idx}]")
            for param_info in nan_grad_params:
                print(f"      {param_info['name']}: {param_info['nan_count']}/{param_info['total']} NaN")
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        # No scheduler.step()

        lrs_per_step.append(optim.param_groups[0]["lr"])
        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

    nan_batches_total += nan_batches
    epoch_mse = running / max(n_obs, 1)
    train_losses.append(epoch_mse)
    lrs_per_epoch.append(optim.param_groups[0]["lr"])
    epoch_rmse = epoch_mse ** 0.5

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

        row["torch_version"] = torch_version
        row["python_version"] = python_version
        row["numpy_version"] = numpy_version

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

# 1) Learning rate plot (per step) — will be a flat line for fixed lr
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
ax.plot(np.arange(len(lrs_per_step)), lrs_per_step, linewidth=1.0)
ax.set_xlabel("Training step (batch)")
ax.set_ylabel("Learning rate")
ax.set_title(f"Learning rate — fixed (dim={LATENT_DIM}, ep={EPOCHS})")
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
    ax.set_title(f"Average RMSE across currencies (bps) — convergence (dim={LATENT_DIM}, fixed lr)")
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
# Save latent coordinates
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

Z = get_latent(model, X_tensor, batch_size=EVAL_BATCH_SIZE)

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
    f"fullmodel_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_fixed_lr.pt"
)

torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {"latent_dim": LATENT_DIM},
    "latent_dim": LATENT_DIM,
    "epochs": EPOCHS,
    "use_data": USE,
    "variant": VARIANT,
    "scheduler": "none (fixed lr)",
    "fixed_lr": FIXED_LR,
}, checkpoint_path)

print("Saved checkpoint:", checkpoint_path)

figures_ckpt_path = os.path.join(FIGURES_DIR, f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt")
torch.save(model.state_dict(), figures_ckpt_path)
print("Saved figures checkpoint:", figures_ckpt_path)

# ==========================================================
# Parameter plots over time
# ==========================================================
print("\nGenerating parameter plots...")

_ccy_colors = {c: custom_palette[i % len(custom_palette)] for i, c in enumerate(ccy_order)}


def extract_parameters(mdl, X_data, meta_df):
    """Extract μ, σ, ρ, r̃ for every observation. Returns a DataFrame."""
    mdl.eval()
    with torch.no_grad():
        X_d = X_data.to(device)
        z       = mdl.encoder(X_d)
        mu      = mdl.K(z)
        sigmas, rhos = mdl.H(z)
        r_til   = mdl.R(z).squeeze(-1)

    d      = mdl.latent_dim
    n_corr = d * (d - 1) // 2

    rec = meta_df.copy().reset_index(drop=True)
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


def param_label(name):
    if name.startswith("mu_"):
        k = name.split("_")[1];  return r"$\mu_{" + k + r"}$"
    if name.startswith("sigma_"):
        k = name.split("_")[1];  return r"$\sigma_{" + k + r"}$"
    if name.startswith("rho_"):
        ij = name.split("_")[1]; return r"$\rho_{" + ",".join(ij) + r"}$"
    if name == "r_tilde":
        return r"$\tilde{r}$"
    return name


# Use only finite rows
_param_mask = torch.isfinite(X_tensor).all(dim=1)
df_params = extract_parameters(model, X_tensor[_param_mask], meta.loc[_param_mask.numpy()].reset_index(drop=True))

d = LATENT_DIM
mu_cols    = [f"mu_{k+1}"    for k in range(d)]
sig_cols   = [f"sigma_{k+1}" for k in range(d)]
rho_cols   = [f"rho_{i+1}{j+1}" for i in range(d) for j in range(i + 1, d)]
param_cols = mu_cols + sig_cols + rho_cols + ["r_tilde"]

params_dir = os.path.join(FIGURES_DIR, "parameters")
os.makedirs(params_dir, exist_ok=True)

for col in param_cols:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for ccy in ccy_order:
        sub = df_params[df_params["ccy"] == ccy].sort_values("as_of_date")
        if sub.empty:
            continue
        ax.plot(sub["as_of_date"], sub[col],
                color=_ccy_colors[ccy], linewidth=0.8, alpha=0.75)
    ax.set_title(param_label(col), fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(params_dir, col + ".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("Parameter plots done.")
