# ==================== Joint Training: Reconstruction + Pricing ====================
"""
Single-stage training that optimizes BOTH objectives simultaneously from scratch:
  1. Reconstruction loss: fit market swap curves (MSE)
  2. Pricing loss: match swaption volatilities (Bachelier price error)

Key features:
  - Combined loss function → learns parameters from both objectives simultaneously
  - Default model priors → let the model learn appropriate volatility and drift from data
  - Per-group learning rates → handle different parameter scales:
    * H (volatility): Full LR (1e-3) - learns large values
    * K (drift): Slow LR (1e-4) - keeps drift stable and prevents overshoot
    * G (decoder): Moderate LR (5e-4) - balances reconstruction quality
  - No regularization (training from scratch, not fine-tuning)
  - Antithetic variates for pricing loss → reduces MC variance
  - Per-group gradient norm tracking → monitor each component's learning
  
Rationale for per-group LR:
  Empirically, H (diffusion/volatility) learns larger magnitude values than K (drift).
  Using slower LR for K prevents drift from becoming too weak relative to diffusion,
  which would cause simulation instability. This is NOT just for fine-tuning - it's
  about handling inherently different parameter scales and sensitivities.
"""

import time
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import math

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import json
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader

# ============================= Environment Setup ===============================
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
from Code.load_swapdata import my_data
from Code.model.full_model_stable import FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import bachelier_price_torch, swap_rate_torch
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Active model variant:", config.VARIANT)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

torch.backends.mkldnn.enabled = True
USE_SET_TO_NONE = True

# ==========================================================
# Settings
# ==========================================================

SHOW_PLOTS = False

LATENT_DIM = 4
EPOCHS = 3500
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256

EVAL_EVERY = 1
LOG_EVERY = 100
TARGET_MSE = 1e-8

# ── Joint training weights ──
LAMBDA_RECON = 1.0  # Reconstruction loss weight
LAMBDA_PRICE = 0.3  # Pricing loss weight (tune this: 0.05-0.5)

# ── Per-group learning rates ──
# Different components have different scales and sensitivities:
# - H (volatility): Learns larger values, needs full learning rate
# - K (drift): Smaller magnitude, slower learning → use lower LR to avoid instability
# - G (decoder): Core reconstruction, moderate LR
LR_H = 1e-3          # Volatility network (H_sigma) - full LR
LR_K = 1e-4          # Drift network (K_mu) - 10× slower to keep drift stable
LR_G = 5e-4          # Decoder (G) - moderate LR
LR_ENCODER = 1e-3    # Encoder - full LR
LR_R = 1e-3          # Short rate (R) - full LR

# ── Pricing mini-batch settings ──
N_SWAPTIONS_PER_BATCH = 2  # How many (expiry, tenor) pairs per batch
N_PATHS_PRICING = 256  # MC paths for pricing loss
DT_PRICING = 1 / 12  # Time step for simulation

USE = "bbg"
CCY_FILTER = "EUR"  # Currency for pricing calibration

FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", "TrainingResults",
                           f"dim{LATENT_DIM}_{config.VARIANT}_joint", f"ep{EPOCHS}")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================================
# Load data
# ==========================================================

meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

# Filter to CCY for pricing
meta_ccy = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# ==========================================================
# Initialize model
# ==========================================================

SEED = 0
PCT_START = 0.3
DIV_FACTOR = 1.0
FINAL_DIV_FACTOR = 3000.0

torch.manual_seed(SEED)
np.random.seed(SEED)

model = FullModel(
    latent_dim=LATENT_DIM
).to(device)
model.train()

# ── Per-group parameter groups ──────────────────────────────────────────────
# Each component has a distinct LR matching the hyperparameter table in the thesis.
param_groups = [
    {"params": list(model.H.parameters()), "lr": LR_H,       "name": "H"},
    {"params": list(model.K.parameters()), "lr": LR_K,       "name": "K"},
    {"params": list(model.G.parameters()), "lr": LR_G,       "name": "G"},
    {"params": list(model.encoder.parameters()), "lr": LR_ENCODER, "name": "encoder"},
    {"params": list(model.R.parameters()), "lr": LR_R,       "name": "R"},
]
optim = torch.optim.Adam(param_groups)

# OneCycleLR needs max_lr as a list (one per group)
scheduler = OneCycleLR(
    optim,
    max_lr=[LR_H, LR_K, LR_G, LR_ENCODER, LR_R],
    steps_per_epoch=len(loader),
    epochs=EPOCHS,
    pct_start=PCT_START,
    div_factor=DIV_FACTOR,
    final_div_factor=FINAL_DIV_FACTOR,
)

loss_fn_recon = nn.MSELoss()

# ==========================================================
# Load swaption volatility data
# ==========================================================

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0  # bp → absolute

# Match dates between swap curves and swaption vols
dates_swap = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
df_vol = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()

if df_vol.empty:
    print("WARNING: No overlapping dates for swaption vols. Pricing loss disabled.")
    LAMBDA_PRICE = 0.0
else:
    print(f"Loaded {len(df_vol)} swaption vol targets from {df_vol['as_of_date'].nunique()} dates")

# Date to index mapping for fast lookup
date_to_idx = {}
for i, row in meta_ccy.iterrows():
    date = pd.Timestamp(row["as_of_date"]).normalize()
    date_to_idx[date] = i


# ==========================================================
# Helper functions
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


def grad_norm(params) -> float:
    """Compute L2 norm of gradients."""
    sq = sum(
        float(p.grad.detach().norm() ** 2)
        for p in params if p.grad is not None
    )
    return math.sqrt(sq)


# ==========================================================
# Pricing loss function
# ==========================================================

def compute_pricing_loss(
        model,
        X_batch: torch.Tensor,
        meta_batch: pd.DataFrame,
        df_vol: pd.DataFrame,
        date_to_idx: dict,
        n_swaptions: int,
        n_paths: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
        return_diagnostics: bool = False,
) -> tuple[torch.Tensor, list]:
    """
    Sample swaption targets and compute pricing loss.

    Returns:
        loss: scalar tensor
        diagnostics: list of dicts with per-swaption stats (if requested)
    """
    if len(df_vol) == 0 or n_swaptions == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), []

    sample = df_vol.sample(n=min(n_swaptions, len(df_vol)))

    total_loss = torch.zeros(1, device=device, dtype=dtype)
    n_valid = 0
    diagnostics = []

    for _, row in sample.iterrows():
        date = pd.Timestamp(row["as_of_date"]).normalize()
        expiry = int(row["option_maturity"])
        tenor = int(row["swap_tenor"])
        sigma_mkt = float(row["market_vol"])

        if date not in date_to_idx:
            continue

        idx = date_to_idx[date]
        xb = X_batch[idx:idx + 1].to(device)

        with torch.no_grad():
            z0 = model.encoder(xb)

        dt_eff = min(dt, 1 / 12, expiry / 10.0)
        n_steps = max(1, int(round(expiry / dt_eff)))

        # Antithetic variates
        half = n_paths // 2
        eps_half = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)
        eps = torch.cat([eps_half, -eps_half], dim=0)

        try:
            z_T, D_T = simulate_to_expiry_differentiable(
                model=model, z0=z0, n_steps=n_steps, dt=dt_eff,
                n_paths=n_paths, eps=eps,
            )

            if torch.isnan(z_T).any():
                continue

            _, aux_T = model.decode_from_z(z_T, tau=None, return_aux=True)
            P_full_T = aux_T["P_full"]

            if torch.isnan(P_full_T).any():
                continue

            F_T, A_T = swap_rate_torch(P_full_T, tenor=tenor)

            if torch.isnan(F_T).any() or torch.isnan(A_T).any():
                continue

            # Time-0 reference
            with torch.no_grad():
                _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
                P_full_0 = aux0["P_full"]
                F_0_t, A_0_t = swap_rate_torch(P_full_0, tenor=tenor)
                F_0 = float(F_0_t[0].item())
                A_0 = float(A_0_t[0].item())

            K = F_0  # ATM strike
            V_market = bachelier_price_torch(F_0, K, sigma_mkt, expiry, A_0)
            V_market_t = torch.tensor(V_market, device=device, dtype=dtype)

            payoff = A_T * torch.relu(F_T - K)
            V_MC = (D_T * payoff).mean()

            loss_ij = ((V_MC - V_market_t) * 10_000) ** 2

            if torch.isfinite(loss_ij):
                total_loss = total_loss + loss_ij
                n_valid += 1

                if return_diagnostics:
                    # Implied vol for logging
                    phi0 = 1.0 / math.sqrt(2.0 * math.pi)
                    denom = A_0 * math.sqrt(expiry) * phi0 + 1e-8
                    sigma_mod = float(V_MC.detach()) / max(denom, 1e-12)

                    diagnostics.append({
                        "date": date.date(),
                        "exp": expiry,
                        "ten": tenor,
                        "mkt_bp": round(sigma_mkt * 10_000, 2),
                        "mod_bp": round(sigma_mod * 10_000, 2),
                        "err_bp": round((sigma_mod - sigma_mkt) * 10_000, 2),
                    })

        except Exception:
            continue

    if n_valid > 0:
        return total_loss / n_valid, diagnostics
    else:
        return torch.tensor(0.0, device=device, dtype=dtype), diagnostics


# ==========================================================
# CSV logger setup
# ==========================================================

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
csv_path = os.path.join(FIGURES_DIR, f"train_joint_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")

torch_version = torch.__version__
python_version = sys.version.split()[0]
numpy_version = np.__version__

csv_cols = (
        ["epoch", "time_total_sec", "time_interval_sec",
         "loss_total", "loss_recon", "loss_price",
         "train_mse", "train_rmse", "avg_rmse_bps", "n_good", "n_bad",
         "grad_norm_total"]
        + [f"rmse_bps_{ccy}" for ccy in ccy_order]
        + ["torch_version", "python_version", "numpy_version"]
)

pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "seed": SEED,
    "latent_dim": LATENT_DIM,
    "variant": config.VARIANT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr_encoder": LR_ENCODER,
    "lr_g": LR_G,
    "lr_h": LR_H,
    "lr_r": LR_R,
    "lr_k": LR_K,
    "lambda_recon": LAMBDA_RECON,
    "lambda_price": LAMBDA_PRICE,
    "n_swaptions_per_batch": N_SWAPTIONS_PER_BATCH,
    "n_paths_pricing": N_PATHS_PRICING,
    "dt_pricing": DT_PRICING,
    "ccy_filter": CCY_FILTER,
    "note": "Using default model priors (h_sigma_max=2.0, k_epsilon=0.001)",
    "per_group_lr_rationale": "H learns larger values than K, so K needs slower LR to stay stable",
}
config_path = os.path.join(FIGURES_DIR, "run_config.json")
with open(config_path, "w") as f:
    json.dump(run_config, f, indent=2)  # type: ignore
print("Saved run config:", config_path)

# ==========================================================
# Training loop
# ==========================================================

train_losses_recon = []
train_losses_price = []
train_losses_total = []
lrs_per_step = []
avg_rmse_bps_hist = []
nan_batches_total = 0

t0 = time.perf_counter()
t_last_log = t0

print("\n" + "=" * 80)
print("JOINT TRAINING: Reconstruction + Pricing")
print("=" * 80)
print(f"Reconstruction weight (λ_recon): {LAMBDA_RECON}")
print(f"Pricing weight (λ_price)       : {LAMBDA_PRICE}")
print(f"Swaptions per batch            : {N_SWAPTIONS_PER_BATCH}")
print(f"MC paths for pricing           : {N_PATHS_PRICING}")
print("=" * 80 + "\n")

for epoch in range(EPOCHS):
    model.train()
    running_recon = 0.0
    running_price = 0.0
    n_obs = 0
    nan_batches = 0
    batch_diagnostics = []

    for batch_idx, (xb_cpu,) in enumerate(loader):
        xb = xb_cpu.to(device)

        optim.zero_grad(set_to_none=USE_SET_TO_NONE)

        # 1. Reconstruction loss
        try:
            S_hat = model(xb)
        except Exception as e:
            nan_batches += 1
            continue

        if not torch.isfinite(S_hat).all():
            nan_batches += 1
            continue

        loss_recon = loss_fn_recon(S_hat, xb)

        if not torch.isfinite(loss_recon):
            nan_batches += 1
            continue

        # 2. Pricing loss
        loss_price = torch.tensor(0.0, device=device, dtype=xb.dtype)

        if LAMBDA_PRICE > 0 and len(df_vol) > 0:
            # Return diagnostics occasionally for logging
            return_diag = (batch_idx == 0 and epoch % (LOG_EVERY // 2) == 0)
            loss_price, batch_diag = compute_pricing_loss(
                model=model,
                X_batch=X_tensor_ccy,
                meta_batch=meta_ccy,
                df_vol=df_vol,
                date_to_idx=date_to_idx,
                n_swaptions=N_SWAPTIONS_PER_BATCH,
                n_paths=N_PATHS_PRICING,
                dt=DT_PRICING,
                device=device,
                dtype=xb.dtype,
                return_diagnostics=return_diag,
            )
            if batch_diag:
                batch_diagnostics = batch_diag

        # 3. Combined loss
        loss_total = LAMBDA_RECON * loss_recon + LAMBDA_PRICE * loss_price

        if not torch.isfinite(loss_total):
            nan_batches += 1
            continue

        loss_total.backward()

        # Check gradients
        has_nan_grad = any(
            param.grad is not None and not torch.isfinite(param.grad).all()
            for param in model.parameters()
        )

        if has_nan_grad:
            nan_batches += 1
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

        lrs_per_step.append(optim.param_groups[0]["lr"])
        running_recon += float(loss_recon.detach().cpu()) * xb.shape[0]
        running_price += float(loss_price.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

    nan_batches_total += nan_batches
    epoch_mse = running_recon / max(n_obs, 1)
    epoch_price = running_price / max(n_obs, 1)
    epoch_total = LAMBDA_RECON * epoch_mse + LAMBDA_PRICE * epoch_price

    train_losses_recon.append(epoch_mse)
    train_losses_price.append(epoch_price)
    train_losses_total.append(epoch_total)
    epoch_rmse = epoch_mse ** 0.5

    # Evaluation
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
        gn_total = grad_norm(model.parameters())
    else:
        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = (None, np.nan, np.nan, np.nan)
        gn_total = 0.0

    # Logging
    if do_log:
        t_now = time.perf_counter()
        time_total = t_now - t0
        time_interval = t_now - t_last_log
        t_last_log = t_now

        row = {
            "epoch": epoch,
            "time_total_sec": time_total,
            "time_interval_sec": time_interval,
            "loss_total": epoch_total,
            "loss_recon": epoch_mse,
            "loss_price": epoch_price,
            "train_mse": epoch_mse,
            "train_rmse": epoch_rmse,
            "avg_rmse_bps": float(avg_rmse_bps),
            "n_good": int(n_good) if np.isfinite(n_good) else np.nan,
            "n_bad": int(n_bad) if np.isfinite(n_bad) else np.nan,
            "grad_norm_total": gn_total,
        }

        for ccy in ccy_order:
            row[f"rmse_bps_{ccy}"] = float(rmse_per_ccy.get(ccy, np.nan)) if rmse_per_ccy is not None else np.nan

        row["torch_version"] = torch_version
        row["python_version"] = python_version
        row["numpy_version"] = numpy_version

        pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)

        print(
            f"epoch={epoch:4d} "
            f"total={epoch_total:.6e} "
            f"recon={epoch_mse:.6e} "
            f"price={epoch_price:.6e} "
            f"rmse={avg_rmse_bps:.2f}bp "
            f"||g||={gn_total:.3f} "
            f"lr={optim.param_groups[0]['lr']:.2e}"
        )

        # Show pricing diagnostics if available
        if batch_diagnostics:
            print("  Pricing diagnostics:")
            print(pd.DataFrame(batch_diagnostics).to_string(index=False))

    if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE and n_obs > 0:
        print(f"[STOP] Reached target MSE at epoch={epoch}")
        break

print("\nTraining done.")

# ==========================================================
# Save model
# ==========================================================

checkpoint_path = os.path.join(FIGURES_DIR, f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {
        "latent_dim": LATENT_DIM,
    },
    "latent_dim": LATENT_DIM,
    "epochs": EPOCHS,
    "variant": config.VARIANT,
    "lambda_recon": LAMBDA_RECON,
    "lambda_price": LAMBDA_PRICE,
}, checkpoint_path)
print("Saved checkpoint:", checkpoint_path)

# ==========================================================
# Plots
# ==========================================================

# Loss curves
fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=150)

axes[0].plot(train_losses_total, linewidth=1.0, label="Total", color="black")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Total Loss")
axes[0].set_title("Joint Training: Total Loss")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(train_losses_recon, linewidth=1.0, label="Reconstruction", color="steelblue")
axes[1].plot(train_losses_price, linewidth=1.0, label="Pricing", color="darkorange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Component Losses")
axes[1].set_title("Joint Training: Component Losses")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.tight_layout()
loss_fig_path = os.path.join(FIGURES_DIR, f"joint_loss_dim{LATENT_DIM}_ep{EPOCHS}.png")
fig.savefig(loss_fig_path, dpi=300)
print("Saved loss plot:", loss_fig_path)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

# RMSE convergence
if len(avg_rmse_bps_hist) > 0:
    epochs_logged = [e for e, v in avg_rmse_bps_hist]
    avg_logged = [v for e, v in avg_rmse_bps_hist]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs_logged, avg_logged, linewidth=1.0, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average RMSE (bps)")
    ax.set_title(f"Joint Training: RMSE Convergence (dim={LATENT_DIM})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    rmse_fig_path = os.path.join(FIGURES_DIR, f"rmse_convergence_dim{LATENT_DIM}_ep{EPOCHS}.png")
    fig.savefig(rmse_fig_path, dpi=300)
    print("Saved RMSE plot:", rmse_fig_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

print("\n" + "=" * 80)
print("JOINT TRAINING COMPLETE")
print("=" * 80)
print(f"Final reconstruction MSE  : {train_losses_recon[-1]:.6e}")
print(f"Final pricing loss        : {train_losses_price[-1]:.6e}")
print(f"Final total loss          : {train_losses_total[-1]:.6e}")
if avg_rmse_bps_hist:
    print(f"Final RMSE (bps)          : {avg_rmse_bps_hist[-1][1]:.2f}")
print(f"Checkpoint saved to       : {checkpoint_path}")
print("=" * 80)

