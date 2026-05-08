# ==================== Joint Training: Reconstruction + Pricing ====================
"""
Diagnostic-driven version v4 (2026-05-07) — vol-space Huber loss.

v3 fixed gradient flow but the relative-price loss exploded to 1e28 because
sigma_mod was 9× sigma_mkt at warm start, the loss squared the ratio, and
small V_market values (low-vol swaptions) made the relative error blow up.
Recon RMSE went from 5.3 to 1011 bp in 27 epochs — the decoder was being shredded.

v4 fix: vol-space Huber loss in basis points.
- Compare sigma_mod_bp to sigma_mkt_bp directly (not via prices)
- Huber loss: quadratic up to 50 bp error, linear beyond → bounded gradients
- Rescale by delta² to keep magnitude near O(1)

Key design:
- Decoder G unfrozen, learning from pricing loss
- Eigenvalue floor regularizer (|Re(λ_K)| >= EIG_MIN)
- Two-pass decoding: no_grad probe → grad pass on survivors only
- Vol-space Huber loss
- Lower LR_G (1e-5) and LAMBDA_PRICE (0.05) for stability
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
EPOCHS     = 2000
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256

EVAL_EVERY    = 1
LOG_EVERY     = 1
DIAG_EVERY    = 10
HEADER_EVERY  = 20
SAVE_EVERY    = 200
TARGET_MSE    = 1e-8

PRETRAIN_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)

FREEZE_ENCODER = True
FREEZE_G        = False
FREEZE_R        = True

LAMBDA_RECON = 1.0
LAMBDA_PRICE = 0.001        # CHANGED: was 0.1
LAMBDA_EIG   = 0.1
EIG_MIN      = 0.05

LR_H        = 5e-5
LR_K        = 1e-5
LR_G        = 1e-5         # CHANGED: was 5e-5 — decoder needs gentler updates
LR_ENCODER  = 1e-5
LR_R        = 1e-5

N_SWAPTIONS_PER_BATCH = 4
N_PATHS_PRICING       = 512
DT_PRICING            = 1 / 12

# Per-path masking thresholds
MIN_FINITE_PATHS_FRAC = 0.10
MIN_FINITE_PATHS_ABS  = 32

# Vol-space loss settings (NEW in v4)
HUBER_DELTA_BP   = 50.0     # transition point: quadratic below, linear above
LOSS_SKIP_THRESH = 1e4      # skip individual swaption if its loss exceeds this

USE        = "bbg"
CCY_FILTER = "EUR"

FIGURES_DIR = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                           f"dim{LATENT_DIM}_{config.VARIANT}_joint", f"ep{EPOCHS}")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================================
# Load data
# ==========================================================

meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

meta_ccy     = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]

dataset = TensorDataset(X_tensor)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# ==========================================================
# Initialize model
# ==========================================================

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

model = FullModel(latent_dim=LATENT_DIM).to(device)

if PRETRAIN_CKPT and os.path.isfile(PRETRAIN_CKPT):
    print(f"Loading warm-start checkpoint: {PRETRAIN_CKPT}")
    raw = torch.load(PRETRAIN_CKPT, map_location=device)
    state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw
    model.load_state_dict(state_dict)
    print("Warm start loaded OK.")
else:
    print(f"WARNING: PRETRAIN_CKPT not found ({PRETRAIN_CKPT}). Training from scratch.")

if FREEZE_ENCODER:
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    print("Encoder frozen.")
if FREEZE_G:
    for p in model.G.parameters():
        p.requires_grad_(False)
    print("Decoder G frozen.")
else:
    print("Decoder G TRAINABLE — will learn to handle simulated z_T.")
if FREEZE_R:
    for p in model.R.parameters():
        p.requires_grad_(False)
    print("Short-rate R frozen.")

model.train()

param_groups_all = [
    {"params": list(model.H.parameters()),       "lr": LR_H,       "name": "H"},
    {"params": list(model.K.parameters()),       "lr": LR_K,       "name": "K"},
    {"params": list(model.G.parameters()),       "lr": LR_G,       "name": "G"},
    {"params": list(model.encoder.parameters()), "lr": LR_ENCODER, "name": "encoder"},
    {"params": list(model.R.parameters()),       "lr": LR_R,       "name": "R"},
]
param_groups = [g for g in param_groups_all if any(p.requires_grad for p in g["params"])]
print(f"Trainable groups: {[g['name'] for g in param_groups]}")
optim = torch.optim.Adam(param_groups)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-7)

loss_fn_recon = nn.MSELoss()

# ==========================================================
# Eigenvalue floor regularizer
# ==========================================================

def get_K_matrix(model, dim, device, dtype):
    """Recover linear part M of the drift K(z) = M z + bias (differentiable)."""
    z0 = torch.zeros(1, dim, device=device, dtype=dtype)
    bias = model.K(z0)
    eye  = torch.eye(dim, device=device, dtype=dtype)
    cols = []
    for i in range(dim):
        e_i = eye[i:i+1]
        cols.append((model.K(e_i) - bias).reshape(-1))
    M = torch.stack(cols, dim=1)
    return M


def eigenvalue_floor_loss(M, eig_min=EIG_MIN):
    eigs = torch.linalg.eigvals(M)
    real = eigs.real
    deficit = torch.relu(eig_min - real.abs())
    return deficit.pow(2).mean()


# ==========================================================
# Load swaption volatility data
# ==========================================================

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0

dates_swap = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
df_vol = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()

if df_vol.empty:
    print("WARNING: No overlapping dates for swaption vols. Pricing loss disabled.")
    LAMBDA_PRICE = 0.0
else:
    print(f"Loaded {len(df_vol)} swaption vol targets from {df_vol['as_of_date'].nunique()} dates")

date_to_idx = {pd.Timestamp(row["as_of_date"]).normalize(): i for i, row in meta_ccy.iterrows()}


# ==========================================================
# Helpers
# ==========================================================

def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)


@torch.no_grad()
def predict_S_hat(model, X, batch_size=256):
    was_training = model.training
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        outs.append(model(X[i:i+batch_size].to(device)).detach().cpu())
    if was_training:
        model.train()
    return torch.cat(outs, dim=0)


def eval_rmse_bps(model, X_full, meta_full, batch_size=256):
    S_hat_all = predict_S_hat(model, X_full, batch_size=batch_size)
    mask = row_finite_mask(X_full) & row_finite_mask(S_hat_all)
    n_bad  = int((~mask).sum().item())
    n_good = int(mask.sum().item())
    rmse_per_ccy = H.rmse_bps_per_currency_paper(
        X_full[mask], S_hat_all[mask],
        meta_full.loc[mask.numpy()].reset_index(drop=True),
    )
    return rmse_per_ccy, float(rmse_per_ccy.mean()), n_bad, n_good


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def grad_norm_groups(model) -> dict:
    groups = {"H": model.H, "K": model.K, "G": model.G,
              "enc": model.encoder, "R": model.R}
    return {name: grad_norm(mod.parameters()) for name, mod in groups.items()}


# ==========================================================
# Pricing loss — v4 with vol-space Huber
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
):
    """
    Returns (loss, diagnostics, n_attempted, n_priced, mean_path_finite_frac).

    Loss: Huber on (sigma_mod - sigma_mkt) in basis points.
        - Quadratic for |err| <= 50 bp
        - Linear for |err| > 50 bp
        - Rescaled by delta² so loss is O(1)
    """
    if len(df_vol) == 0 or n_swaptions == 0:
        return torch.tensor(0.0, device=device, dtype=dtype), [], 0, 0, 0.0

    sample = df_vol.sample(n=min(n_swaptions, len(df_vol)))

    total_loss  = torch.zeros(1, device=device, dtype=dtype)
    n_valid     = 0
    n_attempted = 0
    diagnostics = []
    path_finite_fracs = []

    min_finite_paths = max(MIN_FINITE_PATHS_ABS, int(n_paths * MIN_FINITE_PATHS_FRAC))
    sqrt_2pi = math.sqrt(2.0 * math.pi)

    for _, row in sample.iterrows():
        date     = pd.Timestamp(row["as_of_date"]).normalize()
        expiry   = int(row["option_maturity"])
        tenor    = int(row["swap_tenor"])
        sigma_mkt = float(row["market_vol"])

        if date not in date_to_idx:
            continue

        n_attempted += 1
        idx = date_to_idx[date]
        xb  = X_batch[idx:idx + 1].to(device)

        with torch.no_grad():
            z0 = model.encoder(xb)

        dt_eff  = min(dt, 1 / 12, expiry / 10.0)
        n_steps = max(12, int(round(expiry / dt_eff)))

        half     = n_paths // 2
        eps_half = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)
        eps      = torch.cat([eps_half, -eps_half], dim=0)

        try:
            z_T, D_T = simulate_to_expiry_differentiable(
                model=model, z0=z0, n_steps=n_steps, dt=dt_eff,
                n_paths=n_paths, eps=eps,
            )

            # ---- Per-path validity from simulation ----
            z_finite = torch.isfinite(z_T).all(dim=1)
            if D_T.ndim == 1:
                d_finite = torch.isfinite(D_T)
            else:
                d_finite = torch.isfinite(D_T).all(dim=1)
            valid_pre_decode = z_finite & d_finite
            if int(valid_pre_decode.sum().item()) < min_finite_paths:
                continue

            # ---- FIRST PASS (no_grad) — probe decoder ----
            with torch.no_grad():
                _, aux_check = model.decode_from_z(z_T, tau=None, return_aux=True)
                P_check  = aux_check["P_full"]
                p_finite = torch.isfinite(P_check).all(dim=1)

            finite_mask = valid_pre_decode & p_finite
            n_finite    = int(finite_mask.sum().item())
            path_frac   = n_finite / max(n_paths, 1)
            path_finite_fracs.append(path_frac)
            if n_finite < min_finite_paths:
                continue

            # ---- SECOND PASS (with grad) — survivors only ----
            z_T_keep = z_T[finite_mask]
            _, aux_T = model.decode_from_z(z_T_keep, tau=None, return_aux=True)
            P_full_T = aux_T["P_full"]
            F_T, A_T = swap_rate_torch(P_full_T, tenor=tenor)

            fa_finite = torch.isfinite(F_T) & torch.isfinite(A_T)
            if int(fa_finite.sum().item()) < min_finite_paths:
                continue
            F_T = F_T[fa_finite]
            A_T = A_T[fa_finite]

            # Discount factor: detached, slice to surviving paths
            if D_T.ndim == 1:
                D_keep = D_T[finite_mask][fa_finite]
            else:
                D_keep_all = D_T[finite_mask]
                D_keep_all = D_keep_all.squeeze(-1) if D_keep_all.shape[-1] == 1 else D_keep_all
                D_keep = D_keep_all[fa_finite]

            # ---- Time-0 reference (no_grad anchor) ----
            with torch.no_grad():
                _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
                P_full_0 = aux0["P_full"]
                F_0_t, A_0_t = swap_rate_torch(P_full_0, tenor=tenor)
                F_0 = float(F_0_t[0].item())
                A_0 = float(A_0_t[0].item())

            if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 0):
                continue

            K_strike   = F_0
            payoff      = A_T * torch.relu(F_T - K_strike)
            disc_payoff = D_keep * payoff
            V_MC        = disc_payoff.mean()

            if not torch.isfinite(V_MC):
                continue

            # ---- Vol-space Huber loss (NEW in v4) ----
            # Bachelier ATM inversion: sigma_N = V * sqrt(2*pi) / (A_0 * sqrt(T))
            sigma_mod_bp_t = (V_MC * sqrt_2pi) / (A_0 * math.sqrt(expiry)) * 10_000.0
            sigma_mkt_bp   = sigma_mkt * 10_000.0

            vol_err_bp = sigma_mod_bp_t - sigma_mkt_bp
            abs_err    = vol_err_bp.abs()

            # Huber: quadratic up to delta, linear beyond
            huber = torch.where(
                abs_err <= HUBER_DELTA_BP,
                0.5 * vol_err_bp ** 2,
                HUBER_DELTA_BP * (abs_err - 0.5 * HUBER_DELTA_BP),
            )
            loss_ij = huber / (HUBER_DELTA_BP ** 2)   # rescale to ~O(1)

            # Hard guard against pathological values
            if not torch.isfinite(loss_ij):
                continue
            if float(loss_ij.detach()) > LOSS_SKIP_THRESH:
                continue

            total_loss = total_loss + loss_ij
            n_valid   += 1

            if return_diagnostics:
                sigma_mod_bp = float(sigma_mod_bp_t.detach())
                diagnostics.append({
                    "date":   date.date(),
                    "exp":    expiry,
                    "ten":    tenor,
                    "mkt_bp": round(sigma_mkt_bp, 1),
                    "mod_bp": round(sigma_mod_bp, 1),
                    "err_bp": round(sigma_mod_bp - sigma_mkt_bp, 1),
                    "pths%":  round(path_frac * 100, 0),
                })

        except Exception:
            continue

    mean_pfrac = float(np.mean(path_finite_fracs)) if path_finite_fracs else 0.0
    if n_valid > 0:
        return total_loss / n_valid, diagnostics, n_attempted, n_valid, mean_pfrac
    return torch.tensor(0.0, device=device, dtype=dtype), diagnostics, n_attempted, 0, mean_pfrac


# ==========================================================
# CSV logger
# ==========================================================

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
csv_path  = os.path.join(FIGURES_DIR, f"train_joint_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")

torch_version  = torch.__version__
python_version = sys.version.split()[0]
numpy_version  = np.__version__

csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec",
     "loss_total", "loss_recon", "loss_price", "loss_eig",
     "swaption_priced_frac", "path_finite_frac",
     "train_mse", "train_rmse", "avg_rmse_bps", "n_good", "n_bad",
     "nan_batches",
     "grad_norm_total",
     "gnorm_H", "gnorm_K", "gnorm_G", "gnorm_enc", "gnorm_R",
     "lr_H", "lr_K", "lr_G", "lr_enc", "lr_R",
     "lambda_min_K"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
    + ["torch_version", "python_version", "numpy_version"]
)
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "version": "v4_vol_space_huber",
    "seed": SEED, "latent_dim": LATENT_DIM, "variant": config.VARIANT,
    "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "pretrain_ckpt": PRETRAIN_CKPT,
    "freeze_encoder": FREEZE_ENCODER, "freeze_G": FREEZE_G, "freeze_R": FREEZE_R,
    "lr_encoder": LR_ENCODER, "lr_g": LR_G, "lr_h": LR_H, "lr_r": LR_R, "lr_k": LR_K,
    "lambda_recon": LAMBDA_RECON, "lambda_price": LAMBDA_PRICE,
    "lambda_eig": LAMBDA_EIG, "eig_min": EIG_MIN,
    "n_swaptions_per_batch": N_SWAPTIONS_PER_BATCH,
    "n_paths_pricing": N_PATHS_PRICING, "dt_pricing": DT_PRICING,
    "min_finite_paths_frac": MIN_FINITE_PATHS_FRAC,
    "min_finite_paths_abs":  MIN_FINITE_PATHS_ABS,
    "huber_delta_bp": HUBER_DELTA_BP,
    "loss_skip_thresh": LOSS_SKIP_THRESH,
    "ccy_filter": CCY_FILTER, "save_every": SAVE_EVERY,
}
config_path = os.path.join(FIGURES_DIR, "run_config.json")
with open(config_path, "w") as f:
    json.dump(run_config, f, indent=2)
print("Saved run config:", config_path)


# ==========================================================
# Training loop
# ==========================================================

train_losses_recon  = []
train_losses_price  = []
train_losses_eig    = []
train_losses_total  = []
swaption_priced_hist = []
path_finite_hist     = []
lambda_min_hist      = []
avg_rmse_bps_hist    = []
nan_batches_total   = 0

t0         = time.perf_counter()
t_last_log = t0

print("\n" + "=" * 80)
print("JOINT TRAINING v4: vol-space Huber loss")
print("=" * 80)
print(f"Recon weight    : {LAMBDA_RECON}")
print(f"Price weight    : {LAMBDA_PRICE}")
print(f"Eig weight      : {LAMBDA_EIG}   (floor |Re(λ)| ≥ {EIG_MIN})")
print(f"Huber delta     : {HUBER_DELTA_BP} bp  (quadratic below, linear beyond)")
print(f"Per-path mask   : need ≥ {MIN_FINITE_PATHS_FRAC*100:.0f}% paths finite "
      f"(min {MIN_FINITE_PATHS_ABS} of {N_PATHS_PRICING})")
print(f"Decoder G       : {'TRAINABLE' if not FREEZE_G else 'FROZEN'} (LR={LR_G})")
print("=" * 80 + "\n")

best_swaption_priced = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_recon = 0.0
    running_price = 0.0
    running_eig   = 0.0
    n_obs = 0
    nan_batches = 0
    batch_diagnostics = []
    epoch_attempted    = 0
    epoch_priced       = 0
    epoch_path_fracs   = []

    for batch_idx, (xb_cpu,) in enumerate(loader):
        xb = xb_cpu.to(device)
        optim.zero_grad(set_to_none=USE_SET_TO_NONE)

        # 1. Reconstruction loss
        try:
            S_hat = model(xb)
        except Exception:
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
            return_diag = (batch_idx == 0)
            loss_price, batch_diag, n_att, n_pri, p_frac = compute_pricing_loss(
                model=model, X_batch=X_tensor_ccy, meta_batch=meta_ccy,
                df_vol=df_vol, date_to_idx=date_to_idx,
                n_swaptions=N_SWAPTIONS_PER_BATCH,
                n_paths=N_PATHS_PRICING, dt=DT_PRICING,
                device=device, dtype=xb.dtype, return_diagnostics=return_diag,
            )
            epoch_attempted += n_att
            epoch_priced    += n_pri
            if p_frac > 0:
                epoch_path_fracs.append(p_frac)
            if batch_diag:
                batch_diagnostics = batch_diag

        # 3. Eigenvalue floor
        loss_eig = torch.tensor(0.0, device=device, dtype=xb.dtype)
        if LAMBDA_EIG > 0:
            try:
                M = get_K_matrix(model, LATENT_DIM, device, xb.dtype)
                loss_eig = eigenvalue_floor_loss(M, eig_min=EIG_MIN)
                if not torch.isfinite(loss_eig):
                    loss_eig = torch.tensor(0.0, device=device, dtype=xb.dtype)
            except Exception:
                loss_eig = torch.tensor(0.0, device=device, dtype=xb.dtype)

        loss_total = (LAMBDA_RECON * loss_recon
                      + LAMBDA_PRICE * loss_price
                      + LAMBDA_EIG  * loss_eig)

        if not torch.isfinite(loss_total):
            nan_batches += 1
            continue

        loss_total.backward()

        has_nan_grad = any(
            param.grad is not None and not torch.isfinite(param.grad).all()
            for param in model.parameters()
        )
        if has_nan_grad:
            nan_batches += 1
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        running_recon += float(loss_recon.detach().cpu()) * xb.shape[0]
        running_price += float(loss_price.detach().cpu()) * xb.shape[0]
        running_eig   += float(loss_eig.detach().cpu())   * xb.shape[0]
        n_obs += xb.shape[0]

    nan_batches_total += nan_batches
    epoch_mse   = running_recon / max(n_obs, 1)
    epoch_price = running_price / max(n_obs, 1)
    epoch_eig   = running_eig   / max(n_obs, 1)
    epoch_total = (LAMBDA_RECON * epoch_mse
                   + LAMBDA_PRICE * epoch_price
                   + LAMBDA_EIG  * epoch_eig)

    swaption_priced = epoch_priced / max(epoch_attempted, 1)
    path_finite     = float(np.mean(epoch_path_fracs)) if epoch_path_fracs else 0.0

    scheduler.step()

    train_losses_recon.append(epoch_mse)
    train_losses_price.append(epoch_price)
    train_losses_eig.append(epoch_eig)
    train_losses_total.append(epoch_total)
    swaption_priced_hist.append(swaption_priced)
    path_finite_hist.append(path_finite)
    epoch_rmse = epoch_mse ** 0.5

    try:
        with torch.no_grad():
            M_now = get_K_matrix(model, LATENT_DIM, device, torch.float32)
            lambda_min_now = float(torch.linalg.eigvals(M_now).real.abs().min().cpu())
    except Exception:
        lambda_min_now = float('nan')
    lambda_min_hist.append(lambda_min_now)

    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    do_log  = ((epoch + 1) % LOG_EVERY  == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE and n_obs > 0:
        do_eval = do_log = True

    if do_eval:
        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = eval_rmse_bps(
            model, X_tensor, meta, batch_size=EVAL_BATCH_SIZE
        )
        avg_rmse_bps_hist.append((epoch, avg_rmse_bps))
        gn_total  = grad_norm(model.parameters())
        gn_groups = grad_norm_groups(model)
    else:
        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = (None, np.nan, np.nan, np.nan)
        gn_total  = 0.0
        gn_groups = {k: 0.0 for k in ("H", "K", "G", "enc", "R")}

    lrs_now = {pg["name"]: pg["lr"] for pg in optim.param_groups}

    if do_log:
        t_now         = time.perf_counter()
        time_total    = t_now - t0
        time_interval = t_now - t_last_log
        t_last_log    = t_now

        row = {
            "epoch":                epoch,
            "time_total_sec":       round(time_total, 1),
            "time_interval_sec":    round(time_interval, 3),
            "loss_total":           epoch_total,
            "loss_recon":           epoch_mse,
            "loss_price":           epoch_price,
            "loss_eig":             epoch_eig,
            "swaption_priced_frac": swaption_priced,
            "path_finite_frac":     path_finite,
            "train_mse":            epoch_mse,
            "train_rmse":           epoch_rmse,
            "avg_rmse_bps":         float(avg_rmse_bps),
            "n_good":               int(n_good) if np.isfinite(n_good) else np.nan,
            "n_bad":                int(n_bad)  if np.isfinite(n_bad)  else np.nan,
            "nan_batches":          nan_batches,
            "grad_norm_total":      gn_total,
            "gnorm_H":   gn_groups["H"],
            "gnorm_K":   gn_groups["K"],
            "gnorm_G":   gn_groups["G"],
            "gnorm_enc": gn_groups["enc"],
            "gnorm_R":   gn_groups["R"],
            "lr_H":   lrs_now.get("H",       np.nan),
            "lr_K":   lrs_now.get("K",       np.nan),
            "lr_G":   lrs_now.get("G",       np.nan),
            "lr_enc": lrs_now.get("encoder", np.nan),
            "lr_R":   lrs_now.get("R",       np.nan),
            "lambda_min_K": lambda_min_now,
        }
        for ccy in ccy_order:
            row[f"rmse_bps_{ccy}"] = (
                float(rmse_per_ccy.get(ccy, np.nan))
                if rmse_per_ccy is not None else np.nan
            )
        row["torch_version"]  = torch_version
        row["python_version"] = python_version
        row["numpy_version"]  = numpy_version

        pd.DataFrame([row], columns=csv_cols).to_csv(
            csv_path, mode="a", header=False, index=False
        )

        log_idx = epoch // max(LOG_EVERY, 1)
        if log_idx % HEADER_EVERY == 0:
            print(
                f"\n{'ep':>5} {'recon':>10} {'price':>10} {'eig':>9} "
                f"{'swp%':>5} {'pth%':>5} {'rmse':>7} {'|λ|':>7} "
                f"{'gH':>9} {'gK':>9} {'gG':>9} "
                f"{'lrH':>8}  pricing_diag"
            )
            print("-" * 140)

        if batch_diagnostics and (epoch % DIAG_EVERY == 0 or epoch == EPOCHS - 1):
            diag_parts = []
            for d in batch_diagnostics:
                diag_parts.append(
                    f"{d['exp']}x{d['ten']} "
                    f"mkt={d['mkt_bp']:.0f} mod={d['mod_bp']:.0f} "
                    f"err={d['err_bp']:+.0f}bp pth={int(d['pths%'])}%"
                )
            diag_str = " | ".join(diag_parts)
        else:
            diag_str = ""

        print(
            f"{epoch:>5d} "
            f"{epoch_mse:>10.4e} "
            f"{epoch_price:>10.4e} "
            f"{epoch_eig:>9.3e} "
            f"{swaption_priced*100:>4.0f}% "
            f"{path_finite*100:>4.0f}% "
            f"{avg_rmse_bps:>7.2f} "
            f"{lambda_min_now:>7.4f} "
            f"{gn_groups['H']:>9.2e} "
            f"{gn_groups['K']:>9.2e} "
            f"{gn_groups['G']:>9.2e} "
            f"{lrs_now.get('H', 0):>8.2e}  "
            f"{diag_str}"
        )

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt_inter_path = os.path.join(
            FIGURES_DIR, f"checkpoint_dim{LATENT_DIM}_ep{epoch+1}.pt"
        )
        torch.save({
            "model_state_dict":     model.state_dict(),
            "model_config":         {"latent_dim": LATENT_DIM},
            "latent_dim":           LATENT_DIM,
            "epoch":                epoch + 1,
            "variant":              config.VARIANT,
            "lambda_recon":         LAMBDA_RECON,
            "lambda_price":         LAMBDA_PRICE,
            "lambda_eig":           LAMBDA_EIG,
            "swaption_priced_frac": swaption_priced,
            "path_finite_frac":     path_finite,
            "avg_rmse_bps":         float(avg_rmse_bps) if np.isfinite(avg_rmse_bps) else None,
            "lambda_min_K":         lambda_min_now,
        }, ckpt_inter_path)
        print(f"  → checkpoint saved: ep{epoch+1}  "
              f"(swp={swaption_priced*100:.0f}%, pth={path_finite*100:.0f}%, "
              f"rmse={avg_rmse_bps:.2f}bp)")

    if swaption_priced > best_swaption_priced:
        best_swaption_priced = swaption_priced

    if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE and n_obs > 0:
        print(f"[STOP] Reached target MSE at epoch={epoch}")
        break

print("\nTraining done.")
print(f"Best swaption_priced_frac: {best_swaption_priced*100:.1f}%")

# ==========================================================
# Save final + plots
# ==========================================================

checkpoint_path = os.path.join(FIGURES_DIR, f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config":     {"latent_dim": LATENT_DIM},
    "latent_dim":       LATENT_DIM,
    "epochs":           EPOCHS,
    "variant":          config.VARIANT,
    "lambda_recon":     LAMBDA_RECON, "lambda_price": LAMBDA_PRICE,
    "lambda_eig":       LAMBDA_EIG,
    "swaption_priced_frac_final": swaption_priced_hist[-1] if swaption_priced_hist else None,
    "best_swaption_priced":       best_swaption_priced,
}, checkpoint_path)
print("Saved final checkpoint:", checkpoint_path)

fig, axes = plt.subplots(3, 1, figsize=(9, 11), dpi=150)
axes[0].plot(train_losses_total, lw=1.0, label="Total", color="black")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss")
axes[0].set_title("Joint Training: Total Loss"); axes[0].grid(True, alpha=0.3)
axes[0].set_yscale("log"); axes[0].legend()

axes[1].plot(train_losses_recon, lw=1.0, label="Reconstruction", color="steelblue")
axes[1].plot(train_losses_price, lw=1.0, label="Pricing (Huber)", color="darkorange")
axes[1].plot(train_losses_eig,   lw=1.0, label="Eig floor",      color="seagreen")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Component losses")
axes[1].set_title("Joint Training: Component Losses")
axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3); axes[1].legend()

axes[2].plot([100*p for p in swaption_priced_hist], lw=1.2, color="firebrick",
             label="swaption_priced_frac (%)")
axes[2].plot([100*p for p in path_finite_hist], lw=1.0, color="navy",
             label="path_finite_frac (%)")
axes[2].axhline(95, color="grey", ls=":", lw=1, label="target 95%")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("%")
axes[2].set_title("Decoder Coverage During Training"); axes[2].set_ylim(-2, 102)
axes[2].grid(True, alpha=0.3); axes[2].legend(loc="lower right")

fig.tight_layout()
loss_fig_path = os.path.join(FIGURES_DIR, f"joint_loss_dim{LATENT_DIM}_ep{EPOCHS}.png")
fig.savefig(loss_fig_path, dpi=200); plt.close(fig)
print("Saved loss plot:", loss_fig_path)

if avg_rmse_bps_hist:
    epochs_logged = [e for e, v in avg_rmse_bps_hist]
    avg_logged    = [v for e, v in avg_rmse_bps_hist]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs_logged, avg_logged, lw=1.0, color="steelblue")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Average RMSE (bps)")
    ax.set_title(f"Reconstruction RMSE Convergence (dim={LATENT_DIM})")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    rmse_fig_path = os.path.join(FIGURES_DIR, f"rmse_convergence_dim{LATENT_DIM}_ep{EPOCHS}.png")
    fig.savefig(rmse_fig_path, dpi=200); plt.close(fig)
    print("Saved RMSE plot:", rmse_fig_path)

if lambda_min_hist:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(lambda_min_hist, lw=1.0, color="seagreen")
    ax.axhline(EIG_MIN, color="grey", ls=":", lw=1, label=f"floor = {EIG_MIN}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("min |Re(λ_K)|")
    ax.set_title("Smallest mean-reversion eigenvalue"); ax.grid(True, alpha=0.3)
    ax.legend(); fig.tight_layout()
    eig_fig_path = os.path.join(FIGURES_DIR, f"lambda_min_dim{LATENT_DIM}_ep{EPOCHS}.png")
    fig.savefig(eig_fig_path, dpi=200); plt.close(fig)
    print("Saved eigenvalue plot:", eig_fig_path)

print("\n" + "=" * 80)
print("JOINT TRAINING COMPLETE")
print("=" * 80)
print(f"Final reconstruction MSE  : {train_losses_recon[-1]:.6e}")
print(f"Final pricing loss        : {train_losses_price[-1]:.6e}")
print(f"Final eig-floor loss      : {train_losses_eig[-1]:.6e}")
print(f"Final swaption_priced_frac: {swaption_priced_hist[-1]*100:.1f}%")
print(f"Best  swaption_priced_frac: {best_swaption_priced*100:.1f}%")
print(f"Final path_finite_frac    : {path_finite_hist[-1]*100:.1f}%")
print(f"Final |λ|min for K        : {lambda_min_hist[-1]:.4f}")
if avg_rmse_bps_hist:
    print(f"Final RMSE (bps)          : {avg_rmse_bps_hist[-1][1]:.2f}")
print("=" * 80)