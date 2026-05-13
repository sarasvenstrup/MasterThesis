# ==================== Market Price of Risk (Lambda) Training ====================
"""
Girsanov-consistent Lambda calibration (v1, 2026-05-12).

Previous attempts:
  - Training_joint.py    : single shared K in ODE + simulation → reconstruction
                           collapses (5 bp → 2200 bp). Structural coupling.
  - Training_joint_kq.py : separate K_Q module + H unfrozen → H enters ODE
                           → reconstruction degrades (49 → 66 bp). Same
                           coupling problem, now through H.

Root cause: ANY parameter that appears in both the no-arbitrage ODE and the
training objective will face conflicting gradients.

This script is the Lyashenko-consistent fix:

  K^Q(z) = K^P(z) - L(z) @ Lambda @ z

where:
  - K^P (model.K)  : frozen — reconstruction ODE uses K^P, unchanged
  - H   (model.H)  : frozen — L(z) from H is used only as a fixed coupling matrix
  - Lambda         : (d x d) trainable market-price-of-risk matrix, init = 0

Why this works:
  - K^P is frozen → reconstruction ODE never changes → recon RMSE stays at 5.3 bp
  - H is frozen   → ODE trace_cov_hess term never changes → recon ODE intact
  - Lambda does NOT appear directly in the reconstruction ODE
  - Lambda appears only in K^Q = K^P - L @ Lambda @ z, which is used:
      (a) as the drift in simulation (k_override=lambda_mpr)
      (b) as the drift in the ODE when decoding simulated z_T (k_override=lambda_mpr)
  - These two roles are consistent — same K^Q everywhere in pricing ✓

Vol calibration:
  Since H is frozen at original scale, diffusion is ~6.6x too large.
  We apply the pre-calibrated expiry-level scales as eps-scaling in simulation:
      eps_scaled = eps * s*(expiry)
  This is equivalent to a simulation-time diffusion scale without touching H.

Lambda at init = 0 → K^Q = K^P (same forward bias as ep0 of KQ training).
Lambda training → corrects forward bias so E^Q[S_T] ≈ F_0.
"""

import copy
import time
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import math
import json
import pandas as pd
import matplotlib.pyplot as plt

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

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
from Code.model.full_model_price import FullModelPrice as FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import bachelier_price_torch, swap_rate_torch, forward_swap_rate_torch
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable
from Code.model.sigma_matrix import L_from_sigmas_rhos

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
EPOCHS     = 1000

EVAL_EVERY   = 1
LOG_EVERY    = 1
DIAG_EVERY   = 10
HEADER_EVERY = 20
SAVE_EVERY   = 200

N_STEPS_PER_EPOCH = 4   # Lambda has 16 params — fewer steps needed per epoch than K^Q

PRETRAIN_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)

# Lambda regularisation (keep Λ small to preserve stability near K^P)
LAMBDA_PRICE = 1.0
LAMBDA_EIG   = 2.0          # eigenvalue floor on linearised K^Q
LAMBDA_L2    = 1e-4         # L2 penalty on Lambda matrix entries
EIG_MIN      = 0.05

LR_LAMBDA = 5e-4            # Lambda has very few params (d×d=16), can use larger LR

N_SWAPTIONS_PER_BATCH = 8    # 8 is enough signal for 16-param Lambda; halves forward pass time
N_PATHS_PRICING       = 256  # antithetic still gives 128+128; halves simulation cost
DT_PRICING            = 1 / 6  # NOTE: overridden by hardcoded 1/12 cap in compute loop — no effect

MIN_FINITE_PATHS_FRAC = 0.10
MIN_FINITE_PATHS_ABS  = 16

LOSS_SKIP_THRESH = 1e4

USE        = "bbg"
CCY_FILTER = "EUR"

# Pre-calibrated expiry-level diffusion scales from §4 (diffusion-scale calibration).
# Applied as eps-scaling in simulation — H weights NOT touched.
# s*(1Y)=0.129, s*(5Y)=0.133, s*(10Y)=0.141; use 0.135 for other expiries.
EXPIRY_SCALES = {1: 0.129, 5: 0.133, 10: 0.141}
DEFAULT_SCALE = 0.135

FIGURES_DIR = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                           f"dim{LATENT_DIM}_{config.VARIANT}_lambda_mpr", f"ep{EPOCHS}")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================================
# LambdaMPR module
# ==========================================================

class LambdaMPR(nn.Module):
    """
    Market price of risk: K^Q(z) = K^P(z) - L(z) @ Lambda @ z

    K^P (kp_module) and H (h_module) are passed in frozen.
    Only self.Lambda is trainable.

    Girsanov:  dW^Q = dW^P + Lambda @ z  dt
               K^Q(z) = K^P(z) - H(z) @ Lambda @ z
    """

    def __init__(self, kp_module: nn.Module, h_module: nn.Module, latent_dim: int):
        super().__init__()
        # References to frozen modules (frozen externally — we just hold references)
        self.kp = kp_module
        self.h  = h_module
        self.latent_dim = latent_dim
        # Trainable market-price-of-risk matrix, init=0 → K^Q = K^P at start
        self.Lambda = nn.Parameter(torch.zeros(latent_dim, latent_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, d)
        returns : K^Q(z) = K^P(z) - L(z) @ Lambda @ z   shape (B, d)
        """
        # K^P and L(z) — frozen, no grad needed
        with torch.no_grad():
            mu_p = self.kp(z)                              # (B, d)
            sigmas, rhos = self.h(z)
            L = L_from_sigmas_rhos(sigmas, rhos, validate=False)  # (B, d, d)

        # lambda(z) = Lambda @ z  — grad flows through Lambda
        # z: (B, d) -> unsqueeze -> (B, d, 1) -> matmul -> (B, d, 1) -> squeeze -> (B, d)
        lam = torch.matmul(self.Lambda, z.unsqueeze(-1)).squeeze(-1)   # (B, d)

        # correction = L(z) @ lambda(z): (B, d, d) x (B, d) -> (B, d)
        correction = torch.einsum('bij,bj->bi', L, lam)                # (B, d)

        # K^Q = K^P - L @ Lambda @ z
        return mu_p - correction


# ==========================================================
# Load data
# ==========================================================

meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

meta_ccy     = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]

# ==========================================================
# Initialize model + LambdaMPR
# ==========================================================

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

model = FullModel(latent_dim=LATENT_DIM).to(device)

if PRETRAIN_CKPT and os.path.isfile(PRETRAIN_CKPT):
    print(f"Loading checkpoint: {PRETRAIN_CKPT}")
    raw = torch.load(PRETRAIN_CKPT, map_location=device)
    state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw
    model.load_state_dict(state_dict)
    print("Checkpoint loaded OK.")
else:
    print(f"WARNING: PRETRAIN_CKPT not found ({PRETRAIN_CKPT}). Training from scratch.")

# Freeze ALL original model parameters — reconstruction is completely protected
for p in model.parameters():
    p.requires_grad_(False)
print("All original model parameters frozen (encoder, G, K^P, H, R).")
print("H is NOT pre-scaled — reconstruction RMSE starts at ~5.3 bp and stays there.")

# LambdaMPR: only Lambda (d×d = 16 params) is trainable
lambda_mpr = LambdaMPR(
    kp_module=model.K,
    h_module=model.H,
    latent_dim=LATENT_DIM,
).to(device)

n_lambda_params = sum(p.numel() for p in lambda_mpr.parameters())
print(f"LambdaMPR: {n_lambda_params} trainable parameters (Lambda {LATENT_DIM}x{LATENT_DIM}).")
print(f"K^Q(z) = K^P(z) - L(z) @ Lambda @ z   (Lambda initialised to 0 → K^Q = K^P)")

model.train()

optim = torch.optim.Adam(lambda_mpr.parameters(), lr=LR_LAMBDA)

LR_WARMUP_EPOCHS = 200
scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
    optim, start_factor=1e-3, end_factor=1.0, total_iters=LR_WARMUP_EPOCHS
)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=max(EPOCHS - LR_WARMUP_EPOCHS, 1), eta_min=1e-7
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optim, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[LR_WARMUP_EPOCHS]
)

# ==========================================================
# Eigenvalue floor on K^Q (linearised at z=0)
# ==========================================================

def get_K_matrix(k_module, dim, device, dtype):
    """Recover linear part M of k_module(z) ≈ M z + b by finite differences."""
    z0   = torch.zeros(1, dim, device=device, dtype=dtype)
    bias = k_module(z0)
    eye  = torch.eye(dim, device=device, dtype=dtype)
    cols = []
    for i in range(dim):
        e_i = eye[i:i+1]
        cols.append((k_module(e_i) - bias).reshape(-1))
    return torch.stack(cols, dim=1)


def eigenvalue_floor_loss(M, eig_min=EIG_MIN):
    eigs = torch.linalg.eigvals(M)
    real = eigs.real
    deficit = torch.relu(real + eig_min)   # penalises Re(λ) > -eig_min
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
    raise RuntimeError("No swaption vol data — cannot train.")
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
    """Reconstruction RMSE — uses K^P (model.K), entirely unaffected by Lambda."""
    S_hat_all = predict_S_hat(model, X_full, batch_size=batch_size)
    mask = row_finite_mask(X_full) & row_finite_mask(S_hat_all)
    n_bad = int((~mask).sum().item())
    rmse_per_ccy = H.rmse_bps_per_currency_paper(
        X_full[mask], S_hat_all[mask],
        meta_full.loc[mask.numpy()].reset_index(drop=True),
    )
    return rmse_per_ccy, float(rmse_per_ccy.mean()), n_bad


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


# ==========================================================
# Pricing loss — Lambda MPR, consistent K^Q in ODE + simulation
# ==========================================================

def compute_pricing_loss_lambda(
    model,
    lambda_mpr: LambdaMPR,
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
    Pricing loss with lambda_mpr used consistently in:
      1. simulate_to_expiry_differentiable  (k_override=lambda_mpr)
      2. decode_from_z for terminal curves  (k_override=lambda_mpr)

    Diffusion vol is controlled by scaling eps per expiry using EXPIRY_SCALES,
    which apply the pre-calibrated s*(expiry) without touching H weights.

    Time-0 quantities (F_0, A_0) decoded with K^P — reflects encoded market curve.
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
        date      = pd.Timestamp(row["as_of_date"]).normalize()
        expiry    = int(row["option_maturity"])
        tenor     = int(row["swap_tenor"])
        sigma_mkt = float(row["market_vol"])

        if date not in date_to_idx:
            continue

        n_attempted += 1
        idx = date_to_idx[date]
        xb  = X_batch[idx:idx + 1].to(device)

        with torch.no_grad():
            z0 = model.encoder(xb)

        dt_eff  = min(dt, expiry / 10.0)   # dt cap removed — DT_PRICING controls step size
        n_steps = max(12, int(round(expiry / dt_eff)))

        # Antithetic noise — scale by expiry-level diffusion scale
        # (equivalent to simulation-time s* without touching H weights)
        s_scale = EXPIRY_SCALES.get(expiry, DEFAULT_SCALE)
        half     = n_paths // 2
        eps_half = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)
        eps_raw  = torch.cat([eps_half, -eps_half], dim=0)
        eps      = eps_raw * s_scale   # apply pre-calibrated vol scale

        try:
            # Simulate with K^Q = lambda_mpr — consistent Q-measure drift
            z_T, D_T = simulate_to_expiry_differentiable(
                model=model, z0=z0, n_steps=n_steps, dt=dt_eff,
                n_paths=n_paths, eps=eps,
                k_override=lambda_mpr,          # <-- K^Q = K^P - L @ Lambda @ z
            )

            z_finite = torch.isfinite(z_T).all(dim=1)
            if D_T.ndim == 1:
                d_finite = torch.isfinite(D_T)
            else:
                d_finite = torch.isfinite(D_T).all(dim=1)
            valid_pre_decode = z_finite & d_finite
            if int(valid_pre_decode.sum().item()) < min_finite_paths:
                continue

            # First pass: probe decoder with K^Q in ODE (no grad)
            with torch.no_grad():
                _, aux_check = model.decode_from_z(
                    z_T, tau=None, return_aux=True,
                    k_override=lambda_mpr,        # <-- K^Q in ODE
                )
                P_check  = aux_check["P_full"]
                p_finite = torch.isfinite(P_check).all(dim=1)

            finite_mask = valid_pre_decode & p_finite
            n_finite    = int(finite_mask.sum().item())
            path_frac   = n_finite / max(n_paths, 1)
            path_finite_fracs.append(path_frac)
            if n_finite < min_finite_paths:
                continue

            # Second pass: survivors with grad, K^Q in ODE
            z_T_keep = z_T[finite_mask]
            _, aux_T = model.decode_from_z(
                z_T_keep, tau=None, return_aux=True,
                k_override=lambda_mpr,            # <-- K^Q in ODE
            )
            P_full_T = aux_T["P_full"]
            F_T, A_T = swap_rate_torch(P_full_T, tenor=tenor)

            fa_finite = torch.isfinite(F_T) & torch.isfinite(A_T)
            if int(fa_finite.sum().item()) < min_finite_paths:
                continue
            F_T = F_T[fa_finite]
            A_T = A_T[fa_finite]

            if D_T.ndim == 1:
                D_keep = D_T[finite_mask][fa_finite]
            else:
                D_keep = D_T[finite_mask].squeeze(-1)[fa_finite]

            # Time-0 reference: K^P (unchanged) — reflects the encoded market curve
            with torch.no_grad():
                _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
                P_full_0 = aux0["P_full"]
                F_0, A_0 = forward_swap_rate_torch(P_full_0[0], expiry, tenor)

            if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 0):
                continue

            payoff      = A_T * torch.relu(F_T - F_0)
            disc_payoff = D_keep * payoff
            V_MC        = disc_payoff.mean()

            if not torch.isfinite(V_MC):
                continue

            sigma_mod_bp_t = (V_MC * sqrt_2pi) / (A_0 * math.sqrt(expiry)) * 10_000.0
            sigma_mkt_bp   = sigma_mkt * 10_000.0

            # MSE in vol space: 50 bp error → loss = 0.25
            loss_ij = ((sigma_mod_bp_t - sigma_mkt_bp) / 100.0) ** 2

            if not torch.isfinite(loss_ij):
                continue
            if float(loss_ij.detach()) > LOSS_SKIP_THRESH:
                continue

            total_loss = total_loss + loss_ij
            n_valid   += 1

            if return_diagnostics:
                diagnostics.append({
                    "date":   date.date(),
                    "exp":    expiry,
                    "ten":    tenor,
                    "mkt_bp": round(sigma_mkt_bp, 1),
                    "mod_bp": round(float(sigma_mod_bp_t.detach()), 1),
                    "err_bp": round(float(sigma_mod_bp_t.detach()) - sigma_mkt_bp, 1),
                    "pths%":  round(path_frac * 100, 0),
                    "scale":  s_scale,
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
csv_path  = os.path.join(FIGURES_DIR, f"train_lambda_mpr_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")

csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec",
     "loss_price", "loss_eig", "loss_l2",
     "swaption_priced_frac", "path_finite_frac",
     "recon_rmse_bps",
     "nan_batches",
     "gnorm_Lambda",
     "lr_Lambda",
     "lambda_min_KQ",
     "Lambda_norm_fro"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
)
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "version": "lambda_mpr_v1",
    "seed": SEED, "latent_dim": LATENT_DIM, "variant": config.VARIANT,
    "epochs": EPOCHS, "n_steps_per_epoch": N_STEPS_PER_EPOCH,
    "pretrain_ckpt": PRETRAIN_CKPT,
    "lr_lambda": LR_LAMBDA,
    "lambda_price": LAMBDA_PRICE,
    "lambda_eig": LAMBDA_EIG,
    "lambda_l2": LAMBDA_L2,
    "eig_min": EIG_MIN,
    "n_swaptions_per_batch": N_SWAPTIONS_PER_BATCH,
    "n_paths_pricing": N_PATHS_PRICING,
    "dt_pricing": DT_PRICING,
    "loss": "mse_vol_bp_div100",
    "ccy_filter": CCY_FILTER,
    "save_every": SAVE_EVERY,
    "expiry_scales": EXPIRY_SCALES,
    "default_scale": DEFAULT_SCALE,
    "h_frozen": True,
    "kp_frozen": True,
    "n_lambda_params": n_lambda_params,
}
with open(os.path.join(FIGURES_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

# ==========================================================
# Training loop
# ==========================================================

train_losses_price = []
train_losses_eig   = []
train_losses_l2    = []
swaption_priced_hist = []
path_finite_hist     = []
lambda_min_hist      = []
lambda_norm_hist     = []

t0         = time.perf_counter()
t_last_log = t0

print("\n" + "=" * 80)
print("LAMBDA MPR TRAINING: K^Q(z) = K^P(z) - L(z) @ Lambda @ z")
print("=" * 80)
print(f"Price weight  : {LAMBDA_PRICE}")
print(f"Eig weight    : {LAMBDA_EIG}   (floor |Re(λ_KQ)| ≥ {EIG_MIN})")
print(f"L2 weight     : {LAMBDA_L2}   (regularise Lambda matrix)")
print(f"Lambda params : {n_lambda_params}  (d×d = {LATENT_DIM}×{LATENT_DIM})")
print(f"H frozen      : YES — reconstruction RMSE stays at ~5.3 bp")
print(f"Vol scales    : {EXPIRY_SCALES} (eps-scaled, H untouched)")
print("=" * 80 + "\n")

for epoch in range(EPOCHS):
    model.train()
    lambda_mpr.train()
    running_price = 0.0
    running_eig   = 0.0
    running_l2    = 0.0
    n_batches = 0
    nan_batches = 0
    batch_diagnostics = []
    epoch_attempted = 0
    epoch_priced    = 0
    epoch_path_fracs = []

    for step in range(N_STEPS_PER_EPOCH):
        print(f"\r  ep {epoch}  step {step+1}/{N_STEPS_PER_EPOCH} ...", end="", flush=True)
        optim.zero_grad(set_to_none=USE_SET_TO_NONE)

        return_diag = (step == 0)
        loss_price, batch_diag, n_att, n_pri, p_frac = compute_pricing_loss_lambda(
            model=model, lambda_mpr=lambda_mpr,
            X_batch=X_tensor_ccy, meta_batch=meta_ccy,
            df_vol=df_vol, date_to_idx=date_to_idx,
            n_swaptions=N_SWAPTIONS_PER_BATCH,
            n_paths=N_PATHS_PRICING, dt=DT_PRICING,
            device=device, dtype=torch.float32,
            return_diagnostics=return_diag,
        )
        epoch_attempted += n_att
        epoch_priced    += n_pri
        if p_frac > 0:
            epoch_path_fracs.append(p_frac)
        if batch_diag:
            batch_diagnostics = batch_diag

        # Eigenvalue floor on linearised K^Q
        loss_eig = torch.tensor(0.0, device=device, dtype=torch.float32)
        if LAMBDA_EIG > 0:
            try:
                M = get_K_matrix(lambda_mpr, LATENT_DIM, device, torch.float32)
                loss_eig = eigenvalue_floor_loss(M, eig_min=EIG_MIN)
                if not torch.isfinite(loss_eig):
                    loss_eig = torch.tensor(0.0, device=device, dtype=torch.float32)
            except Exception:
                pass

        # L2 regularisation on Lambda (keeps K^Q close to K^P → stability)
        loss_l2 = LAMBDA_L2 * lambda_mpr.Lambda.pow(2).sum()

        loss_total = LAMBDA_PRICE * loss_price + LAMBDA_EIG * loss_eig + loss_l2

        if not torch.isfinite(loss_total):
            nan_batches += 1
            continue

        loss_total.backward()

        has_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in lambda_mpr.parameters()
        )
        if has_nan_grad:
            nan_batches += 1
            optim.zero_grad(set_to_none=USE_SET_TO_NONE)
            continue

        torch.nn.utils.clip_grad_norm_(lambda_mpr.parameters(), max_norm=1.0)
        optim.step()

        running_price += float(loss_price.detach().cpu())
        running_eig   += float(loss_eig.detach().cpu())
        running_l2    += float(loss_l2.detach().cpu())
        n_batches += 1

    print("\r" + " " * 40 + "\r", end="", flush=True)

    scheduler.step()

    epoch_price = running_price / max(n_batches, 1)
    epoch_eig   = running_eig   / max(n_batches, 1)
    epoch_l2    = running_l2    / max(n_batches, 1)
    swaption_priced = epoch_priced / max(epoch_attempted, 1)
    path_finite     = float(np.mean(epoch_path_fracs)) if epoch_path_fracs else 0.0

    train_losses_price.append(epoch_price)
    train_losses_eig.append(epoch_eig)
    train_losses_l2.append(epoch_l2)
    swaption_priced_hist.append(swaption_priced)
    path_finite_hist.append(path_finite)

    # Lambda diagnostics
    try:
        with torch.no_grad():
            M_now = get_K_matrix(lambda_mpr, LATENT_DIM, device, torch.float32)
            lambda_min_now = float(torch.linalg.eigvals(M_now).real.abs().min().cpu())
            lambda_norm_fro = float(lambda_mpr.Lambda.norm().cpu())
    except Exception:
        lambda_min_now = float('nan')
        lambda_norm_fro = float('nan')
    lambda_min_hist.append(lambda_min_now)
    lambda_norm_hist.append(lambda_norm_fro)

    # Reconstruction RMSE (uses K^P only — should stay at ~5.3 bp throughout)
    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    if do_eval:
        rmse_per_ccy, avg_rmse_bps, n_bad = eval_rmse_bps(
            model, X_tensor, meta, batch_size=256
        )
        gn_lam = grad_norm(lambda_mpr.parameters())
    else:
        rmse_per_ccy, avg_rmse_bps = None, float('nan')
        gn_lam = 0.0

    lrs_now = {pg["name"] if "name" in pg else "Lambda": pg["lr"]
               for pg in optim.param_groups}
    lr_lambda_now = optim.param_groups[0]["lr"]

    t_now      = time.perf_counter()
    time_total = t_now - t0
    time_int   = t_now - t_last_log
    t_last_log = t_now

    row = {
        "epoch":                epoch,
        "time_total_sec":       round(time_total, 1),
        "time_interval_sec":    round(time_int, 3),
        "loss_price":           epoch_price,
        "loss_eig":             epoch_eig,
        "loss_l2":              epoch_l2,
        "swaption_priced_frac": swaption_priced,
        "path_finite_frac":     path_finite,
        "recon_rmse_bps":       float(avg_rmse_bps),
        "nan_batches":          nan_batches,
        "gnorm_Lambda":         gn_lam,
        "lr_Lambda":            lr_lambda_now,
        "lambda_min_KQ":        lambda_min_now,
        "Lambda_norm_fro":      lambda_norm_fro,
    }
    for ccy in ccy_order:
        row[f"rmse_bps_{ccy}"] = (
            float(rmse_per_ccy.get(ccy, float('nan')))
            if rmse_per_ccy is not None else float('nan')
        )
    pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)

    # ETA
    epochs_remaining = EPOCHS - epoch - 1
    eta_sec = time_int * epochs_remaining
    if eta_sec >= 3600:
        eta_str = f"{int(eta_sec//3600)}h{int((eta_sec%3600)//60):02d}m"
    elif eta_sec >= 60:
        eta_str = f"{int(eta_sec//60)}m{int(eta_sec%60):02d}s"
    else:
        eta_str = f"{int(eta_sec)}s"

    log_idx = epoch // max(LOG_EVERY, 1)
    if log_idx % HEADER_EVERY == 0:
        print(
            f"\n{'ep':>5} {'price':>10} {'eig':>9} {'l2':>8} "
            f"{'swp%':>5} {'pth%':>5} {'recon':>7} "
            f"{'|λ|min':>7} {'‖Λ‖F':>7} "
            f"{'gLam':>9} {'lrLam':>8} {'t/ep':>6} {'ETA':>8}  pricing_diag"
        )
        print("-" * 155)

    if batch_diagnostics and (epoch % DIAG_EVERY == 0 or epoch == EPOCHS - 1):
        diag_str = " | ".join(
            f"{d['exp']}x{d['ten']} mkt={d['mkt_bp']:.0f} mod={d['mod_bp']:.0f} "
            f"err={d['err_bp']:+.0f}bp pth={int(d['pths%'])}% s={d['scale']}"
            for d in batch_diagnostics
        )
    else:
        diag_str = ""

    print(
        f"{epoch:>5d} "
        f"{epoch_price:>10.4e} "
        f"{epoch_eig:>9.3e} "
        f"{epoch_l2:>8.4e} "
        f"{swaption_priced*100:>4.0f}% "
        f"{path_finite*100:>4.0f}% "
        f"{avg_rmse_bps:>7.2f} "
        f"{lambda_min_now:>7.4f} "
        f"{lambda_norm_fro:>7.4f} "
        f"{gn_lam:>9.2e} "
        f"{lr_lambda_now:>8.2e} "
        f"{time_int:>5.1f}s "
        f"{eta_str:>8}  "
        f"{diag_str}"
    )

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt_path = os.path.join(FIGURES_DIR, f"checkpoint_lambda_ep{epoch+1}.pt")
        torch.save({
            "model_state_dict":    model.state_dict(),
            "Lambda_state_dict":   lambda_mpr.state_dict(),
            "model_config":        {"latent_dim": LATENT_DIM},
            "latent_dim":          LATENT_DIM,
            "epoch":               epoch + 1,
            "variant":             config.VARIANT,
            "Lambda_matrix":       lambda_mpr.Lambda.detach().cpu(),
            "lambda_min_KQ":       lambda_min_now,
            "lambda_norm_fro":     lambda_norm_fro,
            "path_finite_frac":    path_finite,
            "swaption_priced_frac": swaption_priced,
        }, ckpt_path)
        print(f"  → checkpoint saved: ep{epoch+1}  "
              f"(‖Λ‖F={lambda_norm_fro:.4f}, |λ|min={lambda_min_now:.4f}, "
              f"pth={path_finite*100:.0f}%)")

print("\nTraining done.")

# ==========================================================
# Final checkpoint + plots
# ==========================================================

final_ckpt = os.path.join(FIGURES_DIR, f"checkpoint_lambda_ep{EPOCHS}.pt")
torch.save({
    "model_state_dict":  model.state_dict(),
    "Lambda_state_dict": lambda_mpr.state_dict(),
    "Lambda_matrix":     lambda_mpr.Lambda.detach().cpu(),
    "latent_dim":        LATENT_DIM,
    "epochs":            EPOCHS,
    "variant":           config.VARIANT,
}, final_ckpt)
print("Saved final checkpoint:", final_ckpt)

fig, axes = plt.subplots(3, 1, figsize=(9, 10), dpi=150)

axes[0].semilogy(train_losses_price, lw=1.0, color="darkorange", label="Pricing loss")
axes[0].semilogy(train_losses_eig,   lw=1.0, color="seagreen",   label="Eig floor")
axes[0].semilogy(train_losses_l2,    lw=1.0, color="royalblue",  label="L2 on Lambda")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log scale)")
axes[0].set_title("Lambda MPR Training: Loss Components")
axes[0].grid(True, alpha=0.3); axes[0].legend()

axes[1].plot([100*p for p in swaption_priced_hist], lw=1.2, color="firebrick",
             label="swaption_priced_frac (%)")
axes[1].plot([100*p for p in path_finite_hist], lw=1.0, color="navy",
             label="path_finite_frac (%)")
axes[1].axhline(95, color="grey", ls=":", lw=1, label="target 95%")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("%")
axes[1].set_title("Decoder Coverage (should stay high — H and K^P frozen)")
axes[1].set_ylim(-2, 102); axes[1].grid(True, alpha=0.3); axes[1].legend(loc="lower right")

axes[2].plot(lambda_norm_hist, lw=1.2, color="purple", label="‖Λ‖_F (Frobenius norm)")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("‖Λ‖_F")
axes[2].set_title("Lambda Matrix Growth (Frobenius norm)")
axes[2].grid(True, alpha=0.3); axes[2].legend()

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"lambda_mpr_loss_dim{LATENT_DIM}_ep{EPOCHS}.png"), dpi=200)
plt.close(fig)

print("\n" + "=" * 80)
print("LAMBDA MPR TRAINING COMPLETE")
print("=" * 80)
print(f"Final pricing loss     : {train_losses_price[-1]:.6e}")
print(f"Final eig-floor loss   : {train_losses_eig[-1]:.6e}")
print(f"Final L2 loss          : {train_losses_l2[-1]:.6e}")
print(f"Final ‖Lambda‖_F       : {lambda_norm_hist[-1]:.4f}")
print(f"Final swaption_priced  : {swaption_priced_hist[-1]*100:.1f}%")
print(f"Final path_finite_frac : {path_finite_hist[-1]*100:.1f}%")
print(f"Final |λ|min K^Q       : {lambda_min_hist[-1]:.4f}")
print(f"Reconstruction RMSE    : {avg_rmse_bps:.2f} bp  (should be ~5.3 bp)")
print(f"Lambda matrix:\n{lambda_mpr.Lambda.detach().cpu().numpy()}")
print("=" * 80)
