# ==================== CIR Stochastic Vol + Lambda MPR Training ====================
"""
Train CIR stochastic volatility process + Lambda market-price-of-risk jointly.

Q-measure model:
  dz_t = K^Q(z_t) dt  +  sqrt(v_t) * L(z_t) dW_t^z
  dv_t = kappa*(theta - v_t) dt  +  sigma_v * sqrt(v_t) dW_t^v

  K^Q(z) = K^P(z) - L(z) @ Lambda @ z    [Girsanov drift correction]

Trainable parameters (19 total):
  Lambda      4x4 matrix (16)  — drift correction
  log_kappa   scalar  (1)      — CIR mean-reversion speed  (kappa   = exp(log_kappa))
  log_theta   scalar  (1)      — CIR long-run variance     (theta   = exp(log_theta))
  log_sigma_v scalar  (1)      — CIR vol-of-vol            (sigma_v = exp(log_sigma_v))

Base model (K^P, H, R, encoder, decoder) fully frozen.
No s* anywhere — vol level is entirely controlled by the CIR process.

Antithetic variates: +eps_z and -eps_z pairs share the same v_t path.

Output: Figures/TrainingResults/dim4_sv_lambda/ep{EPOCHS}/
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
import matplotlib
matplotlib.use("Agg")
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
from Code.Pricing.pricing import swap_rate_torch, forward_swap_rate_torch
from Code.model.sigma_matrix import L_from_sigmas_rhos

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Active model variant:", config.VARIANT)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

torch.backends.mkldnn.enabled = True

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

N_STEPS_PER_EPOCH     = 4
N_SWAPTIONS_PER_BATCH = 8
N_PATHS_PRICING       = 256   # antithetic: 128+128
DT_PRICING            = 1 / 6

PRETRAIN_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)

# Loss weights
LAMBDA_PRICE  = 1.0
LAMBDA_EIG    = 2.0    # eigenvalue floor on K^Q
LAMBDA_L2     = 1e-4   # L2 on Lambda entries
LAMBDA_FELLER = 1.0    # Feller condition: 2*kappa*theta > sigma_v^2
LAMBDA_BIAS   = 0.02   # put-call parity: (V_pay - V_rec)/A0 -> 0
EIG_MIN       = 0.05

LR            = 5e-4

MIN_FINITE_PATHS_FRAC = 0.10
MIN_FINITE_PATHS_ABS  = 16
LOSS_SKIP_THRESH      = 1e4

USE        = "bbg"
CCY_FILTER = "EUR"

# CIR initial values (log-parameterised)
# theta_init ≈ 0.018  (sqrt ≈ 0.134, close to s*≈0.13 range for compatibility check)
# kappa_init = 1.0    (1-year mean-reversion)
# sigma_v_init = 0.05 (vol-of-vol; satisfies Feller: 2*1*0.018 = 0.036 >> 0.0025)
CIR_LOG_KAPPA_INIT   =  0.0      # exp(0)   = 1.0
CIR_LOG_THETA_INIT   = -4.02     # exp(-4.02) ≈ 0.018
CIR_LOG_SIGMAV_INIT  = -3.0      # exp(-3)  ≈ 0.05

FIGURES_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_sv_lambda_v2", f"ep{EPOCHS}"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================================
# SVLambdaMPR module
# ==========================================================

class SVLambdaMPR(nn.Module):
    """
    Combined CIR stochastic vol + Lambda market-price-of-risk module.

    K^Q(z) = K^P(z) - L(z) @ Lambda @ z

    CIR variance process (for simulation only):
      dv = kappa*(theta - v)*dt + sigma_v*sqrt(v)*dW_v
      v_0 = theta  (stationary initialisation)

    K^P and H are frozen references. Only Lambda and CIR log-params are trainable.
    """

    def __init__(self, kp_module, h_module, latent_dim):
        super().__init__()
        self.kp         = kp_module
        self.h          = h_module
        self.latent_dim = latent_dim

        # Drift correction — init 0 → K^Q = K^P at epoch 0
        self.Lambda = nn.Parameter(torch.zeros(latent_dim, latent_dim))

        # CIR log-parameters (all positive via exp)
        self.log_kappa   = nn.Parameter(torch.tensor(CIR_LOG_KAPPA_INIT))
        self.log_theta   = nn.Parameter(torch.tensor(CIR_LOG_THETA_INIT))
        self.log_sigma_v = nn.Parameter(torch.tensor(CIR_LOG_SIGMAV_INIT))

    def forward(self, z):
        """K^Q(z) = K^P(z) - L(z) @ Lambda @ z   (B, d)"""
        with torch.no_grad():
            mu_p = self.kp(z)
            sigmas, rhos = self.h(z)
            L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam        = torch.matmul(self.Lambda, z.unsqueeze(-1)).squeeze(-1)
        correction = torch.einsum('bij,bj->bi', L, lam)
        return mu_p - correction

    @property
    def kappa(self):
        return self.log_kappa.exp()

    @property
    def theta(self):
        return self.log_theta.exp()

    @property
    def sigma_v(self):
        return self.log_sigma_v.exp()


# ==========================================================
# CIR + Lambda MPR simulation
# ==========================================================

def simulate_sv_to_expiry(
    model,
    sv_lambda,
    z0,         # (1, d)
    n_steps,
    dt,
    half,       # n_paths = 2 * half
    eps_z,      # (half, n_steps, d)  z-noise
    eps_v,      # (half, n_steps)     CIR noise
):
    """
    Euler-Maruyama with CIR stochastic vol and Lambda MPR drift.

    dz = K^Q(z)*dt + sqrt(v)*L(z)*sqrt(dt)*eps_z
    dv = kappa*(theta-v)*dt + sigma_v*sqrt(v)*sqrt(dt)*eps_v

    Antithetic: eps_z and -eps_z share the same v path per pair.

    Returns
    -------
    z_T : (2*half, d)   terminal state WITH gradient
    D_T : (2*half,)     pathwise discount factor, detached
    """
    kappa   = sv_lambda.kappa    # scalar, grad OK
    theta   = sv_lambda.theta
    sigma_v = sv_lambda.sigma_v

    n_paths = 2 * half
    sqrt_dt = math.sqrt(dt)
    dtype   = z0.dtype
    dev     = z0.device

    # Expand z0 to n_paths copies
    z = z0.expand(n_paths, -1).clone()

    # Initialise v at stationarity (detached — gradient enters through CIR steps)
    v = torch.full((half,), theta.item(), dtype=dtype, device=dev)

    with torch.no_grad():
        r_prev = model.R(z).squeeze(-1)
    log_D = torch.zeros(n_paths, device=dev, dtype=dtype)

    for t in range(n_steps):
        # ---- CIR step (gradient flows through kappa, theta, sigma_v) ----
        kappa_t   = sv_lambda.kappa
        theta_t   = sv_lambda.theta
        sigma_v_t = sv_lambda.sigma_v

        v_safe = v.clamp(min=1e-10)
        dv = (kappa_t * (theta_t - v_safe) * dt
              + sigma_v_t * v_safe.sqrt() * sqrt_dt * eps_v[:, t])
        v = (v_safe + dv).clamp(min=1e-10)

        # Antithetic z noise scaled by sqrt(v)
        sqrt_v     = v.sqrt()
        sqrt_v_all = torch.cat([sqrt_v, sqrt_v], dim=0)            # (n_paths,)
        eps_z_t    = torch.cat([eps_z[:, t], -eps_z[:, t]], dim=0) # (n_paths, d)
        dW         = eps_z_t * sqrt_dt

        # Diffusion — H frozen (no_grad for L)
        with torch.no_grad():
            sigmas, rhos = model.H(z.detach())
            L = L_from_sigmas_rhos(sigmas, rhos, validate=False)

        shock = sqrt_v_all.unsqueeze(-1) * torch.bmm(L, dW.unsqueeze(-1)).squeeze(-1)

        # K^Q drift — gradient through Lambda
        drift = sv_lambda(z) * dt

        z = z + drift + shock

        # Trapezoidal discount — detached
        with torch.no_grad():
            r_next = model.R(z.detach()).squeeze(-1)
            log_D  = log_D - 0.5 * (r_prev + r_next) * dt
            r_prev = r_next

    D_T = log_D.clamp(min=-30.0, max=30.0).exp().detach()
    return z, D_T


# ==========================================================
# Load data
# ==========================================================

meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

meta_ccy     = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]

# ==========================================================
# Initialise model + SVLambdaMPR
# ==========================================================

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

model = FullModel(latent_dim=LATENT_DIM).to(device)

if PRETRAIN_CKPT and os.path.isfile(PRETRAIN_CKPT):
    print(f"Loading checkpoint: {PRETRAIN_CKPT}")
    raw = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw
    model.load_state_dict(state_dict)
    print("Checkpoint loaded OK.")
else:
    raise FileNotFoundError(f"Pretrained checkpoint not found: {PRETRAIN_CKPT}")

for p in model.parameters():
    p.requires_grad_(False)
print("All original model parameters frozen (encoder, G, K^P, H, R).")

sv_lambda = SVLambdaMPR(
    kp_module=model.K,
    h_module=model.H,
    latent_dim=LATENT_DIM,
).to(device)

n_params = sum(p.numel() for p in sv_lambda.parameters() if p.requires_grad)
print(f"SVLambdaMPR: {n_params} trainable parameters "
      f"(Lambda {LATENT_DIM}x{LATENT_DIM}=16 + CIR 3).")
print(f"CIR init: kappa={sv_lambda.kappa.item():.4f}  "
      f"theta={sv_lambda.theta.item():.6f}  "
      f"sigma_v={sv_lambda.sigma_v.item():.4f}")
print(f"Vol init: sqrt(theta)={sv_lambda.theta.item()**0.5:.4f}  "
      f"(equivalent s*-level: {sv_lambda.theta.item()**0.5:.4f})")

model.train()

optim = torch.optim.Adam(sv_lambda.parameters(), lr=LR)

LR_WARMUP_EPOCHS = 200
scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
    optim, start_factor=1e-3, end_factor=1.0, total_iters=LR_WARMUP_EPOCHS
)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=max(EPOCHS - LR_WARMUP_EPOCHS, 1), eta_min=1e-7
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optim, schedulers=[scheduler_warmup, scheduler_cosine],
    milestones=[LR_WARMUP_EPOCHS]
)

# ==========================================================
# Eigenvalue floor (same as Lambda MPR training)
# ==========================================================

def get_K_matrix(k_module, dim, device, dtype):
    z0   = torch.zeros(1, dim, device=device, dtype=dtype)
    bias = k_module(z0)
    eye  = torch.eye(dim, device=device, dtype=dtype)
    cols = []
    for i in range(dim):
        e_i = eye[i:i+1]
        cols.append((k_module(e_i) - bias).reshape(-1))
    return torch.stack(cols, dim=1)


def eigenvalue_floor_loss(M, eig_min=EIG_MIN):
    eigs   = torch.linalg.eigvals(M)
    real   = eigs.real
    deficit = torch.relu(real + eig_min)
    return deficit.pow(2).mean()


# ==========================================================
# Load swaption vol data
# ==========================================================

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0

dates_swap = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
df_vol = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()

if df_vol.empty:
    raise RuntimeError("No swaption vol data — cannot train.")
print(f"Loaded {len(df_vol)} swaption vol targets from {df_vol['as_of_date'].nunique()} dates")

date_to_idx = {pd.Timestamp(row["as_of_date"]).normalize(): i
               for i, row in meta_ccy.iterrows()}

# ==========================================================
# Helpers
# ==========================================================

def row_finite_mask(t):
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
    rmse_per_ccy = H.rmse_bps_per_currency_paper(
        X_full[mask], S_hat_all[mask],
        meta_full.loc[mask.numpy()].reset_index(drop=True),
    )
    return rmse_per_ccy, float(rmse_per_ccy.mean()), int((~mask).sum().item())


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


# ==========================================================
# Pricing loss
# ==========================================================

def compute_pricing_loss_sv(
    model,
    sv_lambda,
    X_batch,
    meta_batch,
    df_vol,
    date_to_idx,
    n_swaptions,
    n_paths,
    dt,
    device,
    dtype,
    return_diagnostics=False,
):
    """
    Monte Carlo pricing loss with CIR stochastic vol + Lambda MPR.

    Loss has two components:
      loss_price : MSE of ATM payer vol vs market  (gets the width right)
      loss_bias  : (V_pay - V_rec)^2 penalty       (centres distribution on F_0)

    eps_z and -eps_z pairs share the same CIR v-path (antithetic variance reduction).
    No s* scaling anywhere — vol level comes entirely from CIR theta.
    """
    if len(df_vol) == 0 or n_swaptions == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, [], 0, 0, 0.0

    sample = df_vol.sample(n=min(n_swaptions, len(df_vol)))

    total_loss       = torch.zeros(1, device=device, dtype=dtype)
    total_bias       = torch.zeros(1, device=device, dtype=dtype)
    n_valid          = 0
    n_attempted      = 0
    diagnostics      = []
    path_finite_fracs = []
    sqrt_2pi         = math.sqrt(2.0 * math.pi)
    min_finite_paths = max(MIN_FINITE_PATHS_ABS, int(n_paths * MIN_FINITE_PATHS_FRAC))

    for _, row in sample.iterrows():
        date      = pd.Timestamp(row["as_of_date"]).normalize()
        expiry    = int(row["option_maturity"])
        tenor     = int(row["swap_tenor"])
        sigma_mkt = float(row["market_vol"])

        if date not in date_to_idx:
            continue

        n_attempted += 1
        idx  = date_to_idx[date]
        xb   = X_batch[idx:idx + 1].to(device)

        with torch.no_grad():
            z0 = model.encoder(xb)

        dt_eff  = min(dt, expiry / 10.0)
        n_steps = max(12, int(round(expiry / dt_eff)))
        half    = n_paths // 2

        # Draw noise (no gradient)
        with torch.no_grad():
            eps_z = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)
            eps_v = torch.randn(half, n_steps,              device=device, dtype=dtype)

        try:
            # Simulate with CIR vol + K^Q drift
            z_T, D_T = simulate_sv_to_expiry(
                model, sv_lambda, z0,
                n_steps=n_steps, dt=dt_eff,
                half=half, eps_z=eps_z, eps_v=eps_v,
            )

            z_finite = torch.isfinite(z_T).all(dim=1)
            d_finite = torch.isfinite(D_T)
            ok_pre   = z_finite & d_finite
            if int(ok_pre.sum().item()) < min_finite_paths:
                continue

            # Probe decoder (no grad)
            with torch.no_grad():
                _, aux_check = model.decode_from_z(
                    z_T, tau=None, return_aux=True, k_override=sv_lambda
                )
                p_finite = torch.isfinite(aux_check["P_full"]).all(dim=1)

            finite_mask = ok_pre & p_finite
            n_finite    = int(finite_mask.sum().item())
            path_frac   = n_finite / max(n_paths, 1)
            path_finite_fracs.append(path_frac)
            if n_finite < min_finite_paths:
                continue

            # Decode survivors with grad
            z_T_keep = z_T[finite_mask]
            _, aux_T  = model.decode_from_z(
                z_T_keep, tau=None, return_aux=True, k_override=sv_lambda
            )
            F_T, A_T = swap_rate_torch(aux_T["P_full"], tenor=tenor)

            fa_ok = (torch.isfinite(F_T) & torch.isfinite(A_T)
                     & (F_T > -0.5) & (F_T < 0.5)
                     & (A_T > 1e-6) & (A_T < 50.0))
            if int(fa_ok.sum().item()) < min_finite_paths:
                continue

            F_T   = F_T[fa_ok]
            A_T   = A_T[fa_ok]
            D_keep = D_T[finite_mask][fa_ok]

            # Time-0 reference under K^P (encoded market curve)
            with torch.no_grad():
                _, aux0  = model.decode_from_z(z0, tau=None, return_aux=True)
                F_0, A_0 = forward_swap_rate_torch(aux0["P_full"][0], expiry, tenor)

            if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 0):
                continue

            V_pay = (D_keep * A_T * torch.relu(F_T - F_0)).mean()
            V_rec = (D_keep * A_T * torch.relu(F_0 - F_T)).mean()
            if not (torch.isfinite(V_pay) and torch.isfinite(V_rec)):
                continue

            sigma_mod_bp = (V_pay * sqrt_2pi) / (A_0 * math.sqrt(expiry)) * 10_000.0
            sigma_mkt_bp = sigma_mkt * 10_000.0

            # Vol MSE loss
            loss_ij = ((sigma_mod_bp - sigma_mkt_bp) / 100.0) ** 2
            if not torch.isfinite(loss_ij):
                continue
            if float(loss_ij.detach()) > LOSS_SKIP_THRESH:
                continue

            # Put-call parity bias loss: (V_pay - V_rec) / A0 -> 0
            fwd_bias_bp = (V_pay - V_rec) / A_0 * 1e4
            bias_ij     = (fwd_bias_bp / 100.0) ** 2

            total_loss = total_loss + loss_ij
            total_bias = total_bias + bias_ij
            n_valid   += 1

            if return_diagnostics:
                diagnostics.append({
                    "date":     date.date(),
                    "exp":      expiry,
                    "ten":      tenor,
                    "mkt_bp":   round(sigma_mkt_bp, 1),
                    "mod_bp":   round(float(sigma_mod_bp.detach()), 1),
                    "err_bp":   round(float(sigma_mod_bp.detach()) - sigma_mkt_bp, 1),
                    "bias_bp":  round(float(fwd_bias_bp.detach()), 1),
                    "pths%":    round(path_frac * 100, 0),
                })

        except Exception:
            continue

    mean_pfrac = float(np.mean(path_finite_fracs)) if path_finite_fracs else 0.0
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    if n_valid > 0:
        return total_loss / n_valid, total_bias / n_valid, diagnostics, n_attempted, n_valid, mean_pfrac
    return zero, zero, diagnostics, n_attempted, 0, mean_pfrac


# ==========================================================
# CSV logger
# ==========================================================

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
csv_path  = os.path.join(FIGURES_DIR, f"train_sv_lambda_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")

csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec",
     "loss_price", "loss_bias", "loss_eig", "loss_l2", "loss_feller",
     "swaption_priced_frac", "path_finite_frac",
     "recon_rmse_bps", "nan_batches",
     "gnorm", "lr",
     "lambda_min_KQ", "Lambda_norm_fro",
     "kappa", "theta", "sigma_v", "sqrt_theta"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
)
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "version":              "sv_lambda_v1",
    "seed":                 SEED,
    "latent_dim":           LATENT_DIM,
    "variant":              config.VARIANT,
    "epochs":               EPOCHS,
    "n_steps_per_epoch":    N_STEPS_PER_EPOCH,
    "pretrain_ckpt":        PRETRAIN_CKPT,
    "lr":                   LR,
    "lambda_price":         LAMBDA_PRICE,
    "lambda_eig":           LAMBDA_EIG,
    "lambda_l2":            LAMBDA_L2,
    "lambda_feller":        LAMBDA_FELLER,
    "lambda_bias":          LAMBDA_BIAS,
    "eig_min":              EIG_MIN,
    "n_swaptions_per_batch": N_SWAPTIONS_PER_BATCH,
    "n_paths_pricing":      N_PATHS_PRICING,
    "dt_pricing":           DT_PRICING,
    "cir_log_kappa_init":   CIR_LOG_KAPPA_INIT,
    "cir_log_theta_init":   CIR_LOG_THETA_INIT,
    "cir_log_sigmav_init":  CIR_LOG_SIGMAV_INIT,
    "ccy_filter":           CCY_FILTER,
    "no_s_star":            True,
    "h_frozen":             True,
    "kp_frozen":            True,
    "n_trainable_params":   n_params,
}
with open(os.path.join(FIGURES_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

# ==========================================================
# Training loop
# ==========================================================

hist = {k: [] for k in [
    "price", "bias", "eig", "l2", "feller",
    "swp_priced", "path_finite",
    "lambda_min", "lambda_norm",
    "kappa", "theta", "sigma_v",
]}

t0         = time.perf_counter()
t_last_log = t0

print("\n" + "=" * 90)
print("SV + LAMBDA MPR TRAINING")
print("  dz = K^Q(z) dt + sqrt(v_t) L(z) dW^z")
print("  dv = kappa*(theta-v) dt + sigma_v*sqrt(v) dW^v")
print("  K^Q(z) = K^P(z) - L(z) @ Lambda @ z")
print("=" * 90)
print(f"No s* — vol level from CIR theta  (init sqrt(theta)={sv_lambda.theta.item()**0.5:.4f})")
print(f"Trainable params: {n_params}  (Lambda 16 + CIR 3)")
print("=" * 90 + "\n")

for epoch in range(EPOCHS):
    model.train()
    sv_lambda.train()
    running_price  = 0.0
    running_bias   = 0.0
    running_eig    = 0.0
    running_l2     = 0.0
    running_feller = 0.0
    n_batches      = 0
    nan_batches    = 0
    batch_diag     = []
    ep_attempted   = 0
    ep_priced      = 0
    ep_pfracs      = []

    for step in range(N_STEPS_PER_EPOCH):
        print(f"\r  ep {epoch}  step {step+1}/{N_STEPS_PER_EPOCH} ...", end="", flush=True)
        optim.zero_grad(set_to_none=True)

        loss_price, loss_bias_raw, diag, n_att, n_pri, p_frac = compute_pricing_loss_sv(
            model=model, sv_lambda=sv_lambda,
            X_batch=X_tensor_ccy, meta_batch=meta_ccy,
            df_vol=df_vol, date_to_idx=date_to_idx,
            n_swaptions=N_SWAPTIONS_PER_BATCH,
            n_paths=N_PATHS_PRICING, dt=DT_PRICING,
            device=device, dtype=torch.float32,
            return_diagnostics=(step == 0),
        )
        ep_attempted += n_att
        ep_priced    += n_pri
        if p_frac > 0:
            ep_pfracs.append(p_frac)
        if diag:
            batch_diag = diag

        # Eigenvalue floor
        loss_eig = torch.tensor(0.0, device=device, dtype=torch.float32)
        if LAMBDA_EIG > 0:
            try:
                M = get_K_matrix(sv_lambda, LATENT_DIM, device, torch.float32)
                loss_eig = eigenvalue_floor_loss(M, eig_min=EIG_MIN)
                if not torch.isfinite(loss_eig):
                    loss_eig = torch.tensor(0.0, device=device, dtype=torch.float32)
            except Exception:
                pass

        # L2 on Lambda
        loss_l2 = LAMBDA_L2 * sv_lambda.Lambda.pow(2).sum()

        # Feller condition: penalise if sigma_v^2 > 2*kappa*theta
        kappa_v   = sv_lambda.kappa
        theta_v   = sv_lambda.theta
        sigma_v_v = sv_lambda.sigma_v
        feller_deficit = torch.relu(sigma_v_v ** 2 - 2.0 * kappa_v * theta_v + 1e-5)
        loss_feller = LAMBDA_FELLER * feller_deficit ** 2

        # Put-call parity bias penalty
        loss_bias = LAMBDA_BIAS * loss_bias_raw

        loss_total = (LAMBDA_PRICE * loss_price
                      + loss_bias
                      + LAMBDA_EIG * loss_eig
                      + loss_l2
                      + loss_feller)

        if not torch.isfinite(loss_total):
            nan_batches += 1
            continue

        loss_total.backward()

        has_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in sv_lambda.parameters()
        )
        if has_nan_grad:
            nan_batches += 1
            optim.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(sv_lambda.parameters(), max_norm=1.0)
        optim.step()

        running_price  += float(loss_price.detach().cpu())
        running_bias   += float(loss_bias_raw.detach().cpu())
        running_eig    += float(loss_eig.detach().cpu())
        running_l2     += float(loss_l2.detach().cpu())
        running_feller += float(loss_feller.detach().cpu())
        n_batches += 1

    print("\r" + " " * 40 + "\r", end="", flush=True)
    scheduler.step()

    ep_price  = running_price  / max(n_batches, 1)
    ep_bias   = running_bias   / max(n_batches, 1)
    ep_eig    = running_eig    / max(n_batches, 1)
    ep_l2     = running_l2     / max(n_batches, 1)
    ep_feller = running_feller / max(n_batches, 1)
    swp_priced  = ep_priced / max(ep_attempted, 1)
    path_finite = float(np.mean(ep_pfracs)) if ep_pfracs else 0.0

    # CIR diagnostics
    with torch.no_grad():
        kappa_now   = float(sv_lambda.kappa.cpu())
        theta_now   = float(sv_lambda.theta.cpu())
        sigma_v_now = float(sv_lambda.sigma_v.cpu())

    for k, v in [("price", ep_price), ("bias", ep_bias), ("eig", ep_eig),
                 ("l2", ep_l2), ("feller", ep_feller),
                 ("swp_priced", swp_priced), ("path_finite", path_finite),
                 ("kappa", kappa_now), ("theta", theta_now), ("sigma_v", sigma_v_now)]:
        hist[k].append(v)

    # Lambda diagnostics
    try:
        with torch.no_grad():
            M_now = get_K_matrix(sv_lambda, LATENT_DIM, device, torch.float32)
            lambda_min_now  = float(torch.linalg.eigvals(M_now).real.abs().min().cpu())
            lambda_norm_fro = float(sv_lambda.Lambda.norm().cpu())
    except Exception:
        lambda_min_now  = float('nan')
        lambda_norm_fro = float('nan')
    hist["lambda_min"].append(lambda_min_now)
    hist["lambda_norm"].append(lambda_norm_fro)

    # Reconstruction RMSE
    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    if do_eval:
        rmse_per_ccy, avg_rmse_bps, _ = eval_rmse_bps(model, X_tensor, meta)
        gn = grad_norm(sv_lambda.parameters())
    else:
        rmse_per_ccy, avg_rmse_bps = None, float('nan')
        gn = 0.0

    lr_now = optim.param_groups[0]["lr"]
    t_now  = time.perf_counter()
    dt_ep  = t_now - t_last_log
    t_last_log = t_now
    eta_sec = dt_ep * (EPOCHS - epoch - 1)
    eta_str = (f"{int(eta_sec//3600)}h{int((eta_sec%3600)//60):02d}m"
               if eta_sec >= 3600 else
               f"{int(eta_sec//60)}m{int(eta_sec%60):02d}s"
               if eta_sec >= 60 else f"{int(eta_sec)}s")

    row = {
        "epoch":                epoch,
        "time_total_sec":       round(t_now - t0, 1),
        "time_interval_sec":    round(dt_ep, 3),
        "loss_price":           ep_price,
        "loss_bias":            ep_bias,
        "loss_eig":             ep_eig,
        "loss_l2":              ep_l2,
        "loss_feller":          ep_feller,
        "swaption_priced_frac": swp_priced,
        "path_finite_frac":     path_finite,
        "recon_rmse_bps":       float(avg_rmse_bps),
        "nan_batches":          nan_batches,
        "gnorm":                gn,
        "lr":                   lr_now,
        "lambda_min_KQ":        lambda_min_now,
        "Lambda_norm_fro":      lambda_norm_fro,
        "kappa":                kappa_now,
        "theta":                theta_now,
        "sigma_v":              sigma_v_now,
        "sqrt_theta":           math.sqrt(max(theta_now, 0)),
    }
    for ccy in ccy_order:
        row[f"rmse_bps_{ccy}"] = (
            float(rmse_per_ccy.get(ccy, float('nan')))
            if rmse_per_ccy is not None else float('nan')
        )
    pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)

    # Header
    if (epoch // max(LOG_EVERY, 1)) % HEADER_EVERY == 0:
        print(
            f"\n{'ep':>5} {'price':>10} {'bias':>9} {'eig':>9} {'l2':>8} {'fell':>8} "
            f"{'swp%':>5} {'pth%':>5} {'recon':>7} "
            f"{'|l|min':>7} {'||L||':>7} "
            f"{'kappa':>7} {'sqrt(th)':>9} {'sv':>7} "
            f"{'gnorm':>9} {'lr':>8} {'t/ep':>6} {'ETA':>8}  diag"
        )
        print("-" * 182)

    if batch_diag and (epoch % DIAG_EVERY == 0 or epoch == EPOCHS - 1):
        diag_str = " | ".join(
            f"{d['exp']}x{d['ten']} mkt={d['mkt_bp']:.0f} mod={d['mod_bp']:.0f} "
            f"err={d['err_bp']:+.0f}bp pth={int(d['pths%'])}%"
            for d in batch_diag
        )
    else:
        diag_str = ""

    print(
        f"{epoch:>5d} "
        f"{ep_price:>10.4e} {ep_bias:>9.3e} {ep_eig:>9.3e} {ep_l2:>8.4e} {ep_feller:>8.3e} "
        f"{swp_priced*100:>4.0f}% {path_finite*100:>4.0f}% "
        f"{avg_rmse_bps:>7.2f} "
        f"{lambda_min_now:>7.4f} {lambda_norm_fro:>7.4f} "
        f"{kappa_now:>7.4f} {math.sqrt(max(theta_now,0)):>9.5f} {sigma_v_now:>7.5f} "
        f"{gn:>9.2e} {lr_now:>8.2e} "
        f"{dt_ep:>5.1f}s {eta_str:>8}  {diag_str}"
    )

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt_path = os.path.join(FIGURES_DIR, f"checkpoint_sv_lambda_ep{epoch+1}.pt")
        torch.save({
            "model_state_dict":     model.state_dict(),
            "sv_lambda_state_dict": sv_lambda.state_dict(),
            "Lambda_matrix":        sv_lambda.Lambda.detach().cpu(),
            "log_kappa":            sv_lambda.log_kappa.detach().cpu(),
            "log_theta":            sv_lambda.log_theta.detach().cpu(),
            "log_sigma_v":          sv_lambda.log_sigma_v.detach().cpu(),
            "kappa":                kappa_now,
            "theta":                theta_now,
            "sigma_v":              sigma_v_now,
            "sqrt_theta":           math.sqrt(max(theta_now, 0)),
            "latent_dim":           LATENT_DIM,
            "epoch":                epoch + 1,
            "variant":              config.VARIANT,
            "lambda_min_KQ":        lambda_min_now,
            "lambda_norm_fro":      lambda_norm_fro,
            "path_finite_frac":     path_finite,
        }, ckpt_path)
        print(f"  -> checkpoint: ep{epoch+1}  "
              f"||L||_F={lambda_norm_fro:.4f}  "
              f"sqrt(theta)={math.sqrt(max(theta_now,0)):.5f}  "
              f"kappa={kappa_now:.4f}  sigma_v={sigma_v_now:.5f}")

print("\nTraining done.")

# ==========================================================
# Final checkpoint + loss curves
# ==========================================================

final_ckpt = os.path.join(FIGURES_DIR, f"checkpoint_sv_lambda_ep{EPOCHS}.pt")
torch.save({
    "model_state_dict":     model.state_dict(),
    "sv_lambda_state_dict": sv_lambda.state_dict(),
    "Lambda_matrix":        sv_lambda.Lambda.detach().cpu(),
    "log_kappa":            sv_lambda.log_kappa.detach().cpu(),
    "log_theta":            sv_lambda.log_theta.detach().cpu(),
    "log_sigma_v":          sv_lambda.log_sigma_v.detach().cpu(),
    "kappa":                float(sv_lambda.kappa.detach()),
    "theta":                float(sv_lambda.theta.detach()),
    "sigma_v":              float(sv_lambda.sigma_v.detach()),
    "latent_dim":           LATENT_DIM,
    "epochs":               EPOCHS,
    "variant":              config.VARIANT,
}, final_ckpt)
print("Saved final checkpoint:", final_ckpt)

fig, axes = plt.subplots(4, 1, figsize=(9, 13), dpi=150)

axes[0].semilogy(hist["price"],  lw=1.0, color="darkorange", label="Pricing loss")
axes[0].semilogy(hist["bias"],   lw=1.0, color="deeppink",   label="Bias (fwd PCP)")
axes[0].semilogy(hist["eig"],    lw=1.0, color="seagreen",   label="Eig floor")
axes[0].semilogy(hist["l2"],     lw=1.0, color="royalblue",  label="L2 Lambda")
axes[0].semilogy(hist["feller"], lw=1.0, color="crimson",    label="Feller")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log)")
axes[0].set_title("SV+Lambda Training: Loss Components")
axes[0].grid(True, alpha=0.3); axes[0].legend()

axes[1].plot([100*p for p in hist["swp_priced"]], lw=1.2, color="firebrick", label="swaption_priced%")
axes[1].plot([100*p for p in hist["path_finite"]], lw=1.0, color="navy",   label="path_finite%")
axes[1].set_ylim(-2, 102); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("%")
axes[1].set_title("Decoder Coverage"); axes[1].grid(True, alpha=0.3); axes[1].legend()

sqrt_theta_hist = [math.sqrt(max(t, 0)) for t in hist["theta"]]
axes[2].plot(sqrt_theta_hist,       lw=1.2, color="purple", label="sqrt(theta)  [= s* equiv]")
axes[2].plot(hist["kappa"],         lw=1.0, color="green",  label="kappa")
axes[2].plot(hist["sigma_v"],       lw=1.0, color="orange", label="sigma_v")
axes[2].set_xlabel("Epoch"); axes[2].set_title("CIR Parameters")
axes[2].grid(True, alpha=0.3); axes[2].legend()

axes[3].plot(hist["lambda_norm"],   lw=1.2, color="purple", label="||Lambda||_F")
axes[3].set_xlabel("Epoch"); axes[3].set_title("Lambda Frobenius Norm")
axes[3].grid(True, alpha=0.3); axes[3].legend()

fig.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, f"sv_lambda_loss_dim{LATENT_DIM}_ep{EPOCHS}.png"),
    dpi=200
)
plt.close(fig)

kappa_f   = float(sv_lambda.kappa.detach())
theta_f   = float(sv_lambda.theta.detach())
sigma_v_f = float(sv_lambda.sigma_v.detach())

print("\n" + "=" * 90)
print("SV + LAMBDA MPR TRAINING COMPLETE")
print("=" * 90)
print(f"Final pricing loss  : {hist['price'][-1]:.6e}")
print(f"Final bias loss     : {hist['bias'][-1]:.6e}  (raw, before LAMBDA_BIAS={LAMBDA_BIAS} weight)")
print(f"Final Feller loss   : {hist['feller'][-1]:.6e}")
print(f"Final ||Lambda||_F  : {hist['lambda_norm'][-1]:.4f}")
print(f"Final kappa         : {kappa_f:.4f}  (mean-reversion timescale: {1/kappa_f:.2f}yr)")
print(f"Final theta         : {theta_f:.6f}  (sqrt(theta) = {theta_f**0.5:.5f})")
print(f"Final sigma_v       : {sigma_v_f:.5f}")
print(f"Feller check        : 2*kappa*theta = {2*kappa_f*theta_f:.5f}  >?  sigma_v^2 = {sigma_v_f**2:.5f}")
print(f"Final path_finite   : {hist['path_finite'][-1]*100:.1f}%")
print(f"Lambda matrix:\n{sv_lambda.Lambda.detach().cpu().numpy()}")
print("=" * 90)
