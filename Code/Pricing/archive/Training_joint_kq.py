# ==================== Joint Training with Separate K^Q ====================
"""
Consistent Q-measure joint training (v1, 2026-05-11).

The fundamental problem with Training_joint.py was that a single K appeared in
both the bond-price ODE and the simulation. The pricing loss pushed K toward
K^Q (risk-neutral), which also changed the ODE bond prices — destroying
reconstruction even with G frozen. Result: reconstruction degraded from ~5 bp
to ~2200 bp and pricing produced 100% NaN.

This script fixes that by introducing a SEPARATE K_Q module:

  - model.K  (K^P): frozen everywhere. Used only for reconstruction via the ODE.
  - K_Q      (K^Q): new trainable module, initialised from K^P. Used in BOTH
                    the simulation drift AND the ODE when decoding simulated
                    z_T for pricing. Trained purely on the pricing loss.

Consistency:
  - Reconstruction: encoder(x) → z → decode_from_z(k_override=None) → K^P in ODE
  - Pricing:        z_0 → simulate(k_override=K_Q) → z_T
                         → decode_from_z(k_override=K_Q) → K_Q in ODE
  Both the simulation and the ODE see the same K — internally consistent.

What K_Q learns:
  The pricing loss E[|σ_model(K_Q) - σ_mkt|] forces K_Q to produce a
  distribution of z_T whose decoded curves give correct swaption prices.
  This is exactly the Girsanov correction: K_Q absorbs the market price of
  risk, shifting z^*_Q away from z^*_P so that E^Q[S_T] ≈ F_0.

What is NOT changed:
  - G (decoder network): frozen
  - encoder: frozen
  - model.K (K^P): frozen
  - R (short rate): frozen
  - H: trainable (vol scaling, as before)
"""

import copy
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
from Code.Pricing.pricing import bachelier_price_torch, swap_rate_torch
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
EVAL_BATCH_SIZE = 256

EVAL_EVERY    = 1
LOG_EVERY     = 1
DIAG_EVERY    = 10
HEADER_EVERY  = 20
SAVE_EVERY    = 200

# Gradient steps per epoch. No DataLoader needed — pricing loss samples
# from df_vol independently of any curve batch.
N_STEPS_PER_EPOCH = 8

PRETRAIN_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)

# All original model components frozen — only K_Q and H are trained
LAMBDA_PRICE = 1.0          # pure pricing loss — no reconstruction conflict
LAMBDA_EIG   = 2.0          # keep K_Q mean-reverting
EIG_MIN      = 0.05

LR_KQ = 1e-4                # K_Q gets a higher LR than old joint training's LR_K
LR_H  = 5e-5                # H scaling (vol level)

N_SWAPTIONS_PER_BATCH = 16
N_PATHS_PRICING       = 512
DT_PRICING            = 1 / 12

MIN_FINITE_PATHS_FRAC = 0.10
MIN_FINITE_PATHS_ABS  = 32

LOSS_SKIP_THRESH = 1e4      # skip individual swaption if loss exceeds this

USE        = "bbg"
CCY_FILTER = "EUR"

FIGURES_DIR = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                           f"dim{LATENT_DIM}_{config.VARIANT}_joint_kq", f"ep{EPOCHS}")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================================
# Load data
# ==========================================================

meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor.float()

meta_ccy     = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]


# ==========================================================
# Initialize model + K_Q
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

# Freeze ALL original model parameters — reconstruction is untouched
for p in model.parameters():
    p.requires_grad_(False)
print("All original model parameters frozen (encoder, G, K^P, H, R).")

# K_Q: separate Q-measure drift, initialised from K^P
# Gradients flow through K_Q only — K^P in the ODE is never touched
K_Q = copy.deepcopy(model.K).to(device)
for p in K_Q.parameters():
    p.requires_grad_(True)
print(f"K_Q initialised from K^P  ({sum(p.numel() for p in K_Q.parameters())} parameters).")

# H_Q: allow H to adjust vol scaling under Q (same approach as Training_joint.py)
# We unfreeze model.H so it can be updated by the pricing loss.
for p in model.H.parameters():
    p.requires_grad_(True)
print("model.H unfrozen for vol scaling.")

# H pre-scaling: H was trained under P-measure (physical vol ~8-10x market).
# Scale down so the first pricing eval starts near market vol.
H_PRESCALE = 0.1
_h_last_linear = None
for _m in model.H.modules():
    if isinstance(_m, nn.Linear):
        _h_last_linear = _m
if _h_last_linear is not None:
    with torch.no_grad():
        _h_last_linear.weight.mul_(H_PRESCALE)
        if _h_last_linear.bias is not None:
            _h_last_linear.bias.mul_(H_PRESCALE)
    print(f"Pre-scaled H output layer by {H_PRESCALE}")
else:
    print("WARNING: could not find H's last linear layer for pre-scaling")

model.train()

param_groups = [
    {"params": list(K_Q.parameters()),     "lr": LR_KQ, "name": "K_Q"},
    {"params": list(model.H.parameters()), "lr": LR_H,  "name": "H"},
]
optim = torch.optim.Adam(param_groups)

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
# Eigenvalue floor on K_Q
# ==========================================================

def get_K_matrix(k_module, dim, device, dtype):
    """Recover linear part M of drift k_module(z) = M z + bias (differentiable)."""
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
    print("WARNING: No overlapping dates for swaption vols. Check data.")
    raise RuntimeError("No swaption vol data — cannot train.")
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
    """Reconstruction RMSE — uses K^P (model.K), unaffected by K_Q."""
    S_hat_all = predict_S_hat(model, X_full, batch_size=batch_size)
    mask = row_finite_mask(X_full) & row_finite_mask(S_hat_all)
    n_bad  = int((~mask).sum().item())
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
# Pricing loss — consistent K_Q in ODE + simulation
# ==========================================================

def compute_pricing_loss_kq(
    model,
    K_Q,
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
    Pricing loss with K_Q used consistently in:
      1. simulate_to_expiry_differentiable (k_override=K_Q)
      2. decode_from_z for terminal curves   (k_override=K_Q)

    model.K (K^P) is never called here — only K_Q and model.H are used.
    Time-0 quantities (F_0, A_0) use model.K (K^P) since they reflect the
    current market curve as encoded by the pretrained model.
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

        dt_eff  = min(dt, 1 / 12, expiry / 10.0)
        n_steps = max(12, int(round(expiry / dt_eff)))

        half     = n_paths // 2
        eps_half = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)
        eps      = torch.cat([eps_half, -eps_half], dim=0)

        try:
            # Simulate with K_Q — consistent Q-measure drift
            z_T, D_T = simulate_to_expiry_differentiable(
                model=model, z0=z0, n_steps=n_steps, dt=dt_eff,
                n_paths=n_paths, eps=eps,
                k_override=K_Q,          # <-- K^Q in simulation
            )

            z_finite = torch.isfinite(z_T).all(dim=1)
            if D_T.ndim == 1:
                d_finite = torch.isfinite(D_T)
            else:
                d_finite = torch.isfinite(D_T).all(dim=1)
            valid_pre_decode = z_finite & d_finite
            if int(valid_pre_decode.sum().item()) < min_finite_paths:
                continue

            # First pass: probe decoder with K_Q in ODE (no grad)
            with torch.no_grad():
                _, aux_check = model.decode_from_z(
                    z_T, tau=None, return_aux=True,
                    k_override=K_Q,       # <-- K^Q in ODE
                )
                P_check  = aux_check["P_full"]
                p_finite = torch.isfinite(P_check).all(dim=1)

            finite_mask = valid_pre_decode & p_finite
            n_finite    = int(finite_mask.sum().item())
            path_frac   = n_finite / max(n_paths, 1)
            path_finite_fracs.append(path_frac)
            if n_finite < min_finite_paths:
                continue

            # Second pass: survivors with grad, K_Q in ODE
            z_T_keep = z_T[finite_mask]
            _, aux_T = model.decode_from_z(
                z_T_keep, tau=None, return_aux=True,
                k_override=K_Q,           # <-- K^Q in ODE
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

            # Time-0 reference: use K^P (original model) — reflects encoded market curve
            with torch.no_grad():
                _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
                P_full_0 = aux0["P_full"]
                F_0_t, A_0_t = swap_rate_torch(P_full_0, tenor=tenor)
                F_0 = float(F_0_t[0].item())
                A_0 = float(A_0_t[0].item())

            if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 0):
                continue

            payoff      = A_T * torch.relu(F_T - F_0)
            disc_payoff = D_keep * payoff
            V_MC        = disc_payoff.mean()

            if not torch.isfinite(V_MC):
                continue

            sigma_mod_bp_t = (V_MC * sqrt_2pi) / (A_0 * math.sqrt(expiry)) * 10_000.0
            sigma_mkt_bp   = sigma_mkt * 10_000.0

            # MSE in vol space, scaled to O(1): 50 bp error → loss = 0.25
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
csv_path  = os.path.join(FIGURES_DIR, f"train_joint_kq_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")

csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec",
     "loss_price", "loss_eig",
     "swaption_priced_frac", "path_finite_frac",
     "recon_rmse_bps",
     "nan_batches",
     "gnorm_KQ", "gnorm_H",
     "lr_KQ", "lr_H",
     "lambda_min_KQ"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
)
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "version": "kq_v1",
    "seed": SEED, "latent_dim": LATENT_DIM, "variant": config.VARIANT,
    "epochs": EPOCHS, "n_steps_per_epoch": N_STEPS_PER_EPOCH,
    "pretrain_ckpt": PRETRAIN_CKPT,
    "lr_kq": LR_KQ, "lr_h": LR_H,
    "lambda_price": LAMBDA_PRICE, "lambda_eig": LAMBDA_EIG, "eig_min": EIG_MIN,
    "n_swaptions_per_batch": N_SWAPTIONS_PER_BATCH,
    "n_paths_pricing": N_PATHS_PRICING, "dt_pricing": DT_PRICING,
    "loss": "mse_vol_bp_div100",
    "ccy_filter": CCY_FILTER, "save_every": SAVE_EVERY,
}
with open(os.path.join(FIGURES_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

# ==========================================================
# Training loop
# ==========================================================

train_losses_price = []
train_losses_eig   = []
swaption_priced_hist = []
path_finite_hist     = []
lambda_min_hist      = []

t0         = time.perf_counter()
t_last_log = t0

print("\n" + "=" * 80)
print("JOINT TRAINING K_Q v1: separate Q-measure drift, consistent ODE + simulation")
print("=" * 80)
print(f"Price weight : {LAMBDA_PRICE}")
print(f"Eig weight   : {LAMBDA_EIG}   (floor |Re(λ_KQ)| ≥ {EIG_MIN})")
print(f"Loss         : MSE in vol space (bp / 100)²")
print(f"K_Q params   : {sum(p.numel() for p in K_Q.parameters())}")
print(f"H   params   : {sum(p.numel() for p in model.H.parameters())}")
print("=" * 80 + "\n")

for epoch in range(EPOCHS):
    model.train()
    K_Q.train()
    running_price = 0.0
    running_eig   = 0.0
    n_batches = 0
    nan_batches = 0
    batch_diagnostics = []
    epoch_attempted    = 0
    epoch_priced       = 0
    epoch_path_fracs   = []

    # No DataLoader needed — pricing loss samples from df_vol independently.
    # N_STEPS_PER_EPOCH gradient steps per epoch, each pricing N_SWAPTIONS_PER_BATCH swaptions.
    for step in range(N_STEPS_PER_EPOCH):
        print(f"\r  ep {epoch}  step {step+1}/{N_STEPS_PER_EPOCH} ...", end="", flush=True)
        optim.zero_grad(set_to_none=USE_SET_TO_NONE)

        # Pricing loss — K_Q used in both ODE and simulation
        return_diag = (step == 0)
        loss_price, batch_diag, n_att, n_pri, p_frac = compute_pricing_loss_kq(
            model=model, K_Q=K_Q,
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

        # Eigenvalue floor on K_Q
        loss_eig = torch.tensor(0.0, device=device, dtype=torch.float32)
        if LAMBDA_EIG > 0:
            try:
                M = get_K_matrix(K_Q, LATENT_DIM, device, torch.float32)
                loss_eig = eigenvalue_floor_loss(M, eig_min=EIG_MIN)
                if not torch.isfinite(loss_eig):
                    loss_eig = torch.tensor(0.0, device=device, dtype=torch.float32)
            except Exception:
                pass

        loss_total = LAMBDA_PRICE * loss_price + LAMBDA_EIG * loss_eig

        if not torch.isfinite(loss_total):
            nan_batches += 1
            continue

        loss_total.backward()

        has_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in list(K_Q.parameters()) + list(model.H.parameters())
        )
        if has_nan_grad:
            nan_batches += 1
            optim.zero_grad(set_to_none=USE_SET_TO_NONE)
            continue

        torch.nn.utils.clip_grad_norm_(
            list(K_Q.parameters()) + list(model.H.parameters()), max_norm=1.0
        )
        optim.step()

        running_price += float(loss_price.detach().cpu())
        running_eig   += float(loss_eig.detach().cpu())
        n_batches += 1

    print("\r" + " " * 40 + "\r", end="", flush=True)  # clear progress line

    scheduler.step()

    epoch_price = running_price / max(n_batches, 1)
    epoch_eig   = running_eig   / max(n_batches, 1)
    swaption_priced = epoch_priced / max(epoch_attempted, 1)
    path_finite     = float(np.mean(epoch_path_fracs)) if epoch_path_fracs else 0.0

    train_losses_price.append(epoch_price)
    train_losses_eig.append(epoch_eig)
    swaption_priced_hist.append(swaption_priced)
    path_finite_hist.append(path_finite)

    try:
        with torch.no_grad():
            M_now = get_K_matrix(K_Q, LATENT_DIM, device, torch.float32)
            lambda_min_now = float(torch.linalg.eigvals(M_now).real.abs().min().cpu())
    except Exception:
        lambda_min_now = float('nan')
    lambda_min_hist.append(lambda_min_now)

    # Reconstruction RMSE (uses K^P — should stay constant)
    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    if do_eval:
        rmse_per_ccy, avg_rmse_bps, n_bad = eval_rmse_bps(
            model, X_tensor, meta, batch_size=EVAL_BATCH_SIZE
        )
        gn_kq = grad_norm(K_Q.parameters())
        gn_h  = grad_norm(model.H.parameters())
    else:
        rmse_per_ccy, avg_rmse_bps = None, float('nan')
        gn_kq = gn_h = 0.0

    lrs_now = {pg["name"]: pg["lr"] for pg in optim.param_groups}

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
        "swaption_priced_frac": swaption_priced,
        "path_finite_frac":     path_finite,
        "recon_rmse_bps":       float(avg_rmse_bps),
        "nan_batches":          nan_batches,
        "gnorm_KQ":             gn_kq,
        "gnorm_H":              gn_h,
        "lr_KQ":                lrs_now.get("K_Q", float('nan')),
        "lr_H":                 lrs_now.get("H",   float('nan')),
        "lambda_min_KQ":        lambda_min_now,
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
            f"\n{'ep':>5} {'price':>10} {'eig':>9} "
            f"{'swp%':>5} {'pth%':>5} {'recon':>7} {'|λ|':>7} "
            f"{'gKQ':>9} {'gH':>9} {'lrKQ':>8} {'t/ep':>6} {'ETA':>8}  pricing_diag"
        )
        print("-" * 145)

    if batch_diagnostics and (epoch % DIAG_EVERY == 0 or epoch == EPOCHS - 1):
        diag_str = " | ".join(
            f"{d['exp']}x{d['ten']} mkt={d['mkt_bp']:.0f} mod={d['mod_bp']:.0f} "
            f"err={d['err_bp']:+.0f}bp pth={int(d['pths%'])}%"
            for d in batch_diagnostics
        )
    else:
        diag_str = ""

    print(
        f"{epoch:>5d} "
        f"{epoch_price:>10.4e} "
        f"{epoch_eig:>9.3e} "
        f"{swaption_priced*100:>4.0f}% "
        f"{path_finite*100:>4.0f}% "
        f"{avg_rmse_bps:>7.2f} "
        f"{lambda_min_now:>7.4f} "
        f"{gn_kq:>9.2e} "
        f"{gn_h:>9.2e} "
        f"{lrs_now.get('K_Q', 0):>8.2e} "
        f"{time_int:>5.1f}s "
        f"{eta_str:>8}  "
        f"{diag_str}"
    )

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt_path = os.path.join(FIGURES_DIR, f"checkpoint_kq_ep{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "K_Q_state_dict":   K_Q.state_dict(),
            "model_config":     {"latent_dim": LATENT_DIM},
            "latent_dim":       LATENT_DIM,
            "epoch":            epoch + 1,
            "variant":          config.VARIANT,
            "lambda_price":     LAMBDA_PRICE,
            "lambda_eig":       LAMBDA_EIG,
            "swaption_priced_frac": swaption_priced,
            "path_finite_frac":     path_finite,
            "lambda_min_KQ":        lambda_min_now,
        }, ckpt_path)
        print(f"  → checkpoint saved: ep{epoch+1}  "
              f"(swp={swaption_priced*100:.0f}%, pth={path_finite*100:.0f}%, "
              f"|λ|={lambda_min_now:.4f})")

print("\nTraining done.")

# ==========================================================
# Save final checkpoint + plots
# ==========================================================

final_ckpt = os.path.join(FIGURES_DIR, f"checkpoint_kq_ep{EPOCHS}.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "K_Q_state_dict":   K_Q.state_dict(),
    "latent_dim":       LATENT_DIM,
    "epochs":           EPOCHS,
    "variant":          config.VARIANT,
}, final_ckpt)
print("Saved final checkpoint:", final_ckpt)

fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150)
axes[0].plot(train_losses_price, lw=1.0, color="darkorange", label="Pricing (Huber)")
axes[0].plot(train_losses_eig,   lw=1.0, color="seagreen",   label="Eig floor K_Q")
axes[0].set_yscale("log"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("K_Q Joint Training: Pricing + Eigenvalue Loss")
axes[0].grid(True, alpha=0.3); axes[0].legend()

axes[1].plot([100*p for p in swaption_priced_hist], lw=1.2, color="firebrick",
             label="swaption_priced_frac (%)")
axes[1].plot([100*p for p in path_finite_hist], lw=1.0, color="navy",
             label="path_finite_frac (%)")
axes[1].axhline(95, color="grey", ls=":", lw=1, label="target 95%")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("%")
axes[1].set_title("Decoder Coverage During K_Q Training")
axes[1].set_ylim(-2, 102); axes[1].grid(True, alpha=0.3); axes[1].legend(loc="lower right")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"kq_loss_dim{LATENT_DIM}_ep{EPOCHS}.png"), dpi=200)
plt.close(fig)

print("\n" + "=" * 80)
print("K_Q TRAINING COMPLETE")
print("=" * 80)
print(f"Final pricing loss     : {train_losses_price[-1]:.6e}")
print(f"Final eig-floor loss   : {train_losses_eig[-1]:.6e}")
print(f"Final swaption_priced  : {swaption_priced_hist[-1]*100:.1f}%")
print(f"Final path_finite_frac : {path_finite_hist[-1]*100:.1f}%")
print(f"Final |λ|min for K_Q   : {lambda_min_hist[-1]:.4f}")
print("Reconstruction RMSE is unchanged (K^P frozen) — verify with eval_joint.py")
print("=" * 80)
