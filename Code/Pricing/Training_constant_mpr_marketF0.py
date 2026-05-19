# ============== Constant MPR Pricing Calibration (market F_0) =================
"""
Constant Market Price of Risk (MPR) pricing adjustment — market-curve variant.

F_0 and A_0 (the swaption strike and forward annuity) are taken DIRECTLY
from the market par-swap-rate curve via standard bootstrap, NOT from the
decoder.  This is the "calibration to today's curve" approach used
throughout the interest-rate options literature (Hull, Andersen-Piterbarg,
Brigo-Mercurio): the market discount curve is treated as exogenous, and
the model is asked only to provide the F_{T_e} distribution.

  - t = 0:    P(0, .) = market bootstrap from par swap rates  (exact)
  - t > 0:    z_t simulated under (μ + σ η, diag(s) σ)
  - t = T_e:  P^*(T_e, .) from the bond ODE under (μ + σ η, diag(s) σ)

Because the t=0 curve is the market curve by construction, there is no
anchoring step and no convexity-vs-vol tension in the bond ODE at t=0.
The optimizer is free to choose any (η, s) that fits the F_{T_e}
distribution.

This model directly addresses the instability found in Lambda v2.
The Lambda v2 correction  L(z) @ Lambda @ z  grows with |z|, creating
a position-dependent feedback loop that amplifies small perturbations
into explosive F_T variance (diagnosed in _path_diagnostics.py).

This model replaces Lambda @ z with a CONSTANT vector lambda_0:

    K_price(z) = K_base(z) + L_base(z) @ lambda_0

    where lambda_0 in R^d is a learnable constant (does not depend on z).

Key properties vs Lambda v2:
  - No position-dependent feedback: the correction is the same everywhere
  - Cannot amplify z perturbations (L(z) varies with z, but lambda_0 is fixed)
  - Can still correct systematic forward bias via the L(z) @ lambda_0 term
  - 8 trainable parameters (lambda_0 [d] + log_sigma_vec [d])
  - Easy to explain: constant drift adjustment + per-factor vol scaling

Interpretation: lambda_0 is a constant risk-premium vector in latent space.
L(z) @ lambda_0 maps it to the physical drift correction.

Output: Figures/TrainingResults/dim4_constant_mpr_marketF0/ep{EPOCHS}/
"""

import os, sys

_N_TORCH   = 4
_N_INTEROP = 2
os.environ.setdefault("OMP_NUM_THREADS",        str(_N_TORCH))
os.environ.setdefault("MKL_NUM_THREADS",        str(_N_TORCH))
os.environ.setdefault("OPENBLAS_NUM_THREADS",   str(_N_TORCH))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_N_TORCH))
os.environ.setdefault("NUMEXPR_NUM_THREADS",    str(_N_TORCH))

import copy, time, math, json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(_N_TORCH)
torch.set_num_interop_threads(_N_INTEROP)

try:
    import psutil
    _proc = psutil.Process()
    _pin_cores = list(range(os.cpu_count()))[:_N_TORCH * 2]
    _proc.cpu_affinity(_pin_cores)
    print(f"CPU affinity pinned to cores {_pin_cores}")
except Exception as _e:
    print(f"CPU affinity not set ({_e})")

# ── paths ──────────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code import config
config.confirm_variant()

from Code.utils import helpers as H
from Code.load_swapdata import my_data
from Code.model.full_model_price import FullModelPrice as FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import swap_rate_torch, forward_swap_rate_torch
from Code.Pricing.bootstrap_market_curve import bootstrap_discount_curve
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Variant:", config.VARIANT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ── settings ───────────────────────────────────────────────────────────────────
LATENT_DIM = 4

EPOCHS                = 1000
EVAL_EVERY            = 100
LOG_EVERY             = 1
DIAG_EVERY            = 10
HEADER_EVERY          = 20
SAVE_EVERY            = 200
N_STEPS_PER_EPOCH     = 4
N_SWAPTIONS_PER_BATCH = 8
N_PATHS_PRICING       = 512
DT_PRICING            = 1 / 6

# Par-swap-rate tenors in X_tensor and max maturity needed for swaptions.
# expiry + tenor up to 10 + 10 = 20.
SWAP_TENORS       = [1, 2, 3, 5, 10, 15, 20, 30]
MAX_TENOR_NEEDED  = 20

PRETRAIN_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)

# Loss weights
LAMBDA_VOL  = 1.0    # straddle-implied ATM vol MSE
LAMBDA_BIAS = 0.5    # ATM payer-receiver parity penalty
LAMBDA_L2   = 1e-3   # L2 on lambda_0  (light regularisation, keeps it bounded)

LR = 5e-4
LR_WARMUP_EPOCHS = 50

MIN_FINITE_PATHS_FRAC = 0.10
MIN_FINITE_PATHS_ABS  = 16
LOSS_SKIP_THRESH      = 1e4


USE        = "bbg"
CCY_FILTER = "EUR"

FIGURES_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_constant_mpr_marketF0", f"ep{EPOCHS}"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── module ─────────────────────────────────────────────────────────────────────

class ConstantMPRAdjustment(nn.Module):
    """
    Constant Market Price of Risk pricing adjustment.

    K_price(z) = K_base(z) + L_base(z) @ lambda_0

    lambda_0 in R^d is a learnable CONSTANT vector — it does not depend on z.
    This eliminates the position-dependent feedback loop present in Lambda v2:

        Lambda v2:      L(z) @ Lambda @ z   (grows with |z| -> explosive)
        Constant MPR:   L(z) @ lambda_0     (constant correction -> stable)

    L(z) itself varies with z (it is the Cholesky factor of the diffusion
    matrix), but since lambda_0 is fixed, the correction cannot amplify
    deviations of z from its initial value.

    Together with sigma_vec, this model can:
      - Shift the distribution of F_T to correct systematic forward bias
      - Adjust the width of the distribution via sigma_vec
      - Do so without creating explosive SDE dynamics

    Trainable: lambda_0 (d) + log_sigma_vec (d) = 2d = 8 parameters.
    """

    def __init__(self, kp_module, h_module, latent_dim):
        super().__init__()
        self.kp         = kp_module
        self.h          = h_module
        self.latent_dim = latent_dim

        # Constant drift bias in latent space.  Init = 0 (base dynamics).
        self.lambda_0      = nn.Parameter(torch.zeros(latent_dim))
        # Per-dimension diffusion scale.  Init exp(-1.8) ~ 0.165.
        self.log_sigma_vec = nn.Parameter(torch.full((latent_dim,), -1.8))

    def forward(self, z):
        """K_price(z) = K_base(z) + L_base(z) @ lambda_0

        lambda_0 is constant — it does NOT grow with |z|.
        L(z) varies with z (it encodes the diffusion geometry),
        but the product L(z) @ lambda_0 cannot amplify z perturbations
        because lambda_0 is fixed.
        """
        k_base       = self.kp(z)                                         # (batch, d)
        sigmas, rhos = self.h(z)
        L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)   # (batch, d, d)
        # Broadcast lambda_0 over the batch dimension
        lam0 = self.lambda_0.unsqueeze(0).expand(z.shape[0], -1)          # (batch, d)
        return k_base + torch.einsum('bij,bj->bi', L, lam0)

    @property
    def sigma_vec(self):
        """Per-dimension diffusion scales: shape (d,)"""
        return self.log_sigma_vec.exp()


# ── helpers ────────────────────────────────────────────────────────────────────

def row_finite_mask(t):
    return torch.isfinite(t).all(dim=1)


@torch.no_grad()
def predict_S_hat(model, X, batch_size=256):
    was_train = model.training
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        outs.append(model(X[i:i+batch_size].to(device)).detach().cpu())
    if was_train:
        model.train()
    return torch.cat(outs, dim=0)


def eval_rmse_bps(model, X_full, meta_full, batch_size=256):
    S_hat = predict_S_hat(model, X_full, batch_size)
    mask  = row_finite_mask(X_full) & row_finite_mask(S_hat)
    rmse  = H.rmse_bps_per_currency_paper(
        X_full[mask], S_hat[mask],
        meta_full.loc[mask.numpy()].reset_index(drop=True),
    )
    return rmse, float(rmse.mean()), int((~mask).sum())


def grad_norm(params):
    total = sum(
        float(p.grad.detach().pow(2).sum().cpu())
        for p in params if p.grad is not None
    )
    return total ** 0.5



# ── pricing loss ───────────────────────────────────────────────────────────────

def compute_pricing_loss(
    model, lm,
    X_batch, meta_batch,
    df_vol, date_to_idx,
    n_swaptions, n_paths, dt,
    device, dtype,
    return_diagnostics=False,
):
    """
    Straddle-implied vol loss + ATM parity bias penalty.

    sigma_str = (V_pay + V_rec)/2 * sqrt(2*pi) / (A_0 * sqrt(T))
    loss_vol  = ((sigma_str - sigma_mkt) / 100)^2
    loss_bias = ((V_pay - V_rec) / A_0 * 1e4 / 100)^2
    """
    if len(df_vol) == 0 or n_swaptions == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, [], 0, 0, 0.0, float("nan")

    sample = df_vol.sample(n=min(n_swaptions, len(df_vol)))

    total_vol  = torch.zeros(1, device=device, dtype=dtype)
    total_bias = torch.zeros(1, device=device, dtype=dtype)
    n_valid      = 0
    n_attempted  = 0
    diagnostics  = []
    path_fracs   = []
    anchor_errors = []
    min_paths    = max(MIN_FINITE_PATHS_ABS, int(n_paths * MIN_FINITE_PATHS_FRAC))

    for _, row in sample.iterrows():
        date      = pd.Timestamp(row["as_of_date"]).normalize()
        expiry    = int(row["option_maturity"])
        tenor     = int(row["swap_tenor"])
        sigma_mkt = float(row["market_vol"])
        sigma_mkt_bp = sigma_mkt * 1e4

        if date not in date_to_idx:
            continue

        n_attempted += 1
        idx = date_to_idx[date]
        xb  = X_batch[idx:idx + 1].to(device)

        # z0 = E(S_0) is the starting latent state for the simulation.
        # F_0 and A_0, however, are read DIRECTLY from the market par-swap
        # curve via bootstrap — not from the decoder.  This decouples the
        # strike from the encoder/decoder roundtrip and removes the
        # convexity-vs-vol tension we'd hit if we required the bond ODE at
        # t=0 to also reproduce the market curve.
        with torch.no_grad():
            z0 = model.encoder(xb)

        try:
            P0 = bootstrap_discount_curve(
                xb.squeeze(0).detach(), SWAP_TENORS, max_tenor=MAX_TENOR_NEEDED,
            ).to(device)
        except Exception:
            continue
        if not torch.isfinite(P0).all():
            continue


            if not torch.isfinite(P0).all():
                continue

        max_idx = P0.shape[0] - 1
        if expiry + tenor > max_idx:
            continue
        F_0, A_0 = forward_swap_rate_torch(P0, expiry, tenor)
        if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 1e-6):
            continue

        dt_eff  = min(dt, expiry / 10.0)
        n_steps = max(12, int(round(expiry / dt_eff)))
        half    = n_paths // 2

        with torch.no_grad():
            eps_z = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)

        try:
            z_T, D_T = simulate_to_expiry_differentiable(
                model, z0,
                n_steps=n_steps, dt=dt_eff,
                n_paths=2 * half,
                eps=eps_z,
                k_override=lm,
                sigma_scale=lm.sigma_vec,
                antithetic=True,
                freeze_H=True,
            )

            # Two-stage decode to prevent invalid paths from poisoning gradients:
            # Stage 1: Probe under no_grad to find valid terminal states
            with torch.no_grad():
                z_ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
                if int(z_ok.sum()) < min_paths:
                    continue

                z_probe = z_T[z_ok].detach()
                _, aux_probe = model.decode_from_z(
                    z_probe, tau=None, return_aux=True,
                    k_override=lm, sigma_scale=lm.sigma_vec,
                )

                p_ok_local = torch.isfinite(aux_probe["P_full"]).all(1)
                if int(p_ok_local.sum()) < min_paths:
                    continue

                # Map local indices back to global
                global_idx = torch.nonzero(z_ok, as_tuple=False).squeeze(1)
                keep_idx = global_idx[p_ok_local]

            # Stage 2: Re-decode only valid paths WITH gradients
            # This ensures clean gradient flow from loss → F_T/A_T → P^*(T,·) → z_T → lm
            z_keep = z_T[keep_idx]
            D_keep = D_T[keep_idx]

            _, aux_keep = model.decode_from_z(
                z_keep, tau=None, return_aux=True,
                k_override=lm, sigma_scale=lm.sigma_vec,
            )

            path_fracs.append(float(len(keep_idx) / len(z_T)))

            F_T, A_T = swap_rate_torch(aux_keep["P_full"], tenor=tenor)
            D_keep   = D_keep

            fa_ok = (torch.isfinite(F_T) & torch.isfinite(A_T)
                     & (F_T > -0.5) & (F_T < 0.5)
                     & (A_T > 1e-6) & (A_T < 50.0))
            if int(fa_ok.sum()) < min_paths:
                continue

            F_T, A_T, D_keep = F_T[fa_ok], A_T[fa_ok], D_keep[fa_ok]

            V_pay = (D_keep * A_T * torch.relu(F_T - F_0)).mean()
            V_rec = (D_keep * A_T * torch.relu(F_0 - F_T)).mean()
            if not (torch.isfinite(V_pay) and torch.isfinite(V_rec)
                    and float(V_pay.detach()) >= 0 and float(V_rec.detach()) >= 0):
                continue

            sqrt_2pi     = math.sqrt(2 * math.pi)
            V_str        = (V_pay + V_rec) * 0.5
            sigma_str_bp = V_str * sqrt_2pi / (A_0 * math.sqrt(expiry)) * 1e4

            loss_vol_ij = ((sigma_str_bp - sigma_mkt_bp) / 100.0).pow(2)
            if not torch.isfinite(loss_vol_ij) or float(loss_vol_ij.detach()) > LOSS_SKIP_THRESH:
                continue

            fwd_bias_bp  = (V_pay - V_rec) / A_0 * 1e4
            loss_bias_ij = (fwd_bias_bp / 100.0).pow(2)
            if not torch.isfinite(loss_bias_ij):
                continue

            total_vol  = total_vol  + loss_vol_ij
            total_bias = total_bias + loss_bias_ij
            n_valid   += 1

            if return_diagnostics:
                diagnostics.append({
                    "date":    date.date(),
                    "exp":     expiry,
                    "ten":     tenor,
                    "mkt_bp":  round(sigma_mkt_bp, 1),
                    "mod_bp":  round(float(sigma_str_bp.detach()), 1),
                    "err_bp":  round(float(sigma_str_bp.detach()) - sigma_mkt_bp, 1),
                    "bias_bp": round(float(fwd_bias_bp.detach()), 1),
                    "F0":      round(F_0 * 1e4, 1),
                    "scale":   [round(v, 4) for v in lm.sigma_vec.detach().cpu().tolist()],
                    "lambda0": [round(v, 4) for v in lm.lambda_0.detach().cpu().tolist()],
                })

        except Exception:
            continue

    mean_pfrac = float(np.mean(path_fracs)) if path_fracs else 0.0
    mean_anchor_bp = float(np.mean(anchor_errors)) if anchor_errors else float("nan")
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    if n_valid > 0:
        return (total_vol / n_valid, total_bias / n_valid,
                diagnostics, n_attempted, n_valid, mean_pfrac, mean_anchor_bp)
    return zero, zero, diagnostics, n_attempted, 0, mean_pfrac, mean_anchor_bp


# ── load data ──────────────────────────────────────────────────────────────────

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT \
    = my_data(use=USE)
X_tensor = X_tensor.float()

meta_ccy     = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]

# ── model ──────────────────────────────────────────────────────────────────────

model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
for p in model.parameters():
    p.requires_grad_(False)
print("Base model loaded and frozen.")

lm = ConstantMPRAdjustment(model.K, model.H, LATENT_DIM).to(device)

print("ConstantMPRAdjustment initialised.")
print(f"  lambda_0 init  = {lm.lambda_0.detach().cpu().numpy().round(4)}  (zeros)")
sv_init = lm.sigma_vec.detach().cpu().numpy()
print(f"  sigma_vec init = {sv_init.round(4)}  (mean={sv_init.mean():.4f})")

n_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
print(f"Trainable params: {n_params}  (lambda_0 {LATENT_DIM} + log_sigma_vec {LATENT_DIM})")

model.train()

# Two param groups: lambda_0 (drift) and sigma_vec (vol scale).
# sigma_vec gets a higher LR — vol gradients can be small relative to bias gradients.
LR_SCALE_MULT = 10.0
optim = torch.optim.Adam([
    {'params': [lm.lambda_0],        'lr': LR,                  'name': 'lambda_0'},
    {'params': [lm.log_sigma_vec],   'lr': LR * LR_SCALE_MULT,  'name': 'sigma_vec'},
], lr=LR)
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

# ── swaption data ──────────────────────────────────────────────────────────────

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0

dates_swap = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
df_vol     = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()

if df_vol.empty:
    raise RuntimeError("No swaption vol data.")
print(f"Loaded {len(df_vol)} vol targets from {df_vol['as_of_date'].nunique()} dates")

date_to_idx = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_ccy.iterrows()
}

# ── CSV logger ─────────────────────────────────────────────────────────────────

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
csv_path  = os.path.join(FIGURES_DIR, f"train_constant_mpr_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")
csv_cols  = (
    ["epoch", "time_total_sec", "time_interval_sec",
     "loss_vol", "loss_bias", "loss_l2",
     "swaption_priced_frac", "path_finite_frac", "anchor_rmse_bp",
     "recon_rmse_bps", "nan_batches",
     "gnorm_lam0", "gnorm_scale", "lr",
     "lambda0_norm", "lambda0_l1",
     "sigma_scale_mean", "sigma_s1", "sigma_s2", "sigma_s3", "sigma_s4",
     "lambda0_v1", "lambda0_v2", "lambda0_v3", "lambda0_v4",
     "fwd_bias_diag_bp"]
    + [f"rmse_bps_{c}" for c in ccy_order]
)
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "version":               "constant_mpr_marketF0",
    "description":           "K_price(z) = K_base(z) + L_base(z) @ lambda_0, stable constant correction",
    "seed":                  SEED,
    "latent_dim":            LATENT_DIM,
    "variant":               config.VARIANT,
    "epochs":                EPOCHS,
    "lr":                    LR,
    "lambda_vol":            LAMBDA_VOL,
    "lambda_bias":           LAMBDA_BIAS,
    "lambda_l2":             LAMBDA_L2,
    "n_swaptions_per_batch": N_SWAPTIONS_PER_BATCH,
    "n_paths_pricing":       N_PATHS_PRICING,
    "dt_pricing":            DT_PRICING,
    "n_trainable_params":    n_params,
}
with open(os.path.join(FIGURES_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

# ── training loop ──────────────────────────────────────────────────────────────

hist = {k: [] for k in [
    "vol", "bias", "l2",
    "swp_priced", "path_finite",
    "anchor_rmse",
    "sigma_scale", "lambda0_norm",
]}

t0         = time.perf_counter()
t_last_log = t0

print("\n" + "=" * 100)
print("CONSTANT MPR PRICING CALIBRATION")
print("  K_price(z) = K_base(z) + L_base(z) @ lambda_0")
print("  lambda_0 in R^d is CONSTANT (no position-dependent feedback)")
print("  Trainable: lambda_0 (4 params) + log_sigma_vec (4 params) = 8 total")
print("=" * 100 + "\n")

for epoch in range(EPOCHS):
    model.train()
    lm.train()
    running_vol  = 0.0
    running_bias = 0.0
    running_l2   = 0.0
    n_batches     = 0
    nan_batches   = 0
    batch_diag    = []
    ep_attempted  = 0
    ep_priced     = 0
    ep_pfracs     = []
    ep_anchor = []

    for step in range(N_STEPS_PER_EPOCH):
        print(f"\r  ep {epoch}  step {step+1}/{N_STEPS_PER_EPOCH} ...",
              end="", flush=True)
        optim.zero_grad(set_to_none=True)

        loss_vol, loss_bias_raw, diag, n_att, n_pri, p_frac, anchor_bp = compute_pricing_loss(
            model=model, lm=lm,
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

        if not math.isnan(anchor_bp):
            ep_anchor.append(anchor_bp)

        # Light L2 on lambda_0 to keep it bounded
        loss_l2    = LAMBDA_L2 * lm.lambda_0.pow(2).sum()
        loss_bias  = LAMBDA_BIAS * loss_bias_raw
        loss_total = LAMBDA_VOL * loss_vol + loss_bias + loss_l2

        if not torch.isfinite(loss_total):
            nan_batches += 1
            continue

        loss_total.backward()

        has_nan = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in lm.parameters()
        )
        if has_nan:
            nan_batches += 1
            optim.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_([lm.lambda_0],        max_norm=2.0)
        torch.nn.utils.clip_grad_norm_([lm.log_sigma_vec],   max_norm=2.0)
        optim.step()

        running_vol  += float(loss_vol.detach().cpu())
        running_bias += float(loss_bias_raw.detach().cpu())
        running_l2   += float(loss_l2.detach().cpu())
        n_batches    += 1

    print("\r" + " " * 40 + "\r", end="", flush=True)
    
    # Only step scheduler if we had valid optimizer updates
    if n_batches > 0:
        scheduler.step()
    elif epoch == 0:
        print(f"  WARNING: No valid batches in epoch {epoch}! nan_batches={nan_batches}")

    ep_vol  = running_vol  / max(n_batches, 1)
    ep_bias = running_bias / max(n_batches, 1)
    ep_l2   = running_l2   / max(n_batches, 1)
    swp_priced  = ep_priced  / max(ep_attempted, 1)
    path_finite = float(np.mean(ep_pfracs))   if ep_pfracs   else 0.0
    anchor_rmse_bp = float(np.mean(ep_anchor)) if ep_anchor else float("nan")

    with torch.no_grad():
        sigma_vec_now  = lm.sigma_vec.detach().cpu()
        scale_now      = float(sigma_vec_now.mean())
        lambda0_now    = lm.lambda_0.detach().cpu()
        lambda0_norm   = float(lambda0_now.norm())
        lambda0_l1     = float(lambda0_now.abs().sum())

    for k, v in [("vol", ep_vol), ("bias", ep_bias), ("l2", ep_l2),
                 ("swp_priced", swp_priced), ("path_finite", path_finite), ("anchor_rmse", anchor_rmse_bp),
                 ("sigma_scale", scale_now), ("lambda0_norm", lambda0_norm)]:
        hist[k].append(v)

    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    if do_eval:
        rmse_per_ccy, avg_rmse_bps, _ = eval_rmse_bps(model, X_tensor, meta)
        gn_lam0  = grad_norm([lm.lambda_0])
        gn_scale = grad_norm([lm.log_sigma_vec])
    else:
        rmse_per_ccy, avg_rmse_bps = None, float('nan')
        gn_lam0  = 0.0
        gn_scale = 0.0

    mean_bias_diag = float(np.mean([d["bias_bp"] for d in batch_diag])) if batch_diag else float('nan')

    lr_now  = optim.param_groups[0]["lr"]
    t_now   = time.perf_counter()
    dt_ep   = t_now - t_last_log
    t_last_log = t_now
    eta_sec = dt_ep * (EPOCHS - epoch - 1)
    eta_str = (f"{int(eta_sec//3600)}h{int((eta_sec%3600)//60):02d}m" if eta_sec >= 3600 else
               f"{int(eta_sec//60)}m{int(eta_sec%60):02d}s"           if eta_sec >= 60 else
               f"{int(eta_sec)}s")

    row = {
        "epoch": epoch, "time_total_sec": round(t_now - t0, 1),
        "time_interval_sec": round(dt_ep, 3),
        "loss_vol": ep_vol, "loss_bias": ep_bias, "loss_l2": ep_l2,
        "swaption_priced_frac": swp_priced, "path_finite_frac": path_finite, "anchor_rmse_bp": anchor_rmse_bp,
        "recon_rmse_bps": float(avg_rmse_bps),
        "nan_batches": nan_batches,
        "gnorm_lam0": gn_lam0, "gnorm_scale": gn_scale, "lr": lr_now,
        "lambda0_norm": lambda0_norm, "lambda0_l1": lambda0_l1,
        "sigma_scale_mean": scale_now,
        "sigma_s1": float(sigma_vec_now[0]), "sigma_s2": float(sigma_vec_now[1]),
        "sigma_s3": float(sigma_vec_now[2]), "sigma_s4": float(sigma_vec_now[3]),
        "lambda0_v1": float(lambda0_now[0]), "lambda0_v2": float(lambda0_now[1]),
        "lambda0_v3": float(lambda0_now[2]), "lambda0_v4": float(lambda0_now[3]),
        "fwd_bias_diag_bp": mean_bias_diag,
    }
    
    for c in ccy_order:
        row[f"rmse_bps_{c}"] = (
            float(rmse_per_ccy.get(c, float('nan')))
            if rmse_per_ccy is not None else float('nan')
        )
    pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)

    if (epoch // max(LOG_EVERY, 1)) % HEADER_EVERY == 0:
        print(
            f"\n{'ep':>5} {'vol':>10} {'bias':>9} {'l2':>8} {'swp%':>5} {'pth%':>5} "
            f"{'anchor':>8} {'recon':>7} {'|l0|':>7} {'sv_mean':>7} {'sv_min':>6} {'sv_max':>6} "
            f"{'bias_bp':>8} {'gn_l0':>7} {'gn_s':>7} {'nan':>4} {'lr':>8} {'t/ep':>6} {'ETA':>8}  diag"
        )
        print("-" * 180)

    if batch_diag and (epoch % DIAG_EVERY == 0 or epoch == EPOCHS - 1):
        diag_str = " | ".join(
            f"{d['exp']}x{d['ten']} mkt={d['mkt_bp']:.0f} mod={d['mod_bp']:.0f} "
            f"err={d['err_bp']:+.0f} bias={d['bias_bp']:+.0f}bp"
            for d in batch_diag[:3]
        )
    else:
        diag_str = ""

    sv_min = float(sigma_vec_now.min())
    sv_max = float(sigma_vec_now.max())
    print(
        f"{epoch:>5d} "
        f"{ep_vol:>10.4e} {ep_bias:>9.3e} {ep_l2:>8.4e} "
        f"{swp_priced*100:>4.0f}% {path_finite*100:>4.0f}% "
        f"{anchor_rmse_bp:>8.2f} "
        f"{avg_rmse_bps:>7.2f} "
        f"{lambda0_norm:>7.4f} "
        f"{scale_now:>7.4f} {sv_min:>6.4f} {sv_max:>6.4f} "
        f"{mean_bias_diag:>+8.1f} {gn_lam0:>7.2e} {gn_scale:>7.2e} "
        f"{nan_batches:>4d} {lr_now:>8.2e} "
        f"{dt_ep:>5.1f}s {eta_str:>8}  {diag_str}"
    )

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt = os.path.join(FIGURES_DIR, f"checkpoint_constant_mpr_ep{epoch+1}.pt")
        torch.save({
            "lm_state_dict":    lm.state_dict(),
            "lambda_0":         lm.lambda_0.detach().cpu(),
            "log_sigma_vec":    lm.log_sigma_vec.detach().cpu(),
            "sigma_vec":        lm.sigma_vec.detach().cpu(),
            "sigma_scale_mean": scale_now,
            "lambda0_norm":     lambda0_norm,
            "latent_dim":       LATENT_DIM,
            "epoch":            epoch + 1,
            "variant":          config.VARIANT,
        }, ckpt)
        print(f"  -> checkpoint ep{epoch+1}  |lambda_0|={lambda0_norm:.4f}  "
              f"lambda_0={lambda0_now.numpy().round(4)}  "
              f"sigma_vec={lm.sigma_vec.detach().cpu().numpy().round(4)}")

print("\nTraining done.")

# ── final checkpoint + plots ───────────────────────────────────────────────────

torch.save({
    "lm_state_dict":    lm.state_dict(),
    "lambda_0":         lm.lambda_0.detach().cpu(),
    "log_sigma_vec":    lm.log_sigma_vec.detach().cpu(),
    "sigma_vec":        lm.sigma_vec.detach().cpu(),
    "sigma_scale_mean": float(lm.sigma_vec.mean().detach()),
    "lambda0_norm":     float(lm.lambda_0.norm().detach()),
    "latent_dim":       LATENT_DIM,
    "epochs":           EPOCHS,
    "variant":          config.VARIANT,
}, os.path.join(FIGURES_DIR, f"checkpoint_constant_mpr_ep{EPOCHS}.pt"))

fig, axes = plt.subplots(5, 1, figsize=(9, 15), dpi=150)

axes[0].semilogy(hist["vol"],  lw=1.0, color="darkorange", label="Vol loss")
axes[0].semilogy(hist["bias"], lw=1.0, color="deeppink",   label="Bias loss")
axes[0].semilogy(hist["l2"],   lw=1.0, color="royalblue",  label="L2 lambda_0")
axes[0].set_title("Constant MPR: Loss Components"); axes[0].legend()
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log)"); axes[0].grid(True, alpha=0.3)

axes[1].plot([100*p for p in hist["swp_priced"]], lw=1.2, color="firebrick", label="swaption_priced%")
axes[1].plot([100*p for p in hist["path_finite"]], lw=1.0, color="navy",     label="path_finite%")
axes[1].set_ylim(-2, 102); axes[1].legend()
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("%"); axes[1].grid(True, alpha=0.3)

# Filter out NaN values for anchor RMSE plot
anchor_valid = [(i, v) for i, v in enumerate(hist["anchor_rmse"]) if not math.isnan(v)]
if anchor_valid:
    epochs_valid, anchor_vals = zip(*anchor_valid)
    axes[2].plot(epochs_valid, anchor_vals, lw=1.2, color="crimson", 
                 label=r"Anchor RMSE: $\|\widehat{S}^*(E(S_0)) - S_0^{\mathrm{mkt}}\|$", marker='.')
    axes[2].axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5 bp (good)')
    axes[2].axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20 bp (moderate)')
    axes[2].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50 bp (re-anchor needed)')
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("RMSE (bp)");
axes[2].set_title("Anchor RMSE: shifted-decoder reconstruction error at t=0"); axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(hist["sigma_scale"], lw=1.2, color="purple", label="mean(sigma_vec)")
axes[3].set_xlabel("Epoch"); axes[3].set_title("Diffusion Scale (mean of sigma_vec)"); axes[3].legend()
axes[3].grid(True, alpha=0.3)

axes[4].plot(hist["lambda0_norm"], lw=1.2, color="teal", label="||lambda_0||")
axes[4].set_xlabel("Epoch"); axes[4].set_title("Constant Drift Bias Norm"); axes[4].legend()
axes[4].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"constant_mpr_loss_dim{LATENT_DIM}_ep{EPOCHS}.png"), dpi=200)
plt.close(fig)

print("\n" + "=" * 90)
print("CONSTANT MPR CALIBRATION COMPLETE")
print("=" * 90)
print(f"Final vol loss  : {hist['vol'][-1]:.6e}")
print(f"Final bias loss : {hist['bias'][-1]:.6e}")
print(f"Final lambda_0  : {lm.lambda_0.detach().cpu().numpy().round(4)}  (||.||={hist['lambda0_norm'][-1]:.4f})")
print(f"Final sigma_vec : {lm.sigma_vec.detach().cpu().numpy().round(4)}  (mean={hist['sigma_scale'][-1]:.4f})")
print("=" * 90)
