# ==================== Regime-Conditioned MPR Pricing Calibration ====================
"""
Regime-Conditioned Market Price of Risk (MPR) pricing adjustment.

Extends the Constant MPR by making the drift correction depend on the
current rate regime, encoded by the initial latent state z0:

    K_price(z_t) = K_base(z_t) + L_base(z_t) @ (lambda_0 + A @ z0)

where:
  - lambda_0 in R^d  : constant drift bias (warm-started from Constant MPR)
  - A in R^{d x d}   : regime-sensitivity matrix (warm-started at zero)
  - z0 in R^d        : encoder(today's curve) — FIXED per observation, not evolving

Key property: z0 is computed once at t=0 and held constant during simulation.
Since the correction lambda_eff = lambda_0 + A @ z0 does NOT depend on the
evolving z_t, there is no position-dependent feedback loop.  The model is
therefore stable by construction, identical to Constant MPR at initialisation
(A=0), and can only improve by learning regime-specific risk premia.

Trainable: lambda_0 (d) + A (d*d) + log_sigma_vec (d) = 2d + d^2 = 24 params
           (for d=4: 4 + 16 + 4 = 24)

Output: Figures/TrainingResults/dim4_regime_mpr/ep{EPOCHS}/
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

PRETRAIN_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)
# Warm-start A from Constant MPR (lambda_0 + sigma_vec copied, A init = 0)
CMPR_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_constant_mpr", "ep1000", "checkpoint_constant_mpr_ep1000.pt"
)

# Loss weights — same as Constant MPR
LAMBDA_VOL  = 1.0
LAMBDA_BIAS = 0.5
LAMBDA_L2   = 1e-3   # L2 on lambda_0
LAMBDA_A    = 1e-3   # L2 on A matrix (keeps regime correction bounded)

LR = 2e-4            # lower than Constant MPR — already warm-started
LR_WARMUP_EPOCHS = 30

MIN_FINITE_PATHS_FRAC = 0.10
MIN_FINITE_PATHS_ABS  = 16
LOSS_SKIP_THRESH      = 1e4

USE        = "bbg"
CCY_FILTER = "EUR"

FIGURES_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_regime_mpr", f"ep{EPOCHS}"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── module ─────────────────────────────────────────────────────────────────────

class RegimeMPRAdjustment(nn.Module):
    """
    Regime-conditioned Market Price of Risk pricing adjustment.

    K_price(z_t) = K_base(z_t) + L_base(z_t) @ lambda_eff(z0)

    where lambda_eff(z0) = lambda_0 + A @ z0.

    z0 = encoder(today's curve) is FIXED per observation.  It is passed
    as an argument to drift(), not as a time-varying state.

    This breaks the feedback loop: the correction is constant *during* each
    simulation path (z0 does not change as z_t evolves), so it cannot
    amplify latent-space deviations.  At initialisation A=0, the model
    reduces exactly to Constant MPR.

    Trainable parameters:
      lambda_0      [d]    constant bias (warm-started from Constant MPR)
      A             [d, d] regime-sensitivity matrix (init = 0)
      log_sigma_vec [d]    per-factor diffusion scales (warm-started)
    Total: 2d + d^2 = 24 (for d=4)
    """

    def __init__(self, kp_module, h_module, latent_dim):
        super().__init__()
        self.kp         = kp_module
        self.h          = h_module
        self.latent_dim = latent_dim

        self.lambda_0      = nn.Parameter(torch.zeros(latent_dim))
        self.A             = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.log_sigma_vec = nn.Parameter(torch.full((latent_dim,), -1.8))

    def lambda_eff(self, z0):
        """Effective lambda for a batch of initial states.
        z0: [batch, d]  ->  lambda_eff: [batch, d]
        """
        return self.lambda_0.unsqueeze(0) + (z0 @ self.A.T)  # [batch, d]

    def drift(self, z_t, z0_broadcast):
        """Full risk-neutral drift.
        z_t:          [batch, d]  current latent state (may differ from z0)
        z0_broadcast: [batch, d]  initial encoding (same for all paths of one date)
        """
        k_base       = self.kp(z_t)
        sigmas, rhos = self.h(z_t)
        L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam_eff      = self.lambda_eff(z0_broadcast)               # [batch, d]
        return k_base + torch.einsum('bij,bj->bi', L, lam_eff)

    def forward(self, z_t):
        """Fallback: constant MPR with A=0 (used if z0 not available)."""
        k_base       = self.kp(z_t)
        sigmas, rhos = self.h(z_t)
        L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam0 = self.lambda_0.unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam0)

    @property
    def sigma_vec(self):
        return self.log_sigma_vec.exp()


class RegimeDriftWrapper(nn.Module):
    """Wraps RegimeMPRAdjustment with a fixed z0, matching the k_override interface."""

    def __init__(self, regime_mpr, z0):
        """
        regime_mpr: RegimeMPRAdjustment
        z0: [1, d] initial latent state for this observation
        """
        super().__init__()
        self.regime_mpr = regime_mpr
        # store as plain tensor (not parameter) so it doesn't appear in state_dict
        self.z0 = z0  # [1, d]

    def forward(self, z_t):
        """z_t: [n_paths, d]  ->  drift: [n_paths, d]"""
        z0_broadcast = self.z0.expand(z_t.shape[0], -1)
        return self.regime_mpr.drift(z_t, z0_broadcast)

    @property
    def sigma_vec(self):
        return self.regime_mpr.sigma_vec


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
    Uses RegimeDriftWrapper so z0 is passed to the drift at each step.
    """
    if len(df_vol) == 0 or n_swaptions == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, [], 0, 0, 0.0

    sample = df_vol.sample(n=min(n_swaptions, len(df_vol)))

    total_vol  = torch.zeros(1, device=device, dtype=dtype)
    total_bias = torch.zeros(1, device=device, dtype=dtype)
    n_valid     = 0
    n_attempted = 0
    diagnostics = []
    path_fracs  = []
    min_paths   = max(MIN_FINITE_PATHS_ABS, int(n_paths * MIN_FINITE_PATHS_FRAC))

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

        with torch.no_grad():
            z0 = model.encoder(xb)           # [1, d] — fixed per observation
            _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
            P0 = aux0["P_full"][0]

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

        # Wrap drift so z0 is visible inside simulate_to_expiry_differentiable
        drift_wrapper = RegimeDriftWrapper(lm, z0)

        try:
            z_T, D_T = simulate_to_expiry_differentiable(
                model, z0,
                n_steps=n_steps, dt=dt_eff,
                n_paths=2 * half,
                eps=eps_z,
                k_override=drift_wrapper,
                sigma_scale=lm.sigma_vec,
                antithetic=True,
                freeze_H=True,
            )

            z_ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
            if int(z_ok.sum()) < min_paths:
                continue

            with torch.no_grad():
                _, aux_T = model.decode_from_z(z_T, tau=None, return_aux=True)
                p_ok = torch.isfinite(aux_T["P_full"]).all(1)

            mask = z_ok & p_ok
            if int(mask.sum()) < min_paths:
                continue

            path_fracs.append(float(mask.float().mean()))

            z_keep = z_T[mask]
            _, aux_keep = model.decode_from_z(z_keep, tau=None, return_aux=True)
            F_T, A_T = swap_rate_torch(aux_keep["P_full"], tenor=tenor)
            D_keep   = D_T[mask]

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
                lam_eff_val = lm.lambda_eff(z0).squeeze(0).detach().cpu().numpy()
                diagnostics.append({
                    "date":     date.date(),
                    "exp":      expiry,
                    "ten":      tenor,
                    "mkt_bp":   round(sigma_mkt_bp, 1),
                    "mod_bp":   round(float(sigma_str_bp.detach()), 1),
                    "err_bp":   round(float(sigma_str_bp.detach()) - sigma_mkt_bp, 1),
                    "bias_bp":  round(float(fwd_bias_bp.detach()), 1),
                    "F0":       round(F_0 * 1e4, 1),
                    "lam_eff":  lam_eff_val.round(4).tolist(),
                })

        except Exception:
            continue

    mean_pfrac = float(np.mean(path_fracs)) if path_fracs else 0.0
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    if n_valid > 0:
        return (total_vol / n_valid, total_bias / n_valid,
                diagnostics, n_attempted, n_valid, mean_pfrac)
    return zero, zero, diagnostics, n_attempted, 0, mean_pfrac


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

lm = RegimeMPRAdjustment(model.K, model.H, LATENT_DIM).to(device)

# ── warm-start from Constant MPR ───────────────────────────────────────────────
if os.path.exists(CMPR_CKPT):
    raw_cmpr = torch.load(CMPR_CKPT, map_location=device, weights_only=False)
    cmpr_state = raw_cmpr.get("lm_state_dict", raw_cmpr)
    # Copy lambda_0 and log_sigma_vec; leave A at zero
    with torch.no_grad():
        if "lambda_0" in cmpr_state:
            lm.lambda_0.copy_(cmpr_state["lambda_0"].to(device))
        if "log_sigma_vec" in cmpr_state:
            lm.log_sigma_vec.copy_(cmpr_state["log_sigma_vec"].to(device))
    print(f"Warm-started lambda_0 and log_sigma_vec from: {CMPR_CKPT}")
    print(f"  lambda_0  = {lm.lambda_0.detach().cpu().numpy().round(4)}")
    print(f"  sigma_vec = {lm.sigma_vec.detach().cpu().numpy().round(4)}")
    print(f"  A         = (zeros — will be learnt)")
else:
    print(f"WARNING: Constant MPR checkpoint not found at {CMPR_CKPT}")
    print("  Starting from scratch (lambda_0=0, A=0, sigma_vec~0.165)")

n_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
print(f"Trainable params: {n_params}  "
      f"(lambda_0 {LATENT_DIM} + A {LATENT_DIM}x{LATENT_DIM} + log_sigma_vec {LATENT_DIM})")

model.train()

# Three param groups with different learning rates
LR_SCALE_MULT = 10.0
LR_A_MULT     = 0.5   # A learns more slowly (starts at 0, don't rush)
optim = torch.optim.Adam([
    {'params': [lm.lambda_0],        'lr': LR,                  'name': 'lambda_0'},
    {'params': [lm.A],               'lr': LR * LR_A_MULT,      'name': 'A'},
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
csv_path  = os.path.join(FIGURES_DIR, f"train_regime_mpr_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")
csv_cols  = (
    ["epoch", "time_total_sec", "time_interval_sec",
     "loss_vol", "loss_bias", "loss_l2", "loss_A",
     "swaption_priced_frac", "path_finite_frac",
     "recon_rmse_bps", "nan_batches",
     "gnorm_lam0", "gnorm_A", "gnorm_scale", "lr",
     "lambda0_norm", "A_norm", "A_frob",
     "sigma_scale_mean", "sigma_s1", "sigma_s2", "sigma_s3", "sigma_s4",
     "lambda0_v1", "lambda0_v2", "lambda0_v3", "lambda0_v4",
     "fwd_bias_diag_bp"]
    + [f"rmse_bps_{c}" for c in ccy_order]
)
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "version":               "regime_mpr",
    "description":           "K_price(z_t) = K_base(z_t) + L(z_t) @ (lambda_0 + A @ z0)",
    "seed":                  SEED,
    "latent_dim":            LATENT_DIM,
    "variant":               config.VARIANT,
    "epochs":                EPOCHS,
    "lr":                    LR,
    "lambda_vol":            LAMBDA_VOL,
    "lambda_bias":           LAMBDA_BIAS,
    "lambda_l2":             LAMBDA_L2,
    "lambda_A":              LAMBDA_A,
    "n_swaptions_per_batch": N_SWAPTIONS_PER_BATCH,
    "n_paths_pricing":       N_PATHS_PRICING,
    "dt_pricing":            DT_PRICING,
    "n_trainable_params":    n_params,
    "warm_start":            CMPR_CKPT,
}
with open(os.path.join(FIGURES_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

# ── training loop ──────────────────────────────────────────────────────────────

hist = {k: [] for k in [
    "vol", "bias", "l2", "A_reg",
    "swp_priced", "path_finite",
    "sigma_scale", "lambda0_norm", "A_frob",
]}

t0         = time.perf_counter()
t_last_log = t0

print("\n" + "=" * 100)
print("REGIME-CONDITIONED MPR PRICING CALIBRATION")
print("  K_price(z_t) = K_base(z_t) + L(z_t) @ (lambda_0 + A @ z0)")
print("  z0 = encoder(today's curve) — FIXED per observation, no feedback loop")
print(f"  Trainable: lambda_0 ({LATENT_DIM}) + A ({LATENT_DIM}x{LATENT_DIM}) "
      f"+ log_sigma_vec ({LATENT_DIM}) = {n_params} total")
print("=" * 100 + "\n")

for epoch in range(EPOCHS):
    model.train()
    lm.train()
    running_vol  = 0.0
    running_bias = 0.0
    running_l2   = 0.0
    running_Areg = 0.0
    n_batches     = 0
    nan_batches   = 0
    batch_diag    = []
    ep_attempted  = 0
    ep_priced     = 0
    ep_pfracs     = []

    for step in range(N_STEPS_PER_EPOCH):
        print(f"\r  ep {epoch}  step {step+1}/{N_STEPS_PER_EPOCH} ...", end="", flush=True)
        optim.zero_grad(set_to_none=True)

        loss_vol, loss_bias_raw, diag, n_att, n_pri, p_frac = compute_pricing_loss(
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

        loss_l2    = LAMBDA_L2 * lm.lambda_0.pow(2).sum()
        loss_A_reg = LAMBDA_A  * lm.A.pow(2).sum()
        loss_bias  = LAMBDA_BIAS * loss_bias_raw
        loss_total = LAMBDA_VOL * loss_vol + loss_bias + loss_l2 + loss_A_reg

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

        torch.nn.utils.clip_grad_norm_([lm.lambda_0],      max_norm=2.0)
        torch.nn.utils.clip_grad_norm_([lm.A],             max_norm=2.0)
        torch.nn.utils.clip_grad_norm_([lm.log_sigma_vec], max_norm=2.0)
        optim.step()

        running_vol  += float(loss_vol.detach().cpu())
        running_bias += float(loss_bias_raw.detach().cpu())
        running_l2   += float(loss_l2.detach().cpu())
        running_Areg += float(loss_A_reg.detach().cpu())
        n_batches    += 1

    print("\r" + " " * 40 + "\r", end="", flush=True)
    scheduler.step()

    ep_vol   = running_vol  / max(n_batches, 1)
    ep_bias  = running_bias / max(n_batches, 1)
    ep_l2    = running_l2   / max(n_batches, 1)
    ep_Areg  = running_Areg / max(n_batches, 1)
    swp_priced  = ep_priced  / max(ep_attempted, 1)
    path_finite = float(np.mean(ep_pfracs)) if ep_pfracs else 0.0

    with torch.no_grad():
        sigma_vec_now = lm.sigma_vec.detach().cpu()
        scale_now     = float(sigma_vec_now.mean())
        lambda0_now   = lm.lambda_0.detach().cpu()
        lambda0_norm  = float(lambda0_now.norm())
        A_now         = lm.A.detach().cpu()
        A_frob        = float(A_now.norm(p='fro'))

    for k, v in [("vol", ep_vol), ("bias", ep_bias), ("l2", ep_l2), ("A_reg", ep_Areg),
                 ("swp_priced", swp_priced), ("path_finite", path_finite),
                 ("sigma_scale", scale_now), ("lambda0_norm", lambda0_norm),
                 ("A_frob", A_frob)]:
        hist[k].append(v)

    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    if do_eval:
        rmse_per_ccy, avg_rmse_bps, _ = eval_rmse_bps(model, X_tensor, meta)
        gn_lam0  = grad_norm([lm.lambda_0])
        gn_A     = grad_norm([lm.A])
        gn_scale = grad_norm([lm.log_sigma_vec])
    else:
        rmse_per_ccy, avg_rmse_bps = None, float('nan')
        gn_lam0 = gn_A = gn_scale = 0.0

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
        "loss_vol": ep_vol, "loss_bias": ep_bias, "loss_l2": ep_l2, "loss_A": ep_Areg,
        "swaption_priced_frac": swp_priced, "path_finite_frac": path_finite,
        "recon_rmse_bps": float(avg_rmse_bps), "nan_batches": nan_batches,
        "gnorm_lam0": gn_lam0, "gnorm_A": gn_A, "gnorm_scale": gn_scale, "lr": lr_now,
        "lambda0_norm": lambda0_norm, "A_norm": A_frob, "A_frob": A_frob,
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
            f"\n{'ep':>5} {'vol':>10} {'bias':>9} {'l2':>8} {'|A|F':>7} "
            f"{'swp%':>5} {'pth%':>5} {'recon':>7} "
            f"{'|l0|':>6} {'|A|F':>6} {'sv_mean':>7} "
            f"{'bias_bp':>8} {'gn_l0':>7} {'gn_A':>7} {'lr':>8} {'t/ep':>6} {'ETA':>8}  diag"
        )
        print("-" * 185)

    if batch_diag and (epoch % DIAG_EVERY == 0 or epoch == EPOCHS - 1):
        diag_str = " | ".join(
            f"{d['exp']}x{d['ten']} mkt={d['mkt_bp']:.0f} mod={d['mod_bp']:.0f} "
            f"err={d['err_bp']:+.0f} bias={d['bias_bp']:+.0f}bp"
            for d in batch_diag[:3]
        )
    else:
        diag_str = ""

    print(
        f"{epoch:>5d} "
        f"{ep_vol:>10.4e} {ep_bias:>9.3e} {ep_l2:>8.4e} {ep_Areg:>7.4e} "
        f"{swp_priced*100:>4.0f}% {path_finite*100:>4.0f}% "
        f"{avg_rmse_bps:>7.2f} "
        f"{lambda0_norm:>6.4f} {A_frob:>6.4f} {scale_now:>7.4f} "
        f"{mean_bias_diag:>+8.1f} {gn_lam0:>7.2e} {gn_A:>7.2e} {lr_now:>8.2e} "
        f"{dt_ep:>5.1f}s {eta_str:>8}  {diag_str}"
    )

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt = os.path.join(FIGURES_DIR, f"checkpoint_regime_mpr_ep{epoch+1}.pt")
        torch.save({
            "lm_state_dict":    lm.state_dict(),
            "lambda_0":         lm.lambda_0.detach().cpu(),
            "A":                lm.A.detach().cpu(),
            "log_sigma_vec":    lm.log_sigma_vec.detach().cpu(),
            "sigma_vec":        lm.sigma_vec.detach().cpu(),
            "A_frob":           A_frob,
            "lambda0_norm":     lambda0_norm,
            "latent_dim":       LATENT_DIM,
            "epoch":            epoch + 1,
            "variant":          config.VARIANT,
        }, ckpt)
        print(f"  -> checkpoint ep{epoch+1}  |lambda_0|={lambda0_norm:.4f}  "
              f"|A|_F={A_frob:.4f}  "
              f"sigma_vec={lm.sigma_vec.detach().cpu().numpy().round(4)}")

print("\nTraining done.")

# ── final checkpoint + plots ───────────────────────────────────────────────────

torch.save({
    "lm_state_dict":    lm.state_dict(),
    "lambda_0":         lm.lambda_0.detach().cpu(),
    "A":                lm.A.detach().cpu(),
    "log_sigma_vec":    lm.log_sigma_vec.detach().cpu(),
    "sigma_vec":        lm.sigma_vec.detach().cpu(),
    "A_frob":           A_frob,
    "lambda0_norm":     lambda0_norm,
    "latent_dim":       LATENT_DIM,
    "epochs":           EPOCHS,
    "variant":          config.VARIANT,
}, os.path.join(FIGURES_DIR, f"checkpoint_regime_mpr_ep{EPOCHS}.pt"))

fig, axes = plt.subplots(5, 1, figsize=(9, 16), dpi=150)

axes[0].semilogy(hist["vol"],   lw=1.0, color="darkorange", label="Vol loss")
axes[0].semilogy(hist["bias"],  lw=1.0, color="deeppink",   label="Bias loss")
axes[0].semilogy(hist["l2"],    lw=1.0, color="royalblue",  label="L2 lambda_0")
axes[0].semilogy(hist["A_reg"], lw=1.0, color="seagreen",   label="L2 A")
axes[0].set_title("Regime MPR: Loss Components"); axes[0].legend()
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log)"); axes[0].grid(True, alpha=0.3)

axes[1].plot([100*p for p in hist["swp_priced"]], lw=1.2, color="firebrick", label="swaption_priced%")
axes[1].plot([100*p for p in hist["path_finite"]], lw=1.0, color="navy",     label="path_finite%")
axes[1].set_ylim(-2, 102); axes[1].legend()
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("%"); axes[1].grid(True, alpha=0.3)

axes[2].plot(hist["sigma_scale"], lw=1.2, color="purple", label="mean(sigma_vec)")
axes[2].set_xlabel("Epoch"); axes[2].set_title("Diffusion Scale (mean of sigma_vec)"); axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(hist["lambda0_norm"], lw=1.2, color="teal",   label="||lambda_0||")
axes[3].set_xlabel("Epoch"); axes[3].set_title("||lambda_0|| over training"); axes[3].legend()
axes[3].grid(True, alpha=0.3)

axes[4].plot(hist["A_frob"], lw=1.2, color="crimson", label="||A||_F (Frobenius)")
axes[4].set_xlabel("Epoch"); axes[4].set_title("Regime matrix ||A||_F over training"); axes[4].legend()
axes[4].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"regime_mpr_loss_dim{LATENT_DIM}_ep{EPOCHS}.png"), dpi=200)
plt.close(fig)

print("\n" + "=" * 90)
print("REGIME MPR CALIBRATION COMPLETE")
print("=" * 90)
print(f"Final vol loss  : {hist['vol'][-1]:.6e}")
print(f"Final bias loss : {hist['bias'][-1]:.6e}")
print(f"Final lambda_0  : {lm.lambda_0.detach().cpu().numpy().round(4)}")
print(f"Final ||A||_F   : {A_frob:.4f}")
print(f"Final A matrix  :\n{A_now.numpy().round(4)}")
print(f"Final sigma_vec : {lm.sigma_vec.detach().cpu().numpy().round(4)}")
print("=" * 90)
