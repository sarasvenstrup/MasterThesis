# ==================== Lambda MPR v2 Training ====================
"""
Clean Lambda MPR training with three key fixes over v1:

  1. CORRECT STRIKE: Use forward swap rate F_0 = (P(0,exp) - P(0,exp+ten)) / A_fwd
     instead of the spot swap rate. This puts options ATM from the start.

  2. VARIANCE-BASED VOL LOSS: Match std^A(F_T)/sqrt(T) to market vol directly.
     Avoids gradient vanishing when paths are OTM (which killed v1 training).

  3. ANISOTROPIC DIFFUSION SCALE: Per-dimension log_sigma_vec (d-vector) replaces
     scalar log_sigma_scale.  Each z-dimension has its own vol scale, so the model
     can fit the *shape* of the swaption vol surface, not just the overall level.

Two decoupled losses:
  loss_vol   : (sigma_mod - sigma_mkt)^2  -- trains sigma_scale
  loss_drift : (E^A[F_T] - F_0)^2        -- trains Lambda

Lambda initialised to zero; base Q-dynamics (K) active from epoch 0.

Trainable parameters (17 total):
  Lambda          4x4  (16)   K^{Q^A} = K(z) + L(z) @ Lambda @ z
  log_sigma_vec    (d)         per-dim diffusion scaling (init=-1.8 -> scale≈0.165 per dim)

Output: Figures/TrainingResults/dim4_lambda_v2/ep{EPOCHS}/
"""

import os, sys

# ── CPU threading: set BEFORE numpy/torch import so OpenMP picks them up ───────
_N_TORCH   = 4   # intra-op parallelism (matmul, etc.)
_N_INTEROP = 2   # inter-op parallelism (parallel ops graph)
os.environ.setdefault("OMP_NUM_THREADS",   str(_N_TORCH))
os.environ.setdefault("MKL_NUM_THREADS",   str(_N_TORCH))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_N_TORCH))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_N_TORCH))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_N_TORCH))

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

# ── Pin process to a fixed core set so OS doesn't scatter threads ───────────────
try:
    import psutil
    _proc = psutil.Process()
    _all_cores = list(range(os.cpu_count()))
    _pin_cores = _all_cores[:_N_TORCH * 2]   # reserve 8 cores for the process
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
LATENT_DIM  = 4
EPOCHS      = 1000

EVAL_EVERY   = 1
LOG_EVERY    = 1
DIAG_EVERY   = 10
HEADER_EVERY = 20
SAVE_EVERY   = 200

N_STEPS_PER_EPOCH     = 4
N_SWAPTIONS_PER_BATCH = 8
N_PATHS_PRICING       = 512    # antithetic 256+256  (variance ~ O(1/N): 2x paths = sqrt(2) better std)
DT_PRICING            = 1 / 6

PRETRAIN_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)

# Loss weights
LAMBDA_VOL    = 1.0    # V_pay-based ATM vol MSE
LAMBDA_DRIFT  = 0.5    # mild drift regulariser: E^A[F_T] ≈ F_0
LAMBDA_EIG    = 0.0    # disabled: eig floor not meaningful for K^Q = -L@Lambda@z
LAMBDA_L2     = 1e-4   # L2 on Lambda entries
EIG_MIN       = 0.05

LR = 5e-4
LR_WARMUP_EPOCHS = 50

MIN_FINITE_PATHS_FRAC = 0.10
MIN_FINITE_PATHS_ABS  = 16
LOSS_SKIP_THRESH      = 1e4

USE        = "bbg"
CCY_FILTER = "EUR"

FIGURES_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_lambda_v2", f"ep{EPOCHS}"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── module ─────────────────────────────────────────────────────────────────────

class LambdaMPR_v2(nn.Module):
    """
    Pricing-measure drift:  K^{Q^A}(z) = K(z) + L(z) @ Lambda @ z

    Per Poulsen et al. (2025), z is defined under the risk-neutral measure Q:
    the base model's K network is already K^Q, calibrated via the no-arbitrage
    ODE system so that bond prices are Q-martingales.

    Lambda parameterises the Q -> Q^A (annuity measure) market price of risk:
        lambda(z) = Lambda @ z   (linear Girsanov kernel)

    so the full drift under Q^A is K^Q(z) + L(z) @ Lambda @ z.

    At Lambda = 0 the simulation runs under the base Q-dynamics; the forward
    bias is then only the convexity adjustment (~50-200 bp), not the full
    risk-premium offset seen when K is excluded entirely.

    Trainable: Lambda (d×d) + log_sigma_vec (d-vector, anisotropic diffusion scale).
    K and H are frozen (from pre-trained base model).
    """

    def __init__(self, kp_module, h_module, latent_dim):
        super().__init__()
        self.kp         = kp_module   # base Q-drift  K(z) — frozen
        self.h          = h_module    # diffusion H(z)      — frozen
        self.latent_dim = latent_dim

        self.Lambda      = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        # log_sigma_vec: per-dimension diffusion scale (replaces scalar log_sigma_scale).
        # Each z-dimension gets its own scale, letting the model fit the *shape*
        # of the swaption vol surface rather than just the overall level.
        # Init exp(-1.8) ~ 0.165 per dim: base Q-diffusion at scale=1 is ~6x market vol.
        self.log_sigma_vec = nn.Parameter(torch.full((latent_dim,), -1.8))

    def forward(self, z):
        """K^{Q^A}(z) = K(z) + L(z) @ Lambda @ z"""
        with torch.no_grad():
            k_base       = self.kp(z)                                         # (batch, d)
            sigmas, rhos = self.h(z)
            L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)   # (batch, d, d)
        lam = torch.matmul(self.Lambda, z.unsqueeze(-1)).squeeze(-1)          # (batch, d)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    @property
    def sigma_vec(self):
        """Per-dimension diffusion scales: shape (d,)"""
        return self.log_sigma_vec.exp()


# ── helpers ────────────────────────────────────────────────────────────────────

def get_K_matrix(k_module, dim, device, dtype):
    z0   = torch.zeros(1, dim, device=device, dtype=dtype)
    bias = k_module(z0)
    eye  = torch.eye(dim, device=device, dtype=dtype)
    cols = []
    for i in range(dim):
        cols.append((k_module(eye[i:i+1]) - bias).reshape(-1))
    return torch.stack(cols, dim=1)


def eigenvalue_floor_loss(M, eig_min=EIG_MIN):
    eigs    = torch.linalg.eigvals(M)
    deficit = torch.relu(eigs.real + eig_min)
    return deficit.pow(2).mean()


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


# forward_swap_rate_torch imported from pricing.py — canonical implementation


# ── pricing loss ───────────────────────────────────────────────────────────────

def compute_pricing_loss_v2(
    model, lm,
    X_batch, meta_batch,
    df_vol, date_to_idx,
    n_swaptions, n_paths, dt,
    device, dtype,
    return_diagnostics=False,
):
    """
    V_pay-based vol loss with correct forward-rate ATM strike.

    sigma_mod = V_pay * sqrt(2*pi) / (A_0 * sqrt(expiry)) * 1e4   [bp/yr]
    loss_vol  = ((sigma_mod - sigma_mkt) / 100)^2

    This is the Bachelier ATM vol implied directly from the payer price.
    It naturally penalises both wrong vol level AND any forward bias
    (a drifted distribution inflates V_pay), so Lambda and sigma_scale
    are trained jointly on the quantity we actually care about.

    A mild drift regulariser is kept to prevent explosive forward bias:
    loss_drift = ((E^A[F_T] - F_0) * 1e4 / 100)^2   (weight LAMBDA_DRIFT)
    """
    if len(df_vol) == 0 or n_swaptions == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, [], 0, 0, 0.0

    sample = df_vol.sample(n=min(n_swaptions, len(df_vol)))

    total_vol   = torch.zeros(1, device=device, dtype=dtype)
    total_drift = torch.zeros(1, device=device, dtype=dtype)
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
            z0 = model.encoder(xb)                                    # (1, d)
            _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
            P0 = aux0["P_full"][0]                                    # (tau_max+1,)

        # ── CORRECT ATM FORWARD RATE ──────────────────────────────────────
        max_idx = P0.shape[0] - 1
        if expiry + tenor > max_idx:
            continue
        F_0, A_0 = forward_swap_rate_torch(P0, expiry, tenor)
        if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 1e-6):
            continue
        # ─────────────────────────────────────────────────────────────────

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

            # Path validity
            z_ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
            if int(z_ok.sum()) < min_paths:
                continue

            with torch.no_grad():
                _, aux_T = model.decode_from_z(
                    z_T, tau=None, return_aux=True
                )
                p_ok = torch.isfinite(aux_T["P_full"]).all(1)

            mask = z_ok & p_ok
            if int(mask.sum()) < min_paths:
                continue

            path_fracs.append(float(mask.float().mean()))

            # F_T from simulated paths (spot swap rate at T — correct payoff variable)
            z_keep = z_T[mask]
            _, aux_keep = model.decode_from_z(
                z_keep, tau=None, return_aux=True
            )
            F_T, A_T = swap_rate_torch(aux_keep["P_full"], tenor=tenor)
            D_keep   = D_T[mask]

            fa_ok = (torch.isfinite(F_T) & torch.isfinite(A_T)
                     & (F_T > -0.5) & (F_T < 0.5)
                     & (A_T > 1e-6) & (A_T < 50.0))
            if int(fa_ok.sum()) < min_paths:
                continue

            F_T, A_T, D_keep = F_T[fa_ok], A_T[fa_ok], D_keep[fa_ok]

            # ── V_PAY-BASED VOL (Bachelier ATM implied vol) ───────────────
            V_pay = (D_keep * A_T * torch.relu(F_T - F_0)).mean()
            if not torch.isfinite(V_pay) or float(V_pay.detach()) < 0:
                continue

            sqrt_2pi     = math.sqrt(2 * math.pi)
            sigma_mod_bp = V_pay * sqrt_2pi / (A_0 * math.sqrt(expiry)) * 1e4
            # ─────────────────────────────────────────────────────────────

            loss_vol_ij = ((sigma_mod_bp - sigma_mkt_bp) / 100.0).pow(2)
            if not torch.isfinite(loss_vol_ij) or float(loss_vol_ij.detach()) > LOSS_SKIP_THRESH:
                continue

            # ── DRIFT REGULARISER: mild penalty on E^A[F_T] ≠ F_0 ───────
            w         = D_keep * A_T                         # D_keep already detached from sim; A_T grad flows to Lambda
            w_norm    = w / w.sum().clamp(min=1e-10)
            F_bar     = (w_norm * F_T).sum()
            fwd_bias_bp = (F_bar - F_0) * 1e4
            loss_drift_ij = (fwd_bias_bp / 100.0).pow(2)
            if not torch.isfinite(loss_drift_ij):
                continue
            # ─────────────────────────────────────────────────────────────

            total_vol   = total_vol   + loss_vol_ij
            total_drift = total_drift + loss_drift_ij
            n_valid    += 1

            if return_diagnostics:
                # ── Q-martingale checks (bond pricing consistency) ────────
                # Under Q: E[D_T] = P(0,T)  and  E[D_T * A_T] = A_0^fwd
                # These verify that the simulation is consistent with the
                # no-arbitrage bond price conditions from the base model.
                with torch.no_grad():
                    P0_T      = float(P0[int(round(expiry))].item())    # P(0, T)
                    E_DT      = float(D_keep.mean().item())              # E[D_T]
                    E_DT_AT   = float((D_keep * A_T).mean().item())      # E[D_T * A_T]
                    dt_err_bp = (E_DT - P0_T) * 1e4                     # should be ~0
                    ann_err   = (E_DT_AT - A_0) / A_0                   # should be ~0 (relative)

                diagnostics.append({
                    "date":     date.date(),
                    "exp":      expiry,
                    "ten":      tenor,
                    "mkt_bp":   round(sigma_mkt_bp, 1),
                    "mod_bp":   round(float(sigma_mod_bp.detach()), 1),
                    "err_bp":   round(float(sigma_mod_bp.detach()) - sigma_mkt_bp, 1),
                    "bias_bp":  round(float(fwd_bias_bp.detach()), 1),
                    "F0":       round(F_0 * 1e4, 1),
                    "Fbar":     round(float(F_bar.detach()) * 1e4, 1),
                    "scale":    [round(v, 4) for v in lm.sigma_vec.detach().cpu().tolist()],
                    # Q-martingale checks
                    "E_DT":     round(E_DT, 5),
                    "P0T":      round(P0_T, 5),
                    "DT_err_bp":round(dt_err_bp, 1),   # E[D_T] - P(0,T) in bp; target 0
                    "ann_err":  round(ann_err * 100, 2), # (E[D_T*A_T]-A_0)/A_0 in %; target 0
                })

        except Exception:
            continue

    mean_pfrac = float(np.mean(path_fracs)) if path_fracs else 0.0
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    if n_valid > 0:
        return (total_vol / n_valid, total_drift / n_valid,
                diagnostics, n_attempted, n_valid, mean_pfrac)
    return zero, zero, diagnostics, n_attempted, 0, mean_pfrac


# ── Lambda initialisation ──────────────────────────────────────────────────────

def init_lambda_zero(latent_dim, device, dtype):
    """
    Lambda = 0  →  K^{Q^A} = K(z)  →  pure Q-dynamics at epoch 0.
    Bond martingale conditions E[D_T]=P(0,T) and E[D_T*A_T]=A_0 should hold
    approximately. Training adjusts Lambda to correct the Q -> Q^A tilt needed
    for swaption vol calibration.
    """
    return torch.zeros(latent_dim, latent_dim, device=device, dtype=dtype)


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

lm = LambdaMPR_v2(model.K, model.H, LATENT_DIM).to(device)

# Lambda = 0: start at base Q-dynamics (K already calibrated under Q by ODE system)
print("Lambda initialised to zero (base Q-dynamics active from epoch 0).")
print(f"  ||Lambda_init||_F = {lm.Lambda.norm():.4f}")
sv_init = lm.sigma_vec.detach().cpu().numpy()
print(f"  sigma_vec init    = {sv_init.round(4)}  (mean={sv_init.mean():.4f})")

n_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
print(f"LambdaMPR_v2: {n_params} trainable params  (Lambda 16 + log_sigma_vec {LATENT_DIM})")

model.train()

# Separate param groups: sigma_vec gets 20x higher LR so its gradient (from vol loss)
# is not drowned out by Lambda's gradient (from drift loss, which is ~10x larger).
# Separate clip_grad_norm_ calls below prevent joint clipping from silencing sigma_vec.
LR_SCALE_MULT = 20.0
optim = torch.optim.Adam([
    {'params': [lm.Lambda],         'lr': LR,                  'name': 'Lambda'},
    {'params': [lm.log_sigma_vec],  'lr': LR * LR_SCALE_MULT,  'name': 'sigma_vec'},
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
csv_path  = os.path.join(FIGURES_DIR, f"train_lambda_v2_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")
csv_cols  = (
    ["epoch", "time_total_sec", "time_interval_sec",
     "loss_vol", "loss_drift", "loss_eig", "loss_l2",
     "swaption_priced_frac", "path_finite_frac",
     "recon_rmse_bps", "nan_batches",
     "gnorm", "gnorm_scale", "lr",
     "lambda_min_KQ", "Lambda_norm_fro",
     "sigma_scale_mean", "sigma_s1", "sigma_s2", "sigma_s3", "sigma_s4",
     "fwd_bias_diag_bp"]
    + [f"rmse_bps_{c}" for c in ccy_order]
)
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "version":              "lambda_v2",
    "fixes":                ["correct_fwd_strike", "variance_vol_loss", "trainable_sigma_scale",
                             "lambda_init_cancel_kp"],
    "seed":                 SEED,
    "latent_dim":           LATENT_DIM,
    "variant":              config.VARIANT,
    "epochs":               EPOCHS,
    "lr":                   LR,
    "lambda_vol":           LAMBDA_VOL,
    "lambda_drift":         LAMBDA_DRIFT,
    "lambda_eig":           LAMBDA_EIG,
    "lambda_l2":            LAMBDA_L2,
    "eig_min":              EIG_MIN,
    "n_swaptions_per_batch": N_SWAPTIONS_PER_BATCH,
    "n_paths_pricing":      N_PATHS_PRICING,
    "dt_pricing":           DT_PRICING,
    "n_trainable_params":   n_params,
}
with open(os.path.join(FIGURES_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

# ── training loop ──────────────────────────────────────────────────────────────

hist = {k: [] for k in [
    "vol", "drift", "eig", "l2",
    "swp_priced", "path_finite",
    "lambda_min", "lambda_norm", "sigma_scale",
]}

t0         = time.perf_counter()
t_last_log = t0

print("\n" + "=" * 100)
print("LAMBDA MPR v2 TRAINING")
print("  1. Correct ATM strike: forward swap rate F_0 = (P(0,exp)-P(0,exp+ten)) / A_fwd")
print("  2. V_pay-based vol loss: sigma_mod = V_pay*sqrt(2pi)/(A_0*sqrt(T))*1e4")
print("  3. Anisotropic sigma_vec (4-vector, init≈0.165/dim): fits vol *shape*, not just level")
print("  4. K^{Q^A}(z) = K(z) + L(z)@Lambda@z  (K is Q-drift per Poulsen et al. 2025)")
print("  5. Lambda init = 0 -> pure diffusion at epoch 0")
print("=" * 100)
print(f"Trainable params: {n_params}  (Lambda 16 + log_sigma_vec {LATENT_DIM})")
print("=" * 100 + "\n")

for epoch in range(EPOCHS):
    model.train()
    lm.train()
    running_vol   = 0.0
    running_drift = 0.0
    running_eig   = 0.0
    running_l2    = 0.0
    n_batches     = 0
    nan_batches   = 0
    batch_diag    = []
    ep_attempted  = 0
    ep_priced     = 0
    ep_pfracs     = []

    for step in range(N_STEPS_PER_EPOCH):
        print(f"\r  ep {epoch}  step {step+1}/{N_STEPS_PER_EPOCH} ...", end="", flush=True)
        optim.zero_grad(set_to_none=True)

        loss_vol, loss_drift_raw, diag, n_att, n_pri, p_frac = compute_pricing_loss_v2(
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

        # Eigenvalue floor on K^Q
        loss_eig = torch.tensor(0.0, device=device, dtype=torch.float32)
        if LAMBDA_EIG > 0:
            try:
                M = get_K_matrix(lm, LATENT_DIM, device, torch.float32)
                loss_eig = eigenvalue_floor_loss(M, eig_min=EIG_MIN)
                if not torch.isfinite(loss_eig):
                    loss_eig = torch.tensor(0.0, device=device, dtype=torch.float32)
            except Exception:
                pass

        # L2 regularisation on Lambda
        loss_l2 = LAMBDA_L2 * lm.Lambda.pow(2).sum()

        # Drift penalty
        loss_drift = LAMBDA_DRIFT * loss_drift_raw

        loss_total = (LAMBDA_VOL  * loss_vol
                      + loss_drift
                      + LAMBDA_EIG * loss_eig
                      + loss_l2)

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

        # Clip Lambda and sigma_vec gradients SEPARATELY so sigma_vec update
        # is not diluted by the much larger drift-driven Lambda gradient.
        torch.nn.utils.clip_grad_norm_([lm.Lambda],        max_norm=5.0)
        torch.nn.utils.clip_grad_norm_([lm.log_sigma_vec], max_norm=2.0)
        optim.step()

        running_vol   += float(loss_vol.detach().cpu())
        running_drift += float(loss_drift_raw.detach().cpu())
        running_eig   += float(loss_eig.detach().cpu())
        running_l2    += float(loss_l2.detach().cpu())
        n_batches     += 1

    print("\r" + " " * 40 + "\r", end="", flush=True)
    scheduler.step()

    ep_vol   = running_vol   / max(n_batches, 1)
    ep_drift = running_drift / max(n_batches, 1)
    ep_eig   = running_eig   / max(n_batches, 1)
    ep_l2    = running_l2    / max(n_batches, 1)
    swp_priced  = ep_priced  / max(ep_attempted, 1)
    path_finite = float(np.mean(ep_pfracs)) if ep_pfracs else 0.0

    with torch.no_grad():
        sigma_vec_now   = lm.sigma_vec.detach().cpu()          # (d,)
        scale_now       = float(sigma_vec_now.mean())           # scalar for display
        lambda_norm_fro = float(lm.Lambda.norm().cpu())
        try:
            M_now          = get_K_matrix(lm, LATENT_DIM, device, torch.float32)
            lambda_min_now = float(torch.linalg.eigvals(M_now).real.abs().min().cpu())
        except Exception:
            lambda_min_now = float('nan')

    for k, v in [("vol", ep_vol), ("drift", ep_drift), ("eig", ep_eig), ("l2", ep_l2),
                 ("swp_priced", swp_priced), ("path_finite", path_finite),
                 ("lambda_min", lambda_min_now), ("lambda_norm", lambda_norm_fro),
                 ("sigma_scale", scale_now)]:   # sigma_scale stores mean of sigma_vec
        hist[k].append(v)

    # Reconstruction RMSE
    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    if do_eval:
        rmse_per_ccy, avg_rmse_bps, _ = eval_rmse_bps(model, X_tensor, meta)
        gn        = grad_norm([lm.Lambda])
        gn_scale  = grad_norm([lm.log_sigma_vec])
    else:
        rmse_per_ccy, avg_rmse_bps = None, float('nan')
        gn       = 0.0
        gn_scale = 0.0

    # Bias diagnostic from diag list
    if batch_diag:
        mean_bias_diag = np.mean([d["bias_bp"] for d in batch_diag])
    else:
        mean_bias_diag = float('nan')

    lr_now = optim.param_groups[0]["lr"]
    t_now  = time.perf_counter()
    dt_ep  = t_now - t_last_log
    t_last_log = t_now
    eta_sec = dt_ep * (EPOCHS - epoch - 1)
    eta_str = (f"{int(eta_sec//3600)}h{int((eta_sec%3600)//60):02d}m" if eta_sec >= 3600 else
               f"{int(eta_sec//60)}m{int(eta_sec%60):02d}s"           if eta_sec >= 60 else
               f"{int(eta_sec)}s")

    # CSV row
    row = {
        "epoch": epoch, "time_total_sec": round(t_now - t0, 1),
        "time_interval_sec": round(dt_ep, 3),
        "loss_vol": ep_vol, "loss_drift": ep_drift,
        "loss_eig": ep_eig, "loss_l2": ep_l2,
        "swaption_priced_frac": swp_priced, "path_finite_frac": path_finite,
        "recon_rmse_bps": float(avg_rmse_bps), "nan_batches": nan_batches,
        "gnorm": gn, "gnorm_scale": gn_scale, "lr": lr_now,
        "lambda_min_KQ": lambda_min_now, "Lambda_norm_fro": lambda_norm_fro,
        "sigma_scale_mean": scale_now,
        "sigma_s1": float(sigma_vec_now[0]), "sigma_s2": float(sigma_vec_now[1]),
        "sigma_s3": float(sigma_vec_now[2]), "sigma_s4": float(sigma_vec_now[3]),
        "fwd_bias_diag_bp": mean_bias_diag,
    }
    for c in ccy_order:
        row[f"rmse_bps_{c}"] = (
            float(rmse_per_ccy.get(c, float('nan')))
            if rmse_per_ccy is not None else float('nan')
        )
    pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)

    # Console header
    if (epoch // max(LOG_EVERY, 1)) % HEADER_EVERY == 0:
        print(
            f"\n{'ep':>5} {'vol':>10} {'drift':>9} {'eig':>9} {'l2':>8} "
            f"{'swp%':>5} {'pth%':>5} {'recon':>7} "
            f"{'|l|min':>7} {'||L||':>7} {'sv_mean':>7} {'sv_min':>6} {'sv_max':>6} "
            f"{'bias_bp':>8} {'gn_L':>9} {'gn_s':>7} {'lr':>8} {'t/ep':>6} {'ETA':>8}  diag"
        )
        print("-" * 215)

    # Diagnostics string
    if batch_diag and (epoch % DIAG_EVERY == 0 or epoch == EPOCHS - 1):
        diag_str = " | ".join(
            f"{d['exp']}x{d['ten']} mkt={d['mkt_bp']:.0f} mod={d['mod_bp']:.0f} "
            f"err={d['err_bp']:+.0f} bias={d['bias_bp']:+.0f}bp F0={d['F0']:.0f}"
            for d in batch_diag[:3]
        )
    else:
        diag_str = ""

    sv_min = float(sigma_vec_now.min())
    sv_max = float(sigma_vec_now.max())
    print(
        f"{epoch:>5d} "
        f"{ep_vol:>10.4e} {ep_drift:>9.3e} {ep_eig:>9.3e} {ep_l2:>8.4e} "
        f"{swp_priced*100:>4.0f}% {path_finite*100:>4.0f}% "
        f"{avg_rmse_bps:>7.2f} "
        f"{lambda_min_now:>7.4f} {lambda_norm_fro:>7.4f} "
        f"{scale_now:>7.4f} {sv_min:>6.4f} {sv_max:>6.4f} "
        f"{mean_bias_diag:>+8.1f} {gn:>9.2e} {gn_scale:>7.2e} {lr_now:>8.2e} "
        f"{dt_ep:>5.1f}s {eta_str:>8}  {diag_str}"
    )

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt = os.path.join(FIGURES_DIR, f"checkpoint_lambda_v2_ep{epoch+1}.pt")
        torch.save({
            "lm_state_dict":    lm.state_dict(),
            "Lambda_matrix":    lm.Lambda.detach().cpu(),
            "log_sigma_vec":    lm.log_sigma_vec.detach().cpu(),
            "sigma_vec":        lm.sigma_vec.detach().cpu(),
            "sigma_scale_mean": scale_now,
            "lambda_norm_fro":  lambda_norm_fro,
            "latent_dim":       LATENT_DIM,
            "epoch":            epoch + 1,
            "variant":          config.VARIANT,
        }, ckpt)
        print(f"  -> checkpoint ep{epoch+1}  ||L||={lambda_norm_fro:.4f}  scale={scale_now:.4f}")

print("\nTraining done.")

# ── final checkpoint + plots ───────────────────────────────────────────────────

torch.save({
    "lm_state_dict":    lm.state_dict(),
    "Lambda_matrix":    lm.Lambda.detach().cpu(),
    "log_sigma_vec":    lm.log_sigma_vec.detach().cpu(),
    "sigma_vec":        lm.sigma_vec.detach().cpu(),
    "sigma_scale_mean": float(lm.sigma_vec.mean().detach()),
    "latent_dim":       LATENT_DIM,
    "epochs":           EPOCHS,
    "variant":          config.VARIANT,
}, os.path.join(FIGURES_DIR, f"checkpoint_lambda_v2_ep{EPOCHS}.pt"))

fig, axes = plt.subplots(4, 1, figsize=(9, 13), dpi=150)

axes[0].semilogy(hist["vol"],   lw=1.0, color="darkorange", label="Vol loss (variance)")
axes[0].semilogy(hist["drift"], lw=1.0, color="deeppink",   label="Drift loss (E^A[F_T]-F_0)")
axes[0].semilogy(hist["eig"],   lw=1.0, color="seagreen",   label="Eig floor")
axes[0].semilogy(hist["l2"],    lw=1.0, color="royalblue",  label="L2 Lambda")
axes[0].set_title("Lambda v2: Loss Components"); axes[0].legend()
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log)"); axes[0].grid(True, alpha=0.3)

axes[1].plot([100*p for p in hist["swp_priced"]], lw=1.2, color="firebrick", label="swaption_priced%")
axes[1].plot([100*p for p in hist["path_finite"]], lw=1.0, color="navy",     label="path_finite%")
axes[1].set_ylim(-2, 102); axes[1].legend()
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("%"); axes[1].grid(True, alpha=0.3)

axes[2].plot(hist["sigma_scale"], lw=1.2, color="purple", label="mean(sigma_vec)")
axes[2].set_xlabel("Epoch"); axes[2].set_title("Diffusion Scale (mean of sigma_vec)"); axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(hist["lambda_norm"], lw=1.2, color="purple", label="||Lambda||_F")
axes[3].set_xlabel("Epoch"); axes[3].set_title("Lambda Frobenius Norm"); axes[3].legend()
axes[3].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"lambda_v2_loss_dim{LATENT_DIM}_ep{EPOCHS}.png"), dpi=200)
plt.close(fig)

print("\n" + "=" * 90)
print("LAMBDA v2 TRAINING COMPLETE")
print("=" * 90)
print(f"Final vol loss    : {hist['vol'][-1]:.6e}")
print(f"Final drift loss  : {hist['drift'][-1]:.6e}  (implies ~{math.sqrt(hist['drift'][-1])*100:.0f} bp bias)")
print(f"Final sigma_vec   : {lm.sigma_vec.detach().cpu().numpy().round(4)}  (mean={hist['sigma_scale'][-1]:.4f})")
print(f"Final ||Lambda||_F: {hist['lambda_norm'][-1]:.4f}")
print(f"Lambda matrix:\n{lm.Lambda.detach().cpu().numpy()}")
print("=" * 90)
