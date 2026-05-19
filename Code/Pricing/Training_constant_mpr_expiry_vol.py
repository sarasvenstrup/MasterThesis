# ==================== Constant MPR — Per-Expiry Vol Scaling ====================
"""
Extension of Training_constant_mpr.py with two changes that address the
structural term-structure mismatch diagnosed in _term_structure_vol_diagnostics.py:

CHANGE 1 — Per-expiry diffusion scale (sigma_vecs)
----------------------------------------------------
The base model σ_F(T_e) is strongly decreasing with expiry (Vasicek-like variance
saturation), while market σ_N is roughly flat.  A single σ_vec cannot reconcile
both short and long expiries.  Solution: one σ_vec per expiry bucket.

    sigma_vecs[i] ∈ R^d  for i in {0=1Y, 1=5Y, 2=10Y}

Required scale factors to match market (from diagnostics):
  1Y: ~0.05–0.09   5Y: ~0.16–0.24   10Y: ~0.24–0.33
→ 3 independent per-expiry scales remove the term-structure lock-in.

Trainable params: lambda_0 (d=4) + log_sigma_vecs (n_expiries × d = 12) = 16.

CHANGE 2 — LAMBDA_BIAS = 1.0  (was 0.5)
-----------------------------------------
With per-expiry vol, the bias penalty must be strong enough so that the forward
is correctly centred *before* the optimizer bends σ_vec to compensate for bias.
Equal weight (1.0 : 1.0) prevents vol from absorbing residual bias errors.

Output: Figures/TrainingResults/dim4_constant_mpr_expvol/ep{EPOCHS}/
"""

import os, sys

_N_TORCH   = 4
_N_INTEROP = 2
os.environ.setdefault("OMP_NUM_THREADS",        str(_N_TORCH))
os.environ.setdefault("MKL_NUM_THREADS",        str(_N_TORCH))
os.environ.setdefault("OPENBLAS_NUM_THREADS",   str(_N_TORCH))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_N_TORCH))
os.environ.setdefault("NUMEXPR_NUM_THREADS",    str(_N_TORCH))

import time, math, json
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

# Expiry buckets — must match the expiries present in the vol data.
# Each bucket gets its own sigma_vec.
EXPIRY_BUCKETS = [1, 5, 10]          # years

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

# Loss weights — bias = 1.0 (equal to vol) to prevent vol absorbing forward errors
LAMBDA_VOL  = 1.0
LAMBDA_BIAS = 1.0    # raised from 0.5 → 1.0
LAMBDA_L2   = 1e-3

LR = 5e-4
LR_WARMUP_EPOCHS = 50

MIN_FINITE_PATHS_FRAC = 0.10
MIN_FINITE_PATHS_ABS  = 16
LOSS_SKIP_THRESH      = 1e4

USE        = "bbg"
CCY_FILTER = "EUR"

FIGURES_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_constant_mpr_expvol", f"ep{EPOCHS}"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── module ─────────────────────────────────────────────────────────────────────

class ConstantMPRExpiryVol(nn.Module):
    """
    Constant MPR drift adjustment + per-expiry diffusion scaling.

    K_price(z) = K_base(z) + L_base(z) @ lambda_0          (shared across expiries)
    sigma_vec(expiry) = exp(log_sigma_vecs[expiry_idx])     (per-expiry)

    lambda_0 is shared: the risk-premium direction in latent space is the same
    regardless of how far we're simulating — it shifts the drift uniformly.

    sigma_vecs are per-expiry: different expiries need different compression
    factors because Var(z_T) saturates (Vasicek-like mean reversion), making the
    base model's σ_F strongly decreasing while the market is roughly flat.

    Trainable: lambda_0 (d=4) + log_sigma_vecs (n_expiries × d = 12) = 16 params.
    """

    def __init__(self, kp_module, h_module, latent_dim, expiry_buckets):
        super().__init__()
        self.kp             = kp_module
        self.h              = h_module
        self.latent_dim     = latent_dim
        self.expiry_buckets = expiry_buckets          # list[int], sorted
        n_exp               = len(expiry_buckets)

        # Shared constant drift correction in latent space.
        self.lambda_0 = nn.Parameter(torch.zeros(latent_dim))

        # Per-expiry diffusion scale.
        # Initialise from diagnostics: required scale ≈ 0.06 / 0.20 / 0.30
        # → log values ≈ -2.8 / -1.6 / -1.2
        # Use the same init as before (-1.8) as a neutral starting point.
        init_log = torch.full((n_exp, latent_dim), -1.8)
        self.log_sigma_vecs = nn.Parameter(init_log)    # (n_exp, d)

    def forward(self, z):
        """K_price(z) — drift only; sigma_vec is looked up separately per call."""
        k_base       = self.kp(z)
        sigmas, rhos = self.h(z)
        L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam0         = self.lambda_0.unsqueeze(0).expand(z.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam0)

    def sigma_vec_for(self, expiry: int) -> torch.Tensor:
        """Return sigma_vec (shape d,) for the given expiry year."""
        try:
            idx = self.expiry_buckets.index(expiry)
        except ValueError:
            # Fall back to nearest bucket
            diffs = [abs(expiry - b) for b in self.expiry_buckets]
            idx   = int(np.argmin(diffs))
        return self.log_sigma_vecs[idx].exp()           # (d,)

    @property
    def sigma_vecs(self) -> torch.Tensor:
        """All per-expiry scales: shape (n_expiries, d)"""
        return self.log_sigma_vecs.exp()


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
    sigma_vec is looked up per-expiry from lm.sigma_vec_for(expiry).
    """
    if len(df_vol) == 0 or n_swaptions == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, [], 0, 0, 0.0

    sample = df_vol.sample(n=min(n_swaptions, len(df_vol)))

    total_vol   = torch.zeros(1, device=device, dtype=dtype)
    total_bias  = torch.zeros(1, device=device, dtype=dtype)
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
            z0 = model.encoder(xb)
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

        # Per-expiry sigma_vec — gradient flows through this
        sv = lm.sigma_vec_for(expiry)                    # (d,)

        with torch.no_grad():
            eps_z = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)

        try:
            z_T, D_T = simulate_to_expiry_differentiable(
                model, z0,
                n_steps=n_steps, dt=dt_eff,
                n_paths=2 * half,
                eps=eps_z,
                k_override=lm,
                sigma_scale=sv,          # ← per-expiry scale
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
                sv_vals = sv.detach().cpu().tolist()
                diagnostics.append({
                    "date":    date.date(),
                    "exp":     expiry,
                    "ten":     tenor,
                    "mkt_bp":  round(sigma_mkt_bp, 1),
                    "mod_bp":  round(float(sigma_str_bp.detach()), 1),
                    "err_bp":  round(float(sigma_str_bp.detach()) - sigma_mkt_bp, 1),
                    "bias_bp": round(float(fwd_bias_bp.detach()), 1),
                    "F0":      round(F_0 * 1e4, 1),
                    "scale":   [round(v, 4) for v in sv_vals],
                    "lambda0": [round(v, 4) for v in lm.lambda_0.detach().cpu().tolist()],
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

lm = ConstantMPRExpiryVol(model.K, model.H, LATENT_DIM, EXPIRY_BUCKETS).to(device)

n_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
print(f"ConstantMPRExpiryVol initialised  ({n_params} trainable params)")
print(f"  lambda_0 init  = {lm.lambda_0.detach().cpu().numpy().round(4)}")
print(f"  sigma_vecs init (mean per expiry):")
for i, exp in enumerate(EXPIRY_BUCKETS):
    sv = lm.sigma_vecs[i].detach().cpu()
    print(f"    {exp}Y: {sv.numpy().round(4)}  mean={float(sv.mean()):.4f}")

model.train()

LR_SCALE_MULT = 10.0
optim = torch.optim.Adam([
    {'params': [lm.lambda_0],          'lr': LR,                 'name': 'lambda_0'},
    {'params': [lm.log_sigma_vecs],    'lr': LR * LR_SCALE_MULT, 'name': 'sigma_vecs'},
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

# Filter vol data to only known expiry buckets
df_vol = df_vol[df_vol["option_maturity"].isin(EXPIRY_BUCKETS)].copy()
print(f"After expiry filter: {len(df_vol)} rows")

# ── CSV logger ─────────────────────────────────────────────────────────────────

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
csv_path  = os.path.join(FIGURES_DIR, f"train_expvol_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")

sv_cols = []
for exp in EXPIRY_BUCKETS:
    for d in range(LATENT_DIM):
        sv_cols.append(f"sv_{exp}Y_d{d+1}")

csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec",
     "loss_vol", "loss_bias", "loss_l2",
     "swaption_priced_frac", "path_finite_frac",
     "recon_rmse_bps", "nan_batches",
     "gnorm_lam0", "gnorm_svecs", "lr",
     "lambda0_norm", "lambda0_l1",
     "fwd_bias_diag_bp"]
    + [f"sv_mean_{exp}Y" for exp in EXPIRY_BUCKETS]
    + sv_cols
    + [f"lambda0_v{i+1}" for i in range(LATENT_DIM)]
    + [f"rmse_bps_{c}" for c in ccy_order]
)
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "version":               "constant_mpr_expiry_vol",
    "description":           "Per-expiry sigma_vec + LAMBDA_BIAS=1.0",
    "seed":                  SEED,
    "latent_dim":            LATENT_DIM,
    "expiry_buckets":        EXPIRY_BUCKETS,
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
    "lambda0_norm",
] + [f"sv_mean_{exp}Y" for exp in EXPIRY_BUCKETS]}

t0         = time.perf_counter()
t_last_log = t0

print("\n" + "=" * 110)
print("CONSTANT MPR — PER-EXPIRY VOL SCALING")
print("  K_price(z) = K_base(z) + L_base(z) @ lambda_0  (shared)")
print(f"  sigma_vec[exp] per expiry in {EXPIRY_BUCKETS}  (independent)")
print(f"  LAMBDA_BIAS = {LAMBDA_BIAS}  (raised from 0.5 to prevent bias-vol confusion)")
print(f"  Trainable: lambda_0 ({LATENT_DIM}) + log_sigma_vecs ({len(EXPIRY_BUCKETS)}×{LATENT_DIM}) = {n_params} total")
print("=" * 110 + "\n")

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
        torch.nn.utils.clip_grad_norm_([lm.log_sigma_vecs],  max_norm=2.0)
        optim.step()

        running_vol  += float(loss_vol.detach().cpu())
        running_bias += float(loss_bias_raw.detach().cpu())
        running_l2   += float(loss_l2.detach().cpu())
        n_batches    += 1

    print("\r" + " " * 40 + "\r", end="", flush=True)
    scheduler.step()

    ep_vol  = running_vol  / max(n_batches, 1)
    ep_bias = running_bias / max(n_batches, 1)
    ep_l2   = running_l2   / max(n_batches, 1)
    swp_priced  = ep_priced  / max(ep_attempted, 1)
    path_finite = float(np.mean(ep_pfracs)) if ep_pfracs else 0.0

    with torch.no_grad():
        sv_now     = lm.sigma_vecs.detach().cpu()        # (n_exp, d)
        lambda0_now = lm.lambda_0.detach().cpu()
        lambda0_norm = float(lambda0_now.norm())
        lambda0_l1   = float(lambda0_now.abs().sum())
        sv_means     = {exp: float(sv_now[i].mean()) for i, exp in enumerate(EXPIRY_BUCKETS)}

    for k, v in [("vol", ep_vol), ("bias", ep_bias), ("l2", ep_l2),
                 ("swp_priced", swp_priced), ("path_finite", path_finite),
                 ("lambda0_norm", lambda0_norm)]:
        hist[k].append(v)
    for exp in EXPIRY_BUCKETS:
        hist[f"sv_mean_{exp}Y"].append(sv_means[exp])

    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    if do_eval:
        rmse_per_ccy, avg_rmse_bps, _ = eval_rmse_bps(model, X_tensor, meta)
        gn_lam0  = grad_norm([lm.lambda_0])
        gn_svecs = grad_norm([lm.log_sigma_vecs])
    else:
        rmse_per_ccy, avg_rmse_bps = None, float('nan')
        gn_lam0  = 0.0
        gn_svecs = 0.0

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
        "swaption_priced_frac": swp_priced, "path_finite_frac": path_finite,
        "recon_rmse_bps": float(avg_rmse_bps), "nan_batches": nan_batches,
        "gnorm_lam0": gn_lam0, "gnorm_svecs": gn_svecs, "lr": lr_now,
        "lambda0_norm": lambda0_norm, "lambda0_l1": lambda0_l1,
        "fwd_bias_diag_bp": mean_bias_diag,
    }
    for exp in EXPIRY_BUCKETS:
        row[f"sv_mean_{exp}Y"] = sv_means[exp]
    for i, exp in enumerate(EXPIRY_BUCKETS):
        for d in range(LATENT_DIM):
            row[f"sv_{exp}Y_d{d+1}"] = float(sv_now[i, d])
    for d in range(LATENT_DIM):
        row[f"lambda0_v{d+1}"] = float(lambda0_now[d])
    for c in ccy_order:
        row[f"rmse_bps_{c}"] = (
            float(rmse_per_ccy.get(c, float('nan')))
            if rmse_per_ccy is not None else float('nan')
        )
    pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)

    if (epoch // max(LOG_EVERY, 1)) % HEADER_EVERY == 0:
        sv_hdr = "  ".join(f"sv{e}Y" for e in EXPIRY_BUCKETS)
        print(
            f"\n{'ep':>5} {'vol':>10} {'bias':>9} {'l2':>8} {'swp%':>5} {'pth%':>5} "
            f"{'recon':>7} {'|l0|':>7}  {sv_hdr}  "
            f"{'bias_bp':>8} {'gn_l0':>7} {'gn_sv':>7} {'lr':>8} {'t/ep':>6} {'ETA':>8}  diag"
        )
        print("-" * 200)

    sv_str = "  ".join(f"{sv_means[e]:.4f}" for e in EXPIRY_BUCKETS)
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
        f"{ep_vol:>10.4e} {ep_bias:>9.3e} {ep_l2:>8.4e} "
        f"{swp_priced*100:>4.0f}% {path_finite*100:>4.0f}% "
        f"{avg_rmse_bps:>7.2f} "
        f"{lambda0_norm:>7.4f}  {sv_str}  "
        f"{mean_bias_diag:>+8.1f} {gn_lam0:>7.2e} {gn_svecs:>7.2e} {lr_now:>8.2e} "
        f"{dt_ep:>5.1f}s {eta_str:>8}  {diag_str}"
    )

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        ckpt = os.path.join(FIGURES_DIR, f"checkpoint_expvol_ep{epoch+1}.pt")
        torch.save({
            "lm_state_dict":    lm.state_dict(),
            "lambda_0":         lm.lambda_0.detach().cpu(),
            "log_sigma_vecs":   lm.log_sigma_vecs.detach().cpu(),
            "sigma_vecs":       lm.sigma_vecs.detach().cpu(),
            "expiry_buckets":   EXPIRY_BUCKETS,
            "latent_dim":       LATENT_DIM,
            "epoch":            epoch + 1,
            "variant":          config.VARIANT,
        }, ckpt)
        sv_summary = {exp: sv_now[i].numpy().round(4).tolist()
                      for i, exp in enumerate(EXPIRY_BUCKETS)}
        print(f"  -> checkpoint ep{epoch+1}  |lambda_0|={lambda0_norm:.4f}  "
              f"lambda_0={lambda0_now.numpy().round(4)}")
        for exp, sv in sv_summary.items():
            print(f"     sigma_vec[{exp}Y] = {sv}")

print("\nTraining done.")

# ── final checkpoint ───────────────────────────────────────────────────────────

torch.save({
    "lm_state_dict":    lm.state_dict(),
    "lambda_0":         lm.lambda_0.detach().cpu(),
    "log_sigma_vecs":   lm.log_sigma_vecs.detach().cpu(),
    "sigma_vecs":       lm.sigma_vecs.detach().cpu(),
    "expiry_buckets":   EXPIRY_BUCKETS,
    "latent_dim":       LATENT_DIM,
    "epochs":           EPOCHS,
    "variant":          config.VARIANT,
}, os.path.join(FIGURES_DIR, f"checkpoint_expvol_ep{EPOCHS}.pt"))

# ── final plots ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(9, 14), dpi=150)

axes[0].semilogy(hist["vol"],  lw=1.0, color="darkorange", label="Vol loss")
axes[0].semilogy(hist["bias"], lw=1.0, color="deeppink",   label="Bias loss (raw)")
axes[0].semilogy(hist["l2"],   lw=1.0, color="royalblue",  label="L2 lambda_0")
axes[0].set_title("Per-Expiry Vol: Loss Components"); axes[0].legend()
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log)"); axes[0].grid(True, alpha=0.3)

colors_exp = ["#d62728", "#ff7f0e", "#2ca02c"]
for i, exp in enumerate(EXPIRY_BUCKETS):
    axes[1].plot(hist[f"sv_mean_{exp}Y"], lw=1.2, color=colors_exp[i],
                 label=f"σ_vec mean — {exp}Y expiry")
axes[1].set_xlabel("Epoch"); axes[1].set_title("Per-Expiry σ_vec Mean")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(hist["lambda0_norm"], lw=1.2, color="teal", label="||lambda_0||")
axes[2].set_xlabel("Epoch"); axes[2].set_title("Drift Bias Norm"); axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot([100*p for p in hist["swp_priced"]], lw=1.2, color="firebrick",
             label="swaption_priced%")
axes[3].plot([100*p for p in hist["path_finite"]], lw=1.0, color="navy",
             label="path_finite%")
axes[3].set_ylim(-2, 102); axes[3].legend()
axes[3].set_xlabel("Epoch"); axes[3].set_ylabel("%"); axes[3].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"expvol_loss_dim{LATENT_DIM}_ep{EPOCHS}.png"), dpi=200)
plt.close(fig)

# ── final sigma_vec convergence table ─────────────────────────────────────────
print("\n" + "=" * 90)
print("FINAL SIGMA_VECS PER EXPIRY")
print("=" * 90)
with torch.no_grad():
    sv_final = lm.sigma_vecs.detach().cpu()
for i, exp in enumerate(EXPIRY_BUCKETS):
    sv = sv_final[i]
    print(f"  {exp:2d}Y:  {sv.numpy().round(5)}  mean={float(sv.mean()):.5f}")
print()
print(f"Final lambda_0: {lm.lambda_0.detach().cpu().numpy().round(4)}")
print("=" * 90)


