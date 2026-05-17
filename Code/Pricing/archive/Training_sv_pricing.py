# ==================== Stochastic-Vol MPR (Pricing) Training ====================
"""
Stochastic-volatility pricing model.

Combines everything we've learned:
  - Per-expiry drift correction         (Expiry MPR)
  - Per-(expiry, tenor) vol surface      (Expiry-Tenor Vol MPR)
  - State-conditioned vol regime         (State-Cond Vol MPR)
  - CIR stochastic variance state        (NEW — this script)
  - State-conditioned initial variance   (NEW — from yield curve)

Q-dynamics:
    dz_t = K*_e(z_t) dt + sqrt(v_t/theta) * diag(σ_eff(z_0)) * L(z_t) * dW^z
    dv_t = κ(θ − v_t) dt + σ_v * sqrt(v_t) * dW^v

Drift (per-expiry):
    K*_e(z) = K(z) + L(z) · λ_e

Static vol surface (z_0-conditioned, evaluated once per swaption):
    log σ_eff,k(z_0) = a_k + b_{e,k} + c_{n,k} + δ_k · tanh(W · z_0)

Initial variance (state-conditioned):
    v_0(z_0) = θ · exp(α_v · tanh(w_v · z_0))

At init (W~small random, δ=0, α_v=0): model reduces to ETV MPR baseline.
Saddle-points avoided by small-random initialisation of W and w_v.

Output: Figures/TrainingResults/dim4_sv_pricing/ep{EPOCHS}/
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
    psutil.Process().cpu_affinity(list(range(os.cpu_count()))[:_N_TORCH * 2])
except Exception:
    pass

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
from Code.model.CIR_vol_pricing import CIRVolPricing
from Code.Simulation.simulate_sv_pricing import simulate_sv_to_expiry_differentiable

print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| Variant:", config.VARIANT)

# ── settings ───────────────────────────────────────────────────────────────────
LATENT_DIM  = 4
EXPIRY_VALS = [1, 5, 10]
TENOR_VALS  = [1, 5, 10]

EPOCHS                = 1000
EVAL_EVERY            = 100
DIAG_EVERY            = 10
HEADER_EVERY          = 20
SAVE_EVERY            = 200
N_STEPS_PER_EPOCH     = 4
N_SWAPTIONS_PER_BATCH = 8
N_PATHS_PRICING       = 512
DT_PRICING            = 1 / 6

PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt")
ETV_CKPT      = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_expiry_tenor_vol_mpr", "ep1000",
                              "checkpoint_expiry_tenor_vol_mpr_ep1000.pt")

LAMBDA_VOL     = 1.0
LAMBDA_BIAS    = 0.5
LAMBDA_L2_LAM  = 1e-3
LAMBDA_L2_VOL  = 1e-3
LAMBDA_L2_REG  = 1e-3   # δ, W (state-cond static)
LAMBDA_L2_VBASE= 1e-3   # w_v, α_v (state-cond v_0)
LAMBDA_FELLER  = 1.0    # penalty if 2κθ − σ_v² < 0

LR             = 2e-4
LR_SCALE_MULT  = 10.0   # log_sigma_base
LR_OFFSET_MULT = 5.0    # log_sigma_expiry, log_sigma_tenor
LR_REG_MULT    = 5.0    # W, δ, w_v, α_v
LR_CIR_MULT    = 5.0    # CIR scalars
LR_WARMUP      = 30

MIN_FINITE_PATHS_ABS  = 16
MIN_FINITE_PATHS_FRAC = 0.10
LOSS_SKIP_THRESH      = 1e4

USE        = "bbg"
CCY_FILTER = "EUR"

FIGURES_DIR = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                            f"dim{LATENT_DIM}_sv_pricing", f"ep{EPOCHS}")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── module ─────────────────────────────────────────────────────────────────────

class SVPricingMPR(nn.Module):
    """
    SV pricing module:
      lambda_expiry    [n_exp, d]  per-expiry drift correction      (warm-start ETV)
      log_sigma_base   [d]         base log-σ                        (warm-start ETV)
      log_sigma_expiry [n_exp, d]  expiry log-σ offsets              (warm-start ETV)
      log_sigma_tenor  [n_ten, d]  tenor log-σ offsets               (warm-start ETV)
      W                [d, d]      state-cond static-vol projection  (init small random)
      delta            [d]         state-cond static-vol scale       (init 0)
      cir              -           CIRVolPricing (κ, θ, σ_v, w_v, α_v)
    """

    def __init__(self, kp_module, h_module, latent_dim, expiry_vals, tenor_vals):
        super().__init__()
        self.kp            = kp_module
        self.h             = h_module
        self.latent_dim    = latent_dim
        self.expiry_vals   = expiry_vals
        self.tenor_vals    = tenor_vals
        self.expiry_to_idx = {e: i for i, e in enumerate(expiry_vals)}
        self.tenor_to_idx  = {t: i for i, t in enumerate(tenor_vals)}

        n_exp = len(expiry_vals)
        n_ten = len(tenor_vals)

        # From ETV MPR (warm-started)
        self.lambda_expiry    = nn.Parameter(torch.zeros(n_exp, latent_dim))
        self.log_sigma_base   = nn.Parameter(torch.full((latent_dim,), -1.8))
        self.log_sigma_expiry = nn.Parameter(torch.zeros(n_exp, latent_dim))
        self.log_sigma_tenor  = nn.Parameter(torch.zeros(n_ten, latent_dim))

        # State-conditioned static vol regime (small random W breaks saddle)
        self.W     = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.delta = nn.Parameter(torch.zeros(latent_dim))

        # CIR stochastic vol process
        self.cir = CIRVolPricing(latent_dim=latent_dim)

    # ── static σ_eff(z_0, expiry, tenor) ────────────────────────────────
    def get_sigma_eff(self, expiry: int, tenor: int, z0: torch.Tensor) -> torch.Tensor:
        e = self.expiry_to_idx[expiry]
        n = self.tenor_to_idx[tenor]
        z = z0.squeeze(0) if z0.dim() == 2 else z0
        regime = self.delta * torch.tanh(self.W @ z)
        return (self.log_sigma_base
                + self.log_sigma_expiry[e]
                + self.log_sigma_tenor[n]
                + regime).exp()

    # ── per-expiry Q-drift ─────────────────────────────────────────────────
    def drift(self, z_t: torch.Tensor, expiry: int) -> torch.Tensor:
        k_base       = self.kp(z_t)
        sigmas, rhos = self.h(z_t)
        L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam = self.lambda_expiry[self.expiry_to_idx[expiry]].unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    def forward(self, z_t):
        k_base       = self.kp(z_t)
        sigmas, rhos = self.h(z_t)
        L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam = self.lambda_expiry.mean(0).unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    @property
    def sigma_vec(self):
        return self.log_sigma_base.exp()


class SVPricingDriftWrapper(nn.Module):
    """Wraps SVPricingMPR with fixed (expiry, tenor, z0).  sigma_vec is the STATIC σ_eff."""

    def __init__(self, model: SVPricingMPR, expiry: int, tenor: int, z0: torch.Tensor):
        super().__init__()
        self.model  = model
        self.expiry = expiry
        self.tenor  = tenor
        self._z0    = z0

    def forward(self, z_t):
        return self.model.drift(z_t, self.expiry)

    @property
    def sigma_vec(self) -> torch.Tensor:
        """Static σ_eff (no v scaling) — v is applied per-path in simulate_sv."""
        return self.model.get_sigma_eff(self.expiry, self.tenor, self._z0)


# ── helpers ────────────────────────────────────────────────────────────────────

def row_finite_mask(t):
    return torch.isfinite(t).all(dim=1)

@torch.no_grad()
def predict_S_hat(model, X, batch_size=256):
    was_train = model.training; model.eval()
    outs = [model(X[i:i+batch_size].to(device)).detach().cpu()
            for i in range(0, X.shape[0], batch_size)]
    if was_train: model.train()
    return torch.cat(outs, 0)

def eval_rmse_bps(model, X_full, meta_full):
    S_hat = predict_S_hat(model, X_full)
    mask  = row_finite_mask(X_full) & row_finite_mask(S_hat)
    rmse  = H.rmse_bps_per_currency_paper(
        X_full[mask], S_hat[mask],
        meta_full.loc[mask.numpy()].reset_index(drop=True))
    return rmse, float(rmse.mean())

def grad_norm(params):
    return sum(float(p.grad.detach().pow(2).sum().cpu())
               for p in params if p.grad is not None) ** 0.5

# ── pricing loss ───────────────────────────────────────────────────────────────

def compute_pricing_loss(model, lm, X_batch, df_vol, date_to_idx,
                         n_swaptions, n_paths, dt, device, dtype,
                         return_diagnostics=False):
    if len(df_vol) == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, zero, [], 0, 0, 0.0

    sample      = df_vol.sample(n=min(n_swaptions, len(df_vol)))
    total_vol   = torch.zeros(1, device=device, dtype=dtype)
    total_bias  = torch.zeros(1, device=device, dtype=dtype)
    n_valid     = 0; n_attempted = 0
    diagnostics = []; path_fracs  = []
    min_paths   = max(MIN_FINITE_PATHS_ABS, int(n_paths * MIN_FINITE_PATHS_FRAC))

    for _, row in sample.iterrows():
        date      = pd.Timestamp(row["as_of_date"]).normalize()
        expiry    = int(row["option_maturity"])
        tenor     = int(row["swap_tenor"])
        sigma_mkt = float(row["market_vol"])
        sigma_mkt_bp = sigma_mkt * 1e4

        if (date not in date_to_idx
                or expiry not in lm.expiry_to_idx
                or tenor  not in lm.tenor_to_idx):
            continue

        n_attempted += 1
        idx = date_to_idx[date]
        xb  = X_batch[idx:idx + 1].to(device)

        with torch.no_grad():
            z0 = model.encoder(xb)
            _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
            P0 = aux0["P_full"][0]

        if expiry + tenor > P0.shape[0] - 1:
            continue
        F_0, A_0 = forward_swap_rate_torch(P0, expiry, tenor)
        if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 1e-6):
            continue

        dt_eff  = min(dt, expiry / 10.0)
        n_steps = max(12, int(round(expiry / dt_eff)))
        half    = n_paths // 2

        with torch.no_grad():
            eps_z = torch.randn(half, n_steps, LATENT_DIM, device=device, dtype=dtype)
            eps_v = torch.randn(half, n_steps, device=device, dtype=dtype)

        wrapper = SVPricingDriftWrapper(lm, expiry, tenor, z0)

        try:
            z_T, v_T, D_T = simulate_sv_to_expiry_differentiable(
                model, lm.cir, z0,
                n_steps=n_steps, dt=dt_eff, n_paths=2*half,
                eps_z=eps_z, eps_v=eps_v,
                k_override=wrapper,
                sigma_scale=wrapper.sigma_vec,   # static σ_eff (z_0-conditioned)
                v0=None,                          # use CIR.initial_v(z0)
                antithetic=True, freeze_H=True,
            )

            z_ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T) & torch.isfinite(v_T)
            if int(z_ok.sum()) < min_paths: continue

            # Terminal decode: use MEAN v_T to set the σ-scale of the no-arbitrage ODE.
            # Per-path v_T would require per-path sigma_scale in decode_from_z (more work).
            v_T_mean = v_T[z_ok].mean().detach()
            sv_factor_T = (v_T_mean / lm.cir.theta.detach()).sqrt()
            sigma_scale_terminal = wrapper.sigma_vec * sv_factor_T   # (d,)

            with torch.no_grad():
                _, aux_T = model.decode_from_z(z_T, tau=None, return_aux=True,
                                               k_override=wrapper,
                                               sigma_scale=sigma_scale_terminal)
                p_ok = torch.isfinite(aux_T["P_full"]).all(1)

            mask = z_ok & p_ok
            if int(mask.sum()) < min_paths: continue

            path_fracs.append(float(mask.float().mean()))
            _, aux_k = model.decode_from_z(z_T[mask], tau=None, return_aux=True,
                                           k_override=wrapper,
                                           sigma_scale=sigma_scale_terminal)
            F_T, A_T = swap_rate_torch(aux_k["P_full"], tenor=tenor)
            D_keep   = D_T[mask]

            fa_ok = (torch.isfinite(F_T) & torch.isfinite(A_T)
                     & (F_T > -0.5) & (F_T < 0.5)
                     & (A_T > 1e-6) & (A_T < 50.0))
            if int(fa_ok.sum()) < min_paths: continue
            F_T, A_T, D_keep = F_T[fa_ok], A_T[fa_ok], D_keep[fa_ok]

            V_pay = (D_keep * A_T * torch.relu(F_T - F_0)).mean()
            V_rec = (D_keep * A_T * torch.relu(F_0 - F_T)).mean()
            if not (torch.isfinite(V_pay) and torch.isfinite(V_rec)
                    and float(V_pay.detach()) >= 0 and float(V_rec.detach()) >= 0):
                continue

            sqrt_2pi     = math.sqrt(2 * math.pi)
            sigma_str_bp = (V_pay + V_rec) * 0.5 * sqrt_2pi / (A_0 * math.sqrt(expiry)) * 1e4
            loss_vol_ij  = ((sigma_str_bp - sigma_mkt_bp) / 100.0).pow(2)
            if not torch.isfinite(loss_vol_ij) or float(loss_vol_ij.detach()) > LOSS_SKIP_THRESH:
                continue

            fwd_bias_bp  = (V_pay - V_rec) / A_0 * 1e4
            loss_bias_ij = (fwd_bias_bp / 100.0).pow(2)
            if not torch.isfinite(loss_bias_ij): continue

            total_vol  = total_vol  + loss_vol_ij
            total_bias = total_bias + loss_bias_ij
            n_valid   += 1

            if return_diagnostics:
                diagnostics.append({
                    "date": date.date(), "exp": expiry, "ten": tenor,
                    "mkt_bp": round(sigma_mkt_bp, 1),
                    "mod_bp": round(float(sigma_str_bp.detach()), 1),
                    "err_bp": round(float(sigma_str_bp.detach()) - sigma_mkt_bp, 1),
                    "bias_bp": round(float(fwd_bias_bp.detach()), 1),
                    "v_T_mean": round(float(v_T_mean), 5),
                    "sv_factor": round(float(sv_factor_T), 4),
                })

        except Exception:
            continue

    mean_pfrac = float(np.mean(path_fracs)) if path_fracs else 0.0
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    if n_valid > 0:
        return total_vol/n_valid, total_bias/n_valid, diagnostics, n_attempted, n_valid, mean_pfrac
    return zero, zero, diagnostics, n_attempted, 0, mean_pfrac

# ── load data ──────────────────────────────────────────────────────────────────
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

meta, X_tensor, meta_full, X_tensor_full, *_ = my_data(use=USE)
X_tensor = X_tensor.float()
meta_ccy     = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]

# ── load model ─────────────────────────────────────────────────────────────────
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
model.load_state_dict(raw.get("model_state_dict", raw))
for p in model.parameters(): p.requires_grad_(False)
print("Base model loaded and frozen.")

lm = SVPricingMPR(model.K, model.H, LATENT_DIM, EXPIRY_VALS, TENOR_VALS).to(device)

# ── warm-start from ETV MPR ────────────────────────────────────────────────────
if os.path.exists(ETV_CKPT):
    raw_e = torch.load(ETV_CKPT, map_location=device, weights_only=False)
    es    = raw_e.get("lm_state_dict", raw_e)
    with torch.no_grad():
        for key in ["lambda_expiry", "log_sigma_base", "log_sigma_expiry", "log_sigma_tenor"]:
            if key in es:
                getattr(lm, key).copy_(es[key].to(device))
    print(f"Warm-started static-vol params from ETV MPR: {ETV_CKPT}")
else:
    print(f"WARNING: ETV MPR checkpoint not found — starting from scratch")

n_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
print(f"Trainable params: {n_params}")
print(f"  CIR init: κ={float(lm.cir.kappa):.4f}  θ={float(lm.cir.theta):.6f}  σ_v={float(lm.cir.sigma_v):.4f}")
print(f"  Feller (2κθ − σ_v²) = {float(lm.cir.feller):+.4f}  (positive = strict positivity)")
print(f"  sqrt(θ) = {float(lm.cir.theta.sqrt()):.4f}  (this is the SV-baseline vol level)")

model.train()

optim = torch.optim.Adam([
    {'params': [lm.lambda_expiry],    'lr': LR,                   'name': 'lambda'},
    {'params': [lm.log_sigma_base],   'lr': LR * LR_SCALE_MULT,   'name': 'sig_base'},
    {'params': [lm.log_sigma_expiry], 'lr': LR * LR_OFFSET_MULT,  'name': 'sig_exp'},
    {'params': [lm.log_sigma_tenor],  'lr': LR * LR_OFFSET_MULT,  'name': 'sig_ten'},
    {'params': [lm.W, lm.delta],      'lr': LR * LR_REG_MULT,     'name': 'regime'},
    {'params': [lm.cir.log_kappa, lm.cir.log_theta, lm.cir.log_sigma_v],
                                       'lr': LR * LR_CIR_MULT,    'name': 'cir'},
    {'params': [lm.cir.w_v, lm.cir.alpha_v],
                                       'lr': LR * LR_REG_MULT,    'name': 'v0_state'},
], lr=LR)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optim,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optim, 1e-3, 1.0, LR_WARMUP),
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, max(EPOCHS-LR_WARMUP,1), 1e-7),
    ],
    milestones=[LR_WARMUP]
)

# ── swaption data ──────────────────────────────────────────────────────────────
df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0
df_vol = df_vol[df_vol["option_maturity"].isin(EXPIRY_VALS)
                & df_vol["swap_tenor"].isin(TENOR_VALS)].copy()

dates_swap  = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
df_vol      = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {pd.Timestamp(r["as_of_date"]).normalize(): i for i, r in meta_ccy.iterrows()}

if df_vol.empty: raise RuntimeError("No swaption vol data.")
print(f"Loaded {len(df_vol)} vol targets from {df_vol['as_of_date'].nunique()} dates")

# ── CSV logger ─────────────────────────────────────────────────────────────────
ccy_order = ["AUD","CAD","DKK","EUR","JPY","NOK","SEK","GBP","USD"]
csv_path  = os.path.join(FIGURES_DIR, f"train_log_dim{LATENT_DIM}_ep{EPOCHS}.csv")
lam_cols  = [f"lam_{e}Y_norm" for e in EXPIRY_VALS]
csv_cols  = (["epoch","time_total_sec","time_interval_sec",
              "loss_vol","loss_bias","loss_l2_lam","loss_l2_vol","loss_l2_reg",
              "loss_l2_vbase","loss_feller",
              "swaption_priced_frac","path_finite_frac","recon_rmse_bps","nan_batches",
              "gnorm_lam","gnorm_sig","gnorm_reg","gnorm_cir","gnorm_v0","lr",
              "fwd_bias_diag_bp",
              "delta_norm","W_norm","w_v_norm","alpha_v",
              "kappa","theta","sigma_v","feller","sqrt_theta"]
             + lam_cols + [f"rmse_bps_{c}" for c in ccy_order])
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)

with open(os.path.join(FIGURES_DIR, "run_config.json"), "w") as f:
    json.dump({"version":"sv_pricing","latent_dim":LATENT_DIM,
               "expiry_vals":EXPIRY_VALS,"tenor_vals":TENOR_VALS,
               "epochs":EPOCHS,"n_params":n_params,
               "warm_start":ETV_CKPT,
               "notes":"CIR + state-cond v0 + state-cond static vol + per-expiry drift"},
              f, indent=2)

# ── training loop ──────────────────────────────────────────────────────────────
t0 = time.perf_counter(); t_last = t0

print("\n" + "="*100)
print("SV PRICING MPR")
print("  Drift:   K*_e(z) = K(z) + L(z) · λ_e")
print("  Vol:     σ_eff(e,n,z_0) = exp(base + exp + ten + δ·tanh(W·z_0))")
print("  CIR:     dv = κ(θ−v)dt + σ_v·√v dW_v   (per-path SV multiplier sqrt(v/θ))")
print("  v_0:     θ · exp(α_v · tanh(w_v · z_0))")
print(f"  Params:  {n_params}")
print("="*100 + "\n")

for epoch in range(EPOCHS):
    model.train(); lm.train()
    r_vol=r_bias=r_l2l=r_l2v=r_l2r=r_l2vb=r_fel=0.0; n_bat=nan_bat=0
    ep_att=ep_pri=0; ep_pf=[]; batch_diag=[]

    for step in range(N_STEPS_PER_EPOCH):
        print(f"\r  ep {epoch}  step {step+1}/{N_STEPS_PER_EPOCH} ...", end="", flush=True)
        optim.zero_grad(set_to_none=True)

        lv, lb, diag, na, np_, pf = compute_pricing_loss(
            model, lm, X_tensor_ccy, df_vol, date_to_idx,
            N_SWAPTIONS_PER_BATCH, N_PATHS_PRICING, DT_PRICING,
            device, torch.float32, return_diagnostics=(step==0))
        ep_att+=na; ep_pri+=np_
        if pf>0: ep_pf.append(pf)
        if diag: batch_diag=diag

        l2_lam = LAMBDA_L2_LAM * lm.lambda_expiry.pow(2).sum()
        l2_vol = LAMBDA_L2_VOL * (lm.log_sigma_expiry.pow(2).sum()
                                   + lm.log_sigma_tenor.pow(2).sum())
        l2_reg = LAMBDA_L2_REG * (lm.delta.pow(2).sum() + lm.W.pow(2).sum())
        l2_vb  = LAMBDA_L2_VBASE * (lm.cir.w_v.pow(2).sum() + lm.cir.alpha_v.pow(2))
        feller_pen = LAMBDA_FELLER * torch.relu(-lm.cir.feller).pow(2)
        loss   = LAMBDA_VOL*lv + LAMBDA_BIAS*lb + l2_lam + l2_vol + l2_reg + l2_vb + feller_pen

        if not torch.isfinite(loss): nan_bat+=1; continue
        loss.backward()

        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in lm.parameters()):
            nan_bat+=1; optim.zero_grad(set_to_none=True); continue

        for pg in optim.param_groups:
            torch.nn.utils.clip_grad_norm_(pg['params'], max_norm=2.0)
        optim.step()

        r_vol+=float(lv.detach()); r_bias+=float(lb.detach())
        r_l2l+=float(l2_lam.detach()); r_l2v+=float(l2_vol.detach())
        r_l2r+=float(l2_reg.detach()); r_l2vb+=float(l2_vb.detach())
        r_fel+=float(feller_pen.detach())
        n_bat+=1

    print("\r"+" "*40+"\r", end="", flush=True)
    scheduler.step()

    n_bat = max(n_bat, 1)
    ep_vol=r_vol/n_bat; ep_bias=r_bias/n_bat
    ep_l2l=r_l2l/n_bat; ep_l2v=r_l2v/n_bat
    ep_l2r=r_l2r/n_bat; ep_l2vb=r_l2vb/n_bat; ep_fel=r_fel/n_bat
    swp=ep_pri/max(ep_att,1); pth=float(np.mean(ep_pf)) if ep_pf else 0.0

    with torch.no_grad():
        lam_norms  = [float(lm.lambda_expiry[i].norm()) for i in range(len(EXPIRY_VALS))]
        delta_norm = float(lm.delta.norm())
        W_norm     = float(lm.W.norm())
        w_v_norm   = float(lm.cir.w_v.norm())
        alpha_v    = float(lm.cir.alpha_v)
        kappa      = float(lm.cir.kappa)
        theta      = float(lm.cir.theta)
        sigma_v    = float(lm.cir.sigma_v)
        feller     = float(lm.cir.feller)
        sqrt_theta = float(lm.cir.theta.sqrt())

    do_eval = ((epoch+1)%EVAL_EVERY==0) or epoch==0 or epoch==EPOCHS-1
    rmse_per_ccy, avg_rmse = eval_rmse_bps(model, X_tensor, meta) if do_eval else (None, float('nan'))
    gn_lam = grad_norm([lm.lambda_expiry])
    gn_sig = grad_norm([lm.log_sigma_base, lm.log_sigma_expiry, lm.log_sigma_tenor])
    gn_reg = grad_norm([lm.W, lm.delta])
    gn_cir = grad_norm([lm.cir.log_kappa, lm.cir.log_theta, lm.cir.log_sigma_v])
    gn_v0  = grad_norm([lm.cir.w_v, lm.cir.alpha_v])

    mean_bias = float(np.mean([d["bias_bp"] for d in batch_diag])) if batch_diag else float('nan')
    lr_now    = optim.param_groups[0]["lr"]
    t_now     = time.perf_counter(); dt_ep=t_now-t_last; t_last=t_now
    eta       = dt_ep*(EPOCHS-epoch-1)
    eta_str   = (f"{int(eta//3600)}h{int((eta%3600)//60):02d}m" if eta>=3600 else
                 f"{int(eta//60)}m{int(eta%60):02d}s" if eta>=60 else f"{int(eta)}s")

    row = {"epoch":epoch,"time_total_sec":round(t_now-t0,1),"time_interval_sec":round(dt_ep,3),
           "loss_vol":ep_vol,"loss_bias":ep_bias,"loss_l2_lam":ep_l2l,
           "loss_l2_vol":ep_l2v,"loss_l2_reg":ep_l2r,
           "loss_l2_vbase":ep_l2vb,"loss_feller":ep_fel,
           "swaption_priced_frac":swp,"path_finite_frac":pth,
           "recon_rmse_bps":avg_rmse,"nan_batches":nan_bat,
           "gnorm_lam":gn_lam,"gnorm_sig":gn_sig,"gnorm_reg":gn_reg,
           "gnorm_cir":gn_cir,"gnorm_v0":gn_v0,
           "lr":lr_now,"fwd_bias_diag_bp":mean_bias,
           "delta_norm":delta_norm,"W_norm":W_norm,
           "w_v_norm":w_v_norm,"alpha_v":alpha_v,
           "kappa":kappa,"theta":theta,"sigma_v":sigma_v,
           "feller":feller,"sqrt_theta":sqrt_theta}
    for i,e in enumerate(EXPIRY_VALS): row[f"lam_{e}Y_norm"]=lam_norms[i]
    for c in ccy_order:
        row[f"rmse_bps_{c}"] = float(rmse_per_ccy.get(c,float('nan'))) if rmse_per_ccy is not None else float('nan')
    pd.DataFrame([row],columns=csv_cols).to_csv(csv_path,mode="a",header=False,index=False)

    if (epoch//1)%HEADER_EVERY==0:
        print(f"\n{'ep':>5} {'vol':>10} {'bias':>8} {'feller':>8} "
              f"{'swp%':>5} {'pth%':>5} {'recon':>6} "
              f"{'|δ|':>6} {'|W|':>6} {'α_v':>6} {'κ':>6} {'√θ':>6} {'σ_v':>6} "
              f"{'lr':>8} {'ETA':>7}")
        print("-"*135)

    diag_str = (" | ".join(f"{d['exp']}x{d['ten']} err={d['err_bp']:+.0f}bp v_T={d['v_T_mean']:.4f}"
                            for d in batch_diag[:3])
                if batch_diag and epoch%DIAG_EVERY==0 else "")
    print(f"{epoch:>5d} {ep_vol:>10.4e} {ep_bias:>8.3e} {feller:>+8.4f} "
          f"{swp*100:>4.0f}% {pth*100:>4.0f}% {avg_rmse:>6.2f} "
          f"{delta_norm:>6.4f} {W_norm:>6.4f} {alpha_v:>+6.3f} "
          f"{kappa:>6.3f} {sqrt_theta:>6.4f} {sigma_v:>6.4f} "
          f"{lr_now:>8.2e} {eta_str:>7}  {diag_str}")

    if (epoch+1)%SAVE_EVERY==0 or epoch==EPOCHS-1:
        ckpt_path = os.path.join(FIGURES_DIR, f"checkpoint_sv_pricing_ep{epoch+1}.pt")
        torch.save({"lm_state_dict": lm.state_dict(),
                    "latent_dim": LATENT_DIM, "expiry_vals": EXPIRY_VALS,
                    "tenor_vals": TENOR_VALS, "epoch": epoch+1,
                    "variant": config.VARIANT}, ckpt_path)
        s = lm.cir.summary()
        print(f"  -> ep{epoch+1}  κ={s['kappa']:.3f}  √θ={s['sqrt_theta']:.4f}  "
              f"σ_v={s['sigma_v']:.4f}  α_v={s['alpha_v']:+.3f}  |w_v|={s['w_v_norm']:.4f}")

print("\nTraining done.")

# ── final save + plots ─────────────────────────────────────────────────────────
torch.save({"lm_state_dict": lm.state_dict(), "latent_dim": LATENT_DIM,
            "expiry_vals": EXPIRY_VALS, "tenor_vals": TENOR_VALS,
            "epochs": EPOCHS, "variant": config.VARIANT},
           os.path.join(FIGURES_DIR, f"checkpoint_sv_pricing_ep{EPOCHS}.pt"))

log_df = pd.read_csv(csv_path)
fig, axes = plt.subplots(4, 1, figsize=(9, 13), dpi=150)
axes[0].semilogy(log_df["loss_vol"],  lw=1, color="darkorange", label="Vol loss")
axes[0].semilogy(log_df["loss_bias"], lw=1, color="deeppink",   label="Bias loss")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(log_df["delta_norm"], lw=1.2, color="#7c3aed", label="|δ|")
axes[1].plot(log_df["W_norm"],     lw=1.2, color="#0891b2", label="|W|", ls="--")
axes[1].plot(log_df["w_v_norm"],   lw=1.2, color="#16a34a", label="|w_v|")
axes[1].set_title("State-conditioning norms"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(log_df["sqrt_theta"], lw=1.5, color="#dc2626", label="√θ")
axes[2].plot(log_df["sigma_v"],    lw=1.2, color="#ea580c", label="σ_v")
axes[2].set_title("CIR vol level + vol-of-vol"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

axes[3].plot(log_df["swaption_priced_frac"]*100, color="firebrick", label="swp priced%")
axes[3].plot(log_df["path_finite_frac"]*100,     color="navy",      label="path finite%")
axes[3].set_ylim(-2, 102); axes[3].legend(); axes[3].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"training_curves_dim{LATENT_DIM}_ep{EPOCHS}.png"), dpi=150)
plt.close(fig)

s = lm.cir.summary()
print("\n" + "="*80)
print("SV PRICING MPR COMPLETE")
print(f"  CIR:   κ={s['kappa']:.4f}  θ={s['theta']:.6f}  √θ={s['sqrt_theta']:.4f}  σ_v={s['sigma_v']:.4f}")
print(f"  v_0:   α_v={s['alpha_v']:+.4f}  |w_v|={s['w_v_norm']:.4f}")
print(f"  Static: |δ|={float(lm.delta.norm()):.4f}  |W|={float(lm.W.norm()):.4f}")
print(f"  Feller (2κθ − σ_v²) = {s['feller']:+.6f}")
print("="*80)
