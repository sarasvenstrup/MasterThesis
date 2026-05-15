"""
Generate swaption implied-vol smile figures for the pricing chapter.

For three representative dates (low-rate / ZIRP / rate-hike) and three
(expiry x tenor) cells, prices payer swaptions at a grid of strikes
  K = F0 + delta,  delta in {-200, -150, ..., +150, +200} bp
using the Constant MPR model, then inverts each MC price to a Bachelier
(normal) implied vol.  Also plots the empirical distribution of F_{T_e}.

Output
------
  Figures/TrainingResults/comparison/fig_pricing_smile.pdf  (.png)
  Figures/TrainingResults/comparison/fig_pricing_smile_dist.pdf  (.png)

Run from repo root:
    python Code/Pricing/_make_smile_figures.py
"""

import math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import optimize
from scipy.stats import norm

# ── path setup ────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import os as _os
_os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

from Code import config
from Code.load_swapdata import my_data
from Code.model.full_model_price import FullModelPrice as FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import swap_rate_torch, forward_swap_rate_torch
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable
from Code.model.sigma_matrix import L_from_sigmas_rhos

# ── settings ──────────────────────────────────────────────────────────────────
LATENT_DIM  = 4
N_PATHS     = 2048          # more paths for cleaner smile
SEED        = 42
CCY_FILTER  = "EUR"
RATE_CLIP   = 0.50
ANNUITY_MAX = 50.0

# Strike offsets in basis points (symmetric around ATM=0)
DELTAS_BP = list(range(-200, 201, 25))

# Target dates per regime (nearest available date will be used)
TARGET_DATES = {
    "Low rate (2015)":  pd.Timestamp("2015-06-30"),
    "ZIRP (2019)":      pd.Timestamp("2019-06-28"),
    "Rate hike (2022)": pd.Timestamp("2022-06-30"),
}

# Cells to show
SMILE_CELLS = [(1, 5), (5, 5), (10, 10)]   # (expiry, tenor)

PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt")
LM_CKPT       = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              f"dim{LATENT_DIM}_constant_mpr", "ep1000",
                              "checkpoint_constant_mpr_ep1000.pt")
OUT_DIR       = os.path.join(PROJECT_ROOT, "Figures", "pricing", "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device   = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)

# ── Constant MPR module (must match Training_constant_mpr.py exactly) ─────────
class ConstantMPRAdjustment(nn.Module):
    def __init__(self, kp_module, h_module, latent_dim):
        super().__init__()
        self.kp         = kp_module
        self.h          = h_module
        self.latent_dim = latent_dim
        self.lambda_0      = nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma_vec = nn.Parameter(torch.full((latent_dim,), -1.8))

    def forward(self, z):
        with torch.no_grad():
            k_base       = self.kp(z)
            sigmas, rhos = self.h(z)
            L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam0 = self.lambda_0.unsqueeze(0).expand(z.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam0)

    @property
    def sigma_vec(self):
        return self.log_sigma_vec.exp()

# ── load model ────────────────────────────────────────────────────────────────
print("Loading base model ...")
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

lm = ConstantMPRAdjustment(model.K, model.H, LATENT_DIM).to(device)
raw_lm   = torch.load(LM_CKPT, map_location=device, weights_only=False)
lm.load_state_dict(raw_lm["lm_state_dict"])
lm.eval()
print(f"lambda_0  = {lm.lambda_0.detach().numpy().round(4)}")
print(f"sigma_vec = {lm.sigma_vec.detach().numpy().round(4)}")

# ── data ──────────────────────────────────────────────────────────────────────
meta, X_tensor, *_ = my_data(use="bbg")
X_tensor = X_tensor.float()
meta_eur = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_eur    = X_tensor[meta["ccy"] == CCY_FILTER]

all_swap_dates = pd.to_datetime(meta_eur["as_of_date"]).dt.normalize().sort_values()
date_to_idx    = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_eur.iterrows()
}

def nearest_date(target, available):
    """Return the closest available date to target."""
    diffs = [(abs((d - target).days), d) for d in available]
    return min(diffs, key=lambda x: x[0])[1]

SELECTED_DATES = {
    label: nearest_date(t, date_to_idx.keys())
    for label, t in TARGET_DATES.items()
}
for label, d in SELECTED_DATES.items():
    print(f"  {label}: using {d.date()}")

# ── Bachelier inversion ───────────────────────────────────────────────────────
def bachelier_payer(F0, K, sigma_N, T):
    """Bachelier payer price per unit (A0=1, N=1).  sigma_N in rate units."""
    if sigma_N <= 0 or T <= 0:
        return max(F0 - K, 0.0)
    d = (F0 - K) / (sigma_N * math.sqrt(T))
    return (F0 - K) * norm.cdf(d) + sigma_N * math.sqrt(T) * norm.pdf(d)

def implied_vol_bachelier(V_norm, F0, K, T):
    """
    Invert Bachelier formula.
    V_norm = V_pay / A0  (normalised price, same units as F0).
    Returns sigma_N in rate units, or nan if not invertible.
    """
    intrinsic = max(F0 - K, 0.0)
    if V_norm <= intrinsic + 1e-12:
        return float("nan")
    def obj(s):
        return bachelier_payer(F0, K, s, T) - V_norm
    try:
        # bracket: upper bound — 10x the ATM vol estimate
        atm_est = V_norm * sqrt_2pi / math.sqrt(T)
        hi = max(atm_est * 10, 2.0)
        lo = 1e-8
        if obj(lo) * obj(hi) > 0:
            return float("nan")
        return optimize.brentq(obj, lo, hi, xtol=1e-12, maxiter=300)
    except Exception:
        return float("nan")

# ── simulate and price smile ──────────────────────────────────────────────────
def run_smile(date, expiry, tenor):
    """
    Simulate N_PATHS paths to expiry and price payer at each delta in DELTAS_BP.
    Returns dict with F0, A0, F_T array, and smile arrays (delta_bp, sigma_bp).
    """
    idx = date_to_idx.get(date)
    if idx is None:
        return None
    xb = X_eur[idx:idx+1].to(device)

    with torch.no_grad():
        z0 = model.encoder(xb)
        _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
        P0 = aux0["P_full"][0]

    if expiry + tenor > P0.shape[0] - 1:
        return None
    F0, A0 = forward_swap_rate_torch(P0, expiry, tenor)
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 1e-6):
        return None

    dt_eff  = min(1.0 / 12.0, expiry / 10.0)
    n_steps = max(12, int(round(expiry / dt_eff)))
    half    = N_PATHS // 2

    with torch.no_grad():
        eps_half = torch.randn(half, n_steps, LATENT_DIM, device=device)
        z_T, D_T = simulate_to_expiry_differentiable(
            model, z0, n_steps=n_steps, dt=dt_eff,
            n_paths=N_PATHS, eps=eps_half,
            k_override=lm, sigma_scale=lm.sigma_vec,
            antithetic=True, freeze_H=True,
        )

    ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 32:
        return None
    z_k, D_k = z_T[ok], D_T[ok]

    with torch.no_grad():
        _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True)
    P_T = aux_T["P_full"]
    dok = torch.isfinite(P_T).all(1)
    if dok.sum() < 32:
        return None

    F_T, A_T = swap_rate_torch(P_T[dok], tenor=tenor)
    D_k = D_k[dok]
    sane = (torch.isfinite(F_T) & torch.isfinite(A_T)
            & (F_T > -RATE_CLIP) & (F_T < RATE_CLIP)
            & (A_T > 1e-6) & (A_T < ANNUITY_MAX))
    if sane.sum() < 32:
        return None
    F_T, A_T, D_k = F_T[sane], A_T[sane], D_k[sane]

    F_T_np = F_T.numpy()
    # effective annuity for normalisation (E[D_T * A_T])
    eff_A0 = float((D_k * A_T).mean())
    if eff_A0 <= 0:
        eff_A0 = A0

    deltas_out, sigmas_out = [], []
    for delta_bp in DELTAS_BP:
        K = F0 + delta_bp / 1e4
        V_pay = float((D_k * A_T * torch.relu(F_T - K)).mean())
        if not math.isfinite(V_pay):
            deltas_out.append(delta_bp)
            sigmas_out.append(float("nan"))
            continue
        V_norm = V_pay / eff_A0
        sigma  = implied_vol_bachelier(V_norm, F0, K, expiry)
        deltas_out.append(delta_bp)
        sigmas_out.append(sigma * 1e4 if math.isfinite(sigma) else float("nan"))

    return {
        "F0":     F0,
        "A0":     A0,
        "F_T":    F_T_np,
        "deltas": deltas_out,
        "sigmas": sigmas_out,
        "expiry": expiry,
        "tenor":  tenor,
    }

# ── Figure 1: smile grid (3 dates x 3 cells) ─────────────────────────────────
print("\nComputing smiles ...")

date_labels = list(SELECTED_DATES.keys())
date_vals   = list(SELECTED_DATES.values())
row_colors  = ["#2563eb", "#16a34a", "#dc2626"]
cell_labels = [f"{e}Yx{t}Y" for e, t in SMILE_CELLS]

fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=False)

for j, (expiry, tenor) in enumerate(SMILE_CELLS):
    for i, (label, date) in enumerate(zip(date_labels, date_vals)):
        ax = axes[i][j]
        print(f"  {label}  {expiry}Yx{tenor}Y  ...", end=" ", flush=True)
        res = run_smile(date, expiry, tenor)

        if res is None:
            print("skipped")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
            ax.set_title(f"{label}\n{expiry}Yx{tenor}Y", fontsize=8)
            continue

        deltas = np.array(res["deltas"])
        sigmas = np.array(res["sigmas"])
        atm_vol = sigmas[deltas == 0][0] if 0 in deltas else float("nan")

        ax.plot(deltas, sigmas, color=row_colors[i], lw=2.0, marker="o",
                markersize=3.5, label="Model smile")
        if math.isfinite(atm_vol):
            ax.axhline(atm_vol, color="black", lw=0.8, ls="--", alpha=0.6,
                       label=f"ATM = {atm_vol:.0f} bp")
        ax.axvline(0, color="grey", lw=0.6, ls=":")

        ax.set_title(f"{label}\n{expiry}Yx{tenor}Y  "
                     f"(F₀ = {res['F0']*100:.2f}%)", fontsize=8, fontweight="bold")
        ax.set_xlabel("Strike offset (bp)", fontsize=7)
        ax.set_ylabel("Implied normal vol (bp)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right")
        print(f"ATM={atm_vol:.1f} bp")

fig.suptitle("Model-implied Bachelier Vol Smile — Constant MPR\n"
             "(rows = rate regime, columns = swaption cell)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
out1 = os.path.join(OUT_DIR, "fig_pricing_smile.pdf")
fig.savefig(out1, bbox_inches="tight")
fig.savefig(out1.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out1}")

# ── Figure 2: empirical distribution of F_T vs fitted normal ─────────────────
print("\nComputing distributions ...")

fig, axes = plt.subplots(3, 3, figsize=(13, 9))

for j, (expiry, tenor) in enumerate(SMILE_CELLS):
    for i, (label, date) in enumerate(zip(date_labels, date_vals)):
        ax = axes[i][j]
        print(f"  {label}  {expiry}Yx{tenor}Y  ...", end=" ", flush=True)
        res = run_smile(date, expiry, tenor)

        if res is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
            ax.set_title(f"{label}\n{expiry}Yx{tenor}Y", fontsize=8)
            continue

        F_T_bp = res["F_T"] * 1e4
        F0_bp  = res["F0"]  * 1e4

        # ATM vol from smile
        deltas = np.array(res["deltas"])
        sigmas = np.array(res["sigmas"])
        atm_vol = sigmas[deltas == 0][0] if 0 in deltas else float("nan")

        # Empirical histogram
        ax.hist(F_T_bp, bins=60, density=True, alpha=0.55,
                color=row_colors[i], label="Model paths")

        # Fitted normal with ATM vol
        if math.isfinite(atm_vol):
            x_range = np.linspace(F_T_bp.min(), F_T_bp.max(), 300)
            sigma_bp_sqrt_T = atm_vol * math.sqrt(expiry)
            pdf_normal = norm.pdf(x_range, loc=F0_bp, scale=sigma_bp_sqrt_T)
            ax.plot(x_range, pdf_normal, color="black", lw=1.5, ls="--",
                    label=f"Bachelier N(F₀, σ√T)\nσ = {atm_vol:.0f} bp")

        ax.axvline(F0_bp, color="black", lw=1.0, ls=":", alpha=0.7,
                   label=f"F₀ = {res['F0']*100:.2f}%")

        ax.set_title(f"{label}\n{expiry}Yx{tenor}Y", fontsize=8, fontweight="bold")
        ax.set_xlabel(r"$F_{T_e}$ (bp)", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right")
        print("done")

fig.suptitle("Empirical Distribution of $F_{T_e}$ vs Bachelier Normal — Constant MPR\n"
             "(rows = rate regime, columns = swaption cell; dashed = fitted normal at ATM vol)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
out2 = os.path.join(OUT_DIR, "fig_pricing_smile_dist.pdf")
fig.savefig(out2, bbox_inches="tight")
fig.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")

print(f"\nAll smile figures written to: {OUT_DIR}")
