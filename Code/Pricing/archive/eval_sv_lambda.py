# ==================== SV+Lambda MPR Evaluation ====================
"""
Post-training evaluation of the CIR Stochastic Vol + Lambda MPR model.

Loads the final checkpoint from Training_sv_lambda.py and prices the full
EUR swaption test set. Reports per-cell ATM vol errors over historical dates.

Outputs -> Figures/TrainingResults/dim4_sv_lambda/ep1000/eval/
  per_cell_final.csv              raw (date, expiry, tenor) results
  tab_sv_lambda_per_cell.tex      LaTeX table: per-cell MAE/RMSE/bias/fwd-bias
  fig_vol_surface.png             3x3 scatter: model vs market vol
  fig_vol_error_timeseries.png    3x3 time-series: vol error over calendar dates
  fig_vol_heatmap.png             heatmap of vol MAE per cell
  fig_forward_bias_timeseries.png forward bias over calendar time
  fig_cir_vol_level.png           sqrt(theta) convergence diagnostic
"""

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
from Code.model.sigma_matrix import L_from_sigmas_rhos

# ================================================================
# Settings
# ================================================================

LATENT_DIM  = 4
N_PATHS     = 512      # antithetic: 256+256
TRAIN_FRAC  = 0.70
SEED        = 42
CCY_FILTER  = "EUR"

RATE_CLIP   = 0.50
ANNUITY_MAX = 50.0

CKPT_DIR      = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              f"dim{LATENT_DIM}_sv_lambda", "ep1000")
PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt")
SV_CKPT       = os.path.join(CKPT_DIR, "checkpoint_sv_lambda_ep1000.pt")
OUT_DIR       = os.path.join(CKPT_DIR, "eval")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device   = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)

print(f"Output directory: {OUT_DIR}")

# ================================================================
# SVLambdaMPR module (must match Training_sv_lambda.py)
# ================================================================

class SVLambdaMPR(nn.Module):
    """
    K^Q(z) = K^P(z) - L(z) @ Lambda @ z
    CIR: dv = kappa*(theta-v)*dt + sigma_v*sqrt(v)*dW_v
    """

    def __init__(self, kp_module, h_module, latent_dim,
                 log_kappa=0.0, log_theta=-4.02, log_sigma_v=-3.0):
        super().__init__()
        self.kp         = kp_module
        self.h          = h_module
        self.latent_dim = latent_dim
        self.Lambda     = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.log_kappa   = nn.Parameter(torch.tensor(float(log_kappa)))
        self.log_theta   = nn.Parameter(torch.tensor(float(log_theta)))
        self.log_sigma_v = nn.Parameter(torch.tensor(float(log_sigma_v)))

    def forward(self, z):
        with torch.no_grad():
            mu_p = self.kp(z)
            sigmas, rhos = self.h(z)
            L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam        = torch.matmul(self.Lambda, z.unsqueeze(-1)).squeeze(-1)
        correction = torch.einsum('bij,bj->bi', L, lam)
        return mu_p - correction

    @property
    def kappa(self):   return self.log_kappa.exp()
    @property
    def theta(self):   return self.log_theta.exp()
    @property
    def sigma_v(self): return self.log_sigma_v.exp()


# ================================================================
# Load base model
# ================================================================

print(f"\nLoading base model ...")
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

# ================================================================
# Load SV+Lambda checkpoint
# ================================================================

print(f"Loading SV+Lambda checkpoint: {SV_CKPT}")
raw_sv = torch.load(SV_CKPT, map_location=device, weights_only=False)

sv_lambda = SVLambdaMPR(
    kp_module=model.K,
    h_module=model.H,
    latent_dim=LATENT_DIM,
    log_kappa=float(raw_sv["log_kappa"]),
    log_theta=float(raw_sv["log_theta"]),
    log_sigma_v=float(raw_sv["log_sigma_v"]),
).to(device)

lm = raw_sv.get("Lambda_matrix")
if lm is not None:
    with torch.no_grad():
        sv_lambda.Lambda.copy_(lm)

sv_lambda.eval()

kappa_f   = float(sv_lambda.kappa.detach())
theta_f   = float(sv_lambda.theta.detach())
sigma_v_f = float(sv_lambda.sigma_v.detach())

print(f"Lambda loaded: ||Lambda||_F = {sv_lambda.Lambda.norm().item():.4f}")
print(f"CIR params:  kappa={kappa_f:.4f}  theta={theta_f:.6f}  "
      f"sigma_v={sigma_v_f:.5f}")
print(f"Vol level:   sqrt(theta) = {theta_f**0.5:.5f}  "
      f"(Feller OK: {2*kappa_f*theta_f:.5f} > {sigma_v_f**2:.5f}? "
      f"{'YES' if 2*kappa_f*theta_f > sigma_v_f**2 else 'NO'})")

# ================================================================
# Load data
# ================================================================

meta, X_tensor, *_ = my_data(use="bbg")
X_tensor = X_tensor.float()
meta_eur = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_eur    = X_tensor[meta["ccy"] == CCY_FILTER]

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0

dates_swap  = set(pd.to_datetime(meta_eur["as_of_date"]).dt.normalize())
df_vol      = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_eur.iterrows()
}

all_dates   = sorted(df_vol["as_of_date"].unique())
test_dates  = set(all_dates)   # full sample — price everything
print(f"\nEUR: {len(meta_eur)} curve dates, {len(all_dates)} vol dates")
print(f"Pricing all {len(test_dates)} dates (full in-sample)")

# ================================================================
# Pricing function (mirrors Training_sv_lambda simulate_sv_to_expiry)
# ================================================================

def price_cell(date, expiry, tenor):
    if date not in date_to_idx:
        return None
    idx = date_to_idx[date]
    xb  = X_eur[idx:idx + 1].to(device)

    with torch.no_grad():
        z0 = model.encoder(xb)

    # Time-0 quantities under K^P
    with torch.no_grad():
        _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
    F0, A0 = forward_swap_rate_torch(aux0["P_full"][0], expiry, tenor)
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 1e-6):
        return None

    dt_eff  = min(1.0 / 6.0, expiry / 10.0)
    n_steps = max(12, int(round(expiry / dt_eff)))
    half    = N_PATHS // 2
    sqrt_dt = math.sqrt(dt_eff)

    eps_z = torch.randn(half, n_steps, LATENT_DIM, device=device)
    eps_v = torch.randn(half, n_steps,              device=device)

    with torch.no_grad():
        # ---- inline simulation (same logic as training) ----
        kappa   = sv_lambda.kappa.item()
        theta   = sv_lambda.theta.item()
        sigma_v = sv_lambda.sigma_v.item()

        n_paths = 2 * half
        z = z0.expand(n_paths, -1).clone()
        v = torch.full((half,), theta, device=device)

        r_prev = model.R(z).squeeze(-1)
        log_D  = torch.zeros(n_paths, device=device)

        for t in range(n_steps):
            v_safe = v.clamp(min=1e-10)
            dv = (kappa * (theta - v_safe) * dt_eff
                  + sigma_v * v_safe.sqrt() * sqrt_dt * eps_v[:, t])
            v  = (v_safe + dv).clamp(min=1e-10)

            sqrt_v_all = torch.cat([v.sqrt(), v.sqrt()], dim=0)
            eps_z_t    = torch.cat([eps_z[:, t], -eps_z[:, t]], dim=0)
            dW         = eps_z_t * sqrt_dt

            sigmas, rhos = model.H(z.detach())
            L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
            shock = sqrt_v_all.unsqueeze(-1) * torch.bmm(L, dW.unsqueeze(-1)).squeeze(-1)
            drift = sv_lambda(z) * dt_eff

            z = z + drift + shock

            r_next = model.R(z.detach()).squeeze(-1)
            log_D  = log_D - 0.5 * (r_prev + r_next) * dt_eff
            r_prev = r_next

        D_T = log_D.clamp(-30, 30).exp()

    ok = torch.isfinite(z).all(1) & torch.isfinite(D_T)
    if ok.sum() < 16:
        return None
    z_k, D_k = z[ok], D_T[ok]

    with torch.no_grad():
        _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True,
                                        k_override=sv_lambda)
    P_T = aux_T["P_full"]
    dok = torch.isfinite(P_T).all(1)
    if dok.sum() < 16:
        return None

    F_T, A_T = swap_rate_torch(P_T[dok], tenor=tenor)
    D_k = D_k[dok]
    sane = (torch.isfinite(F_T) & torch.isfinite(A_T)
            & (F_T > -RATE_CLIP) & (F_T < RATE_CLIP)
            & (A_T > 1e-6) & (A_T < ANNUITY_MAX))
    if sane.sum() < 16:
        return None
    F_T, A_T, D_k = F_T[sane], A_T[sane], D_k[sane]

    V_pay = float((D_k * A_T * torch.relu(F_T - F0)).mean())
    V_rec = float((D_k * A_T * torch.relu(F0 - F_T)).mean())
    if not (math.isfinite(V_pay) and math.isfinite(V_rec)):
        return None

    sigma_pay_bp     = V_pay * sqrt_2pi / (A0 * math.sqrt(expiry)) * 1e4
    forward_bias_bp  = (V_pay - V_rec) / A0 * 1e4
    path_frac        = float(ok.float().mean())

    return {
        "F0": F0, "A0": A0,
        "sigma_pay_bp":    sigma_pay_bp,
        "forward_bias_bp": forward_bias_bp,
        "path_frac":       path_frac,
    }


# ================================================================
# Price all test dates
# ================================================================

print(f"\nPricing {len(test_dates)} test dates x 9 cells ...")
t0 = time.time()

combos = df_vol[df_vol["as_of_date"].isin(test_dates)][
    ["as_of_date", "option_maturity", "swap_tenor", "market_vol"]
].drop_duplicates().sort_values("as_of_date")

EXPIRY_VALS = [1, 5, 10]
combos = combos[combos["option_maturity"].isin(EXPIRY_VALS)]

rows = []
for counter, (_, row) in enumerate(combos.iterrows()):
    date   = pd.Timestamp(row["as_of_date"]).normalize()
    expiry = int(row["option_maturity"])
    tenor  = int(row["swap_tenor"])
    mkt_bp = float(row["market_vol"]) * 1e4

    result = price_cell(date, expiry, tenor)
    if counter % 100 == 0:
        print(f"  {counter}/{len(combos)}  ({date.date()}  {expiry}Yx{tenor}Y)  "
              f"elapsed {time.time()-t0:.0f}s")
    if result is None:
        continue

    rows.append({
        "date":            date,
        "expiry":          expiry,
        "tenor":           tenor,
        "mkt_bp":          mkt_bp,
        "sigma_pay_bp":    result["sigma_pay_bp"],
        "vol_error_bp":    result["sigma_pay_bp"] - mkt_bp,
        "forward_bias_bp": result["forward_bias_bp"],
        "path_frac":       result["path_frac"],
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "per_cell_final.csv"), index=False)
print(f"\nPriced {len(df)} observations in {time.time()-t0:.0f}s")

# ================================================================
# Cell summary
# ================================================================

TENOR_VALS = [1, 5, 10]
CELLS      = [(e, t) for e in EXPIRY_VALS for t in TENOR_VALS]

def cell_stats(expiry, tenor):
    sub = df[(df["expiry"] == expiry) & (df["tenor"] == tenor)]
    if len(sub) == 0:
        return None
    return {
        "n":           len(sub),
        "mae_bp":      sub["vol_error_bp"].abs().mean(),
        "rmse_bp":     float(np.sqrt((sub["vol_error_bp"] ** 2).mean())),
        "bias_bp":     sub["vol_error_bp"].mean(),
        "fwd_bias_bp": sub["forward_bias_bp"].mean(),
        "mkt_mean_bp": sub["mkt_bp"].mean(),
        "mod_mean_bp": sub["sigma_pay_bp"].mean(),
    }

print(f"\n{'Cell':>9}  {'MAE':>7}  {'RMSE':>7}  {'Bias':>8}  {'Fwd bias':>10}  {'N':>5}")
print("-" * 58)
for e, t in CELLS:
    s = cell_stats(e, t)
    if s:
        print(f"  {e}Yx{t}Y  {s['mae_bp']:>7.1f}  {s['rmse_bp']:>7.1f}  "
              f"{s['bias_bp']:>+8.1f}  {s['fwd_bias_bp']:>+10.1f}  {s['n']:>5}")

overall_mae = df["vol_error_bp"].abs().mean()
print(f"\nOverall MAE:       {overall_mae:.1f} bp")
print(f"Mean fwd bias:     {df['forward_bias_bp'].mean():+.1f} bp")
print(f"Mean path finite:  {df['path_frac'].mean()*100:.1f}%")
print(f"\nCIR params:  kappa={kappa_f:.4f}  "
      f"sqrt(theta)={theta_f**0.5:.5f}  sigma_v={sigma_v_f:.5f}")

# ================================================================
# LaTeX table
# ================================================================

lines = [
    r"\begin{table}[H]",
    r"\centering",
    (r"\caption{Per-cell ATM vol errors: CIR SV + $\Lambda$-MPR model (ep\,999, EUR full sample). "
     r"CIR long-run vol $\sqrt{\theta}=" + f"{theta_f**0.5:.4f}" + r"$, "
     r"$\kappa=" + f"{kappa_f:.3f}" + r"$, "
     r"$\sigma_v=" + f"{sigma_v_f:.4f}" + r"$. "
     r"Forward bias benchmark: 0 bp.}"),
    r"\label{tab:sv_lambda_per_cell_final}",
    r"\small",
    r"\begin{tabular}{@{}ccrrrrr@{}}",
    r"\toprule",
    (r"\textbf{Exp} & \textbf{Ten} & \textbf{Mkt vol (bp)} & "
     r"\textbf{Mod vol (bp)} & \textbf{MAE (bp)} & \textbf{RMSE (bp)} & "
     r"\textbf{Fwd bias (bp)} \\"),
    r"\midrule",
]

for i, e in enumerate(EXPIRY_VALS):
    for t in TENOR_VALS:
        s = cell_stats(e, t)
        if s is None:
            lines.append(f"  {e}Y & {t}Y & --- & --- & --- & --- & --- \\\\")
        else:
            lines.append(
                f"  {e}Y & {t}Y & {s['mkt_mean_bp']:.0f} & {s['mod_mean_bp']:.0f} "
                f"& {s['mae_bp']:.1f} & {s['rmse_bp']:.1f} & {s['fwd_bias_bp']:+.1f} \\\\"
            )
    if i < len(EXPIRY_VALS) - 1:
        lines.append(r"  \midrule")

lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(OUT_DIR, "tab_sv_lambda_per_cell.tex"), "w") as f:
    f.write("\n".join(lines))
print("Saved: tab_sv_lambda_per_cell.tex")

# ================================================================
# Figure 1: Scatter model vs market vol
# ================================================================

fig, axes = plt.subplots(3, 3, figsize=(11, 9))
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax  = axes[i][j]
        sub = df[(df["expiry"] == e) & (df["tenor"] == t)]
        if len(sub) == 0:
            ax.set_visible(False); continue
        ax.scatter(sub["mkt_bp"], sub["sigma_pay_bp"],
                   s=14, alpha=0.7, color="#2563eb", rasterized=True)
        mn = min(sub["mkt_bp"].min(), sub["sigma_pay_bp"].min()) * 0.92
        mx = max(sub["mkt_bp"].max(), sub["sigma_pay_bp"].max()) * 1.08
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8)
        ax.set_title(f"{e}Yx{t}Y", fontsize=9)
        ax.set_xlabel("Market vol (bp)", fontsize=7)
        ax.set_ylabel("Model vol (bp)", fontsize=7)
        ax.tick_params(labelsize=7)
        s = cell_stats(e, t)
        if s:
            ax.text(0.05, 0.95, f"MAE={s['mae_bp']:.0f}bp",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle(r"CIR SV + $\Lambda$-MPR: Model vs Market ATM Vol, EUR full sample",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_vol_surface.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_vol_surface.png")

# ================================================================
# Figure 2: Vol error time-series
# ================================================================

fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=False)
colors = ["#2563eb", "#16a34a", "#dc2626"]

for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax  = axes[i][j]
        sub = df[(df["expiry"] == e) & (df["tenor"] == t)].sort_values("date")
        if len(sub) == 0:
            ax.set_visible(False); continue
        ax.plot(sub["date"], sub["vol_error_bp"], color=colors[i], lw=1.0, alpha=0.8)
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.fill_between(sub["date"], sub["vol_error_bp"], 0,
                        where=sub["vol_error_bp"] > 0, alpha=0.15, color="#dc2626")
        ax.fill_between(sub["date"], sub["vol_error_bp"], 0,
                        where=sub["vol_error_bp"] < 0, alpha=0.15, color="#2563eb")
        ax.set_title(f"{e}Yx{t}Y", fontsize=9)
        ax.set_ylabel("Error (bp)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        s = cell_stats(e, t)
        if s:
            ax.text(0.04, 0.96, f"MAE={s['mae_bp']:.0f}bp",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle(r"CIR SV + $\Lambda$-MPR: Vol error (model $-$ market) over time",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_vol_error_timeseries.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_vol_error_timeseries.png")

# ================================================================
# Figure 3: Vol MAE heatmap
# ================================================================

mae_grid = np.zeros((3, 3))
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        s = cell_stats(e, t)
        mae_grid[i, j] = s["mae_bp"] if s else float("nan")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(mae_grid, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="Vol MAE (bp)")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([f"{t}Y tenor" for t in TENOR_VALS])
ax.set_yticklabels([f"{e}Y expiry" for e in EXPIRY_VALS])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{mae_grid[i,j]:.0f}",
                ha="center", va="center", fontsize=11, color="black")
ax.set_title(r"CIR SV + $\Lambda$-MPR: Vol MAE per Cell (bp)", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_vol_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_vol_heatmap.png")

# ================================================================
# Figure 4: Forward bias time-series
# ================================================================

fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=False)
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax  = axes[i][j]
        sub = df[(df["expiry"] == e) & (df["tenor"] == t)].sort_values("date")
        if len(sub) == 0:
            ax.set_visible(False); continue
        ax.plot(sub["date"], sub["forward_bias_bp"], color="#7c3aed", lw=1.0, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8, ls="--", label="No bias")
        ax.set_title(f"{e}Yx{t}Y", fontsize=9)
        ax.set_ylabel("Fwd bias (bp)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        mean_b = sub["forward_bias_bp"].mean()
        ax.text(0.04, 0.96, f"mean={mean_b:+.0f}bp",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle(r"CIR SV + $\Lambda$-MPR: Forward bias over time (benchmark: 0)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_forward_bias_timeseries.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_forward_bias_timeseries.png")

# ================================================================
# Figure 5: CIR convergence (sqrt(theta) level diagnostic)
# ================================================================

log_csv = os.path.join(CKPT_DIR, f"train_sv_lambda_log_dim{LATENT_DIM}_ep1000.csv")
if os.path.isfile(log_csv):
    df_log = pd.read_csv(log_csv)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].semilogy(df_log["loss_price"].values, lw=1.0, color="darkorange")
    axes[0, 0].set_title("Pricing loss"); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel("Epoch")

    if "sqrt_theta" in df_log.columns:
        axes[0, 1].plot(df_log["sqrt_theta"].values, lw=1.2, color="purple",
                        label="sqrt(theta)")
        axes[0, 1].set_title("CIR vol level sqrt(theta)")
        axes[0, 1].set_xlabel("Epoch"); axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

    if "kappa" in df_log.columns:
        axes[1, 0].plot(df_log["kappa"].values, lw=1.0, color="green", label="kappa")
        axes[1, 0].plot(df_log["sigma_v"].values, lw=1.0, color="orange", label="sigma_v")
        axes[1, 0].set_title("CIR kappa and sigma_v")
        axes[1, 0].set_xlabel("Epoch"); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

    if "Lambda_norm_fro" in df_log.columns:
        axes[1, 1].plot(df_log["Lambda_norm_fro"].values, lw=1.2, color="purple",
                        label="||Lambda||_F")
        axes[1, 1].set_title("Lambda Frobenius norm")
        axes[1, 1].set_xlabel("Epoch"); axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

    fig.suptitle(r"CIR SV + $\Lambda$-MPR Training Diagnostics", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_cir_vol_level.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig_cir_vol_level.png")

print(f"\nAll outputs written to: {OUT_DIR}")
print(f"\nSummary:")
print(f"  Overall MAE:    {overall_mae:.1f} bp")
print(f"  Mean fwd bias:  {df['forward_bias_bp'].mean():+.1f} bp")
print(f"  sqrt(theta):    {theta_f**0.5:.5f}")
print(f"  kappa:          {kappa_f:.4f}  ({1/kappa_f:.1f}yr mean-reversion)")
print(f"  sigma_v:        {sigma_v_f:.5f}")
