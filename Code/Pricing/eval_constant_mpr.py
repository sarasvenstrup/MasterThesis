# ==================== Constant MPR Evaluation ====================
"""
Evaluation for Training_constant_mpr.py checkpoint.

K_price(z) = K_base(z) + L_base(z) @ lambda_0
  lambda_0 in R^d  (constant, no position-dependent feedback)
  sigma_vec in R^d  (per-factor diffusion scale)

Prices all EUR dates x 9 cells and saves per_cell_final.csv + figures.
Output -> Figures/pricing/eval_constant_mpr/
"""

import math, os, sys, time
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
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable
from Code.model.sigma_matrix import L_from_sigmas_rhos

# ── settings ───────────────────────────────────────────────────────────────────
LATENT_DIM  = 4
N_PATHS     = 512
TRAIN_FRAC  = 0.70
SEED        = 42
CCY_FILTER  = "EUR"
RATE_CLIP   = 0.50
ANNUITY_MAX = 50.0

PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt")
LM_CKPT       = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              f"dim{LATENT_DIM}_constant_mpr", "ep1000",
                              "checkpoint_constant_mpr_ep1000.pt")
OUT_DIR       = os.path.join(PROJECT_ROOT, "Figures", "pricing", "eval_constant_mpr")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device   = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)

print(f"Output directory: {OUT_DIR}")

# ── module (must match Training_constant_mpr.py exactly) ──────────────────────
class ConstantMPRAdjustment(nn.Module):
    """
    K_price(z) = K_base(z) + L_base(z) @ lambda_0
    lambda_0 in R^d is a CONSTANT vector — no feedback loop.
    """
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

# ── load model ─────────────────────────────────────────────────────────────────
print("\nLoading base model ...")
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

lm = ConstantMPRAdjustment(model.K, model.H, LATENT_DIM).to(device)
print(f"Loading checkpoint: {LM_CKPT}")
raw_lm   = torch.load(LM_CKPT, map_location=device, weights_only=False)
lm_state = raw_lm["lm_state_dict"]
lm.load_state_dict(lm_state)
lm.eval()

l0  = lm.lambda_0.detach().cpu().numpy()
sv  = lm.sigma_vec.detach().cpu().numpy()
print(f"lambda_0  = {l0.round(4)}  (||.||={np.linalg.norm(l0):.4f})")
print(f"sigma_vec = {sv.round(4)}  (mean={sv.mean():.4f})")

# ── data ───────────────────────────────────────────────────────────────────────
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
n_train     = int(len(all_dates) * TRAIN_FRAC)
train_dates = set(all_dates[:n_train])
test_dates  = set(all_dates[n_train:])
print(f"EUR: {len(meta_eur)} curve dates, {len(all_dates)} vol dates")
print(f"Train: {n_train}  Test: {len(test_dates)}")

# ── pricing ────────────────────────────────────────────────────────────────────
def price_cell(date, expiry, tenor):
    if date not in date_to_idx:
        return None
    idx = date_to_idx[date]
    xb  = X_eur[idx:idx + 1].to(device)

    with torch.no_grad():
        z0 = model.encoder(xb)
        _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
        P0 = aux0["P_full"][0]

    max_idx = P0.shape[0] - 1
    if expiry + tenor > max_idx:
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
            k_override=lm,
            sigma_scale=lm.sigma_vec,
            antithetic=True,
            freeze_H=True,
        )

    ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 16:
        return None
    z_k, D_k = z_T[ok], D_T[ok]

    with torch.no_grad():
        _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True)
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

    sigma_str_bp    = (V_pay + V_rec) * 0.5 * sqrt_2pi / (A0 * math.sqrt(expiry)) * 1e4
    forward_bias_bp = (V_pay - V_rec) / A0 * 1e4
    path_frac       = float(ok.float().mean())

    return {
        "sigma_str_bp":    sigma_str_bp,
        "forward_bias_bp": forward_bias_bp,
        "path_frac":       path_frac,
    }

# ── price all dates ────────────────────────────────────────────────────────────
EXPIRY_VALS  = [1, 5, 10]
TENOR_VALS   = [1, 5, 10]
VALID_EXPIRY = set(EXPIRY_VALS)

print(f"\nPricing {len(all_dates)} dates ({n_train} train + {len(test_dates)} test) x 9 cells ...")
t0 = time.time()

combos = df_vol[
    ["as_of_date", "option_maturity", "swap_tenor", "market_vol"]
].drop_duplicates().sort_values("as_of_date")

rows = []
for counter, (_, row) in enumerate(combos.iterrows()):
    date   = pd.Timestamp(row["as_of_date"]).normalize()
    expiry = int(row["option_maturity"])
    tenor  = int(row["swap_tenor"])
    mkt_bp = float(row["market_vol"]) * 1e4

    if expiry not in VALID_EXPIRY:
        continue

    result = price_cell(date, expiry, tenor)
    if counter % 100 == 0:
        print(f"  {counter}/{len(combos)}  ({date.date()}  {expiry}Yx{tenor}Y)  "
              f"elapsed {time.time()-t0:.0f}s")
    if result is None:
        continue

    split = "train" if date in train_dates else "test"
    rows.append({
        "date": date, "expiry": expiry, "tenor": tenor,
        "split":           split,
        "mkt_bp":          mkt_bp,
        "sigma_str_bp":    result["sigma_str_bp"],
        "vol_error_bp":    result["sigma_str_bp"] - mkt_bp,
        "forward_bias_bp": result["forward_bias_bp"],
        "path_frac":       result["path_frac"],
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "per_cell_final.csv"), index=False)
print(f"\nPriced {len(df)} observations in {time.time()-t0:.0f}s")

# ── summary ────────────────────────────────────────────────────────────────────
CELLS = [(e, t) for e in EXPIRY_VALS for t in TENOR_VALS]

def cell_stats(e, t, split=None):
    sub = df[(df["expiry"] == e) & (df["tenor"] == t)]
    if split is not None:
        sub = sub[sub["split"] == split]
    if len(sub) == 0:
        return None
    return {
        "n":           len(sub),
        "mae_bp":      sub["vol_error_bp"].abs().mean(),
        "rmse_bp":     float(np.sqrt((sub["vol_error_bp"]**2).mean())),
        "bias_bp":     sub["vol_error_bp"].mean(),
        "fwd_bias_bp": sub["forward_bias_bp"].mean(),
        "mkt_bp":      sub["mkt_bp"].mean(),
        "mod_bp":      sub["sigma_str_bp"].mean(),
    }

for split_label, split_key in [("TEST SET", "test"), ("TRAIN SET", "train")]:
    df_split = df[df["split"] == split_key]
    print(f"\n-- {split_label} ({len(df_split[['date']].drop_duplicates())} dates, {len(df_split)} obs) --")
    print(f"{'Cell':>9}  {'MAE':>7}  {'RMSE':>7}  {'Bias':>8}  {'Fwd bias':>10}  {'N':>5}")
    print("-" * 58)
    for e, t in CELLS:
        s = cell_stats(e, t, split=split_key)
        if s:
            print(f"  {e}Yx{t}Y  {s['mae_bp']:>7.1f}  {s['rmse_bp']:>7.1f}  "
                  f"{s['bias_bp']:>+8.1f}  {s['fwd_bias_bp']:>+10.1f}  {s['n']:>5}")
    if len(df_split):
        print(f"  Overall MAE:   {df_split['vol_error_bp'].abs().mean():.1f} bp")
        print(f"  Mean fwd bias: {df_split['forward_bias_bp'].mean():+.1f} bp")

overall_mae = df["vol_error_bp"].abs().mean()
print(f"\nAll-dates MAE:    {overall_mae:.1f} bp")
print(f"Mean fwd bias:    {df['forward_bias_bp'].mean():+.1f} bp")
print(f"Mean path finite: {df['path_frac'].mean()*100:.1f}%")
l0_final = lm.lambda_0.detach().cpu().numpy()
sv_final = lm.sigma_vec.detach().cpu().numpy()
print(f"\nlambda_0  = {l0_final.round(4)}  (||.||={np.linalg.norm(l0_final):.4f})")
print(f"sigma_vec = {sv_final.round(4)}  (mean={sv_final.mean():.4f})")

# ── LaTeX table ────────────────────────────────────────────────────────────────
for split_label2, split_key2, label_suffix in [
    ("all dates", None,    "all"),
    ("test set",  "test",  "test"),
    ("train set", "train", "train"),
]:
    lines = [
        r"\begin{table}[H]", r"\centering",
        (rf"\caption{{Per-cell ATM straddle vol errors: constant MPR pricing "
         rf"({split_label2}). EUR.}}"),
        rf"\label{{tab:constant_mpr_per_cell_{label_suffix}}}",
        r"\small",
        r"\begin{tabular}{@{}ccrrrrr@{}}",
        r"\toprule",
        (r"\textbf{Exp} & \textbf{Ten} & \textbf{Mkt (bp)} & \textbf{Mod (bp)} & "
         r"\textbf{MAE (bp)} & \textbf{RMSE (bp)} & \textbf{Fwd bias (bp)} \\"),
        r"\midrule",
    ]
    for i, e in enumerate(EXPIRY_VALS):
        for t in TENOR_VALS:
            s = cell_stats(e, t, split=split_key2)
            if s is None:
                lines.append(f"  {e}Y & {t}Y & --- & --- & --- & --- & --- \\\\")
            else:
                lines.append(
                    f"  {e}Y & {t}Y & {s['mkt_bp']:.0f} & {s['mod_bp']:.0f} "
                    f"& {s['mae_bp']:.1f} & {s['rmse_bp']:.1f} & {s['fwd_bias_bp']:+.1f} \\\\"
                )
        if i < len(EXPIRY_VALS) - 1:
            lines.append(r"  \midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    fname = f"tab_constant_mpr_per_cell_{label_suffix}.tex"
    with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {fname}")

# ── figures ────────────────────────────────────────────────────────────────────

# Scatter: model vs market
fig, axes = plt.subplots(3, 3, figsize=(11, 9))
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax  = axes[i][j]
        sub = df[(df["expiry"] == e) & (df["tenor"] == t)]
        if len(sub) == 0:
            ax.set_visible(False); continue
        for sp, c in [("train", "#16a34a"), ("test", "#2563eb")]:
            ss = sub[sub["split"] == sp]
            ax.scatter(ss["mkt_bp"], ss["sigma_str_bp"], s=12, alpha=0.7,
                       color=c, rasterized=True, label=sp)
        mn = min(sub["mkt_bp"].min(), sub["sigma_str_bp"].min()) * 0.90
        mx = max(sub["mkt_bp"].max(), sub["sigma_str_bp"].max()) * 1.10
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8)
        ax.set_title(f"{e}Yx{t}Y", fontsize=9)
        ax.set_xlabel("Market (bp)", fontsize=7); ax.set_ylabel("Model (bp)", fontsize=7)
        ax.tick_params(labelsize=7)
        s = cell_stats(e, t)
        s_tst = cell_stats(e, t, split="test")
        if s:
            lbl = f"MAE={s['mae_bp']:.0f}bp"
            if s_tst:
                lbl += f"\ntest={s_tst['mae_bp']:.0f}bp"
            ax.text(0.05, 0.95, lbl, transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
axes[0][0].legend(fontsize=6, loc="lower right")
fig.suptitle(r"Constant MPR: Model vs Market Straddle Vol ($K_{\mathrm{price}}=K_{\mathrm{base}}+L\lambda_0$)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_vol_surface.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_vol_surface.png")

# Vol error time series
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
colors = ["#2563eb", "#16a34a", "#dc2626"]
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax  = axes[i][j]
        sub = df[(df["expiry"] == e) & (df["tenor"] == t)].sort_values("date")
        if len(sub) == 0:
            ax.set_visible(False); continue
        test_dates_sub = sub[sub["split"] == "test"]["date"]
        if len(test_dates_sub):
            ax.axvspan(test_dates_sub.min(), test_dates_sub.max(), alpha=0.07, color="#f59e0b")
        ax.plot(sub["date"], sub["vol_error_bp"], color=colors[i], lw=1.0, alpha=0.9)
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.fill_between(sub["date"], sub["vol_error_bp"], 0,
                        where=sub["vol_error_bp"] > 0, alpha=0.15, color="#dc2626")
        ax.fill_between(sub["date"], sub["vol_error_bp"], 0,
                        where=sub["vol_error_bp"] < 0, alpha=0.15, color="#2563eb")
        ax.set_title(f"{e}Yx{t}Y", fontsize=9)
        ax.set_ylabel("Error (bp)", fontsize=7); ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        s = cell_stats(e, t)
        if s:
            ax.text(0.04, 0.96, f"MAE={s['mae_bp']:.0f}bp", transform=ax.transAxes,
                    fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
fig.suptitle("Constant MPR: Vol error over time (amber=test)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_vol_error_timeseries.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_vol_error_timeseries.png")

# Heatmap
for split_label3, split_key3, fname3 in [
    ("all dates", None,    "fig_vol_heatmap.png"),
    ("test set",  "test",  "fig_vol_heatmap_test.png"),
]:
    mae_grid = np.full((3, 3), np.nan)
    for i, e in enumerate(EXPIRY_VALS):
        for j, t in enumerate(TENOR_VALS):
            s = cell_stats(e, t, split=split_key3)
            if s:
                mae_grid[i, j] = s["mae_bp"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mae_grid, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Vol MAE (bp)")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"{t}Y" for t in TENOR_VALS])
    ax.set_yticklabels([f"{e}Y" for e in EXPIRY_VALS])
    for i in range(3):
        for j in range(3):
            if not np.isnan(mae_grid[i, j]):
                ax.text(j, i, f"{mae_grid[i,j]:.0f}", ha="center", va="center",
                        fontsize=11, color="black")
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
    ax.set_title(f"Constant MPR: Vol MAE per Cell (bp), {split_label3}")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname3), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname3}")

# Forward bias
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax  = axes[i][j]
        sub = df[(df["expiry"] == e) & (df["tenor"] == t)].sort_values("date")
        if len(sub) == 0:
            ax.set_visible(False); continue
        test_dates_sub = sub[sub["split"] == "test"]["date"]
        if len(test_dates_sub):
            ax.axvspan(test_dates_sub.min(), test_dates_sub.max(), alpha=0.07, color="#f59e0b")
        ax.plot(sub["date"], sub["forward_bias_bp"], color="#7c3aed", lw=1.0)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"{e}Yx{t}Y", fontsize=9)
        ax.set_ylabel("Fwd bias (bp)", fontsize=7); ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.text(0.04, 0.96, f"mean={sub['forward_bias_bp'].mean():+.0f}bp",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
fig.suptitle("Constant MPR: Forward bias (target: 0 bp, amber=test)",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_forward_bias_timeseries.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("Saved: fig_forward_bias_timeseries.png")

print(f"\nAll outputs written to: {OUT_DIR}")
print(f"\nSummary:")
print(f"  Overall MAE:   {overall_mae:.1f} bp")
print(f"  Fwd bias:      {df['forward_bias_bp'].mean():+.1f} bp")
print(f"  lambda_0:      {l0_final.round(4)}  (||.||={np.linalg.norm(l0_final):.4f})")
print(f"  sigma_vec:     {sv_final.round(4)}  (mean={sv_final.mean():.4f})")
