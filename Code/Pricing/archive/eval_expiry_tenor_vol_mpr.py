# ==================== Expiry-Tenor Vol MPR Evaluation ====================
"""
Evaluation for Training_expiry_tenor_vol_mpr.py checkpoint.

K*_e(z)        = K(z) + L(z) @ lambda_e
sigma_eff(e,n) = exp( log_sigma_base + log_sigma_expiry[e] + log_sigma_tenor[n] )

Each (expiry, tenor) cell uses its own effective diffusion scale.

Output -> Figures/pricing/eval_expiry_tenor_vol_mpr/
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
EXPIRY_VALS = [1, 5, 10]
TENOR_VALS  = [1, 5, 10]
N_PATHS     = 512
TRAIN_FRAC  = 0.70
SEED        = 42
CCY_FILTER  = "EUR"
RATE_CLIP   = 0.50
ANNUITY_MAX = 50.0

PRETRAIN_CKPT = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt")
LM_CKPT       = os.path.join(PROJECT_ROOT, "Figures", "TrainingResults",
                              f"dim{LATENT_DIM}_expiry_tenor_vol_mpr", "ep1000",
                              "checkpoint_expiry_tenor_vol_mpr_ep1000.pt")
OUT_DIR       = os.path.join(PROJECT_ROOT, "Figures", "pricing", "eval_expiry_tenor_vol_mpr")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device   = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)

print(f"Output directory: {OUT_DIR}")
print(f"Checkpoint:       {LM_CKPT}")

# ── modules (must match Training_expiry_tenor_vol_mpr.py exactly) ─────────────

class ExpiryTenorVolMPR(nn.Module):
    """
    K*_e(z)        = K(z) + L(z) @ lambda_e
    sigma_eff(e,n) = exp( log_sigma_base + log_sigma_expiry[e] + log_sigma_tenor[n] )
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

        self.lambda_expiry    = nn.Parameter(torch.zeros(n_exp, latent_dim))
        self.log_sigma_base   = nn.Parameter(torch.full((latent_dim,), -1.8))
        self.log_sigma_expiry = nn.Parameter(torch.zeros(n_exp, latent_dim))
        self.log_sigma_tenor  = nn.Parameter(torch.zeros(n_ten, latent_dim))

    def get_sigma_eff(self, expiry: int, tenor: int) -> torch.Tensor:
        """Effective diffusion scale for a given (expiry, tenor) cell. Returns [d]."""
        e = self.expiry_to_idx[expiry]
        n = self.tenor_to_idx[tenor]
        return (self.log_sigma_base
                + self.log_sigma_expiry[e]
                + self.log_sigma_tenor[n]).exp()

    def drift(self, z_t, expiry: int) -> torch.Tensor:
        with torch.no_grad():
            k_base       = self.kp(z_t)
            sigmas, rhos = self.h(z_t)
            L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam = self.lambda_expiry[self.expiry_to_idx[expiry]].unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    def forward(self, z_t):
        with torch.no_grad():
            k_base       = self.kp(z_t)
            sigmas, rhos = self.h(z_t)
            L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam = self.lambda_expiry.mean(0).unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    @property
    def sigma_vec(self):
        return self.log_sigma_base.exp()


class ExpiryTenorDriftWrapper(nn.Module):
    """Wraps ExpiryTenorVolMPR with fixed (expiry, tenor)."""

    def __init__(self, model, expiry: int, tenor: int):
        super().__init__()
        self.model  = model
        self.expiry = expiry
        self.tenor  = tenor

    def forward(self, z_t):
        return self.model.drift(z_t, self.expiry)

    @property
    def sigma_vec(self):
        return self.model.get_sigma_eff(self.expiry, self.tenor)


# ── load model ─────────────────────────────────────────────────────────────────
print("\nLoading base model ...")
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

lm = ExpiryTenorVolMPR(model.K, model.H, LATENT_DIM, EXPIRY_VALS, TENOR_VALS).to(device)
print(f"Loading checkpoint: {LM_CKPT}")
raw_lm   = torch.load(LM_CKPT, map_location=device, weights_only=False)
lm_state = raw_lm["lm_state_dict"]
lm.load_state_dict(lm_state)
lm.eval()

# Print learned parameters
with torch.no_grad():
    sb = lm.log_sigma_base.exp().cpu().numpy()
    print(f"\nsigma_base = {sb.round(4)}  (mean={sb.mean():.4f})")
    for i, e in enumerate(EXPIRY_VALS):
        lv  = lm.lambda_expiry[i].cpu().numpy()
        lse = lm.log_sigma_expiry[i].cpu().numpy()
        print(f"  {e}Y: lambda={lv.round(4)}  ||λ||={np.linalg.norm(lv):.4f}"
              f"  log_sig_exp_offset={lse.round(4)}")
    for i, t in enumerate(TENOR_VALS):
        lst = lm.log_sigma_tenor[i].cpu().numpy()
        print(f"  {t}Y tenor: log_sig_ten_offset={lst.round(4)}")
    print("\nEffective sigma_eff grid (mean per cell):")
    for e in EXPIRY_VALS:
        row_str = "  ".join(f"{t}Y={lm.get_sigma_eff(e,t).mean().item():.4f}" for t in TENOR_VALS)
        print(f"  {e}Y expiry:  {row_str}")

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
print(f"\nEUR: {len(meta_eur)} curve dates, {len(all_dates)} vol dates")
print(f"Train: {n_train}  Test: {len(test_dates)}")

# ── pricing ────────────────────────────────────────────────────────────────────
def price_cell(date, expiry, tenor):
    if (date not in date_to_idx
            or expiry not in lm.expiry_to_idx
            or tenor  not in lm.tenor_to_idx):
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

    wrapper = ExpiryTenorDriftWrapper(lm, expiry, tenor)

    with torch.no_grad():
        eps_half = torch.randn(half, n_steps, LATENT_DIM, device=device)
        z_T, D_T = simulate_to_expiry_differentiable(
            model, z0, n_steps=n_steps, dt=dt_eff,
            n_paths=N_PATHS, eps=eps_half,
            k_override=wrapper,
            sigma_scale=wrapper.sigma_vec,
            antithetic=True,
            freeze_H=True,
        )

    ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 16:
        return None
    z_k, D_k = z_T[ok], D_T[ok]

    with torch.no_grad():
        # Pass wrapper and sigma_vec so the decoder ODE uses the same pricing
        # dynamics as the simulation (consistency fix — Step 2 of advisor roadmap).
        _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True,
                                       k_override=wrapper,
                                       sigma_scale=wrapper.sigma_vec)
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
print(f"\nPricing {len(all_dates)} dates x 9 cells ...")
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

    if expiry not in EXPIRY_VALS or tenor not in TENOR_VALS:
        continue

    result = price_cell(date, expiry, tenor)
    if counter % 100 == 0:
        print(f"  {counter}/{len(combos)}  ({date.date()}  {expiry}Yx{tenor}Y)  "
              f"elapsed {time.time()-t0:.0f}s")
    if result is None:
        continue

    split = "train" if date in train_dates else "test"
    rows.append({
        "date":            date,
        "expiry":          expiry,
        "tenor":           tenor,
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

# ── summary stats ──────────────────────────────────────────────────────────────
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

CELLS = [(e, t) for e in EXPIRY_VALS for t in TENOR_VALS]

for split_label, split_key in [("TEST SET", "test"), ("TRAIN SET", "train")]:
    df_split = df[df["split"] == split_key]
    print(f"\n-- {split_label} ({len(df_split[['date']].drop_duplicates())} dates) --")
    print(f"{'Cell':>9}  {'MAE':>7}  {'RMSE':>7}  {'Bias':>8}  {'Fwd bias':>10}  {'N':>5}")
    print("-" * 58)
    for e, t in CELLS:
        s = cell_stats(e, t, split=split_key)
        if s:
            print(f"  {e}Yx{t}Y  {s['mae_bp']:>7.1f}  {s['rmse_bp']:>7.1f}  "
                  f"{s['bias_bp']:>+8.1f}  {s['fwd_bias_bp']:>+10.1f}  {s['n']:>5}")
    if len(df_split):
        print(f"  Overall MAE: {df_split['vol_error_bp'].abs().mean():.1f} bp")

overall_mae = df["vol_error_bp"].abs().mean()
print(f"\nAll-dates MAE:    {overall_mae:.1f} bp")
print(f"Mean fwd bias:    {df['forward_bias_bp'].mean():+.1f} bp")
print(f"Mean path finite: {df['path_frac'].mean()*100:.1f}%")

# ── LaTeX tables ───────────────────────────────────────────────────────────────
for split_label2, split_key2, label_suffix in [
    ("all dates", None,    "all"),
    ("test set",  "test",  "test"),
    ("train set", "train", "train"),
]:
    lines = [
        r"\begin{table}[H]", r"\centering",
        (rf"\caption{{Per-cell ATM straddle vol errors: expiry-tenor vol MPR "
         rf"({split_label2}). EUR.}}"),
        rf"\label{{tab:expiry_tenor_vol_mpr_per_cell_{label_suffix}}}",
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
    fname = f"tab_expiry_tenor_vol_mpr_per_cell_{label_suffix}.tex"
    with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {fname}")

# ── figures ────────────────────────────────────────────────────────────────────

def gapped(sub):
    """Return (dates, values) inserting NaN where date gap > 14 days."""
    sub = sub.sort_values("date").copy()
    if len(sub) < 2:
        return sub["date"].tolist(), sub["vol_error_bp"].tolist()
    dates, vals = [], []
    for i, (_, r) in enumerate(sub.iterrows()):
        if i > 0:
            gap = (r["date"] - sub.iloc[i-1]["date"]).days
            if gap > 14:
                dates.append(r["date"] - pd.Timedelta(days=1))
                vals.append(float("nan"))
        dates.append(r["date"])
        vals.append(r["vol_error_bp"])
    return dates, vals

# ── 1. Vol error time series ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
colors = ["#2563eb", "#16a34a", "#dc2626"]
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax  = axes[i][j]
        sub = df[(df["expiry"] == e) & (df["tenor"] == t)].sort_values("date")
        if len(sub) == 0:
            ax.set_visible(False); continue
        test_d = sub[sub["split"] == "test"]["date"]
        if len(test_d):
            ax.axvspan(test_d.min(), test_d.max(), alpha=0.07, color="#f59e0b")
        ax.axhline(0, color="black", lw=0.7, ls="--")
        gd, gv = gapped(sub)
        ax.plot(gd, gv, color=colors[i], lw=1.0, alpha=0.9)
        s = cell_stats(e, t)
        if s:
            ax.text(0.04, 0.96, f"MAE={s['mae_bp']:.0f}bp", transform=ax.transAxes,
                    fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.set_title(f"{e}Yx{t}Y", fontsize=9)
        ax.set_ylabel("Error (bp)", fontsize=7); ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
fig.suptitle("Expiry-Tenor Vol MPR: Vol error over time (amber=test)", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_vol_error_timeseries.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_vol_error_timeseries.png")

# ── 2. MAE heatmaps ─────────────────────────────────────────────────────────────
for split_label3, split_key3, fname3 in [
    ("all dates", None,    "fig_vol_heatmap.png"),
    ("test set",  "test",  "fig_vol_heatmap_test.png"),
    ("train set", "train", "fig_vol_heatmap_train.png"),
]:
    mae_grid = np.full((3, 3), np.nan)
    for i, e in enumerate(EXPIRY_VALS):
        for j, t in enumerate(TENOR_VALS):
            s = cell_stats(e, t, split=split_key3)
            if s:
                mae_grid[i, j] = s["mae_bp"]
    vmax = np.nanmax(mae_grid)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mae_grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Vol MAE (bp)")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"{t}Y" for t in TENOR_VALS])
    ax.set_yticklabels([f"{e}Y" for e in EXPIRY_VALS])
    for ii in range(3):
        for jj in range(3):
            if not np.isnan(mae_grid[ii, jj]):
                ax.text(jj, ii, f"{mae_grid[ii,jj]:.0f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if mae_grid[ii, jj] > vmax * 0.6 else "black")
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
    ax.set_title(f"Expiry-Tenor Vol MPR: Vol MAE (bp) — {split_label3}")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname3), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname3}")

# ── 3. sigma_eff heatmap ────────────────────────────────────────────────────────
with torch.no_grad():
    sig_eff_grid = np.array([[lm.get_sigma_eff(e, t).mean().item()
                               for t in TENOR_VALS] for e in EXPIRY_VALS])

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(sig_eff_grid, cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, label="mean(sigma_eff)")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([f"{t}Y" for t in TENOR_VALS])
ax.set_yticklabels([f"{e}Y" for e in EXPIRY_VALS])
for ii in range(3):
    for jj in range(3):
        ax.text(jj, ii, f"{sig_eff_grid[ii, jj]:.4f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if sig_eff_grid[ii, jj] > sig_eff_grid.max() * 0.7 else "black")
ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
ax.set_title("Learned sigma_eff per (expiry, tenor) cell")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_sigma_eff_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_sigma_eff_heatmap.png")

# ── 4. Lambda bar chart ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
lam_vals    = lm.lambda_expiry.detach().cpu().numpy()
x           = np.arange(LATENT_DIM)
width       = 0.25
colors_exp  = ["#2563eb", "#16a34a", "#dc2626"]
for i, e in enumerate(EXPIRY_VALS):
    ax.bar(x + i*width, lam_vals[i], width, label=f"$\\lambda_{{{e}Y}}$",
           color=colors_exp[i], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels([f"dim {k+1}" for k in range(LATENT_DIM)])
ax.set_ylabel("Value"); ax.set_title("Per-Expiry Lambda Vectors")
ax.axhline(0, color="black", lw=0.6)
ax.legend(); plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_lambda_vectors.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_lambda_vectors.png")

# ── 5. Log-sigma offsets (expiry + tenor, per latent dim) ──────────────────────
fig, axes5 = plt.subplots(1, 2, figsize=(11, 4))

lse_vals = lm.log_sigma_expiry.detach().cpu().numpy()   # [3, 4]
lst_vals = lm.log_sigma_tenor.detach().cpu().numpy()    # [3, 4]
colors_d = ["#7c3aed", "#db2777", "#ea580c", "#0891b2"]

# Expiry offsets
ax = axes5[0]
x  = np.arange(len(EXPIRY_VALS))
for k in range(LATENT_DIM):
    ax.plot(x, lse_vals[:, k], marker="o", lw=1.5, color=colors_d[k], label=f"dim {k+1}")
ax.set_xticks(x); ax.set_xticklabels([f"{e}Y" for e in EXPIRY_VALS])
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_xlabel("Expiry"); ax.set_ylabel("log-sigma offset")
ax.set_title("Expiry log-sigma offsets"); ax.legend(fontsize=8)

# Tenor offsets
ax = axes5[1]
x  = np.arange(len(TENOR_VALS))
for k in range(LATENT_DIM):
    ax.plot(x, lst_vals[:, k], marker="s", lw=1.5, color=colors_d[k], label=f"dim {k+1}")
ax.set_xticks(x); ax.set_xticklabels([f"{t}Y" for t in TENOR_VALS])
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_xlabel("Tenor"); ax.set_ylabel("log-sigma offset")
ax.set_title("Tenor log-sigma offsets"); ax.legend(fontsize=8)

fig.suptitle("Additive log-sigma structure: learned offsets", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "fig_log_sigma_offsets.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig_log_sigma_offsets.png")

# ── final summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("EXPIRY-TENOR VOL MPR — EVALUATION SUMMARY")
print(f"{'='*70}")
print(f"Overall MAE:    {overall_mae:.1f} bp")
print(f"Mean fwd bias:  {df['forward_bias_bp'].mean():+.1f} bp")
print(f"Mean path frac: {df['path_frac'].mean()*100:.1f}%")
print(f"\nPer-expiry MAE (all dates):")
for e in EXPIRY_VALS:
    sub = df[df["expiry"] == e]
    if len(sub):
        print(f"  {e}Y expiry:  {sub['vol_error_bp'].abs().mean():.1f} bp")
print(f"\nPer-tenor MAE (all dates):")
for t in TENOR_VALS:
    sub = df[df["tenor"] == t]
    if len(sub):
        print(f"  {t}Y tenor:  {sub['vol_error_bp'].abs().mean():.1f} bp")
print(f"\nsigma_base: {lm.log_sigma_base.exp().detach().cpu().numpy().round(4)}")
print(f"\nEffective sigma_eff grid:")
for e in EXPIRY_VALS:
    row_str = "  ".join(f"{t}Y={sig_eff_grid[EXPIRY_VALS.index(e), TENOR_VALS.index(t)]:.4f}"
                        for t in TENOR_VALS)
    print(f"  {e}Y expiry:  {row_str}")
print(f"\nAll outputs written to: {OUT_DIR}")
print(f"{'='*70}")
