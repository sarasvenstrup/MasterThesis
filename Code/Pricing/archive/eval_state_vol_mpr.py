# ==================== State-Conditioned Vol MPR Evaluation ====================
"""
Evaluation for Training_state_vol_mpr.py checkpoint.

sigma_eff(e,n,z0) = exp( base + expiry_off[e] + tenor_off[n] + delta*tanh(W@z0) )

Output -> Figures/pricing/eval_state_vol_mpr/
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
                              f"dim{LATENT_DIM}_state_vol_mpr", "ep1000",
                              "checkpoint_state_vol_mpr_ep1000.pt")
OUT_DIR       = os.path.join(PROJECT_ROOT, "Figures", "pricing", "eval_state_vol_mpr")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)
device   = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)
print(f"Checkpoint: {LM_CKPT}")
print(f"Output:     {OUT_DIR}")

# ── module definitions (must match Training_state_vol_mpr.py) ──────────────────

class StateCondVolMPR(nn.Module):
    def __init__(self, kp_module, h_module, latent_dim, expiry_vals, tenor_vals):
        super().__init__()
        self.kp = kp_module; self.h = h_module
        self.latent_dim = latent_dim
        self.expiry_vals = expiry_vals; self.tenor_vals = tenor_vals
        self.expiry_to_idx = {e: i for i, e in enumerate(expiry_vals)}
        self.tenor_to_idx  = {t: i for i, t in enumerate(tenor_vals)}
        n_exp = len(expiry_vals); n_ten = len(tenor_vals)
        self.lambda_expiry    = nn.Parameter(torch.zeros(n_exp, latent_dim))
        self.log_sigma_base   = nn.Parameter(torch.full((latent_dim,), -1.8))
        self.log_sigma_expiry = nn.Parameter(torch.zeros(n_exp, latent_dim))
        self.log_sigma_tenor  = nn.Parameter(torch.zeros(n_ten, latent_dim))
        self.W     = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.delta = nn.Parameter(torch.zeros(latent_dim))

    def get_sigma_eff(self, expiry, tenor, z0):
        e = self.expiry_to_idx[expiry]; n = self.tenor_to_idx[tenor]
        z = z0.squeeze(0) if z0.dim() == 2 else z0
        regime = self.delta * torch.tanh(self.W @ z)
        return (self.log_sigma_base + self.log_sigma_expiry[e]
                + self.log_sigma_tenor[n] + regime).exp()

    def drift(self, z_t, expiry):
        with torch.no_grad():
            k_base = self.kp(z_t); sigmas, rhos = self.h(z_t)
            L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam = self.lambda_expiry[self.expiry_to_idx[expiry]].unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    def forward(self, z_t):
        with torch.no_grad():
            k_base = self.kp(z_t); sigmas, rhos = self.h(z_t)
            L = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam = self.lambda_expiry.mean(0).unsqueeze(0).expand(z_t.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam)

    @property
    def sigma_vec(self): return self.log_sigma_base.exp()


class StateCondDriftWrapper(nn.Module):
    def __init__(self, model, expiry, tenor, z0):
        super().__init__()
        self.model = model; self.expiry = expiry; self.tenor = tenor; self._z0 = z0

    def forward(self, z_t): return self.model.drift(z_t, self.expiry)

    @property
    def sigma_vec(self): return self.model.get_sigma_eff(self.expiry, self.tenor, self._z0)


# ── load model ─────────────────────────────────────────────────────────────────
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
model.load_state_dict(raw.get("model_state_dict", raw))
for p in model.parameters(): p.requires_grad_(False)
model.eval()

lm = StateCondVolMPR(model.K, model.H, LATENT_DIM, EXPIRY_VALS, TENOR_VALS).to(device)
raw_lm = torch.load(LM_CKPT, map_location=device, weights_only=False)
lm.load_state_dict(raw_lm["lm_state_dict"])
lm.eval()

with torch.no_grad():
    print(f"delta     = {lm.delta.numpy().round(4)}  (|δ|={float(lm.delta.norm()):.4f})")
    print(f"W norm    = {float(lm.W.norm()):.4f}")
    print(f"sigma_base= {lm.sigma_vec.numpy().round(4)}")
    for i, e in enumerate(EXPIRY_VALS):
        lv = lm.lambda_expiry[i].numpy()
        print(f"lambda_{e}Y = {lv.round(4)}")

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
date_to_idx = {pd.Timestamp(r["as_of_date"]).normalize(): i for i, r in meta_eur.iterrows()}

all_dates   = sorted(df_vol["as_of_date"].unique())
n_train     = int(len(all_dates) * TRAIN_FRAC)
train_dates = set(all_dates[:n_train])
test_dates  = set(all_dates[n_train:])
print(f"EUR: {len(meta_eur)} curve dates, {len(all_dates)} vol dates")
print(f"Train: {n_train}  Test: {len(test_dates)}")

# ── pricing ────────────────────────────────────────────────────────────────────
def price_cell(date, expiry, tenor):
    if (date not in date_to_idx
            or expiry not in lm.expiry_to_idx
            or tenor  not in lm.tenor_to_idx):
        return None
    idx = date_to_idx[date]
    xb  = X_eur[idx:idx+1].to(device)
    with torch.no_grad():
        z0 = model.encoder(xb)
        _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
        P0 = aux0["P_full"][0]
    if expiry + tenor > P0.shape[0] - 1: return None
    F0, A0 = forward_swap_rate_torch(P0, expiry, tenor)
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 1e-6): return None

    dt_eff  = min(1.0/12.0, expiry/10.0)
    n_steps = max(12, int(round(expiry/dt_eff)))
    half    = N_PATHS // 2
    wrapper = StateCondDriftWrapper(lm, expiry, tenor, z0)

    with torch.no_grad():
        eps_half = torch.randn(half, n_steps, LATENT_DIM, device=device)
        z_T, D_T = simulate_to_expiry_differentiable(
            model, z0, n_steps=n_steps, dt=dt_eff, n_paths=N_PATHS, eps=eps_half,
            k_override=wrapper, sigma_scale=wrapper.sigma_vec,
            antithetic=True, freeze_H=True)
    ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 16: return None
    z_k, D_k = z_T[ok], D_T[ok]
    with torch.no_grad():
        _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True,
                                       k_override=wrapper, sigma_scale=wrapper.sigma_vec)
    P_T = aux_T["P_full"]
    dok = torch.isfinite(P_T).all(1)
    if dok.sum() < 16: return None
    F_T, A_T = swap_rate_torch(P_T[dok], tenor=tenor)
    D_k = D_k[dok]
    sane = (torch.isfinite(F_T) & torch.isfinite(A_T)
            & (F_T > -RATE_CLIP) & (F_T < RATE_CLIP)
            & (A_T > 1e-6) & (A_T < ANNUITY_MAX))
    if sane.sum() < 16: return None
    F_T, A_T, D_k = F_T[sane], A_T[sane], D_k[sane]
    V_pay = float((D_k * A_T * torch.relu(F_T - F0)).mean())
    V_rec = float((D_k * A_T * torch.relu(F0 - F_T)).mean())
    if not (math.isfinite(V_pay) and math.isfinite(V_rec)): return None
    return {
        "sigma_str_bp":    (V_pay+V_rec)*0.5*sqrt_2pi/(A0*math.sqrt(expiry))*1e4,
        "forward_bias_bp": (V_pay-V_rec)/A0*1e4,
        "path_frac":       float(ok.float().mean()),
        "sigma_eff_mean":  float(wrapper.sigma_vec.detach().mean()),
    }

# ── price all dates ────────────────────────────────────────────────────────────
print(f"\nPricing {len(all_dates)} dates x 9 cells ...")
t0 = time.time()
combos = df_vol[["as_of_date","option_maturity","swap_tenor","market_vol"]].drop_duplicates().sort_values("as_of_date")
rows = []
for counter, (_, row) in enumerate(combos.iterrows()):
    date   = pd.Timestamp(row["as_of_date"]).normalize()
    expiry = int(row["option_maturity"]); tenor = int(row["swap_tenor"])
    mkt_bp = float(row["market_vol"]) * 1e4
    if expiry not in EXPIRY_VALS or tenor not in TENOR_VALS: continue
    result = price_cell(date, expiry, tenor)
    if counter % 100 == 0:
        print(f"  {counter}/{len(combos)}  {date.date()} {expiry}Yx{tenor}Y  {time.time()-t0:.0f}s")
    if result is None: continue
    rows.append({"date": date, "expiry": expiry, "tenor": tenor,
                 "split": "train" if date in train_dates else "test",
                 "mkt_bp": mkt_bp, "sigma_str_bp": result["sigma_str_bp"],
                 "vol_error_bp": result["sigma_str_bp"] - mkt_bp,
                 "forward_bias_bp": result["forward_bias_bp"],
                 "path_frac": result["path_frac"],
                 "sigma_eff_mean": result["sigma_eff_mean"]})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "per_cell_final.csv"), index=False)
print(f"Priced {len(df)} obs in {time.time()-t0:.0f}s")

# ── summary ────────────────────────────────────────────────────────────────────
def cell_stats(e, t, split=None):
    sub = df[(df["expiry"]==e) & (df["tenor"]==t)]
    if split: sub = sub[sub["split"]==split]
    if len(sub) == 0: return None
    return {"n": len(sub), "mae_bp": sub["vol_error_bp"].abs().mean(),
            "rmse_bp": float(np.sqrt((sub["vol_error_bp"]**2).mean())),
            "bias_bp": sub["vol_error_bp"].mean(),
            "fwd_bias_bp": sub["forward_bias_bp"].mean(),
            "mkt_bp": sub["mkt_bp"].mean(), "mod_bp": sub["sigma_str_bp"].mean()}

CELLS = [(e,t) for e in EXPIRY_VALS for t in TENOR_VALS]
for split_label, split_key in [("TEST SET","test"),("TRAIN SET","train")]:
    df_s = df[df["split"]==split_key]
    print(f"\n-- {split_label} ({len(df_s[['date']].drop_duplicates())} dates) --")
    for e, t in CELLS:
        s = cell_stats(e, t, split=split_key)
        if s: print(f"  {e}Yx{t}Y  MAE={s['mae_bp']:.1f}  bias={s['bias_bp']:+.1f}  N={s['n']}")
    if len(df_s): print(f"  Overall MAE: {df_s['vol_error_bp'].abs().mean():.1f} bp")

overall_mae = df["vol_error_bp"].abs().mean()
print(f"\nAll-dates MAE: {overall_mae:.1f} bp")

# ── LaTeX tables ───────────────────────────────────────────────────────────────
for split_label2, split_key2, label_suffix in [("all dates",None,"all"),("test set","test","test"),("train set","train","train")]:
    lines = [r"\begin{table}[H]", r"\centering",
             rf"\caption{{Per-cell vol errors: state-conditioned vol MPR ({split_label2}). EUR.}}",
             rf"\label{{tab:state_vol_mpr_per_cell_{label_suffix}}}",
             r"\small", r"\begin{tabular}{@{}ccrrrrr@{}}", r"\toprule",
             r"\textbf{Exp} & \textbf{Ten} & \textbf{Mkt (bp)} & \textbf{Mod (bp)} & \textbf{MAE (bp)} & \textbf{RMSE (bp)} & \textbf{Fwd bias (bp)} \\",
             r"\midrule"]
    for i, e in enumerate(EXPIRY_VALS):
        for t in TENOR_VALS:
            s = cell_stats(e, t, split=split_key2)
            if s is None: lines.append(f"  {e}Y & {t}Y & --- & --- & --- & --- & --- \\\\")
            else: lines.append(f"  {e}Y & {t}Y & {s['mkt_bp']:.0f} & {s['mod_bp']:.0f} & {s['mae_bp']:.1f} & {s['rmse_bp']:.1f} & {s['fwd_bias_bp']:+.1f} \\\\")
        if i < len(EXPIRY_VALS)-1: lines.append(r"  \midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    fname = f"tab_state_vol_mpr_per_cell_{label_suffix}.tex"
    with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f: f.write("\n".join(lines))
    print(f"Saved: {fname}")

# ── figures ────────────────────────────────────────────────────────────────────
def gapped(sub):
    sub = sub.sort_values("date").copy()
    if len(sub) < 2: return sub["date"].tolist(), sub["vol_error_bp"].tolist()
    dates, vals = [], []
    for i, (_, r) in enumerate(sub.iterrows()):
        if i > 0 and (r["date"] - sub.iloc[i-1]["date"]).days > 14:
            dates.append(r["date"] - pd.Timedelta(days=1)); vals.append(float("nan"))
        dates.append(r["date"]); vals.append(r["vol_error_bp"])
    return dates, vals

# Time series
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
colors = ["#2563eb","#16a34a","#dc2626"]
for i, e in enumerate(EXPIRY_VALS):
    for j, t in enumerate(TENOR_VALS):
        ax = axes[i][j]
        sub = df[(df["expiry"]==e)&(df["tenor"]==t)].sort_values("date")
        if len(sub) == 0: ax.set_visible(False); continue
        test_d = sub[sub["split"]=="test"]["date"]
        if len(test_d): ax.axvspan(test_d.min(), test_d.max(), alpha=0.07, color="#f59e0b")
        ax.axhline(0, color="black", lw=0.7, ls="--")
        gd, gv = gapped(sub)
        ax.plot(gd, gv, color=colors[i], lw=1.0)
        s = cell_stats(e, t)
        if s: ax.text(0.04, 0.96, f"MAE={s['mae_bp']:.0f}bp", transform=ax.transAxes,
                      fontsize=7, va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.set_title(f"{e}Yx{t}Y", fontsize=9)
        ax.set_ylabel("Error (bp)", fontsize=7); ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
fig.suptitle("State-Conditioned Vol MPR: Vol error (amber=test)", fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_vol_error_timeseries.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print("Saved: fig_vol_error_timeseries.png")

# MAE heatmaps
for split_label3, split_key3, fname3 in [("all dates",None,"fig_vol_heatmap.png"),("test set","test","fig_vol_heatmap_test.png"),("train set","train","fig_vol_heatmap_train.png")]:
    mae_grid = np.full((3,3), np.nan)
    for i,e in enumerate(EXPIRY_VALS):
        for j,t in enumerate(TENOR_VALS):
            s = cell_stats(e, t, split=split_key3)
            if s: mae_grid[i,j] = s["mae_bp"]
    vmax = np.nanmax(mae_grid)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(mae_grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Vol MAE (bp)")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"{t}Y" for t in TENOR_VALS]); ax.set_yticklabels([f"{e}Y" for e in EXPIRY_VALS])
    for ii in range(3):
        for jj in range(3):
            if not np.isnan(mae_grid[ii,jj]):
                ax.text(jj,ii,f"{mae_grid[ii,jj]:.0f}",ha="center",va="center",fontsize=11,fontweight="bold",
                        color="white" if mae_grid[ii,jj]>vmax*0.6 else "black")
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
    ax.set_title(f"State-Cond Vol MPR: MAE — {split_label3}")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname3), dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"Saved: {fname3}")

# Regime term: sigma_eff vs date (scatter per cell)
fig, ax = plt.subplots(figsize=(10, 4))
colors_e = ["#2563eb","#16a34a","#dc2626"]
for i, e in enumerate(EXPIRY_VALS):
    sub = df[df["expiry"]==e].sort_values("date")
    if len(sub): ax.scatter(sub["date"], sub["sigma_eff_mean"], s=3, color=colors_e[i],
                            alpha=0.5, label=f"{e}Y expiry")
ax.set_xlabel("Date"); ax.set_ylabel("mean(sigma_eff)")
ax.set_title("State-conditioned sigma_eff over time — regime variation")
ax.legend(markerscale=4); plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_sigma_eff_timeseries.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print("Saved: fig_sigma_eff_timeseries.png")

print(f"\nAll outputs: {OUT_DIR}")
print(f"Overall MAE: {overall_mae:.1f} bp")
print(f"|delta|={float(lm.delta.norm()):.4f}  |W|={float(lm.W.norm()):.4f}")
print("(Large norms → yield curve carries vol-regime info)")
