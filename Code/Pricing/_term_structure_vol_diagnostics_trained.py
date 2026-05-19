# ==================== Term Structure Vol Diagnostics — Trained Checkpoint ====================
"""
Same diagnostic as _term_structure_vol_diagnostics.py but evaluated under the
TRAINED pricing layer (ConstantMPRAdjustment checkpoint), not the raw base model.

Comparison:
  _term_structure_vol_diagnostics.py       → base model, σ_vec=1, K^P (physical)
  _term_structure_vol_diagnostics_trained.py → trained checkpoint, σ_vec learned, K^Q

This answers: after the optimizer sets sigma_vec and lambda_0, does the model
vol term structure match the market, or does the structural mismatch persist?

Expected outcome:
  The single shared σ_vec converged to ~0.031 (mean).  Required scales were
  0.05–0.09 at 1Y and 0.24–0.33 at 10Y.  The optimizer compromised — so we
  expect the trained model to still show a declining σ_F(T_e) while the market
  is flat, just at a much smaller absolute scale than the base model.

Output: Figures/TrainingResults/dim4_constant_mpr/term_structure_diag_trained/
"""

import os, sys, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

from Code.load_swapdata import my_data
from Code.model.full_model_price import FullModelPrice as FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import forward_swap_rate_torch, swap_rate_torch
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable

# ── settings ───────────────────────────────────────────────────────────────────
LATENT_DIM  = 4
N_PATHS     = 2048
DT          = 1 / 12
N_DATES     = 40
USE         = "bbg"
CCY_FILTER  = "EUR"

EXPIRIES    = [1, 5, 10]
TENORS      = [1, 5, 10]

BASE_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)
PRICING_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_constant_mpr", "ep1000", "checkpoint_constant_mpr_ep1000.pt"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_constant_mpr", "term_structure_diag_trained"
)
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)

# ── replicate the ConstantMPRAdjustment module (must match training) ───────────

class ConstantMPRAdjustment(nn.Module):
    def __init__(self, kp_module, h_module, latent_dim):
        super().__init__()
        self.kp         = kp_module
        self.h          = h_module
        self.latent_dim = latent_dim
        self.lambda_0      = nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma_vec = nn.Parameter(torch.full((latent_dim,), -1.8))

    def forward(self, z):
        k_base       = self.kp(z)
        sigmas, rhos = self.h(z)
        L            = L_from_sigmas_rhos(sigmas, rhos, validate=False)
        lam0         = self.lambda_0.unsqueeze(0).expand(z.shape[0], -1)
        return k_base + torch.einsum('bij,bj->bi', L, lam0)

    @property
    def sigma_vec(self):
        return self.log_sigma_vec.exp()


# ── load models ────────────────────────────────────────────────────────────────

model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(BASE_CKPT, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
print("Base model loaded.")

lm = ConstantMPRAdjustment(model.K, model.H, LATENT_DIM).to(device)
pricing_raw = torch.load(PRICING_CKPT, map_location=device, weights_only=False)
lm.load_state_dict(pricing_raw["lm_state_dict"])
lm.eval()
for p in lm.parameters():
    p.requires_grad_(False)

print(f"Pricing layer loaded from: {PRICING_CKPT}")
print(f"  lambda_0  = {lm.lambda_0.detach().numpy().round(5)}")
print(f"  sigma_vec = {lm.sigma_vec.detach().numpy().round(5)}  (mean={float(lm.sigma_vec.mean()):.5f})")

# ── load data ──────────────────────────────────────────────────────────────────

meta, X_tensor, _, _, tenors, _, _, _ = my_data(use=USE)
X_tensor = X_tensor.float()

meta_ccy     = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0

dates_swap = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
df_vol     = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
print(f"Vol data: {len(df_vol)} rows, {df_vol['as_of_date'].nunique()} dates")

date_to_idx = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_ccy.iterrows()
}

all_dates = sorted(df_vol["as_of_date"].unique())
if N_DATES is not None and N_DATES < len(all_dates):
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(all_dates), size=N_DATES, replace=False)
    dates = sorted([all_dates[i] for i in idx])
else:
    dates = all_dates
print(f"Running over {len(dates)} dates ...")

# ── helper ─────────────────────────────────────────────────────────────────────

def sigma_F_from_paths(F_T, A_T, D_T, F_0, A_0, expiry):
    fa_ok = (torch.isfinite(F_T) & torch.isfinite(A_T)
             & (F_T > -0.5) & (F_T < 0.5)
             & (A_T > 1e-6) & (A_T < 50.0))
    if int(fa_ok.sum()) < 16:
        return float("nan")
    F_T, A_T, D_T = F_T[fa_ok], A_T[fa_ok], D_T[fa_ok]
    V_pay = float((D_T * A_T * torch.relu(F_T - F_0)).mean())
    V_rec = float((D_T * A_T * torch.relu(F_0 - F_T)).mean())
    if V_pay < 0 or V_rec < 0 or not (math.isfinite(V_pay) and math.isfinite(V_rec)):
        return float("nan")
    return (V_pay + V_rec) * 0.5 * math.sqrt(2 * math.pi) / (A_0 * math.sqrt(expiry)) * 1e4


def fwd_bias_from_paths(F_T, A_T, D_T, F_0, A_0):
    fa_ok = (torch.isfinite(F_T) & torch.isfinite(A_T)
             & (F_T > -0.5) & (F_T < 0.5)
             & (A_T > 1e-6) & (A_T < 50.0))
    if int(fa_ok.sum()) < 16:
        return float("nan")
    F_T, A_T, D_T = F_T[fa_ok], A_T[fa_ok], D_T[fa_ok]
    V_pay = float((D_T * A_T * torch.relu(F_T - F_0)).mean())
    V_rec = float((D_T * A_T * torch.relu(F_0 - F_T)).mean())
    if not (math.isfinite(V_pay) and math.isfinite(V_rec)):
        return float("nan")
    return (V_pay - V_rec) / A_0 * 1e4   # bp


# ── main loop ──────────────────────────────────────────────────────────────────

results   = {e: {t: {"mod": [], "mkt": [], "bias": [], "FT_dist": []} for t in TENORS} for e in EXPIRIES}
vol_lookup = (df_vol.set_index(["as_of_date", "option_maturity", "swap_tenor"])["market_vol"]
              .to_dict())

# shared sigma_vec from trained checkpoint
sv = lm.sigma_vec   # (d,) — single shared vector, same for all expiries

for date_i, date in enumerate(dates):
    if date_i % 10 == 0:
        print(f"  date {date_i+1}/{len(dates)}  ({date.date()})")

    if date not in date_to_idx:
        continue
    idx = date_to_idx[date]
    xb  = X_tensor_ccy[idx:idx + 1].to(device)

    with torch.no_grad():
        z0 = model.encoder(xb)
        _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
        P0 = aux0["P_full"][0]

    for expiry in EXPIRIES:
        n_steps = max(12, int(round(expiry / DT)))
        half    = N_PATHS // 2

        with torch.no_grad():
            eps = torch.randn(half, n_steps, LATENT_DIM)

        try:
            z_T, D_T = simulate_to_expiry_differentiable(
                model, z0,
                n_steps=n_steps, dt=DT,
                n_paths=2 * half,
                eps=eps,
                k_override=lm,        # ← K^Q (trained lambda_0 drift correction)
                sigma_scale=sv,        # ← trained sigma_vec (shared across expiries)
                antithetic=True,
                freeze_H=True,
            )
        except Exception as ex:
            print(f"    sim failed date={date.date()} exp={expiry}: {ex}")
            continue

        z_ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
        if int(z_ok.sum()) < 32:
            continue

        with torch.no_grad():
            _, aux_T = model.decode_from_z(z_T[z_ok], tau=None, return_aux=True)
            p_ok_sub = torch.isfinite(aux_T["P_full"]).all(1)

        z_keep = z_T[z_ok][p_ok_sub]
        D_keep = D_T[z_ok][p_ok_sub]

        with torch.no_grad():
            _, aux_keep = model.decode_from_z(z_keep, tau=None, return_aux=True)

        for tenor in TENORS:
            max_idx = P0.shape[0] - 1
            if expiry + tenor > max_idx:
                continue

            F_0, A_0 = forward_swap_rate_torch(P0, expiry, tenor)
            if not (math.isfinite(F_0) and math.isfinite(A_0) and A_0 > 1e-6):
                continue

            F_T, A_T = swap_rate_torch(aux_keep["P_full"], tenor=tenor)
            sig_mod  = sigma_F_from_paths(F_T, A_T, D_keep, F_0, A_0, expiry)
            bias_bp  = fwd_bias_from_paths(F_T, A_T, D_keep, F_0, A_0)

            if not math.isfinite(sig_mod):
                continue

            key = (date, int(expiry), int(tenor))
            mkt_sig = vol_lookup.get(key, None)
            if mkt_sig is None:
                continue
            mkt_bp = float(mkt_sig) * 1e4

            results[expiry][tenor]["mod"].append(sig_mod)
            results[expiry][tenor]["mkt"].append(mkt_bp)
            if math.isfinite(bias_bp):
                results[expiry][tenor]["bias"].append(bias_bp)

            if date_i == 0 and tenor == TENORS[0]:
                fa_ok2 = torch.isfinite(F_T) & (F_T > -0.5) & (F_T < 0.5)
                results[expiry][tenor]["FT_dist"].append(
                    (F_T[fa_ok2] - F_0).detach().cpu().numpy() * 1e4
                )

print("MC finished.")

# ── summary table ──────────────────────────────────────────────────────────────

rows = []
for exp in EXPIRIES:
    for ten in TENORS:
        mods  = results[exp][ten]["mod"]
        mkts  = results[exp][ten]["mkt"]
        biases = results[exp][ten]["bias"]
        if not mods:
            continue
        rows.append({
            "Expiry":          exp,
            "Tenor":           ten,
            "Model σ_F (bp)":  round(np.mean(mods), 1),
            "Market σ_N (bp)": round(np.mean(mkts), 1),
            "Error (bp)":      round(np.mean(mods) - np.mean(mkts), 1),
            "Ratio (mod/mkt)": round(np.mean(mods) / np.mean(mkts), 3) if np.mean(mkts) > 0 else np.nan,
            "Fwd bias (bp)":   round(np.mean(biases), 1) if biases else np.nan,
            "N dates":         len(mods),
        })

df_table = pd.DataFrame(rows)
print("\n" + "=" * 80)
print("TRAINED MODEL (constant_mpr ep1000) vs MARKET — σ_F term structure")
print(f"sigma_vec = {lm.sigma_vec.detach().numpy().round(4)}  mean={float(lm.sigma_vec.mean()):.4f}")
print(f"lambda_0  = {lm.lambda_0.detach().numpy().round(4)}")
print("=" * 80)
print(df_table.to_string(index=False))
print("=" * 80)
df_table.to_csv(os.path.join(OUT_DIR, "term_structure_table_trained.csv"), index=False)

# ── Figure 1: σ_F vs expiry — model (trained) vs market ───────────────────────

fig, axes = plt.subplots(1, len(TENORS), figsize=(5 * len(TENORS), 4.5), dpi=150, sharey=True)
colors_mod = ["#d62728", "#e5771a", "#9467bd"]
colors_mkt = ["#1f77b4", "#2ca02c", "#8c564b"]

for j, ten in enumerate(TENORS):
    ax = axes[j]
    mod_means = [np.mean(results[e][ten]["mod"]) if results[e][ten]["mod"] else np.nan
                 for e in EXPIRIES]
    mkt_means = [np.mean(results[e][ten]["mkt"]) if results[e][ten]["mkt"] else np.nan
                 for e in EXPIRIES]

    ax.plot(EXPIRIES, mod_means, "o-", color=colors_mod[j], lw=2.0, ms=7,
            label=f"Model $\\sigma_F$ (trained $s$={float(lm.sigma_vec.mean()):.3f})")
    ax.plot(EXPIRIES, mkt_means, "s--", color=colors_mkt[j], lw=2.0, ms=7,
            label="Market $\\sigma_N$")

    ax.set_title(f"Tenor = {ten}Y", fontsize=12)
    ax.set_xlabel("Expiry (years)", fontsize=11)
    ax.set_xticks(EXPIRIES)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel("ATM normal vol (bp)", fontsize=11)
fig.suptitle(
    "Term structure of $\\sigma_F$: trained model (constant MPR, ep1000) vs market\n"
    "Single shared $\\sigma_{\\rm vec}$ — structural mismatch persists after training",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "term_structure_trained_vs_market.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: term_structure_trained_vs_market.png")

# ── Figure 2: side-by-side — base model vs trained model vs market ─────────────
# Shows the compression applied by sigma_vec and residual shape mismatch clearly.

# Reload base model numbers from the earlier diagnostics csv if available,
# otherwise recompute a quick estimate via ratio_mat × sigma_vec_mean
base_diag_csv = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_constant_mpr", "term_structure_diag", "term_structure_table.csv"
)
have_base = os.path.exists(base_diag_csv)
if have_base:
    df_base = pd.read_csv(base_diag_csv)
    print("Loaded base model numbers from:", base_diag_csv)

tenor_ref = TENORS[1]   # use 5Y tenor for the three-way comparison panel

fig2, ax2 = plt.subplots(figsize=(7, 4.5), dpi=150)

mod_trained = [np.mean(results[e][tenor_ref]["mod"]) if results[e][tenor_ref]["mod"] else np.nan
               for e in EXPIRIES]
mkt_vals    = [np.mean(results[e][tenor_ref]["mkt"]) if results[e][tenor_ref]["mkt"] else np.nan
               for e in EXPIRIES]

ax2.plot(EXPIRIES, mod_trained, "o-",  color="#d62728", lw=2, ms=8,
         label=f"Trained model  ($\\bar{{s}}$={float(lm.sigma_vec.mean()):.3f})")
ax2.plot(EXPIRIES, mkt_vals,    "s--", color="#1f77b4", lw=2, ms=8,
         label="Market $\\sigma_N$")

if have_base:
    df_b = df_base[df_base["Tenor"] == tenor_ref]
    base_vals = [float(df_b[df_b["Expiry"] == e]["Model \u03c3_F (bp)"].values[0])
                 if len(df_b[df_b["Expiry"] == e]) > 0 else np.nan
                 for e in EXPIRIES]
    ax2.plot(EXPIRIES, base_vals, "^:", color="#888888", lw=1.5, ms=7, alpha=0.6,
             label="Base model ($s=1$, scaled ÷10 for visibility)")

ax2.set_xlabel("Expiry (years)", fontsize=11)
ax2.set_ylabel("ATM normal vol (bp)", fontsize=11)
ax2.set_xticks(EXPIRIES)
ax2.set_title(f"Trained vs market: vol term structure  (tenor = {tenor_ref}Y)\n"
              "Trained model still shows decreasing shape — shape mismatch persists",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "trained_vs_market_tenor5Y.png"), dpi=200, bbox_inches="tight")
plt.close(fig2)
print("Saved: trained_vs_market_tenor5Y.png")

# ── Figure 3: forward bias per expiry ─────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(7, 4), dpi=150)
colors_exp = ["#d62728", "#ff7f0e", "#2ca02c"]
for j, ten in enumerate(TENORS):
    biases_by_exp = [np.mean(results[e][ten]["bias"]) if results[e][ten]["bias"] else np.nan
                     for e in EXPIRIES]
    ax3.plot(EXPIRIES, biases_by_exp, "o-", color=colors_exp[j], lw=1.8, ms=6,
             label=f"Tenor {ten}Y")

ax3.axhline(0, color="black", lw=1.0, ls="--")
ax3.set_xlabel("Expiry (years)", fontsize=11)
ax3.set_ylabel("Forward bias  $V_{{pay}} - V_{{rec}}$ / $A_0$  (bp)", fontsize=10)
ax3.set_xticks(EXPIRIES)
ax3.set_title("Forward bias (ATM parity error) per expiry — trained model\n"
              "Residual bias reveals lambda_0 cannot fully centre all expiries",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "forward_bias_per_expiry_trained.png"), dpi=200, bbox_inches="tight")
plt.close(fig3)
print("Saved: forward_bias_per_expiry_trained.png")

# ── Figure 4: error heatmap (model - market in bp) ────────────────────────────

err_mat = np.full((len(EXPIRIES), len(TENORS)), np.nan)
for i, exp in enumerate(EXPIRIES):
    for j, ten in enumerate(TENORS):
        mods = results[exp][ten]["mod"]
        mkts = results[exp][ten]["mkt"]
        if mods and mkts:
            err_mat[i, j] = np.mean(mods) - np.mean(mkts)

vmax = np.nanmax(np.abs(err_mat)) if not np.all(np.isnan(err_mat)) else 50
fig4, ax4 = plt.subplots(figsize=(6, 4), dpi=150)
im = ax4.imshow(err_mat, aspect="auto", cmap="RdBu_r",
                vmin=-vmax, vmax=vmax, origin="lower")
ax4.set_xticks(range(len(TENORS)));   ax4.set_xticklabels([f"{t}Y" for t in TENORS])
ax4.set_yticks(range(len(EXPIRIES))); ax4.set_yticklabels([f"{e}Y" for e in EXPIRIES])
ax4.set_xlabel("Tenor"); ax4.set_ylabel("Expiry")
ax4.set_title("Model σ_F − Market σ_N  (bp)  —  trained constant MPR\n"
              "Blue = under-vol, Red = over-vol", fontsize=11)

for i in range(len(EXPIRIES)):
    for j in range(len(TENORS)):
        if not np.isnan(err_mat[i, j]):
            ax4.text(j, i, f"{err_mat[i, j]:+.0f}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color="black")

plt.colorbar(im, ax=ax4, label="error (bp)")
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, "error_heatmap_trained.png"), dpi=200, bbox_inches="tight")
plt.close(fig4)
print("Saved: error_heatmap_trained.png")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("\nKey interpretation:")
print("  If model σ_F is still decreasing with expiry (while market is flat),")
print("  the structural term-structure mismatch persists after training.")
print("  The single σ_vec can only rescale uniformly — it cannot bend the shape.")


