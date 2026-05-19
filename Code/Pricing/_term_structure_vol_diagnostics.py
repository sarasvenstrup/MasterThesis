# ==================== Term Structure of Model Variance Diagnostics ====================
"""
Diagnoses the structural mismatch between model σ_F(T_e) and market σ_N(T_e).

Finding: the stable base model produces a *strongly decreasing* σ_F with expiry
because its latent SDE is mean-reverting (Vasicek-like), causing Var(z_T) to
saturate.  Market ATM normal vols are roughly *flat* across expiries.  No single
rescaling of σ_F (via the σ_vec pricing parameter) can reconcile the two shapes.

This script:
  1. Loads the frozen base model (no pricing adjustment, σ_vec = 1).
  2. For each (expiry, tenor) cell over a sample of dates, runs N_PATHS Monte
     Carlo paths under the *physical* measure (K^P, σ_vec = 1) to compute:
         σ_F_model  = straddle-implied bp vol  =  (V_pay + V_rec)/2
                       × √(2π) / (A_0 √T)  × 1e4
  3. Reads the market σ_N for the same cells.
  4. Produces:
       - A table: mean σ_F_model and mean σ_N_mkt per (expiry, tenor) cell
       - Figure 1: σ_F vs expiry for each tenor (model vs market)
       - Figure 2: σ_F / σ_N ratio surface — shows where the model over/under-vol
       - Figure 3: Distribution of F_T - F_0 per expiry — visualises the
                   Var(z_T) saturation effect directly

Output: Figures/TrainingResults/dim4_constant_mpr/term_structure_diag/
"""

import os, sys, math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
from Code.Simulation.simulate_model import simulate_to_expiry_differentiable

# ── settings ───────────────────────────────────────────────────────────────────
LATENT_DIM  = 4
N_PATHS     = 2048          # MC paths per cell
DT          = 1 / 12        # Euler step (monthly)
N_DATES     = 40            # number of dates to sample (None = all)
USE         = "bbg"
CCY_FILTER  = "EUR"

EXPIRIES    = [1, 5, 10]    # option maturities in years
TENORS      = [1, 5, 10]    # swap tenors in years

PRETRAIN_CKPT = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_constant_mpr", "term_structure_diag"
)
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)

# ── load model ─────────────────────────────────────────────────────────────────
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
print("Base model loaded (frozen, no pricing adjustment).")

# ── load swap / vol data ────────────────────────────────────────────────────────
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT \
    = my_data(use=USE)
X_tensor = X_tensor.float()

meta_ccy     = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_tensor_ccy = X_tensor[meta["ccy"] == CCY_FILTER]

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0   # convert bp → fractional (consistent with training)

dates_swap = set(pd.to_datetime(meta_ccy["as_of_date"]).dt.normalize())
df_vol     = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
print(f"Vol data: {len(df_vol)} rows, {df_vol['as_of_date'].nunique()} dates")

date_to_idx = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_ccy.iterrows()
}

# ── select dates ───────────────────────────────────────────────────────────────
all_dates = sorted(df_vol["as_of_date"].unique())
if N_DATES is not None and N_DATES < len(all_dates):
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(all_dates), size=N_DATES, replace=False)
    dates = sorted([all_dates[i] for i in idx])
else:
    dates = all_dates
print(f"Running diagnostics over {len(dates)} dates ...")

# ── helpers ────────────────────────────────────────────────────────────────────

def sigma_F_from_paths(F_T: torch.Tensor, A_T: torch.Tensor,
                       D_T: torch.Tensor, F_0: float, A_0: float,
                       expiry: int) -> float:
    """Straddle-implied normal vol in bps."""
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
    V_str = (V_pay + V_rec) * 0.5
    return V_str * math.sqrt(2 * math.pi) / (A_0 * math.sqrt(expiry)) * 1e4


# ── main loop ──────────────────────────────────────────────────────────────────
# results[exp][ten] = list of (model_sig, mkt_sig, F_T_array)
results   = {e: {t: {"mod": [], "mkt": [], "FT_dist": []} for t in TENORS} for e in EXPIRIES}
vol_lookup = (df_vol.set_index(["as_of_date", "option_maturity", "swap_tenor"])["market_vol"]
              .to_dict())

for date_i, date in enumerate(dates):
    if date_i % 10 == 0:
        print(f"  date {date_i+1}/{len(dates)}  ({date.date()})")

    if date not in date_to_idx:
        continue
    idx = date_to_idx[date]
    xb  = X_tensor_ccy[idx:idx + 1].to(device)

    with torch.no_grad():
        z0 = model.encoder(xb)                          # (1, d)
        _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
        P0 = aux0["P_full"][0]                          # (max_tau+1,)

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
                k_override=None,    # physical measure, no MPR adjustment
                sigma_scale=None,   # σ_vec = 1 (base model, unscaled)
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
            if not math.isfinite(sig_mod):
                continue

            key = (date, int(expiry), int(tenor))
            mkt_sig = vol_lookup.get(key, None)
            if mkt_sig is None:
                continue
            mkt_bp = float(mkt_sig) * 1e4

            results[expiry][tenor]["mod"].append(sig_mod)
            results[expiry][tenor]["mkt"].append(mkt_bp)

            # Store a sample of F_T - F_0 values (first date, first tenor)
            if date_i == 0 and tenor == TENORS[0]:
                fa_ok = (torch.isfinite(F_T) & (F_T > -0.5) & (F_T < 0.5))
                results[expiry][tenor]["FT_dist"].append(
                    (F_T[fa_ok] - F_0).detach().cpu().numpy() * 1e4
                )

print("MC finished.")

# ── summary table ──────────────────────────────────────────────────────────────
rows = []
for exp in EXPIRIES:
    for ten in TENORS:
        mods = results[exp][ten]["mod"]
        mkts = results[exp][ten]["mkt"]
        if not mods:
            continue
        rows.append({
            "Expiry": exp,
            "Tenor":  ten,
            "Model σ_F (bp)": round(np.mean(mods), 1),
            "Market σ_N (bp)": round(np.mean(mkts), 1),
            "Ratio (mod/mkt)": round(np.mean(mods) / np.mean(mkts), 3) if np.mean(mkts) > 0 else np.nan,
            "N dates": len(mods),
        })

df_table = pd.DataFrame(rows)
print("\n" + "=" * 65)
print("TERM STRUCTURE OF MODEL vs MARKET NORMAL VOL  (base model, σ_vec=1)")
print("=" * 65)
print(df_table.to_string(index=False))
print("=" * 65)
df_table.to_csv(os.path.join(OUT_DIR, "term_structure_table.csv"), index=False)

# ── Figure 1: σ_F vs expiry per tenor ─────────────────────────────────────────
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
            label="Model $\\sigma_F$ (base, $s=1$)")
    ax.plot(EXPIRIES, mkt_means, "s--", color=colors_mkt[j], lw=2.0, ms=7,
            label="Market $\\sigma_N$")

    ax.set_title(f"Tenor = {ten}Y", fontsize=12)
    ax.set_xlabel("Expiry (years)", fontsize=11)
    ax.set_xticks(EXPIRIES)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel("ATM normal vol (bp)", fontsize=11)
fig.suptitle(
    "Term structure of $\\sigma_F$: base model vs market\n"
    "(Decreasing model vol reveals structural mean-reversion mismatch)",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "term_structure_vol_by_tenor.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: term_structure_vol_by_tenor.png")

# ── Figure 2: model/market ratio heat-map ─────────────────────────────────────
ratio_mat = np.full((len(EXPIRIES), len(TENORS)), np.nan)
for i, exp in enumerate(EXPIRIES):
    for j, ten in enumerate(TENORS):
        mods = results[exp][ten]["mod"]
        mkts = results[exp][ten]["mkt"]
        if mods and mkts and np.mean(mkts) > 0:
            ratio_mat[i, j] = np.mean(mods) / np.mean(mkts)

fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
im = ax2.imshow(ratio_mat, aspect="auto", cmap="RdYlGn_r", vmin=0.4, vmax=2.0,
                origin="lower")
ax2.set_xticks(range(len(TENORS)));   ax2.set_xticklabels([f"{t}Y" for t in TENORS])
ax2.set_yticks(range(len(EXPIRIES))); ax2.set_yticklabels([f"{e}Y" for e in EXPIRIES])
ax2.set_xlabel("Tenor");  ax2.set_ylabel("Expiry")
ax2.set_title("σ_F(model) / σ_N(market)  —  base model, σ_vec=1\n"
              "Green < 1 (under-vol), Red > 1 (over-vol)", fontsize=11)

for i in range(len(EXPIRIES)):
    for j in range(len(TENORS)):
        if not np.isnan(ratio_mat[i, j]):
            ax2.text(j, i, f"{ratio_mat[i, j]:.2f}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color="black")

plt.colorbar(im, ax=ax2, label="ratio")
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "vol_ratio_heatmap.png"), dpi=200, bbox_inches="tight")
plt.close(fig2)
print("Saved: vol_ratio_heatmap.png")

# ── Figure 3: distribution of F_T - F_0 per expiry ───────────────────────────
fig3, axes3 = plt.subplots(1, len(EXPIRIES), figsize=(5 * len(EXPIRIES), 4), dpi=150, sharey=False)
for i, exp in enumerate(EXPIRIES):
    ten    = TENORS[0]
    chunks = results[exp][ten]["FT_dist"]
    if not chunks:
        axes3[i].set_title(f"Expiry {exp}Y — no data")
        continue
    vals = np.concatenate(chunks)
    ax3  = axes3[i]
    ax3.hist(vals, bins=60, color="#4c72b0", alpha=0.7, density=True)
    ax3.axvline(0, color="red", lw=1.5, ls="--", label="F_0")
    std_bp = np.std(vals)
    ax3.set_title(f"Expiry = {exp}Y  (tenor={ten}Y)\n"
                  f"std(F_T − F_0) = {std_bp:.1f} bp", fontsize=11)
    ax3.set_xlabel("$F_T - F_0$  (bp)", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

fig3.suptitle(
    "Distribution of $F_T - F_0$ per expiry  (base model, one date, tenor=1Y)\n"
    "Variance saturation: spread grows sub-linearly with expiry",
    fontsize=11, fontweight="bold"
)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "FT_distribution_per_expiry.png"), dpi=200, bbox_inches="tight")
plt.close(fig3)
print("Saved: FT_distribution_per_expiry.png")

# ── Figure 4: scaling sensitivity — what s does to each expiry ────────────────
# For each expiry row show: what vol you'd get at various σ_vec scale factors
scale_factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
tenor_ref     = TENORS[0]   # show for first tenor only

fig4, ax4 = plt.subplots(figsize=(7, 4.5), dpi=150)
colors_exp = ["#d62728", "#ff7f0e", "#2ca02c"]
for i, exp in enumerate(EXPIRIES):
    base_mods = results[exp][tenor_ref]["mod"]
    base_mkts = results[exp][tenor_ref]["mkt"]
    if not base_mods:
        continue
    base_sig  = np.mean(base_mods)
    mkt_sig   = np.mean(base_mkts) if base_mkts else None

    # σ_F scales linearly with σ_vec (Euler-Maruyama: F_T ∝ s × diffusion)
    scaled_sigs = base_sig * scale_factors
    ax4.plot(scale_factors, scaled_sigs, "o-", color=colors_exp[i], lw=2, ms=7,
             label=f"Expiry {exp}Y (model)")
    if mkt_sig is not None:
        ax4.axhline(float(mkt_sig), color=colors_exp[i], lw=1.5, ls="--",
                    alpha=0.7, label=f"Market {exp}Y = {mkt_sig:.0f} bp")

ax4.set_xlabel("σ_vec scale factor  $s$", fontsize=11)
ax4.set_ylabel("σ_F (bp)", fontsize=11)
ax4.set_title(
    f"No single $s$ reconciles all expiries  (tenor = {tenor_ref}Y)\n"
    "Dashed lines = market target per expiry",
    fontsize=11, fontweight="bold"
)
ax4.legend(fontsize=8, ncol=2)
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, "scaling_sensitivity.png"), dpi=200, bbox_inches="tight")
plt.close(fig4)
print("Saved: scaling_sensitivity.png")

print(f"\nAll diagnostics saved to: {OUT_DIR}")
print("\nKey finding:")
print("  Model σ_F is strongly DECREASING with expiry (Vasicek-like variance saturation).")
print("  Market σ_N is roughly FLAT across expiries.")
print("  No single rescaling of σ_vec can match both short and long expiries simultaneously.")



