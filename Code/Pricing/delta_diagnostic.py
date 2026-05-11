# ==================== Forward Bias and Effective Exercise Probability ====================
"""
Quantifies the martingale violation in K per (expiry, tenor) cell.

Background
----------
At ATM, the Bachelier Delta is exactly (1/2) * N * A_0 — the distribution of
S_T is symmetric around F_0, giving a 50% exercise probability. A model with
K not satisfying the annuity-measure martingale condition E^{Q_A}[S_T] = F_0
will have a shifted distribution, measurable directly from put-call parity:

    forward_bias = (V_pay - V_rec) / A_0     [decimal rate units]
                 = E^{Q_A}[S_T] - F_0         [under annuity measure]

The effective exercise probability is then:

    d_eff    = forward_bias / (sigma_str * sqrt(T_e))
    p_eff    = Phi(d_eff)

where sigma_str = (V_pay + V_rec)/2 * sqrt(2pi) / (A_0 * sqrt(T_e)) is the
model's straddle vol (measures distribution width without directional bias).

Under a correctly specified Q model:
    forward_bias = 0,  p_eff = 50%,  Delta = (1/2) N A_0

Under our model (K identified cross-sectionally, not from martingale condition):
    forward_bias != 0 per cell,  p_eff != 50%

Pricing is done at the expiry-level calibrated scales (s_1Y, s_5Y, s_10Y)
rather than s=1 to keep paths in the well-behaved regime.

Outputs (to dim4_stable_hscale/delta_diagnostic/)
-------------------------------------------------
    delta_results.csv           per-(date, expiry, tenor) raw results
    tab_delta_diagnostic.tex    LaTeX table: forward bias + p_eff per cell
    fig_forward_bias.png        heatmap of average forward bias per cell
    fig_exercise_prob.png       heatmap of p_eff per cell
"""

import math
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
for p in [PROJECT_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SKIP_VARIANT_CONFIRM", "1")

from Code import config
from Code.load_swapdata import my_data
from Code.model.full_model_stable import FullModel
from Code.Pricing.load_swapvol_ois import load_swaption_vol_data
from Code.Pricing.pricing import swap_rate_torch
from Code.Simulation.simulate_model import simulate_latent_paths, compute_discount_paths

# ================================================================
# Settings
# ================================================================

LATENT_DIM = 4
N_PATHS    = 2048
DT         = 1 / 12
CCY_FILTER = "EUR"
USE        = "bbg"
SEED       = 42
TRAIN_FRAC = 0.70

RATE_CLIP    = 0.50   # 50% rate — far beyond any observed level
ANNUITY_MAX  = 50.0   # annuity > 50 implies implausible discount factors

CKPT_STABLE = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)
HSCALE_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults", "dim4_stable_hscale"
)
OUT_DIR = os.path.join(HSCALE_DIR, "delta_diagnostic")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device   = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)

# ================================================================
# Load model
# ================================================================

print(f"Repo root: {PROJECT_ROOT}")
model = FullModel(latent_dim=LATENT_DIM).to(device)
raw   = torch.load(CKPT_STABLE, map_location=device, weights_only=False)
state = raw["model_state_dict"] if "model_state_dict" in raw else raw
model.load_state_dict(state)
model.eval()
print(f"Loaded: {os.path.basename(CKPT_STABLE)}")

# ================================================================
# Load data
# ================================================================

meta, X_tensor, *_ = my_data(use=USE)
X_tensor  = X_tensor.float()
meta_eur  = meta[meta["ccy"] == CCY_FILTER].reset_index(drop=True)
X_eur     = X_tensor[meta["ccy"] == CCY_FILTER]

df_vol = load_swaption_vol_data(currency=CCY_FILTER)
df_vol["as_of_date"] = pd.to_datetime(df_vol["as_of_date"]).dt.normalize()
df_vol["market_vol"] = df_vol["vol"] / 10_000.0
dates_swap  = set(pd.to_datetime(meta_eur["as_of_date"]).dt.normalize())
df_vol      = df_vol[df_vol["as_of_date"].isin(dates_swap)].copy()
date_to_idx = {
    pd.Timestamp(r["as_of_date"]).normalize(): i
    for i, r in meta_eur.iterrows()
}
combos = df_vol[["as_of_date","option_maturity","swap_tenor","market_vol"]].drop_duplicates()
print(f"EUR: {len(meta_eur)} dates, {len(combos)} vol targets")

all_dates   = sorted(df_vol["as_of_date"].unique())
n_train     = int(len(all_dates) * TRAIN_FRAC)
train_dates = set(all_dates[:n_train])
test_dates  = set(all_dates[n_train:])

# ================================================================
# Load expiry-level calibrated scales
# ================================================================

scales_path = os.path.join(HSCALE_DIR, "expiry_scale", "expiry_scales.json")
with open(scales_path) as f:
    _raw = json.load(f)
expiry_scales = {int(k): float(v) for k, v in _raw.items()}
print(f"\nExpiry-level calibrated scales: {expiry_scales}")

# ================================================================
# MC pricing helper — price BOTH payer and receiver
# ================================================================

@torch.no_grad()
def price_payer_receiver(date, expiry, tenor, s):
    """
    Returns (V_pay_bp, V_rec_bp, A0, F0, sigma_pay, sigma_rec, n_surv) or None.
    All vols in bp. V_pay_bp and V_rec_bp are prices normalised by (A0 * sqrt(T_e/2pi)).
    """
    if date not in date_to_idx:
        return None
    idx  = date_to_idx[date]
    xb   = X_eur[idx:idx+1].to(device)
    z0   = model.encoder(xb)

    n_steps = max(12, int(round(expiry / DT)))
    dt_eff  = expiry / n_steps
    half    = N_PATHS // 2

    z1,r1,_,_ = simulate_latent_paths(model, z0, n_paths=half,
                                       n_steps=n_steps, dt=dt_eff,
                                       device=device, diffusion_scale=s)
    z2,r2,_,_ = simulate_latent_paths(model, z0, n_paths=half,
                                       n_steps=n_steps, dt=dt_eff,
                                       device=device, diffusion_scale=s)
    z_T   = torch.cat([z1[:,-1,:], z2[:,-1,:]], dim=0)
    r_all = torch.cat([r1, r2], dim=0)
    D_T   = compute_discount_paths(r_all, dt_eff)[:,-1]

    ok = torch.isfinite(z_T).all(1) & torch.isfinite(D_T)
    if ok.sum() < 32:
        return None
    z_k, D_k = z_T[ok], D_T[ok]

    _, aux_T = model.decode_from_z(z_k, tau=None, return_aux=True)
    P_T = aux_T["P_full"]
    dok = torch.isfinite(P_T).all(1)
    if dok.sum() < 32:
        return None
    F_T, A_T = swap_rate_torch(P_T[dok], tenor=tenor)
    D_k = D_k[dok]

    fa = torch.isfinite(F_T) & torch.isfinite(A_T)
    F_T, A_T, D_k = F_T[fa], A_T[fa], D_k[fa]

    # Sanity filter: remove explosive paths
    sane = (F_T > -RATE_CLIP) & (F_T < RATE_CLIP) & (A_T > 1e-6) & (A_T < ANNUITY_MAX)
    if sane.sum() < 32:
        return None
    F_T, A_T, D_k = F_T[sane], A_T[sane], D_k[sane]

    _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
    F0, A0  = swap_rate_torch(aux0["P_full"], tenor=tenor)
    F0, A0  = float(F0[0]), float(A0[0])
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 0):
        return None

    # Payer and receiver prices (MC average of discounted payoffs)
    V_pay = float((D_k * A_T * torch.relu(F_T - F0)).mean())
    V_rec = float((D_k * A_T * torch.relu(F0 - F_T)).mean())

    if not (math.isfinite(V_pay) and math.isfinite(V_rec)):
        return None

    # Convert to implied normal vols (bp)
    scale = sqrt_2pi / (A0 * math.sqrt(expiry))
    sigma_pay = V_pay * scale * 10_000.0
    sigma_rec = V_rec * scale * 10_000.0

    n_surv = int(sane.sum()) / N_PATHS
    return V_pay, V_rec, A0, F0, sigma_pay, sigma_rec, n_surv


# ================================================================
# Step 1 — Price all swaptions at calibrated scales
# ================================================================

print("\n=== Pricing payer + receiver at expiry-level scales ===")

rows    = []
counter = 0

for _, row in combos.iterrows():
    date   = pd.Timestamp(row["as_of_date"]).normalize()
    expiry = int(row["option_maturity"])
    tenor  = int(row["swap_tenor"])
    mkt_vol = float(row["market_vol"]) * 10_000.0   # bp

    if expiry not in expiry_scales:
        continue
    s = expiry_scales[expiry]

    result = price_payer_receiver(date, expiry, tenor, s)
    counter += 1
    if counter % 100 == 0:
        print(f"  Priced {counter} ({date.date()} {expiry}Yx{tenor}Y)")

    if result is None:
        continue

    V_pay, V_rec, A0, F0, sigma_pay, sigma_rec, n_surv = result

    # Forward bias in bp: (V_pay - V_rec) / A0 * 10000
    forward_bias_bp = (V_pay - V_rec) / A0 * 10_000.0

    # Straddle vol: symmetric measure of distribution width (bp)
    sqrt_2pi_local  = math.sqrt(2.0 * math.pi)
    sigma_str = (V_pay + V_rec) / 2.0 * sqrt_2pi_local / (A0 * math.sqrt(expiry)) * 10_000.0

    # Standardised bias: how many straddle-vol std devs is the mean off-centre?
    if sigma_str > 1e-4:
        d_eff = forward_bias_bp / (sigma_str * math.sqrt(expiry))
    else:
        d_eff = 0.0

    # Effective ATM exercise probability
    p_eff = float(norm.cdf(d_eff))

    split = "train" if date in train_dates else "test"
    rows.append({
        "date": date, "expiry": expiry, "tenor": tenor,
        "split": split,
        "mkt_vol_bp": mkt_vol,
        "sigma_pay_bp": sigma_pay,
        "sigma_rec_bp": sigma_rec,
        "sigma_str_bp": sigma_str,
        "forward_bias_bp": forward_bias_bp,
        "d_eff": d_eff,
        "p_eff_pct": p_eff * 100.0,
        "n_surv": n_surv,
        "A0": A0,
        "F0": F0,
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "delta_results.csv"), index=False)
print(f"\nPriced {len(df)} swaptions")

# ================================================================
# Step 2 — Per-cell summary
# ================================================================

print("\n=== Per-cell summary ===")
print(f"\n{'Cell':>10}  {'Fwd bias (bp)':>14}  {'Straddle vol':>13}  {'p_eff (%)':>10}  {'d_eff':>7}")
print("-"*62)

expiry_vals = sorted(df["expiry"].unique())
tenor_vals  = sorted(df["tenor"].unique())

cell_stats = {}
for e in expiry_vals:
    for t in tenor_vals:
        sub = df[(df["expiry"] == e) & (df["tenor"] == t)]
        if len(sub) == 0:
            continue
        bias_mean  = sub["forward_bias_bp"].mean()
        str_mean   = sub["sigma_str_bp"].mean()
        p_eff_mean = sub["p_eff_pct"].mean()
        d_eff_mean = sub["d_eff"].mean()
        cell_stats[(e, t)] = {
            "forward_bias_bp": bias_mean,
            "sigma_str_bp": str_mean,
            "p_eff_pct": p_eff_mean,
            "d_eff": d_eff_mean,
            "n": len(sub),
        }
        label = f"{e}Yx{t}Y"
        print(f"  {label:>8}  {bias_mean:>+14.1f}  {str_mean:>13.1f}  {p_eff_mean:>10.1f}  {d_eff_mean:>+7.3f}")

# ================================================================
# Step 3 — LaTeX table
# ================================================================

cell_labels = {(1,1):"1Yx1Y",(1,5):"1Yx5Y",(1,10):"1Yx10Y",
               (5,1):"5Yx1Y",(5,5):"5Yx5Y",(5,10):"5Yx10Y",
               (10,1):"10Yx1Y",(10,5):"10Yx5Y",(10,10):"10Yx10Y"}

lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"\centering")
lines.append(
    r"\caption{Per-cell forward bias and effective ATM exercise probability "
    r"at the expiry-level calibrated scale. The forward bias "
    r"$\mathbb{E}^{\mathbb{Q}_\mathcal{A}}[S_T] - F_0$ is measured directly "
    r"from put-call parity: $(V_{\mathrm{pay}} - V_{\mathrm{rec}})/\mathcal{A}_0$. "
    r"The effective exercise probability $p_{\mathrm{eff}} = \Phi(d_{\mathrm{eff}})$ "
    r"where $d_{\mathrm{eff}} = \text{forward bias} / (\sigma_{\mathrm{str}}\sqrt{T_e})$ "
    r"measures how far the model's $\mathbb{Q}$ distribution is shifted relative to its own width. "
    r"Under a correctly specified risk-neutral model: forward bias $= 0$ and $p_{\mathrm{eff}} = 50\%$. "
    r"Deviations quantify the failure of $\mathcal{K}$ to satisfy the annuity-measure "
    r"martingale condition.}"
)
lines.append(r"\label{tab:delta_diagnostic}")
lines.append(r"\begin{tabular}{@{}ccrrrrr@{}}")
lines.append(r"\toprule")
lines.append(r"\textbf{Exp} & \textbf{Ten} & $\sigma_{\mathrm{mkt}}$ \textbf{(bp)} "
             r"& $\sigma_{\mathrm{str}}$ \textbf{(bp)} & \textbf{Fwd bias (bp)} "
             r"& $d_{\mathrm{eff}}$ & $p_{\mathrm{eff}}$ \textbf{(\%)} \\")
lines.append(r"\midrule")

for i, e in enumerate(expiry_vals):
    for j, t in enumerate(tenor_vals):
        key = (e, t)
        if key not in cell_stats:
            continue
        st = cell_stats[key]
        avg_mkt = df[(df["expiry"]==e) & (df["tenor"]==t)]["mkt_vol_bp"].mean()
        bias    = st["forward_bias_bp"]
        sigma_s = st["sigma_str_bp"]
        d_eff   = st["d_eff"]
        p_eff   = st["p_eff_pct"]
        sign    = "+" if bias >= 0 else ""
        lines.append(
            f"{e} & {t} & {avg_mkt:.0f} & {sigma_s:.0f} & "
            f"${sign}{bias:.0f}$ & ${d_eff:+.2f}$ & {p_eff:.1f} \\\\"
        )
    if i < len(expiry_vals) - 1:
        lines.append(r"\addlinespace[2pt]")

lines.append(r"\midrule")
overall_bias = df["forward_bias_bp"].mean()
overall_peff = df["p_eff_pct"].mean()
overall_str  = df["sigma_str_bp"].mean()
overall_mkt  = df["mkt_vol_bp"].mean()
sign = "+" if overall_bias >= 0 else ""
lines.append(
    rf"\textbf{{Overall}} & & {overall_mkt:.0f} & {overall_str:.0f} & "
    rf"${sign}{overall_bias:.0f}$ & & {overall_peff:.1f} \\"
)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
tex_path = os.path.join(OUT_DIR, "tab_delta_diagnostic.tex")
with open(tex_path, "w") as f:
    f.write(tex)
print(f"Saved {tex_path}")

# ================================================================
# Step 4 — Figures
# ================================================================

def make_heatmap(values, title, cmap, vcenter, vmin, vmax, fmt, label, fname, annotate_50=False):
    grid = np.full((3, 3), np.nan)
    for i, e in enumerate(expiry_vals):
        for j, t in enumerate(tenor_vals):
            key = (e, t)
            if key in cell_stats:
                grid[i, j] = values[key]

    fig, ax = plt.subplots(figsize=(6, 4))
    norm_c  = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(grid, cmap=cmap, norm=norm_c, aspect="auto")
    plt.colorbar(im, ax=ax, label=label)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"{t}Y" for t in tenor_vals])
    ax.set_yticklabels([f"{e}Y" for e in expiry_vals])
    ax.set_xlabel("Swap tenor")
    ax.set_ylabel("Option expiry")
    ax.set_title(title)

    for i in range(3):
        for j in range(3):
            v = grid[i, j]
            if not np.isnan(v):
                txt = fmt(v)
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if abs(norm_c(v) - 0.5) > 0.25 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved {fname}")

# Forward bias heatmap (centred on zero, diverging)
bias_vals = {k: v["forward_bias_bp"] for k, v in cell_stats.items()}
all_bias  = [v for v in bias_vals.values() if not np.isnan(v)]
abs_max   = max(abs(min(all_bias)), abs(max(all_bias))) * 1.05

make_heatmap(
    values=bias_vals,
    title=r"Forward bias $\mathbb{E}^{Q_A}[S_T]-F_0$ (bp)",
    cmap="RdBu_r",
    vcenter=0,
    vmin=-abs_max,
    vmax=abs_max,
    fmt=lambda v: f"{v:+.0f}",
    label="bp",
    fname="fig_forward_bias.png",
)

# Effective exercise probability (centred on 50%)
peff_vals = {k: v["p_eff_pct"] for k, v in cell_stats.items()}
all_peff  = [v for v in peff_vals.values()]
dev_max   = max(abs(min(all_peff) - 50), abs(max(all_peff) - 50)) * 1.05

make_heatmap(
    values=peff_vals,
    title=r"Effective ATM exercise probability $p_\mathrm{eff}$ (%)",
    cmap="RdBu_r",
    vcenter=50,
    vmin=50 - dev_max,
    vmax=50 + dev_max,
    fmt=lambda v: f"{v:.1f}%",
    label="%",
    fname="fig_exercise_prob.png",
)

# ================================================================
# Step 5 — Print interpretation
# ================================================================

print("\n=== Interpretation ===")
print(f"\n{'Cell':>10}  {'Fwd bias':>10}  {'% of mkt vol':>13}  {'p_eff':>8}")
print("-"*48)
for e in expiry_vals:
    for t in tenor_vals:
        key = (e, t)
        if key not in cell_stats:
            continue
        st  = cell_stats[key]
        mkt = df[(df["expiry"]==e) & (df["tenor"]==t)]["mkt_vol_bp"].mean()
        pct_of_vol = st["forward_bias_bp"] / mkt * 100
        label = f"{e}Yx{t}Y"
        print(f"  {label:>8}  {st['forward_bias_bp']:>+10.1f}  {pct_of_vol:>+12.1f}%  {st['p_eff_pct']:>7.1f}%")

print(f"\nBachelier benchmark: bias = 0 bp, p_eff = 50.0% for all cells")
print(f"\nDone. All outputs in: {OUT_DIR}")
