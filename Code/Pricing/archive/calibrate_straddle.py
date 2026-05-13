# ==================== Straddle-Based Drift-Bias Diagnostic ====================
"""
Diagnostic experiment: test whether the residual pricing error is driven by a
directional forward-rate bias in K.

Background
----------
The model assumes Q-measure dynamics  dz = K(z)dt + H(z)dW^Q.  K is identified
by minimising the cross-sectional swap-curve reconstruction loss, NOT by enforcing
the annuity-measure martingale condition E^{Q_A}[S_T] = F_0.  As a result, the
model's Q may not centre S_T on the forward swap rate F_0, creating a forward bias.

For a payer swaption:
  V_pay  = E^Q[D_T A_T max(S_T - F_0, 0)]   -- inflated if E^{Q_A}[S_T] > F_0

For a receiver:
  V_rec  = E^Q[D_T A_T max(F_0 - S_T, 0)]   -- inflated if E^{Q_A}[S_T] < F_0

Put-call parity:
  V_pay - V_rec = A_0 * (E^{Q_A}[S_T] - F_0)  != 0  when K is mis-calibrated

The STRADDLE (V_pay + V_rec) would cancel the directional component IF the bias
were uniformly signed across cells.

Symmetric implied vol:
  sigma_straddle = (V_pay + V_rec)/2 * sqrt(2*pi) / (A_0 * sqrt(T_e))

Under a correctly specified Q:  sigma_straddle == sigma_payer
Under a bias with consistent sign: sigma_straddle < sigma_payer (cancellation)
Finding (result): OOS MAE jumps 75 -> 287 bp under straddle convention, because
the forward bias is SIGN-INCONSISTENT across cells (+239 bp for 5Yx1Y,
-401 bp for 1Yx5Y).  Straddle amplifies rather than cancels the error.

This script:
  1. Re-prices every swaption from baseline_vols_s1.csv as BOTH payer AND receiver
  2. Computes sigma_payer, sigma_receiver, sigma_straddle per (date, expiry, tenor)
  3. Fits OLS scale factors for STRADDLE vol (instead of payer vol)
  4. Evaluates train/OOS MAE for all three vol measures
  5. Checks whether straddle significantly reduces 1Yx1Y and other drift-biased cells

Outputs (to dim4_stable_hscale/straddle/)
-----------------------------------------
  straddle_vols_s1.csv       per-(date,cell) payer/receiver/straddle vols at s=1
  straddle_scales.json       per-cell OLS on straddle vols
  straddle_results.csv       per-(date,cell) all vol errors after calibration
  straddle_summary.json      aggregate and per-cell MAE (payer vs straddle)
  tab_straddle.tex           LaTeX table
  fig_straddle_comparison.png
"""

import math
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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

CKPT_STABLE = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults",
    "dim4_stable", "ep5000", "checkpoint_dim4_ep5000.pt"
)
HSCALE_DIR = os.path.join(
    PROJECT_ROOT, "Figures", "TrainingResults", "dim4_stable_hscale"
)
OUT_DIR = os.path.join(HSCALE_DIR, "straddle")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device   = torch.device("cpu")
sqrt_2pi = math.sqrt(2.0 * math.pi)

# ================================================================
# Load model
# ================================================================

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
# MC pricing: returns BOTH payer and receiver vols at scale s
# ================================================================

@torch.no_grad()
def price_straddle(date, expiry, tenor, s=1.0):
    """
    Returns (sigma_payer, sigma_receiver, forward_bias, path_frac) or None.
    sigma_payer    = payer implied vol (bp)
    sigma_receiver = receiver implied vol (bp)
    forward_bias   = (E^P[S_T] - F_0) in bp — positive means upward drift bias
    """
    if date not in date_to_idx:
        return None
    idx  = date_to_idx[date]
    xb   = X_eur[idx:idx+1].to(device)
    z0   = model.encoder(xb)
    n_steps = max(12, int(round(expiry / DT)))
    dt_eff  = expiry / n_steps
    half    = N_PATHS // 2

    z1, r1, _, _ = simulate_latent_paths(model, z0, n_paths=half,
                                          n_steps=n_steps, dt=dt_eff,
                                          device=device, diffusion_scale=s)
    z2, r2, _, _ = simulate_latent_paths(model, z0, n_paths=half,
                                          n_steps=n_steps, dt=dt_eff,
                                          device=device, diffusion_scale=s)
    z_T   = torch.cat([z1[:, -1, :], z2[:, -1, :]], dim=0)
    r_all = torch.cat([r1, r2], dim=0)
    D_T   = compute_discount_paths(r_all, dt_eff)[:, -1]

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
    fa  = torch.isfinite(F_T) & torch.isfinite(A_T)
    if fa.sum() < 32:
        return None
    F_T, A_T, D_k = F_T[fa], A_T[fa], D_k[fa]

    # Filter explosive paths: swap rates beyond ±50% are model blowup artefacts.
    # At training-time scale (s≈0.14) this is never triggered; at s=1.0 a small
    # fraction of paths diverge and would otherwise dominate the receiver payoff.
    RATE_CLIP = 0.50   # 50% in decimal — far beyond any observed or plausible rate
    ANNUITY_MAX = 50.0  # annuity factor > 50 implies unrealistic tenor/discount
    sane = (F_T > -RATE_CLIP) & (F_T < RATE_CLIP) & (A_T > 1e-6) & (A_T < ANNUITY_MAX)
    if sane.sum() < 32:
        return None
    F_T, A_T, D_k = F_T[sane], A_T[sane], D_k[sane]

    # Time-0 quantities (strike and annuity)
    _, aux0 = model.decode_from_z(z0, tau=None, return_aux=True)
    F0, A0  = swap_rate_torch(aux0["P_full"], tenor=tenor)
    F0, A0  = float(F0[0]), float(A0[0])
    if not (math.isfinite(F0) and math.isfinite(A0) and A0 > 0):
        return None

    # Payer:   max(S_T - F0, 0)
    # Receiver: max(F0 - S_T, 0)
    payer_payoff    = torch.relu(F_T - F0)
    receiver_payoff = torch.relu(F0 - F_T)

    V_pay = (D_k * A_T * payer_payoff).mean()
    V_rec = (D_k * A_T * receiver_payoff).mean()

    if not (torch.isfinite(V_pay) and torch.isfinite(V_rec)):
        return None

    # Implied vols: sigma = V * sqrt(2pi) / (A0 * sqrt(T_e))
    scale = sqrt_2pi / (A0 * math.sqrt(expiry)) * 10_000.0
    sig_pay = float(V_pay) * scale
    sig_rec = float(V_rec) * scale
    sig_str = (float(V_pay) + float(V_rec)) / 2.0 * scale

    # Forward bias: (V_pay - V_rec) / A0 in bp
    # Under Q^A: = (E^{Q^A}[S_T] - F0) * 10000
    # Under P:   = (E^{P,A}[S_T] - F0) * 10000  (approx via discount-weighted mean)
    fwd_bias = float((D_k * A_T * (F_T - F0)).mean()) / A0 * 10_000.0

    pfrac = int(ok.sum()) / N_PATHS
    return sig_pay, sig_rec, sig_str, fwd_bias, pfrac

# ================================================================
# Step 1 — Price everything at s=1: payer, receiver, straddle
# ================================================================
print("\n=== Step 1: Price all swaptions at s=1 (payer + receiver) ===")

STRADDLE_CSV = os.path.join(OUT_DIR, "straddle_vols_s1.csv")

if os.path.isfile(STRADDLE_CSV):
    print(f"Loading cached straddle vols from {STRADDLE_CSV}")
    df_s1 = pd.read_csv(STRADDLE_CSV)
    df_s1["date"] = pd.to_datetime(df_s1["date"])
else:
    rows = []
    n_ok = 0
    for _, row in combos.iterrows():
        date    = pd.Timestamp(row["as_of_date"]).normalize()
        expiry  = int(row["option_maturity"])
        tenor   = int(row["swap_tenor"])
        sig_mkt = float(row["market_vol"]) * 10_000.0

        res = price_straddle(date, expiry, tenor, s=1.0)
        if res is None:
            sp, sr, ss, fb, pf = (float("nan"),)*4 + (0.0,)
        else:
            sp, sr, ss, fb, pf = res
            n_ok += 1

        split = "train" if date in train_dates else "test"
        rows.append({
            "date":     date.date(),
            "expiry":   expiry,
            "tenor":    tenor,
            "split":    split,
            "sigma_mkt":  round(sig_mkt, 1),
            "sig_pay_s1": round(sp, 1) if math.isfinite(sp) else None,
            "sig_rec_s1": round(sr, 1) if math.isfinite(sr) else None,
            "sig_str_s1": round(ss, 1) if math.isfinite(ss) else None,
            "fwd_bias_s1":round(fb, 1) if math.isfinite(fb) else None,
            "path_frac":  round(pf, 3),
        })
        if n_ok % 100 == 0 and n_ok > 0:
            print(f"  Priced {n_ok} ({date.date()} {expiry}Yx{tenor}Y)")

    df_s1 = pd.DataFrame(rows)
    df_s1["date"] = pd.to_datetime(df_s1["date"])
    df_s1.to_csv(STRADDLE_CSV, index=False)
    print(f"Saved straddle_vols_s1.csv ({n_ok} priced)")

# ================================================================
# Step 2 — Report forward bias per cell
# ================================================================
print("\n=== Step 2: Forward bias analysis ===")
print(f"\n{'Cell':>8}  {'avg bias (bp)':>14}  {'avg pay_s1':>11}  "
      f"{'avg rec_s1':>11}  {'avg mkt':>9}  {'% dates bias>0':>15}")
print("-"*72)

cells = sorted(
    df_s1[["expiry","tenor"]].drop_duplicates().itertuples(index=False),
    key=lambda x: (x.expiry, x.tenor)
)

bias_summary = {}
for cell in cells:
    e, t = cell.expiry, cell.tenor
    sub = df_s1[(df_s1["expiry"]==e) & (df_s1["tenor"]==t)
                & df_s1["fwd_bias_s1"].notna()
                & (df_s1["split"]=="train")]
    if sub.empty:
        continue
    avg_bias  = float(sub["fwd_bias_s1"].mean())
    avg_pay   = float(sub["sig_pay_s1"].dropna().mean())
    avg_rec   = float(sub["sig_rec_s1"].dropna().mean())
    avg_mkt   = float(sub["sigma_mkt"].mean())
    pct_pos   = float((sub["fwd_bias_s1"] > 0).mean()) * 100
    bias_summary[(e,t)] = {"bias": avg_bias, "pay": avg_pay, "rec": avg_rec, "mkt": avg_mkt}
    print(f"  {e}Yx{t}Y  {avg_bias:>14.1f}  {avg_pay:>11.1f}  "
          f"{avg_rec:>11.1f}  {avg_mkt:>9.1f}  {pct_pos:>14.0f}%")

# ================================================================
# Step 3 — Per-cell OLS on STRADDLE vol (training dates)
# ================================================================
print("\n=== Step 3: Per-cell OLS on straddle vol ===")

df_valid = df_s1[
    df_s1["sig_str_s1"].notna() &
    (df_s1["sigma_mkt"] > 0)
].copy()
df_train = df_valid[df_valid["split"] == "train"]

# OLS for payer and straddle
scales_pay = {}
scales_str = {}
print(f"\n{'Cell':>8}  {'s*_payer':>10}  {'s*_straddle':>12}  {'ratio':>7}")
print("-"*44)
for cell in cells:
    e, t = cell.expiry, cell.tenor
    sub = df_train[(df_train["expiry"]==e) & (df_train["tenor"]==t)]

    # Payer OLS
    pay = sub["sig_pay_s1"].dropna()
    mkt = sub.loc[pay.index, "sigma_mkt"]
    s_pay = float(np.dot(pay, mkt) / np.dot(pay, pay)) if len(pay) > 0 else 1.0

    # Straddle OLS
    str_ = sub["sig_str_s1"].dropna()
    mkt2 = sub.loc[str_.index, "sigma_mkt"]
    s_str = float(np.dot(str_, mkt2) / np.dot(str_, str_)) if len(str_) > 0 else 1.0

    scales_pay[(e,t)] = s_pay
    scales_str[(e,t)] = s_str
    ratio = s_str / s_pay if s_pay > 0 else float("nan")
    print(f"  {e}Yx{t}Y  {s_pay:>10.4f}  {s_str:>12.4f}  {ratio:>7.3f}")

# Save
with open(os.path.join(OUT_DIR, "straddle_scales.json"), "w") as f:
    json.dump({f"{e}x{t}": {"s_payer": float(v), "s_straddle": float(scales_str[(e,t)])}
               for (e,t), v in scales_pay.items()}, f, indent=2)
print("Saved straddle_scales.json")

# ================================================================
# Step 4 — MC pricing at calibrated scales: both payer and straddle
# ================================================================
print("\n=== Step 4: MC pricing at calibrated scales ===")

result_rows = []
n_done = 0

for _, row in combos.iterrows():
    date    = pd.Timestamp(row["as_of_date"]).normalize()
    expiry  = int(row["option_maturity"])
    tenor   = int(row["swap_tenor"])
    sig_mkt = float(row["market_vol"]) * 10_000.0
    split   = "train" if date in train_dates else "test"

    s_p = scales_pay.get((expiry, tenor), 1.0)
    s_s = scales_str.get((expiry, tenor), 1.0)

    res_p = price_straddle(date, expiry, tenor, s=s_p)
    res_s = price_straddle(date, expiry, tenor, s=s_s)

    def extract(res):
        if res is None or not all(math.isfinite(x) for x in res[:4]):
            return float("nan"), float("nan"), float("nan"), float("nan")
        return res[0], res[1], res[2], res[3]

    pay_p, rec_p, str_p, bias_p = extract(res_p)
    pay_s, rec_s, str_s, bias_s = extract(res_s)
    n_done += 1
    if n_done % 100 == 0:
        print(f"  Priced {n_done} ({date.date()} {expiry}Yx{tenor}Y)")

    result_rows.append({
        "date":    date.date(),
        "expiry":  expiry,
        "tenor":   tenor,
        "split":   split,
        "sigma_mkt": round(sig_mkt, 1),
        # At payer-calibrated scale
        "s_pay":    round(s_p, 5),
        "pay_cal":  round(pay_p, 1) if math.isfinite(pay_p) else None,
        "str_pay":  round(str_p, 1) if math.isfinite(str_p) else None,
        "bias_pay": round(bias_p, 1) if math.isfinite(bias_p) else None,
        # At straddle-calibrated scale
        "s_str":    round(s_s, 5),
        "str_cal":  round(str_s, 1) if math.isfinite(str_s) else None,
        "pay_str":  round(pay_s, 1) if math.isfinite(pay_s) else None,
    })

print(f"Priced {n_done} swaptions")
df_res = pd.DataFrame(result_rows)
df_res["date"] = pd.to_datetime(df_res["date"])
df_res["ae_pay"]       = (df_res["pay_cal"] - df_res["sigma_mkt"]).abs()
df_res["ae_str_pscal"] = (df_res["str_pay"] - df_res["sigma_mkt"]).abs()
df_res["ae_str"]       = (df_res["str_cal"] - df_res["sigma_mkt"]).abs()
df_res.to_csv(os.path.join(OUT_DIR, "straddle_results.csv"), index=False)

# ================================================================
# Step 5 — Summary
# ================================================================
print("\n=== Step 5: MAE summary ===\n")

def agg(sub, col):
    v = sub[sub[col].notna() & (sub["sigma_mkt"] > 0)]
    if v.empty:
        return float("nan"), float("nan")
    return round(v[col].mean(), 1), round((v[col]**2).mean()**0.5, 1)

methods = [
    ("ae_pay",       "Payer (OLS on payer vol)"),
    ("ae_str_pscal", "Straddle @ payer scale"),
    ("ae_str",       "Straddle (OLS on straddle vol)"),
]

summary = {}
for split_n, split_k in [("Train","train"), ("OOS","test"), ("All","all")]:
    sub = df_res if split_k=="all" else df_res[df_res["split"]==split_k]
    print(f"{'--- ' + split_n + ' ---':}")
    print(f"{'Method':<35}  {'MAE':>7}  {'RMSE':>8}")
    print("-"*55)
    for col, label in methods:
        m, r = agg(sub, col)
        summary.setdefault(split_k, {})[col] = {"mae": m, "rmse": r}
        print(f"  {label:<33}  {m:>7.1f}  {r:>8.1f}")
    print()

# Per-cell breakdown for OOS
print(f"\n{'Cell':>8}", end="")
for col, label in methods:
    print(f"  {label[:22]:>22}", end="")
print()
print("-"*(8 + 25*len(methods)))

cell_summary = {}
for cell in cells:
    e, t = cell.expiry, cell.tenor
    sub = df_res[(df_res["expiry"]==e) & (df_res["tenor"]==t) & (df_res["split"]=="test")]
    row_vals = {}
    print(f"  {e}Yx{t}Y", end="")
    for col, label in methods:
        m, _ = agg(sub, col)
        row_vals[col] = m
        print(f"  {m:>22.1f}", end="")
    print()
    cell_summary[f"{e}x{t}"] = row_vals

summary["per_cell"] = cell_summary
with open(os.path.join(OUT_DIR, "straddle_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved straddle_summary.json")

# ================================================================
# Step 6 — LaTeX table
# ================================================================

expiry_vals = sorted(df_s1["expiry"].unique().astype(int))
tenor_vals  = sorted(df_s1["tenor"].unique().astype(int))

lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{OOS vol MAE (bp) under three pricing conventions. "
    r"\emph{Payer}: standard payer swaption implied vol "
    r"(current convention). "
    r"\emph{Straddle @ pay scale}: straddle vol "
    r"$\sigma_{\mathrm{str}} = (V_{\mathrm{pay}}+V_{\mathrm{rec}})/2 "
    r"\cdot \sqrt{2\pi}/(A_0\sqrt{T_e})$ evaluated at the payer-calibrated scale. "
    r"\emph{Straddle}: straddle vol at its own OLS-calibrated scale. "
    r"Under the physical-measure drift bias, $V_{\mathrm{pay}} > V_{\mathrm{rec}}$ "
    r"at ATM; the straddle averages the bias. "
    r"A smaller MAE under the straddle convention indicates that the "
    r"residual error is partly attributable to the $\mathbb{P}$-measure "
    r"drift rather than to vol mis-calibration.}",
    r"\label{tab:straddle_comparison}",
    r"\begin{tabular}{@{}ccrrr@{}}",
    r"\toprule",
    r"\textbf{Exp} & \textbf{Ten} & "
    r"\textbf{Payer (OLS)} & "
    r"\textbf{Straddle @ pay scale} & "
    r"\textbf{Straddle (OLS)} \\",
    r"\midrule",
]
prev_e = None
for cell in cells:
    e, t = cell.expiry, cell.tenor
    if prev_e is not None and e != prev_e:
        lines.append(r"\addlinespace[2pt]")
    prev_e = e
    vals = cell_summary.get(f"{e}x{t}", {})
    def fmt(key):
        v = vals.get(key, float("nan"))
        return "--" if not math.isfinite(v) else str(int(round(v)))
    lines.append(f"{e} & {t} & {fmt('ae_pay')} & {fmt('ae_str_pscal')} & {fmt('ae_str')} \\\\")
lines.append(r"\midrule")
ov = summary.get("test", {})
def ofmt(col):
    v = ov.get(col, {}).get("mae", float("nan"))
    return "--" if not math.isfinite(v) else str(int(round(v)))
lines.append(r"\textbf{Overall} & & " +
             f"{ofmt('ae_pay')} & {ofmt('ae_str_pscal')} & {ofmt('ae_str')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(os.path.join(OUT_DIR, "tab_straddle.tex"), "w") as f:
    f.write("\n".join(lines))
print("Saved tab_straddle.tex")

# ================================================================
# Step 7 — Figure: forward bias per cell
# ================================================================
bias_mat = np.full((len(expiry_vals), len(tenor_vals)), np.nan)
pay_mat  = np.full_like(bias_mat, np.nan)
rec_mat  = np.full_like(bias_mat, np.nan)
mkt_mat  = np.full_like(bias_mat, np.nan)
for (e,t), v in bias_summary.items():
    i = expiry_vals.index(e)
    j = tenor_vals.index(t)
    bias_mat[i,j] = v["bias"]
    pay_mat[i,j]  = v["pay"]
    rec_mat[i,j]  = v["rec"]
    mkt_mat[i,j]  = v["mkt"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
for ax, mat, title, cmap, label in [
    (axes[0], pay_mat,  r"$\sigma_{\rm pay}(s=1)$ bp", "Reds", "Model payer vol (bp)"),
    (axes[1], bias_mat, r"Forward bias $\bar{E}^P[S_T]-F_0$ (bp)", "RdYlGn_r", "Bias (bp)"),
    (axes[2], mkt_mat,  r"Market vol $\sigma_{\rm mkt}$ (bp)", "Blues", "Market vol (bp)"),
]:
    vmax = float(np.nanmax(np.abs(mat))) if not np.all(np.isnan(mat)) else 1
    im = ax.imshow(mat, cmap=cmap, aspect="auto",
                   vmin=-vmax if title.startswith("Forward") else 0,
                   vmax=vmax)
    ax.set_xticks(range(len(tenor_vals)))
    ax.set_xticklabels([f"{c}Y" for c in tenor_vals])
    ax.set_yticks(range(len(expiry_vals)))
    ax.set_yticklabels([f"{r}Y" for r in expiry_vals])
    ax.set_xlabel("Tenor"); ax.set_ylabel("Expiry")
    ax.set_title(title, fontsize=9)
    for i in range(len(expiry_vals)):
        for j in range(len(tenor_vals)):
            v = mat[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                        fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, label=label, fraction=0.04, pad=0.04)
fig.suptitle("P-measure drift bias at s=1: payer vol, forward bias, market vol", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_bias_analysis.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_bias_analysis.png")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
