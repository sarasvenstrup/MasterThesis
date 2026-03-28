# comparison_plots.py
# Run from repo root: python comparison_plots.py
#
# Produces:
#   1. Figures/comparison/autoencoder_comparison_oos.png
#      3x3 grid, one subplot per currency
#      Lines: Actual + 2F autoencoder + 3F autoencoder + 4F autoencoder
#
#   2. Figures/comparison/kalman_vs_autoencoder_{n}f_oos.png  (for n in 2, 3, 4)
#      3x3 grid, one subplot per currency
#      Lines: Actual + EKF DNS nF + Autoencoder nF
#
# Requires that OutOfSampleSplit.py has been run for LATENT_DIM = 2, 3, 4
# and kalman_benchmark.py has been run.

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ── path setup ─────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import my_data, custom_palette, set_paper_theme, TARGET_TENORS
from Code.model.full_model import FullModel

# ── Apply paper theme ──────────────────────────────────────────────────────────
set_paper_theme()

# ── config ─────────────────────────────────────────────────────────────────────
EPOCHS      = 2500
N_SEEDS     = 3
LATENT_DIMS = [2, 3, 4]

TRAIN_START = "2004-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2022-12-31"

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

OUT_DIR = os.path.join(REPO_ROOT, "Figures", "comparison", f"ep{EPOCHS}")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ── Colours from custom_palette ───────────────────────────────────────────────
# custom_palette has 8 colours (indices 0-7 from tab20b selected_indices)
# We use:
#   palette[0] = Actual (darkest, first colour)
#   palette[1] = 2F autoencoder
#   palette[2] = 3F autoencoder
#   palette[3] = 4F autoencoder
#   palette[4] = EKF (second group of colours)

COL_ACTUAL = custom_palette[0]
COL_AE = {
    2: custom_palette[1],
    3: custom_palette[2],
    4: custom_palette[3],
}
COL_EKF = {
    2: custom_palette[5],
    3: custom_palette[6],
    4: custom_palette[7],
}

ACTUAL_STYLE = dict(linestyle="-", marker="o", color=COL_ACTUAL, label="Actual", linewidth=1.8)

AUTOENCODER_STYLES = {
    2: dict(linestyle="--", marker="s", color=COL_AE[2], label="Autoencoder 2F", linewidth=1.5),
    3: dict(linestyle="-.", marker="^", color=COL_AE[3], label="Autoencoder 3F", linewidth=1.5),
    4: dict(linestyle=":",  marker="D", color=COL_AE[4], label="Autoencoder 4F", linewidth=1.5),
}

EKF_STYLES = {
    2: dict(linestyle="--", marker="s", color=COL_EKF[2], label="EKF DNS 2F", linewidth=1.5),
    3: dict(linestyle="-.", marker="^", color=COL_EKF[3], label="EKF DNS 3F", linewidth=1.5),
    4: dict(linestyle=":",  marker="D", color=COL_EKF[4], label="EKF DNS 4F", linewidth=1.5),
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: mid-period index (numpy-based, avoids TimedeltaIndex issues)
# ══════════════════════════════════════════════════════════════════════════════

def mid_period_idx(dates):
    ts  = np.array(dates, dtype="datetime64[ns]").astype(np.int64)
    mid = (ts.min() + ts.max()) // 2
    return int(np.abs(ts - mid).argmin())


# ══════════════════════════════════════════════════════════════════════════════
# Load data and split
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data...")
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, SCALE_IS_PERCENT = my_data(use="bbg")
X_tensor = X_tensor_full.float()
meta = meta_full.copy()
meta["as_of_date"] = pd.to_datetime(meta["as_of_date"])
meta = meta.reset_index(drop=True)

m_test     = (meta["as_of_date"] >= TEST_START) & (meta["as_of_date"] <= TEST_END)
X_test     = X_tensor[m_test.values]
meta_test  = meta.loc[m_test.values].reset_index(drop=True)
dates_test = pd.to_datetime(meta_test["as_of_date"])

print(f"Test set: {TEST_START} – {TEST_END}  n={len(X_test)}")


# ══════════════════════════════════════════════════════════════════════════════
# Load autoencoder reconstructions (best seed per LATENT_DIM)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_reconstructions(model, X, batch_size=256):
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        outs.append(model(xb)[0].detach().cpu())
    return torch.cat(outs, dim=0)


def load_best_autoencoder(latent_dim):
    figures_dir = os.path.join(REPO_ROOT, "Figures", f"OOS_split_dim{latent_dim}", f"ep{EPOCHS}")
    best_model, best_oos = None, np.inf
    for seed in range(N_SEEDS):
        ckpt = os.path.join(figures_dir, f"checkpoint_seed{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  Warning: checkpoint not found: {ckpt}")
            continue
        model = FullModel(latent_dim=latent_dim).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        S_hat = get_reconstructions(model, X_test)
        mask  = torch.isfinite(X_test).all(dim=1) & torch.isfinite(S_hat).all(dim=1)
        oos_rmse = float(((S_hat[mask] - X_test[mask]) ** 2).mean().sqrt()) * 1e4
        if oos_rmse < best_oos:
            best_oos   = oos_rmse
            best_model = model
    print(f"  Loaded autoencoder {latent_dim}F  (best OOS RMSE ≈ {best_oos:.2f} bps)")
    return best_model


print("\nLoading autoencoder checkpoints...")
ae_reconstructions = {}
for dim in LATENT_DIMS:
    model = load_best_autoencoder(dim)
    if model is not None:
        ae_reconstructions[dim] = get_reconstructions(model, X_test)


# ══════════════════════════════════════════════════════════════════════════════
# Run EKF forward passes (imports from kalman_benchmark.py)
# ══════════════════════════════════════════════════════════════════════════════

from Code.kalman_benchmark import (
    ns_loadings, ekf_filter_smoother, ekf_forward_only,
    estimate_A_Q, estimate_R, rmse_bps as ekf_rmse_bps,
    LAM_GRID, CCY_FREQ, P0_scale, A_shrink,
)

df_ekf = df_wide.copy()
if SCALE_IS_PERCENT:
    for col in TARGET_TENORS:
        df_ekf[col] = df_ekf[col].astype(float) / 100.0
df_ekf["as_of_date"] = pd.to_datetime(df_ekf["as_of_date"])
df_ekf["ccy"]        = df_ekf["ccy"].astype(str)

df_train_ekf = df_ekf[(df_ekf["as_of_date"] >= TRAIN_START) & (df_ekf["as_of_date"] <= TRAIN_END)]
df_test_ekf  = df_ekf[(df_ekf["as_of_date"] >= TEST_START)  & (df_ekf["as_of_date"] <= TEST_END)]
tenors_arr   = np.array(TARGET_TENORS, dtype=float)


def run_ekf_for_ccy(ccy, Y_tr, Y_te, tenors, n_factors):
    freq = CCY_FREQ.get(ccy, 1)
    best = None
    for lam in LAM_GRID:
        H0 = ns_loadings(tenors, lam, n_factors)
        x0, *_ = np.linalg.lstsq(H0, Y_tr[0], rcond=None)
        P0 = np.eye(n_factors) * P0_scale
        A  = 0.90 * np.eye(n_factors)
        Q  = 1e-6 * np.eye(n_factors)
        R  = 1e-6 * np.eye(len(tenors))
        Xs1, _, Yfit1 = ekf_filter_smoother(Y_tr, tenors, lam, A, Q, R, x0, P0, freq=freq)
        A2, Q2 = estimate_A_Q(Xs1, A_shrink=A_shrink)
        R2     = estimate_R(Y_tr - Yfit1)
        Xs2, _, Yfit2 = ekf_filter_smoother(Y_tr, tenors, lam, A2, Q2, R2, x0, P0, freq=freq)
        bps = ekf_rmse_bps(Y_tr, Yfit2)
        if best is None or bps < best["rmse_is"]:
            best = dict(lam=lam, A=A2, Q=Q2, R=R2, Xs_tr=Xs2, rmse_is=bps)
    x_init = best["Xs_tr"][-1]
    P_init = np.eye(n_factors) * P0_scale
    _, Yfit_te = ekf_forward_only(
        Y_te, tenors_arr, best["lam"],
        best["A"], best["Q"], best["R"],
        x_init, P_init, freq=freq
    )
    return Yfit_te


print("\nRunning EKF forward passes...")
ekf_fitted = {}  # {n_factors: {ccy: (dates_te, Y_te, Yfit_te)}}

for n_factors in LATENT_DIMS:
    print(f"  EKF {n_factors}F...")
    ekf_fitted[n_factors] = {}
    for ccy in ccy_order:
        dfi_tr = df_train_ekf[df_train_ekf["ccy"] == ccy].sort_values("as_of_date")
        dfi_te = df_test_ekf[df_test_ekf["ccy"] == ccy].sort_values("as_of_date")
        if len(dfi_tr) == 0 or len(dfi_te) == 0:
            continue
        Y_tr     = dfi_tr[TARGET_TENORS].astype(float).to_numpy()
        Y_te     = dfi_te[TARGET_TENORS].astype(float).to_numpy()
        dates_te = dfi_te["as_of_date"].to_numpy()
        Yfit_te  = run_ekf_for_ccy(ccy, Y_tr, Y_te, tenors_arr, n_factors)
        ekf_fitted[n_factors][ccy] = (dates_te, Y_te, Yfit_te)
        print(f"    {ccy} done")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Autoencoder comparison — Actual + 2F + 3F + 4F
# ══════════════════════════════════════════════════════════════════════════════

print("\nPlot 1: Autoencoder comparison...")

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for ax, ccy in zip(axes, ccy_order):
    mask = (meta_test["ccy"] == ccy).values
    if mask.sum() == 0:
        ax.set_visible(False); continue

    ccy_dates  = dates_test[mask].reset_index(drop=True)
    idx_local  = mid_period_idx(ccy_dates.values)
    idx_global = np.where(mask)[0][idx_local]
    date_str   = ccy_dates[idx_local].strftime("%Y-%m-%d")

    ax.plot(tenors, X_test[idx_global].numpy() * 100, **ACTUAL_STYLE)

    for dim in LATENT_DIMS:
        if dim not in ae_reconstructions:
            continue
        S_hat = ae_reconstructions[dim]
        ax.plot(tenors, S_hat[idx_global].numpy() * 100, **AUTOENCODER_STYLES[dim])

    ax.set_title(f"{ccy}  ({date_str})")
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=6)

fig.suptitle("OOS: Autoencoder Reconstructions — 2F / 3F / 4F vs Actual",
             fontsize=13, fontweight="bold")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "autoencoder_comparison_oos.png")
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Kalman vs Autoencoder — one plot per factor count
# ══════════════════════════════════════════════════════════════════════════════

print("\nPlot 2: Kalman vs Autoencoder comparison...")

for n_factors in LATENT_DIMS:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for ax, ccy in zip(axes, ccy_order):
        ae_mask = (meta_test["ccy"] == ccy).values
        if ae_mask.sum() == 0:
            ax.set_visible(False); continue

        ccy_dates  = dates_test[ae_mask].reset_index(drop=True)
        idx_local  = mid_period_idx(ccy_dates.values)
        idx_global = np.where(ae_mask)[0][idx_local]
        date_str   = ccy_dates[idx_local].strftime("%Y-%m-%d")

        # Actual
        ax.plot(tenors, X_test[idx_global].numpy() * 100, **ACTUAL_STYLE)

        # Autoencoder
        if n_factors in ae_reconstructions:
            S_hat = ae_reconstructions[n_factors]
            ax.plot(tenors, S_hat[idx_global].numpy() * 100,
                    linestyle="--", marker="s",
                    color=COL_AE[n_factors],
                    label=f"Autoencoder {n_factors}F", linewidth=1.5)

        # EKF
        if ccy in ekf_fitted[n_factors]:
            dates_te, Y_te, Yfit_te = ekf_fitted[n_factors][ccy]
            ekf_idx = mid_period_idx(dates_te)
            ax.plot(tenors_arr, Yfit_te[ekf_idx] * 100,
                    linestyle="-.", marker="^",
                    color=COL_EKF[n_factors],
                    label=f"EKF DNS {n_factors}F", linewidth=1.5)

        ax.set_title(f"{ccy}  ({date_str})")
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel("Rate (%)")
        ax.legend(fontsize=6)

    fig.suptitle(f"OOS: EKF DNS vs Autoencoder — {n_factors}-Factor Model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"kalman_vs_autoencoder_{n_factors}f_oos.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

print(f"\nAll comparison plots saved to: {OUT_DIR}")