"""
repair_split_dim2.py
Generates the diagnostic plots that OutOfSampleSplit.py missed for dim=2
due to a crash after saving checkpoints and rmse_summary.csv.

Saves results to the same folder OutOfSampleSplit would have used:
  Figures/OOS_split_dim2_baseline/ep2500/

This will be deleted as soon as the split OOS has run!

"""
import os
import sys
import json
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

from Code.utils import helpers as H
from Code.load_swapdata import my_data, TARGET_TENORS
from Code.model.full_model import FullModel
from Code.config import VARIANT

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ── config (must match what OutOfSampleSplit used for dim=2) ───────────────────
LATENT_DIM  = 2
EPOCHS      = 2500
TRAIN_START = "2010-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2022-12-31"
ccy_order   = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", "OOSResults", "Split",
                           f"OOS_split_dim{LATENT_DIM}_{VARIANT}",
                           f"ep{EPOCHS}")

print(f"Repair script — dim={LATENT_DIM}")
print(f"Output dir: {FIGURES_DIR}")

# ── check all checkpoints exist ────────────────────────────────────────────────
manifest_path = os.path.join(FIGURES_DIR, "run_manifest.json")
with open(manifest_path) as f:
    manifest = json.load(f)

best_seed = manifest["best_seed"]
ckpt_path = os.path.join(FIGURES_DIR, f"checkpoint_seed{best_seed}.pt")
print(f"Best seed: {best_seed}  checkpoint: {ckpt_path}")
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

# ── load data ──────────────────────────────────────────────────────────────────
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use="bbg")
X_tensor = X_tensor_full.float()
meta = meta_full.copy()
meta["as_of_date"] = pd.to_datetime(meta["as_of_date"])
meta = meta.reset_index(drop=True)

m_train = (meta["as_of_date"] >= TRAIN_START) & (meta["as_of_date"] <= TRAIN_END)
m_test  = (meta["as_of_date"] >= TEST_START)  & (meta["as_of_date"] <= TEST_END)

X_train    = X_tensor[m_train.values]
X_test     = X_tensor[m_test.values]
meta_train = meta.loc[m_train.values].reset_index(drop=True)
meta_test  = meta.loc[m_test.values].reset_index(drop=True)

print(f"Train n={len(X_train)}, Test n={len(X_test)}")

# ── load best model ────────────────────────────────────────────────────────────
model = FullModel(latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print("Loaded best model checkpoint.")

# ── predict ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_full(mdl, X, batch_size=256):
    all_S, all_z, all_SR = [], [], []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        S_hat, aux = mdl(xb, return_aux=True, do_arb_checks=True)
        all_S.append(S_hat.cpu())
        all_z.append(aux["z"].cpu())
        all_SR.append(aux["arb"]["SR_tau"].cpu())
    return torch.cat(all_S), torch.cat(all_z), torch.cat(all_SR)

S_hat_test,  Z_test,  SR_test  = predict_full(model, X_test)
S_hat_train, Z_train, SR_train = predict_full(model, X_train)
print("Predictions done.")

# ── plot 1: fitted vs actual ───────────────────────────────────────────────────
dates_test = pd.to_datetime(meta_test["as_of_date"])
mid_date   = dates_test.min() + (dates_test.max() - dates_test.min()) / 2

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for ax, ccy in zip(axes, ccy_order):
    mask = (meta_test["ccy"] == ccy).values
    if mask.sum() == 0:
        ax.set_visible(False)
        continue
    ccy_dates = dates_test[mask].reset_index(drop=True)
    idx_local  = (ccy_dates - mid_date).abs().argmin()
    idx_global = np.where(mask)[0][idx_local]

    ax.plot(tenors, X_test[idx_global].numpy() * 100,      "o-",  label="Actual", linewidth=1.8)
    ax.plot(tenors, S_hat_test[idx_global].numpy() * 100,  "s--", label="Fitted", linewidth=1.8)
    ax.set_title(f"{ccy}  ({ccy_dates[idx_local].strftime('%Y-%m-%d')})")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=7)

fig.suptitle("OOS: Fitted vs Actual Swap Curves (best seed)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "oos_fitted_vs_actual.png"), dpi=200)
plt.close(fig)
print("Saved oos_fitted_vs_actual.png")

# ── plot 2: latent factor paths ────────────────────────────────────────────────
for split_label, Z_split, meta_split in [
    ("train", Z_train, meta_train),
    ("oos",   Z_test,  meta_test),
]:
    dates_split = pd.to_datetime(meta_split["as_of_date"])
    fig, axes = plt.subplots(LATENT_DIM, 1, figsize=(12, 3 * LATENT_DIM), sharex=True)
    if LATENT_DIM == 1:
        axes = [axes]

    for ccy in ccy_order:
        mask = (meta_split["ccy"] == ccy).values
        if mask.sum() == 0:
            continue
        ccy_dates = dates_split.values[mask]
        ccy_z     = Z_split[mask].numpy()
        sort_idx  = np.argsort(ccy_dates)
        for dim_i, ax in enumerate(axes):
            ax.plot(ccy_dates[sort_idx], ccy_z[sort_idx, dim_i],
                    linewidth=1.2, label=ccy, alpha=0.85)

    for dim_i, ax in enumerate(axes):
        ax.set_ylabel(f"z{dim_i+1}")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if dim_i == 0:
            ax.legend(ncol=3, fontsize=7)

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Latent factor paths — {split_label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"latent_factors_{split_label}.png"), dpi=200)
    plt.close(fig)

print("Saved latent_factors_train.png and latent_factors_oos.png")

# ── plot 3: Sharpe ratio ───────────────────────────────────────────────────────
tau_grid = torch.arange(1, 31).float().numpy()
SR_mean  = SR_test.mean(dim=0).numpy()
SR_std   = SR_test.std(dim=0).numpy()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tau_grid, SR_mean, linewidth=2, label="Mean OOS Sharpe ratio")
ax.fill_between(tau_grid, SR_mean - SR_std, SR_mean + SR_std, alpha=0.2, label="±1 std")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Maturity")
ax.set_ylabel("Approx. Sharpe ratio")
ax.set_title("OOS arbitrage check — approx. Sharpe ratio across tenors")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "oos_sharpe_ratio.png"), dpi=200)
plt.close(fig)
print("Saved oos_sharpe_ratio.png")

print(f"\nRepair complete. All outputs saved to: {FIGURES_DIR}")
