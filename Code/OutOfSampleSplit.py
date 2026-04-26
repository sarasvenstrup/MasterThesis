# OutOfSampleSplit.py
# Run from repo root: python OutOfSampleSplit.py
# This script will be used to test the robustness of the model.

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# ── path setup ─────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.utils import helpers as H
from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS
from Code.model.full_model import FullModel

VARIANT = "baseline"  # frozen — no config dependency

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.mkldnn.enabled = False  # IMPORTANT: Disable MKLDNN for numerical stability

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
print("MKLDNN enabled:", torch.backends.mkldnn.enabled)

# ── config ─────────────────────────────────────────────────────────────────────
USE        = "bbg"
LATENT_DIM = 4
EPOCHS     = 2500
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
N_SEEDS    = 12        # train N times with different seeds, report average
max_lr     = 3e-3
final_div_factor = 3000.0
LOG_EVERY  = 100

# Train/test split dates
TRAIN_START = "2010-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2022-12-31"

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", "OOSResults", "Split", f"OOS_split_dim{LATENT_DIM}_{VARIANT}", f"ep{EPOCHS}")
os.makedirs(FIGURES_DIR, exist_ok=True)
print("Output dir:", FIGURES_DIR)



# ── load data ──────────────────────────────────────────────────────────────────
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor_full.float()

meta = meta_full.copy()
meta["as_of_date"] = pd.to_datetime(meta["as_of_date"])
meta = meta.reset_index(drop=True)

assert len(meta) == X_tensor.shape[0], "meta and X_tensor length mismatch"

# ── split ──────────────────────────────────────────────────────────────────────
m_train = (meta["as_of_date"] >= TRAIN_START) & (meta["as_of_date"] <= TRAIN_END)
m_test  = (meta["as_of_date"] >= TEST_START)  & (meta["as_of_date"] <= TEST_END)

X_train    = X_tensor[m_train.values]
X_test     = X_tensor[m_test.values]
meta_train = meta.loc[m_train.values].reset_index(drop=True)
meta_test  = meta.loc[m_test.values].reset_index(drop=True)

print(f"Train: {TRAIN_START} – {TRAIN_END}  n={len(X_train)}")
print(f"Test:  {TEST_START}  – {TEST_END}   n={len(X_test)}")

# ── helpers ────────────────────────────────────────────────────────────────────
def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)

@torch.no_grad()
def predict_S_hat(model, X, batch_size=256):
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        outs.append(model(xb).detach().cpu())
    return torch.cat(outs, dim=0)

@torch.no_grad()
def predict_full(model, X, batch_size=256):
    """Returns S_hat, z, SR_tau for the full dataset."""
    model.eval()
    all_S, all_z, all_SR = [], [], []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        S_hat, aux = model(xb, return_aux=True, do_arb_checks=True)
        all_S.append(S_hat.cpu())
        all_z.append(aux["z"].cpu())
        all_SR.append(aux["arb"]["SR_tau"].cpu())
    return torch.cat(all_S), torch.cat(all_z), torch.cat(all_SR)

def rmse_bps_on_subset(model, X_sub, meta_sub):
    S_hat = predict_S_hat(model, X_sub, batch_size=EVAL_BATCH_SIZE)
    mask  = row_finite_mask(X_sub) & row_finite_mask(S_hat)
    rmse_per_ccy = H.rmse_bps_per_currency_paper(
        X_sub[mask], S_hat[mask],
        meta_sub.loc[mask.numpy()].reset_index(drop=True)
    )
    return rmse_per_ccy, float(rmse_per_ccy.mean())

def train_model(X_tr, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model  = FullModel(latent_dim=LATENT_DIM).to(device)
    model.train()
    loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    optim  = torch.optim.Adam(model.parameters(), lr=max_lr)
    sched  = OneCycleLR(optim, max_lr=max_lr, steps_per_epoch=len(loader),
                        epochs=EPOCHS, pct_start=0.3, div_factor=1.0,
                        final_div_factor=final_div_factor)
    loss_fn = nn.MSELoss()
    mse_hist, lr_hist = [], []

    for epoch in range(EPOCHS):
        running, n_obs = 0.0, 0
        for (xb_cpu,) in loader:
            xb = xb_cpu.to(device)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), xb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            sched.step()
            running += float(loss.detach().cpu()) * xb.shape[0]
            n_obs   += xb.shape[0]

        epoch_mse = running / max(n_obs, 1)
        mse_hist.append(epoch_mse)
        lr_hist.append(optim.param_groups[0]["lr"])

        if (epoch == 0) or ((epoch + 1) % LOG_EVERY == 0) or (epoch == EPOCHS - 1):
            print(f"  [seed={seed}] epoch={epoch:4d}  train_rmse={epoch_mse**0.5:.6e}  "
                  f"lr={optim.param_groups[0]['lr']:.2e}")

    return model, np.array(mse_hist), np.array(lr_hist)

# ── train N seeds ──────────────────────────────────────────────────────────────
is_rmse_all,  oos_rmse_all  = [], []   # one pd.Series per seed
is_avg_all,   oos_avg_all   = [], []   # one float per seed
best_model, best_oos = None, np.inf

SEEDS = list(range(1, N_SEEDS + 1))  # explicit — change these integers to reproduce exact runs

# ── run manifest (written before loop, updated after each seed) ────────────────
manifest = {
    "seeds": SEEDS,
    "latent_dim": LATENT_DIM,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_lr": max_lr,
    "final_div_factor": final_div_factor,
    "train_start": TRAIN_START,
    "train_end": TRAIN_END,
    "test_start": TEST_START,
    "test_end": TEST_END,
    "torch_version": torch.__version__,
    "numpy_version": np.__version__,
    "run_started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "seed_results": {},
}
manifest_path = os.path.join(FIGURES_DIR, "run_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest initialised: {manifest_path}")

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"Training seed {seed+1}/{N_SEEDS}  (seed={seed})")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    model, mse_hist, lr_hist = train_model(X_train, seed=seed)
    train_minutes = (time.perf_counter() - t0) / 60
    print(f"  Training time: {train_minutes:.1f} min")

    # save checkpoint per seed
    ckpt = os.path.join(FIGURES_DIR, f"checkpoint_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt)

    # evaluate
    is_rmse,  is_avg  = rmse_bps_on_subset(model, X_train, meta_train)
    oos_rmse, oos_avg = rmse_bps_on_subset(model, X_test,  meta_test)

    is_rmse_all.append(is_rmse)
    oos_rmse_all.append(oos_rmse)
    is_avg_all.append(is_avg)
    oos_avg_all.append(oos_avg)

    print(f"  In-sample avg RMSE:  {is_avg:.2f} bps")
    print(f"  OOS avg RMSE:        {oos_avg:.2f} bps")

    # update manifest after every seed (crash-safe)
    manifest["seed_results"][str(seed)] = {
        "is_avg_bps":       round(is_avg,  4),
        "oos_avg_bps":      round(oos_avg, 4),
        "is_per_ccy_bps":   {k: round(v, 4) for k, v in is_rmse.items()},
        "oos_per_ccy_bps":  {k: round(v, 4) for k, v in oos_rmse.items()},
        "checkpoint":       f"checkpoint_seed{seed}.pt",
        "train_minutes":    round(train_minutes, 2),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # keep best model for plots
    if oos_avg < best_oos:
        best_oos   = oos_avg
        best_model = model
        manifest["best_seed"] = seed
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # convergence plot per seed
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.plot(np.sqrt(mse_hist))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train RMSE")
    ax.set_title(f"Training convergence — seed {seed}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"convergence_seed{seed}.png"), dpi=200)
    plt.close(fig)

# ── finalise manifest ──────────────────────────────────────────────────────────
manifest["run_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest finalised: {manifest_path}")

# ── 1) RMSE table (mean ± std across seeds) ────────────────────────────────────
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")

is_df  = pd.DataFrame(is_rmse_all)
oos_df = pd.DataFrame(oos_rmse_all)

summary = pd.DataFrame({
    "IS mean (bps)":  is_df.mean(),
    "IS std (bps)":   is_df.std(),
    "OOS mean (bps)": oos_df.mean(),
    "OOS std (bps)":  oos_df.std(),
})
summary.loc["Average"] = summary.mean()

print(summary.to_string())
summary.to_csv(os.path.join(FIGURES_DIR, "rmse_summary.csv"))
print("\nSaved RMSE table.")

# ── 2) Fitted vs actual curves (best model) ────────────────────────────────────
S_hat_test, Z_test, SR_test = predict_full(best_model, X_test)
S_hat_train, Z_train, SR_train = predict_full(best_model, X_train)

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

    ax.plot(tenors, X_test[idx_global].numpy() * 100,  "o-",  label="Actual", linewidth=1.8)
    ax.plot(tenors, S_hat_test[idx_global].numpy() * 100, "s--", label="Fitted", linewidth=1.8)
    ax.set_title(f"{ccy}  ({ccy_dates[idx_local].strftime('%Y-%m-%d')})")
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=7)

fig.suptitle("OOS: Fitted vs Actual Swap Curves (best seed)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "oos_fitted_vs_actual.png"), dpi=200)
plt.close(fig)
print("Saved fitted vs actual plot.")

# ── 3) Latent factor paths ─────────────────────────────────────────────────────
for split_label, Z_split, meta_split in [
    ("train", Z_train, meta_train),
    ("oos",   Z_test,  meta_test)
]:
    dates_split = pd.to_datetime(meta_split["as_of_date"])
    fig, axes = plt.subplots(LATENT_DIM, 1, figsize=(12, 3 * LATENT_DIM), sharex=True)
    if LATENT_DIM == 1:
        axes = [axes]

    for ccy in ccy_order:
        mask     = (meta_split["ccy"] == ccy).values
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

print("Saved latent factor plots.")

# ── 4) Sharpe ratio check on OOS ──────────────────────────────────────────────
tau_grid = torch.arange(1, 31).float().numpy()
SR_mean  = SR_test.mean(dim=0).numpy()
SR_std   = SR_test.std(dim=0).numpy()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tau_grid, SR_mean, linewidth=2, label="Mean OOS Sharpe ratio")
ax.fill_between(tau_grid, SR_mean - SR_std, SR_mean + SR_std, alpha=0.2, label="±1 std")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Tenor (years)")
ax.set_ylabel("Approx. Sharpe ratio")
ax.set_title("OOS arbitrage check — approx. Sharpe ratio across tenors")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "oos_sharpe_ratio.png"), dpi=200)
plt.close(fig)
print("Saved Sharpe ratio plot.")

print(f"\nAll outputs saved to: {FIGURES_DIR}")