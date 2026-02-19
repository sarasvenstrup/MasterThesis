# ============================================
# FULL SCRIPT: TRAIN (paper settings) + EVAL + PLOTS + SAVE FIGURES
# Local repo data (SwapDAta/TestData + SwapDAta/Bloombergdata), no Spark.
# Matches Section 2.4:
#  - Adam, lr = 1e-3
#  - batch size = 32
#  - 1000 epochs
#  - in-sample RMSE table (bps) per currency
# Saves all plots to: <repo_root>/figures/<USE>/
# ============================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# 0) Paths + imports
# -----------------------------
# If running as script → __file__ exists
# If running in console → fallback to current working directory
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Your model import (keep as you wrote; adjust if your package name differs)
from Code.model.full_model import FullModel

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 0b) Figure saving utilities
# -----------------------------
USE = "test"   # "test" (debug first) then switch to "bbg"

FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", USE)
os.makedirs(FIGURES_DIR, exist_ok=True)

def save_figure(name: str, dpi: int = 300):
    """Save current matplotlib figure as PNG + PDF into figures/<USE>/"""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))
    png_path = os.path.join(FIGURES_DIR, f"{safe}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{safe}.pdf")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {png_path}")

# Optional: consistent figure style
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
})

# -----------------------------
# 1) Load data from local repo folder SwapDAta
# -----------------------------
# Assumes your loader lives at SwapDAta/load_swapdata.py and defines:
#   - build_all_dataframes()
#   - TARGET_TENORS = [1,2,3,5,10,15,20,30]
from Code.load_swapdata import build_all_dataframes, TARGET_TENORS

data = build_all_dataframes()

if USE == "test":
    df_wide_full = data["df_wide_test_full"].copy()
    df_long = data["df_long_test"].copy()
else:
    df_wide_full = data["df_wide_bbg_full"].copy()
    df_long = data["df_long_bbg"].copy()

print("\nLoaded long rows:", len(df_long))
print("Loaded wide full rows:", len(df_wide_full))
print("Currencies found:", sorted(df_wide_full["ccy"].unique())[:30])

# Optional: filter to one currency folder if you want (e.g. 'sw')
# df_wide_full = df_wide_full[df_wide_full["ccy"].str.lower() == "sw"].copy()

# Tenor grid (what the model expects)
WANTED_TENORS = [float(x) for x in TARGET_TENORS]  # for plotting
tenors = np.array(WANTED_TENORS, dtype=float)

# Ensure wide has exactly these columns (id + 8 tenors)
df_wide = df_wide_full[["as_of_date", "ccy"] + TARGET_TENORS].copy()
df_wide["as_of_date"] = pd.to_datetime(df_wide["as_of_date"])

meta = df_wide[["as_of_date", "ccy"]].reset_index(drop=True)
X = df_wide[TARGET_TENORS].to_numpy(dtype=np.float32)  # may be percent or decimals
print("Wide shape:", X.shape)

# -----------------------------
# 2) Currency rename + colors
# -----------------------------
currency_rename_map = {
    'ad': 'AUD', 'AD': 'AUD',
    'cd': 'CAD', 'CD': 'CAD',
    'dk': 'DKK', 'DK': 'DKK',
    'eu': 'EUR', 'EU': 'EUR',
    'jy': 'JPY', 'JY': 'JPY',
    'nk': 'NOK', 'NK': 'NOK',
    'sk': 'SEK', 'SK': 'SEK',
    'uk': 'GBP', 'UK': 'GBP',
    'us': 'USD', 'US': 'USD'
}
meta["ccy"] = meta["ccy"].map(lambda x: currency_rename_map.get(x, x))
df_wide["ccy"] = df_wide["ccy"].map(lambda x: currency_rename_map.get(x, x))

currency_color_map = {
    'AUD': 'pink', 'CAD': 'grey', 'DKK': 'red', 'EUR': 'blue', 'JPY': 'black',
    'NOK': 'orange', 'SEK': 'purple', 'GBP': 'green', 'USD': 'brown'
}

# -----------------------------
# 3) Scale to decimals (auto-detect)
# -----------------------------
median_abs = float(np.nanmedian(np.abs(X)))
SCALE_IS_PERCENT = median_abs > 0.5
print("Median |swap|:", median_abs, "=> SCALE_IS_PERCENT =", SCALE_IS_PERCENT)

if SCALE_IS_PERCENT:
    X = X / 100.0

X_tensor = torch.from_numpy(X)  # CPU (N,8)
print("X_tensor:", tuple(X_tensor.shape))
print("First row TRUE:", X_tensor[0].numpy())

# -----------------------------
# 4) DataLoader (paper settings)
# -----------------------------
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 32      # paper
LR = 1e-3            # paper
EPOCHS = 1000        # paper

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# -----------------------------
# 5) Train
# -----------------------------
torch.manual_seed(0)

model = FullModel().to(device)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

train_losses = []
nan_batches_total = 0

for epoch in range(EPOCHS):
    running = 0.0
    n_obs = 0
    nan_batches = 0

    for (xb_cpu,) in loader:
        xb = xb_cpu.to(device)

        optim.zero_grad()
        out = model(xb)
        S_hat = out[0]

        loss = loss_fn(S_hat, xb)

        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        loss.backward()
        optim.step()

        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

    nan_batches_total += nan_batches
    epoch_loss = running / max(n_obs, 1)
    train_losses.append(epoch_loss)

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(
            f"epoch={epoch:4d} loss={epoch_loss:.6e} "
            f"used_obs={n_obs} nan_batches={nan_batches} total_nan_batches={nan_batches_total}"
        )

print("Training done.")

# -----------------------------
# 6) Inference (in-sample)
# -----------------------------
@torch.no_grad()
def run_model_batches(model, X_tensor_cpu, batch_size=256, device="cpu"):
    model.eval()
    S_hats, zs = [], []
    N = X_tensor_cpu.shape[0]
    for i in range(0, N, batch_size):
        xb = X_tensor_cpu[i:i+batch_size].to(device)
        out = model(xb)
        S_hat, z = out[0], out[1]
        S_hats.append(S_hat.detach().cpu())
        zs.append(z.detach().cpu())
    return torch.cat(S_hats, dim=0), torch.cat(zs, dim=0)

model.eval()
S_hat_all, z_all = run_model_batches(model, X_tensor, batch_size=256, device=device)

# -----------------------------
# 7) Filter non-finite rows
# -----------------------------
def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)

mask = row_finite_mask(X_tensor) & row_finite_mask(S_hat_all)
n_bad = int((~mask).sum().item())
print(f"Non-finite rows: {n_bad} / {len(mask)}")

X_eval = X_tensor[mask]
S_eval = S_hat_all[mask]
meta_eval = meta.loc[mask.numpy()].reset_index(drop=True)

# -----------------------------
# 8) RMSE per currency (bps) + save table
# -----------------------------
def rmse_bps_per_currency_paper(S_true, S_pred, meta_df):
    if torch.is_tensor(S_true):
        S_true = S_true.detach().cpu().numpy()
    if torch.is_tensor(S_pred):
        S_pred = S_pred.detach().cpu().numpy()

    err = (S_pred - S_true)  # decimals
    tmp = meta_df.copy()

    rmses = {}
    for ccy in tmp["ccy"].unique():
        idx = (tmp["ccy"].values == ccy)
        e = err[idx, :]  # (N_ccy, 8)
        rmses[ccy] = float(np.sqrt(np.mean(e**2)) * 10000.0)  # bps

    out = pd.Series(rmses).sort_values()
    out.loc["Average"] = out.mean()
    return out

rmse_series = rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)
print("\nSteady-state in-sample RMSE (bps) per currency:")
print(rmse_series.to_frame("RMSE (bps)"))

rmse_path = os.path.join(FIGURES_DIR, f"rmse_bps_{USE}.csv")
rmse_series.to_frame("RMSE (bps)").to_csv(rmse_path)
print("Saved RMSE table:", rmse_path)

# -----------------------------
# 9) Plots (all saved)
# -----------------------------
# 9a) Training loss
plt.figure(figsize=(6, 3.5))
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Train MSE")
plt.title(f"Training loss (in-sample) — USE={USE}")
save_figure(f"training_loss_{USE}")
plt.show()

# 9b) Actual vs reconstructed on one date
def plot_recon_on_date(df_wide_used, S_hat_all_eval, meta_eval_df, date_pick):
    m = meta_eval_df.copy()
    idx = (m["as_of_date"] == pd.to_datetime(date_pick)).values
    if idx.sum() == 0:
        raise ValueError("No rows found for that date.")

    X_true = df_wide_used.loc[idx, TARGET_TENORS].to_numpy(dtype=np.float32)
    if SCALE_IS_PERCENT:
        X_true = X_true / 100.0

    X_pred = S_hat_all_eval[idx].detach().cpu().numpy()
    ccys = m.loc[idx, "ccy"].values

    plt.figure(figsize=(8, 4))
    for i, ccy in enumerate(ccys):
        col = currency_color_map.get(ccy, None)
        plt.plot(tenors, X_true[i], marker="o", color=col, alpha=0.6)
        plt.plot(tenors, X_pred[i], marker="x", linestyle="--", color=col, alpha=0.9)

    plt.xlabel("Tenor (years)")
    plt.ylabel("Swap rate (decimals)")
    d = pd.to_datetime(date_pick).date()
    plt.title(f"Actual (o) vs Reconstructed (x) on {d} — USE={USE}")
    plt.grid(True)
    save_figure(f"reconstruction_{USE}_{d}")
    plt.show()

date_pick = meta_eval["as_of_date"].iloc[0]
df_wide_eval = df_wide.loc[mask.numpy()].reset_index(drop=True)
plot_recon_on_date(df_wide_eval, S_hat_all[mask], meta_eval, date_pick)

# 9c) Latent factors over time
def plot_latents_over_time(z_eval_t, meta_eval_df):
    z_np = z_eval_t.detach().cpu().numpy()
    m = meta_eval_df.copy()
    m["z1"] = z_np[:, 0]
    m["z2"] = z_np[:, 1]
    m = m.sort_values(["ccy", "as_of_date"])

    plt.figure(figsize=(11, 4))

    plt.subplot(1, 2, 1)
    for ccy, g in m.groupby("ccy"):
        plt.plot(g["as_of_date"], g["z1"], color=currency_color_map.get(ccy, None), label=ccy)
    plt.title("Latent factor z1")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for ccy, g in m.groupby("ccy"):
        plt.plot(g["as_of_date"], g["z2"], color=currency_color_map.get(ccy, None), label=ccy)
    plt.title("Latent factor z2")
    plt.grid(True)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    save_figure(f"latent_factors_{USE}")
    plt.show()

plot_latents_over_time(z_all[mask], meta_eval)

# -----------------------------
# 10) Parameter-model plots
# -----------------------------
@torch.no_grad()
def run_model_full_batches(model, X_tensor_cpu, batch_size=256, device="cpu"):
    model.eval()
    zs, mus, sigmas, rts = [], [], [], []
    N = X_tensor_cpu.shape[0]
    for i in range(0, N, batch_size):
        xb = X_tensor_cpu[i:i+batch_size].to(device)
        out = model(xb)
        z, mu, sigma, r_tilde = out[1], out[6], out[7], out[8]
        zs.append(z.detach().cpu())
        mus.append(mu.detach().cpu())
        sigmas.append(sigma.detach().cpu())
        rts.append(r_tilde.detach().cpu())
    z_all_ = torch.cat(zs, dim=0)
    mu_all_ = torch.cat(mus, dim=0)
    sigma_all_ = torch.cat(sigmas, dim=0)
    r_all_ = torch.cat(rts, dim=0)
    return z_all_, mu_all_, sigma_all_, r_all_

z_all_full, mu_all_full, sigma_all_full, r_all_full = run_model_full_batches(
    model, X_tensor, batch_size=256, device=device
)

z_eval = z_all_full[mask]
mu_eval = mu_all_full[mask]
sigma_eval = sigma_all_full[mask]
r_eval = r_all_full[mask]

if r_eval.ndim == 2 and r_eval.shape[1] == 1:
    r_eval = r_eval.squeeze(1)

sigma1_eval = sigma_eval[:, 0, 0]
sigma2_eval = torch.sqrt(sigma_eval[:, 1, 0]**2 + sigma_eval[:, 1, 1]**2)
rho_eval = sigma_eval[:, 1, 0] / torch.clamp(sigma2_eval, min=1e-12)

params_df = meta_eval.copy()
params_df["mu1"] = mu_eval[:, 0].numpy()
params_df["mu2"] = mu_eval[:, 1].numpy()
params_df["sigma1"] = sigma1_eval.numpy()
params_df["sigma2"] = sigma2_eval.numpy()
params_df["rho"] = rho_eval.numpy()
params_df["r_tilde"] = r_eval.numpy()
params_df = params_df.sort_values(["ccy", "as_of_date"])

def plot_param_over_time(params_df, col, title=None):
    plt.figure(figsize=(11, 4))
    for ccy, g in params_df.groupby("ccy"):
        plt.plot(g["as_of_date"], g[col], color=currency_color_map.get(ccy, None), label=ccy)
    plt.title(title if title is not None else col)
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    save_figure(f"{col}_{USE}")
    plt.show()

plot_param_over_time(params_df, "r_tilde", title="Short rate mapping r̃(z)")
plot_param_over_time(params_df, "sigma1",  title="Volatility σ1(z)")
plot_param_over_time(params_df, "sigma2",  title="Volatility σ2(z)")
plot_param_over_time(params_df, "rho",     title="Correlation ρ(z)")
plot_param_over_time(params_df, "mu1",     title="Drift μ1(z)")
plot_param_over_time(params_df, "mu2",     title="Drift μ2(z)")

def hist_param(params_df, col, bins=50):
    plt.figure(figsize=(6, 3.5))
    plt.hist(params_df[col].values, bins=bins)
    plt.title(f"Histogram of {col}")
    plt.grid(True)
    save_figure(f"hist_{col}_{USE}")
    plt.show()

hist_param(params_df, "r_tilde")
hist_param(params_df, "sigma1")
hist_param(params_df, "sigma2")
hist_param(params_df, "rho")
hist_param(params_df, "mu1")
hist_param(params_df, "mu2")

print(f"Done. Figures saved to: {FIGURES_DIR}")
