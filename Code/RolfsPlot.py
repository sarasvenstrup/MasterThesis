# ============================================
# FULL SCRIPT: TRAIN (paper settings) + EVAL + PLOTS
# Matches Section 2.4:
#  - Adam, lr = 1e-3
#  - batch size = 32
#  - 1000 epochs
#  - in-sample RMSE table (bps) per currency
# ============================================

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# 0) Project path + imports
# -----------------------------
PROJECT_PATH = "/Workspace/Users/sara@svenstrup.net/Master-Thesis"
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from models.full_model import FullModel

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 1) Load data from Delta table
# -----------------------------
TABLE_NAME = "workspace.default.swap_quotes"
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

df = spark.table(TABLE_NAME).toPandas()
df = df.copy()
df["as_of_date"] = pd.to_datetime(df["as_of_date"])

expected = {"as_of_date", "ccy", "maturity_years", "swap_rate"}
missing = expected - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {TABLE_NAME}: {missing}")

print("Loaded rows:", len(df))
print("Raw swap_rate sample:", df["swap_rate"].head(10).tolist())

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
df["ccy"] = df["ccy"].map(lambda x: currency_rename_map.get(x, x))

currency_color_map = {
    'AUD': 'pink', 'CAD': 'grey', 'DKK': 'red', 'EUR': 'blue', 'JPY': 'black',
    'NOK': 'orange', 'SEK': 'purple', 'GBP': 'green', 'USD': 'brown'
}

# -----------------------------
# 3) Pivot long -> wide (N, 8)
# -----------------------------
WANTED_TENORS = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]
tenors = np.array(WANTED_TENORS, dtype=float)

df_wide = (
    df.pivot_table(
        index=["as_of_date", "ccy"],
        columns="maturity_years",
        values="swap_rate",
        aggfunc="first",
    )
    .sort_index(axis=1)
)

df_wide = df_wide.reindex(columns=WANTED_TENORS)

# Paper does not state missing-tenor handling.
# If you want strict, use dropna(). If you want maximum data, interpolate.
INTERPOLATE_MISSING = True
if INTERPOLATE_MISSING:
    df_wide = df_wide.interpolate(axis=1).ffill(axis=1).bfill(axis=1)
else:
    df_wide = df_wide.dropna(axis=0, how="any")

meta = df_wide.reset_index()[["as_of_date", "ccy"]]
X = df_wide.to_numpy(dtype=np.float32)  # percent units initially
print("Wide shape:", df_wide.shape)

# -----------------------------
# 4) Scale to decimals (your data is in percent)
# -----------------------------
SCALE_IS_PERCENT = True
if SCALE_IS_PERCENT:
    X = X / 100.0

X_tensor = torch.from_numpy(X)  # CPU (N,8)
print("X_tensor:", tuple(X_tensor.shape))
print("First row TRUE:", X_tensor[0].numpy())

# -----------------------------
# 5) DataLoader (paper settings)
# -----------------------------
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 32      # paper
LR = 1e-3            # paper
EPOCHS = 1000        # paper

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# -----------------------------
# 6) Train (paper objective: MSE on swap rates)
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

        # If your model produces NaNs early, training will break.
        # We skip such batches but also count them so you can see if it's happening a lot.
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
# 7) Inference on in-sample data
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
# 8) Filter non-finite rows (so RMSE is meaningful)
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
# 9) Teacher-style RMSE per currency (bps)
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
display(rmse_series.to_frame("RMSE (bps)"))

# -----------------------------
# 10) Plots
# -----------------------------
# 10a) Training loss
plt.figure(figsize=(6, 3.5))
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Train MSE")
plt.title("Training loss (in-sample)")
plt.grid(True)
plt.show()

# 10b) Actual vs reconstructed on one date (model fit sanity)
def plot_recon_on_date(df_wide_used, S_hat_all, meta_df, date_pick):
    m = meta_df.copy()
    idx = (m["as_of_date"] == pd.to_datetime(date_pick)).values
    if idx.sum() == 0:
        raise ValueError("No rows found for that date.")

    X_true = df_wide_used.loc[idx].to_numpy(dtype=np.float32)
    if SCALE_IS_PERCENT:
        X_true = X_true / 100.0

    X_pred = S_hat_all[idx].detach().cpu().numpy()
    ccys = m.loc[idx, "ccy"].values

    plt.figure(figsize=(8, 4))
    for i, ccy in enumerate(ccys):
        col = currency_color_map.get(ccy, None)
        plt.plot(tenors, X_true[i], marker="o", color=col, alpha=0.6)
        plt.plot(tenors, X_pred[i], marker="x", linestyle="--", color=col, alpha=0.9)

    plt.xlabel("Tenor (years)")
    plt.ylabel("Swap rate (decimals)")
    plt.title(f"Actual (o) vs Reconstructed (x) on {pd.to_datetime(date_pick).date()}")
    plt.grid(True)
    plt.show()

date_pick = meta["as_of_date"].iloc[0]
plot_recon_on_date(df_wide, S_hat_all[mask], meta_eval, date_pick)

# 10c) Latent factors over time
def plot_latents_over_time(z_all, meta_df):
    z_np = z_all.detach().cpu().numpy()
    m = meta_df.copy()
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
    plt.show()

plot_latents_over_time(z_all[mask], meta_eval)

# ============================================================
# 11) Parameter-model plots (mu1, mu2, sigma1, sigma2, rho, r̃)
# Copy/paste AFTER your existing code (after plot_latents_over_time)
# ============================================================

@torch.no_grad()
def run_model_full_batches(model, X_tensor_cpu, batch_size=256, device="cpu"):
    """
    Returns everything you need for parameter plots:
    z (N,2), mu (N,2), sigma (N,2,2), r_tilde (N,1), plus sigmas/rho as scalars.
    """
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
    z_all = torch.cat(zs, dim=0)           # (N,2)
    mu_all = torch.cat(mus, dim=0)         # (N,2)
    sigma_all = torch.cat(sigmas, dim=0)   # (N,2,2)
    r_all = torch.cat(rts, dim=0)          # (N,1) or (N,)
    return z_all, mu_all, sigma_all, r_all

# --- run model again to get mu/sigma/r on ALL rows (then filter with mask) ---
z_all_full, mu_all_full, sigma_all_full, r_all_full = run_model_full_batches(
    model, X_tensor, batch_size=256, device=device
)

# apply same finite mask as used for S_hat evaluation
z_eval = z_all_full[mask]
mu_eval = mu_all_full[mask]
sigma_eval = sigma_all_full[mask]
r_eval = r_all_full[mask]

# ensure shapes
if r_eval.ndim == 2 and r_eval.shape[1] == 1:
    r_eval = r_eval.squeeze(1)  # (N,)

# Extract sigma1, sigma2, rho from sigma matrix consistent with paper form:
# sigma = [[sigma1, 0],
#          [rho*sigma2, sqrt(1-rho^2)*sigma2]]
sigma1_eval = sigma_eval[:, 0, 0]                      # (N,)
sigma2_eval = torch.sqrt(sigma_eval[:, 1, 0]**2 + sigma_eval[:, 1, 1]**2)  # (N,)
rho_eval = sigma_eval[:, 1, 0] / torch.clamp(sigma2_eval, min=1e-12)       # (N,)

# Build a plotting frame
params_df = meta_eval.copy()
params_df["mu1"] = mu_eval[:, 0].numpy()
params_df["mu2"] = mu_eval[:, 1].numpy()
params_df["sigma1"] = sigma1_eval.numpy()
params_df["sigma2"] = sigma2_eval.numpy()
params_df["rho"] = rho_eval.numpy()
params_df["r_tilde"] = r_eval.numpy()

# Sort by time for clean lines
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
    plt.show()

# --- Plots like "parameter modeling" figures ---
plot_param_over_time(params_df, "r_tilde", title="Short rate mapping r̃(z)")
plot_param_over_time(params_df, "sigma1",  title="Volatility σ1(z)")
plot_param_over_time(params_df, "sigma2",  title="Volatility σ2(z)")
plot_param_over_time(params_df, "rho",     title="Correlation ρ(z)")
plot_param_over_time(params_df, "mu1",     title="Drift μ1(z)")
plot_param_over_time(params_df, "mu2",     title="Drift μ2(z)")

# Optional: quick distribution sanity (histograms)
def hist_param(params_df, col, bins=50):
    plt.figure(figsize=(6, 3.5))
    plt.hist(params_df[col].values, bins=bins)
    plt.title(f"Histogram of {col}")
    plt.grid(True)
    plt.show()

# Uncomment if you want:
hist_param(params_df, "r_tilde")
hist_param(params_df, "sigma1")
hist_param(params_df, "sigma2")
hist_param(params_df, "rho")
hist_param(params_df, "mu1")
hist_param(params_df, "mu2")


print("Done.")
