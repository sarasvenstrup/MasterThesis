# ============================================
# SCRIPT: train_vasicek_simulated_reconstruction.py
# Purpose:
#   1) Simulate swap curves from Vasicek (closed form)
#   2) Train a simple encoder that maps swap curve -> r0
#   3) Decode with fixed Vasicek closed-form and reconstruct swaps
#   4) Report RMSE (bps) + plots
#
# Saves figures to: <repo_root>/Figures/sim_vasicek/
# ============================================

import os
import sys

try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # save figures only
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

from Code.load_swapdata import TARGET_TENORS
from Code.Vasicek.simulate_vasicek import simulate_vasicek_curves
from Code.Vasicek.vasicek_decoder_model import VasicekDecoderModel

# -----------------------------
# Config
# -----------------------------
USE_TAG = "sim_vasicek"
FIG_DIR = os.path.join(REPO_ROOT, "Figures", USE_TAG)
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

torch.manual_seed(0)
np.random.seed(0)

plt.rcParams.update({"font.size": 11, "axes.grid": True})

# Simulation params (truth)
KAPPA_TRUE = 0.7
THETA_TRUE = 0.02
SIGMA_TRUE = 0.01
NOISE_STD_BPS = 0.0          # set 1-3 to test quote noise

N = 5000
BATCH_SIZE = 64
EPOCHS = 1000
LR = 1e-3

TENORS = list(TARGET_TENORS)  # [1,2,3,5,10,15,20,30]
TENORS_YEARS = np.array([float(t) for t in TENORS], dtype=float)

print("Device:", DEVICE)
print("Tenors:", TENORS)

# -----------------------------
# 1) Simulate dataset
# -----------------------------
Y_sim, r_sim = simulate_vasicek_curves(
    N=N,
    tenors=TENORS,
    kappa=KAPPA_TRUE,
    theta=THETA_TRUE,
    sigma=SIGMA_TRUE,
    noise_std_bps=NOISE_STD_BPS,
    seed=0,
    device="cpu",
)

X = Y_sim.astype(np.float32)            # swaps in decimals
X_tensor = torch.from_numpy(X)          # (N,8) CPU float32
r_true = r_sim.astype(np.float64)       # (N,) decimals

meta = pd.DataFrame({
    "as_of_date": pd.date_range("2000-01-01", periods=N, freq="D"),
    "ccy": ["SIM"] * N
})

print("Simulated X shape:", X_tensor.shape)
print("First curve (decimals):", X_tensor[0].numpy())
print("First r_true:", r_true[0])

# -----------------------------
# 2) Dataloader
# -----------------------------
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# -----------------------------
# 3) Model (encoder learns r0; decoder fixed Vasicek)
# -----------------------------
model = VasicekDecoderModel(
    input_dim=8,
    tenors=TENORS,
    kappa=KAPPA_TRUE,
    theta=THETA_TRUE,
    sigma=SIGMA_TRUE,
    tau_max=30,
    dtype=DTYPE,
    device=DEVICE
).to(DEVICE)

model.train()
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# -----------------------------
# 4) Train
# -----------------------------
train_losses = []

for epoch in range(EPOCHS):
    running = 0.0
    n_obs = 0

    for (xb_cpu,) in loader:
        xb = xb_cpu.to(DEVICE, dtype=DTYPE)  # decoder uses float64

        opt.zero_grad(set_to_none=True)
        S_hat, r0_hat = model(xb)            # S_hat (B,8), r0_hat (B,)
        loss = loss_fn(S_hat, xb)
        loss.backward()
        opt.step()

        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

    epoch_loss = running / max(n_obs, 1)
    train_losses.append(epoch_loss)

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"epoch={epoch:4d} loss={epoch_loss:.6e}")

print("Training done.")
model.eval()

# -----------------------------
# 5) Inference on full dataset
# -----------------------------
@torch.no_grad()
def run_batches(model, X_cpu, batch_size=512, device=DEVICE, dtype=DTYPE):
    S_list, r_list = [], []
    N_ = X_cpu.shape[0]
    for i in range(0, N_, batch_size):
        xb = X_cpu[i:i+batch_size].to(device, dtype=dtype)
        S_hat, r0_hat = model(xb)
        S_list.append(S_hat.cpu())
        r_list.append(r0_hat.cpu())
    return torch.cat(S_list, dim=0), torch.cat(r_list, dim=0)

S_hat_all, r_hat_all = run_batches(model, X_tensor, batch_size=512)

# -----------------------------
# 6) Metrics (RMSE bps)
# -----------------------------
X_eval = X_tensor.to(torch.float64)
err = (S_hat_all - X_eval).numpy()   # decimals
rmse_total_bps = 1e4 * float(np.sqrt(np.mean(err**2)))
rmse_by_tenor_bps = 1e4 * np.sqrt(np.mean(err**2, axis=0))

print("\n=== Reconstruction RMSE (Vasicek truth, fixed decoder) ===")
print(f"Total RMSE (bps): {rmse_total_bps:.6f}")
for t, v in zip(TENORS, rmse_by_tenor_bps):
    print(f"  {t:>2}Y: {v:.6f} bps")

# r0 fit quality
r_hat_np = r_hat_all.numpy()
r_mse = float(np.mean((r_hat_np - r_true)**2))
r_rmse = float(np.sqrt(r_mse))
print("\nState (r0) fit:")
print("  r0 RMSE (decimal):", r_rmse)
print("  r0 RMSE (bps):", 1e4 * r_rmse)

# Save metrics CSV
metrics_path = os.path.join(FIG_DIR, "vasicek_sim_metrics.csv")
metrics_df = pd.DataFrame({
    "metric": ["rmse_total_bps"] + [f"rmse_{t}y_bps" for t in TENORS] + ["r0_rmse_bps"],
    "value": [rmse_total_bps] + list(rmse_by_tenor_bps) + [1e4 * r_rmse]
})
metrics_df.to_csv(metrics_path, index=False)
print("\nSaved metrics:", metrics_path)

# -----------------------------
# 7) Plots
# -----------------------------

# 7a) training loss
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(train_losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Train MSE")
ax.set_title("Training loss (Vasicek simulated data)")
fig.tight_layout()
fig_path = os.path.join(FIG_DIR, "training_loss.png")
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print("Saved:", fig_path)

# 7b) observed vs reconstructed for a few curves
pick_idxs = [0, N//2, N-1]
fig, ax = plt.subplots(figsize=(8, 4))
for idx in pick_idxs:
    ax.plot(TENORS_YEARS, X_eval[idx].numpy(), marker="o", alpha=0.8, label=f"Obs idx={idx}")
    ax.plot(TENORS_YEARS, S_hat_all[idx].numpy(), marker="x", alpha=0.8, linestyle="--", label=f"Recon idx={idx}")
ax.set_xlabel("Tenor (years)")
ax.set_ylabel("Par swap rate (decimals)")
ax.set_title("Observed vs reconstructed swap curves (Vasicek sim)")
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig_path = os.path.join(FIG_DIR, "obs_vs_recon_curves.png")
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print("Saved:", fig_path)

# 7c) residuals by tenor (hist)
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot(1e4 * err, labels=[f"{t}Y" for t in TENORS])  # bps
ax.set_xlabel("Tenor")
ax.set_ylabel("Error (bps)")
ax.set_title("Reconstruction error by tenor (bps)")
fig.tight_layout()
fig_path = os.path.join(FIG_DIR, "error_by_tenor_boxplot.png")
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print("Saved:", fig_path)

# 7d) true vs estimated r0 over time
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(meta["as_of_date"], r_true, label="True r0", alpha=0.9)
ax.plot(meta["as_of_date"], r_hat_np, label="Estimated r0 (encoder)", alpha=0.8)
ax.set_xlabel("Date")
ax.set_ylabel("Short rate r0 (decimals)")
ax.set_title("True vs estimated Vasicek state r0")
ax.legend()
fig.tight_layout()
fig_path = os.path.join(FIG_DIR, "r0_true_vs_hat.png")
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print("Saved:", fig_path)

print(f"\nDone. Figures saved to: {FIG_DIR}")
