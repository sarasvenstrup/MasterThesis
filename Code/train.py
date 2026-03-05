
# ============================= Import Packages ===============================

import os
import sys
import torch
import torch.nn as nn
torch.set_num_threads(4) # --- Torch thread settings MUST be first Torch-related thing ---
torch.set_num_interop_threads(2)
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from torch.optim.lr_scheduler import OneCycleLR

# ============================= Environment Setup & Imports ===============================

# First we set out working directory, in order for all our outputs to be saved in the same folder.

try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# We now import the needed components, like objects, models, helper functions and data in order to train the model.

from Code.utils import helpers as H
from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS
from Code.model.full_model import FullModel


print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# The following line accelerates deep learning operations on CPU, helping us to improve performance when training.
torch.backends.mkldnn.enabled = True

# The following line sets all .grad attributes to None instead of zero in order to lessen memory traffic.
USE_SET_TO_NONE = True
print("CPU threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())


#### LOAD DATA AND CONFIG PLOTTING

# Use your theme palette for consistent currency colors
ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
currency_color_map = {ccy: custom_palette[i % len(custom_palette)] for i, ccy in enumerate(ccy_order)}

USE = "bbg"  # "test" first, then "bbg"

# Where we save our figures, according to the dataset used to train.
FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", USE)
os.makedirs(FIGURES_DIR, exist_ok=True)

meta, X_tensor, tenors, df_wide, SCALE_IS_PERCENT = my_data(use=USE)

X_tensor = X_tensor.float()

plot_cfg = H.PlotConfig(
    figures_dir=FIGURES_DIR,
    use_tag=USE,
    currency_colors=currency_color_map,
    dpi=300,
)

data_cfg = H.DataConfig(
    target_tenors=list(TARGET_TENORS),
    tenor_years=tenors,
    scale_is_percent=SCALE_IS_PERCENT,
)

# -----------------------------
# 4) DataLoader (paper settings)
# -----------------------------
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 32
TARGET_MSE = 1e-6

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# -----------------------------
# 5) Train
# -----------------------------
torch.manual_seed(0)

LATENT_DIM = 3
model = FullModel(latent_dim=LATENT_DIM).to(device)
model.train()

max_lr = 3e-3
optim = torch.optim.Adam(model.parameters(), lr=max_lr)

EPOCHS = 15
max_lr = 3e-3

scheduler = OneCycleLR(
    optim,
    max_lr=max_lr,
    steps_per_epoch=len(loader),
    epochs=EPOCHS,
    pct_start=0.3,
    div_factor=1.0,
    final_div_factor=3000.0
)

loss_fn = nn.MSELoss()

train_losses = []
lrs = []
nan_batches_total = 0

def batch_diagnostics(out, latent_dim: int):
    """
    Returns a dict of cheap diagnostics from one forward pass output tuple.
    Assumes out = (S_hat, z, P_mkt, A_vals, B_vals, G_vals, mu, sigma, r_tilde, arb)
    """
    S_hat, z, P_mkt, A_vals, B_vals, G_vals, mu, sigma, r_tilde, arb, rhos = out

    d = latent_dim
    diag = {}

    # --- G at tau=0 is the sensitive one for beta=r/G ---
    G0 = G_vals[:, 0]
    diag["min_abs_G0"] = float(G0.abs().min().detach().cpu())
    diag["median_abs_G0"] = float(G0.abs().median().detach().cpu())

    # Also keep your existing min|G| across the curve
    diag["min_abs_G_alltau"] = float(G_vals.abs().min().detach().cpu())

    # --- detR feasibility (only meaningful for d=3 if rhos are plain correlations) ---
    if d == 3:
        rho12, rho13, rho23 = rhos[:, 0], rhos[:, 1], rhos[:, 2]
        detR = 1.0 - rho12 ** 2 - rho13 ** 2 - rho23 ** 2 + 2.0 * rho12 * rho13 * rho23

        diag["frac_detR_neg"] = float((detR < 0).float().mean().detach().cpu())
        diag["min_detR"] = float(detR.min().detach().cpu())

    # --- sigma scale sanity ---
    cov = sigma @ sigma.transpose(-1, -2)
    diag["cov_trace_mean"] = float(torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).mean().detach().cpu())

    # --- mu, r scale sanity ---
    diag["mu_norm_mean"] = float(mu.norm(dim=-1).mean().detach().cpu())
    rt = r_tilde.squeeze(-1) if r_tilde.ndim == 2 else r_tilde
    diag["r_tilde_mean"] = float(rt.mean().detach().cpu())
    diag["r_tilde_std"] = float(rt.std().detach().cpu())

    # z usage: per-dim std
    z_std = z.std(dim=0)  # (d,)
    for k in range(d):
        diag[f"zstd{k + 1}"] = float(z_std[k].detach().cpu())

    # --- arb stats (note: your R_tau is currently tautological; SR_tau too) ---
    diag["SR_max_abs_mean"] = float(arb["SR_tau"][:, 1:].abs().max(dim=1).values.mean().detach().cpu())

    return diag


global_step = 0

for epoch in range(EPOCHS):
    running = 0.0
    n_obs = 0
    nan_batches = 0

    for (xb_cpu,) in loader:
        xb = xb_cpu.to(device)

        optim.zero_grad(set_to_none=True)
        out = model(xb)
        S_hat = out[0]

        loss = loss_fn(S_hat, xb)
        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()
        scheduler.step()

        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

        # ---- Diagnostics every N steps ----
        if global_step % 200 == 0:
            with torch.no_grad():
                diag = batch_diagnostics(out, latent_dim=LATENT_DIM)
            print(
                f"step={global_step:6d} "
                f"loss={float(loss.detach().cpu()):.3e} "
                f"grad_norm={float(grad_norm):.3e} "
                f"zstd={diag['zstd1']:.3e},{diag['zstd2']:.3e},{diag['zstd3']:.3e} "
                f"min|G0|={diag['min_abs_G0']:.3e} "
                f"med|G0|={diag['median_abs_G0']:.3e} "
                f"min|G|={diag['min_abs_G_alltau']:.3e} "
                + (
                    f"frac_detR_neg={diag['frac_detR_neg']:.3f} min_detR={diag['min_detR']:.3e} " if LATENT_DIM == 3 else "")
                + f"mu_norm={diag['mu_norm_mean']:.3e} "
                  f"r_mean={diag['r_tilde_mean']:.3e} r_std={diag['r_tilde_std']:.3e} "
                  f"cov_trace_mean={diag['cov_trace_mean']:.3e}"
            )

        global_step += 1

    nan_batches_total += nan_batches
    epoch_loss = running / max(n_obs, 1)
    train_losses.append(epoch_loss)
    lrs.append(optim.param_groups[0]["lr"])

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        rmse_epoch = epoch_loss ** 0.5
        print(
            f"epoch={epoch:4d} rmse={rmse_epoch:.6e} lr={optim.param_groups[0]['lr']:.2e} "
            f"used_obs={n_obs} nan_batches={nan_batches} total_nan_batches={nan_batches_total}"
        )

print("Training done.")

model.eval()
