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
from Code.utils.sharpe_ratio import final_term_2factor_from_model, approx_sharpe_from_final_term

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
LR = 1e-3
EPOCHS = 200

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# -----------------------------
# 5) Train
# -----------------------------
torch.manual_seed(0)

LATENT_DIM = 2
model = FullModel(latent_dim=LATENT_DIM).to(device)
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

        optim.zero_grad(set_to_none=True)
        out = model(xb)
        S_hat = out[0]

        loss = loss_fn(S_hat, xb)

        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs += xb.shape[0]

    nan_batches_total += nan_batches
    epoch_loss = running / max(n_obs, 1)
    train_losses.append(epoch_loss)

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(
            f"epoch={epoch:4d} loss={epoch_loss:.6e} "
            f"used_obs={n_obs} nan_batches={nan_batches} total_nan_batches={nan_batches_total}"
        )

print("Training done.")

model.eval()

# -----------------------------
# 6) Inference (in-sample)
# -----------------------------
@torch.no_grad()
def run_model(
    model,
    X_tensor_cpu: torch.Tensor,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
    return_full: bool = False,
    return_dict: bool = True,
):
    """
    Batched forward pass for your FullModel.

    Assumes model(x) returns a tuple/list where:
      out[0] = S_hat
      out[1] = z
      out[6] = mu
      out[7] = sigma_or_L
      out[8] = r_tilde

    Parameters
    ----------
    model : torch.nn.Module
    X_tensor_cpu : torch.Tensor
        Input on CPU (or any device, name kept from your code).
    batch_size : int
    device : str or torch.device
    return_full : bool
        If False: return only (S_hat, z).
        If True : also return (mu, sigma_or_L, r_tilde).
    return_dict : bool
        If True: returns a dict with named outputs.
        If False: returns a tuple.

    Returns
    -------
    If return_full == False:
        dict: {"S_hat": (N,8), "z": (N,latent_dim)}
        or tuple: (S_hat, z)

    If return_full == True:
        dict: {"S_hat": ..., "z": ..., "mu": ..., "sigma_or_L": ..., "r_tilde": ...}
        or tuple: (S_hat, z, mu, sigma_or_L, r_tilde)
    """
    model.eval()

    S_hats, zs = [], []
    mus, sigmas_or_Ls, rts = [], [], []

    N = X_tensor_cpu.shape[0]

    for i in range(0, N, batch_size):
        xb = X_tensor_cpu[i : i + batch_size].to(device)
        out = model(xb)

        # Always collect these
        S_hat = out[0]
        z = out[1]
        S_hats.append(S_hat.detach().cpu())
        zs.append(z.detach().cpu())

        # Optionally collect dynamics pieces
        if return_full:
            mu = out[6]
            sigma_or_L = out[7]
            r_tilde = out[8]
            mus.append(mu.detach().cpu())
            sigmas_or_Ls.append(sigma_or_L.detach().cpu())
            rts.append(r_tilde.detach().cpu())

    S_hat_all = torch.cat(S_hats, dim=0)
    z_all     = torch.cat(zs, dim=0)

    if not return_full:
        if return_dict:
            return {"S_hat": S_hat_all, "z": z_all}
        return S_hat_all, z_all

    mu_all          = torch.cat(mus, dim=0)
    sigma_or_L_all  = torch.cat(sigmas_or_Ls, dim=0)
    r_tilde_all     = torch.cat(rts, dim=0)

    if return_dict:
        return {
            "S_hat": S_hat_all,
            "z": z_all,
            "mu": mu_all,
            "sigma_or_L": sigma_or_L_all,
            "r_tilde": r_tilde_all,
        }

    return S_hat_all, z_all, mu_all, sigma_or_L_all, r_tilde_all

S_hat_all, z_all_full, mu_all_full, sigma_all_full, r_all_full = run_model(
    model,
    X_tensor,
    batch_size=256,
    device=device,
    return_full=True,
    return_dict=False
)

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
rmse_series = H.rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)

print("\nSteady-state in-sample RMSE (bps) per currency:")
print(rmse_series.to_frame("RMSE (bps)"))

rmse_path = os.path.join(FIGURES_DIR, f"rmse_bps_{USE}.csv")
rmse_series.to_frame("RMSE (bps)").to_csv(rmse_path)
print("Saved RMSE table:", rmse_path)

# -----------------------------
# 9) Plots
# -----------------------------
# 9a) Training loss
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(train_losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Train MSE")
ax.set_title(f"Training loss (in-sample) — USE={USE}")
H.save_figure(fig, plot_cfg, "training_loss")

# 9b) Actual vs reconstructed on one date
date_pick = meta_eval["as_of_date"].iloc[0]
df_wide_eval = df_wide.loc[mask.numpy()].reset_index(drop=True)

H.plot_recon_on_date(
    df_wide_used=df_wide_eval,
    S_hat_all_eval=S_hat_all[mask],
    meta_eval_df=meta_eval,
    date_pick=date_pick,
    data_cfg=data_cfg,
    cfg=plot_cfg,
)

# 9c) Latent factors over time (fix ordering)
def plot_latents_over_time(z_eval_t: torch.Tensor, meta_eval_df: pd.DataFrame, cfg: H.PlotConfig):
    order = meta_eval_df.sort_values(["ccy", "as_of_date"]).index.to_numpy()
    m = meta_eval_df.loc[order].reset_index(drop=True)
    z_np = z_eval_t.detach().cpu().numpy()[order]

    d = z_np.shape[1]
    fig, axes = plt.subplots(nrows=d, ncols=1, figsize=(11, 3.5 * d), sharex=False)
    if d == 1:
        axes = [axes]

    for k in range(d):
        ax = axes[k]
        m_k = m.copy()
        m_k[f"z{k+1}"] = z_np[:, k]

        for ccy, g in m_k.groupby("ccy"):
            color = cfg.currency_colors.get(ccy) if cfg.currency_colors else None
            ax.plot(
                g["as_of_date"], g[f"z{k + 1}"],
                color=color,
                label=ccy,
                alpha=0.9,
            )

        ax.set_title(f"Latent factor z{k+1}")
        ax.grid(True)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    H.save_figure(fig, cfg, "latent_factors")

plot_latents_over_time(z_all_full[mask], meta_eval, plot_cfg)

# -----------------------------
# 10) Parameter-model plots
# -----------------------------

mu_eval = mu_all_full[mask]
sigma_eval = sigma_all_full[mask]
r_eval = r_all_full[mask]
if r_eval.ndim == 2 and r_eval.shape[1] == 1:
    r_eval = r_eval.squeeze(1)

# Build params_df (auto-detect sigma vs L)
if sigma_eval.ndim == 3:
    params_df = H.build_params_df_from_L(meta_eval, mu_eval, sigma_eval, r_eval)
elif sigma_eval.ndim == 2:
    params_df = H.build_params_df_from_diag_vol(meta_eval, mu_eval, sigma_eval, r_eval)
else:
    raise ValueError(f"Unexpected sigma/L shape: {tuple(sigma_eval.shape)}")

mu_cols = sorted(H.cols_matching(params_df, r"^mu\d+$"), key=lambda s: int(s[2:]))
sigma_cols = sorted(H.cols_matching(params_df, r"^sigma\d+$"), key=lambda s: int(s[5:]))
rho_cols = sorted(H.cols_matching(params_df, r"^rho\d+\d+$"))

H.plot_param_over_time(params_df, "r_tilde", cfg=plot_cfg, title="Short rate mapping r̃(z)")

for col in sigma_cols:
    H.plot_param_over_time(params_df, col, cfg=plot_cfg, title=f"Volatility {col}(z)")

for col in rho_cols:
    H.plot_param_over_time(params_df, col, cfg=plot_cfg, title=f"Correlation {col}")

for col in mu_cols:
    H.plot_param_over_time(params_df, col, cfg=plot_cfg, title=f"Drift {col}(z)")

H.hist_param(params_df, "r_tilde", cfg=plot_cfg)
for col in sigma_cols:
    H.hist_param(params_df, col, cfg=plot_cfg)
for col in rho_cols:
    H.hist_param(params_df, col, cfg=plot_cfg)
for col in mu_cols:
    H.hist_param(params_df, col, cfg=plot_cfg)

print(f"Done. Figures saved to: {FIGURES_DIR}")

# SHARPE RATIO

# Pick ONE date to match the paper style
date_pick = meta_eval["as_of_date"].iloc[0]  # or choose a specific date

meta_eval["as_of_date"] = pd.to_datetime(meta_eval["as_of_date"])
date_pick = pd.Timestamp("2014-12-28")
df_date = meta_eval[meta_eval["as_of_date"] == date_pick]
print("Rows on that date:", len(df_date))

# df_date = meta_eval[meta_eval["as_of_date"] == date_pick].copy()

print("Sharpe ratio plot date:", date_pick, "rows:", len(df_date))

# We’ll take one curve per currency on that date (first occurrence)
rows_idx = []
for ccy in sorted(df_date["ccy"].unique()):
    idx_ccy = df_date.index[df_date["ccy"] == ccy]
    if len(idx_ccy) > 0:
        rows_idx.append(idx_ccy[0])

rows_idx = torch.tensor(rows_idx, dtype=torch.long)
S_in = X_eval[rows_idx].to(device)  # (n_ccy, 8)

# Run model to get P(τ) (we need N(τ)=P(τ))
with torch.no_grad():
    out = model(S_in)
    P_full = out[2]          # (B, tau_max+1) includes τ=0
P_1T = P_full[:, 1:]         # (B, 30)

# Compute PDE residual ("final_term") as diagnostic
final_term, dbg = final_term_2factor_from_model(
    model,
    S_in,
    tau_max=30,
    use_no_grad_AB=True,     # diagnostic mode; fast
)

# Convert to approx Sharpe ratio term structure
sigma_bar = 0.006  # Andreasen uses 0.60% proxy often; adjust if you want
SR = approx_sharpe_from_final_term(final_term, P_1T, sigma_bar=sigma_bar)  # (B,30)

# Plot
tau_plot = torch.arange(1, 31).cpu().numpy()

fig, ax = plt.subplots(figsize=(7.2, 4.2))

for i, idx in enumerate(rows_idx.tolist()):
    ccy = meta_eval.loc[idx, "ccy"]
    color = currency_color_map.get(ccy, None)
    ax.plot(tau_plot, SR[i].detach().cpu().numpy(), label=ccy, color=color, linewidth=1.6, alpha=0.95)

ax.axhline(0.0, linestyle="--", linewidth=1)
ax.set_xlabel("Tenor (year)")
ax.set_ylabel("Approximate Sharpe ratio")
ax.set_title(f"Approx Sharpe ratio on {date_pick}")
ax.grid(True)
ax.legend(ncol=3, fontsize=9, frameon=True)

H.save_figure(fig, plot_cfg, f"approx_sharpe_ratio_{pd.to_datetime(date_pick).date()}")