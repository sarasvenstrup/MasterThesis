import os
import sys
import torch
import torch.nn as nn
torch.set_num_threads(4) # --- Torch thread settings MUST be first Torch-related thing ---
torch.set_num_interop_threads(2)
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler



def set_paper_theme():
    # 1) Use seaborn only to define a nice clean theme (works for matplotlib plots too)
    sns.set_theme(context="paper", style="darkgrid", font_scale=1.05)


    # Customize tab20b palette
    full_palette = sns.color_palette("tab20b", 20)
    selected_indices = [0, 1, 2, 3, 12, 13, 14, 15]
    palette = [full_palette[i] for i in selected_indices]


    # 3) Global matplotlib defaults (applies to ALL figures you create afterwards)
    mpl.rcParams.update({
        # Figure / saving
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",


        # Light grey full frame
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "0.8",  # light grey frame
        "axes.linewidth": 1.0,

        # Grid styling
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 1.0,

        # Text
        "font.size": 11,
        "axes.labelcolor": "0.2",
        "xtick.color": "0.2",
        "ytick.color": "0.2",

        # Legend
        "legend.frameon": False,

        # Lines default
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,


    })

    # 4) Make the palette the default color cycle for matplotlib
    mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)

    return palette


def style_axis(ax, title=None, xlabel=None, ylabel=None, legend=True, legend_kwargs=None):
    """Optional helper you can call per-figure for consistent finishing touches."""
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Ensure consistent grid/spines (in case some plots override)
    ax.grid(True, which="major", axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if legend:
        kw = dict(frameon=False)
        if legend_kwargs:
            kw.update(legend_kwargs)
        ax.legend(**kw)

# Call this once, early in your script
custom_palette = set_paper_theme()


# ABOVE IS THE FIGURE SETTINGS =================




try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)



from Code.utils import helpers as H
from Code.load_swapdata import build_all_dataframes, TARGET_TENORS
from Code.model.full_model import FullModel


print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Helps on many CPUs for convs etc.
torch.backends.mkldnn.enabled = True

# Less overhead in optimizer.zero_grad
USE_SET_TO_NONE = True

print("CPU threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())

# -----------------------------
# 0b) Run config
# -----------------------------
USE = "bbg"  # "test" first, then "bbg"

FIGURES_DIR = os.path.join(REPO_ROOT, "Figures", USE)
os.makedirs(FIGURES_DIR, exist_ok=True)



# -----------------------------
# 1) Load data
# -----------------------------
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

# Tenor grid
tenors = np.array([float(x) for x in TARGET_TENORS], dtype=float)

# Ensure columns
df_wide = df_wide_full[["as_of_date", "ccy"] + list(TARGET_TENORS)].copy()
df_wide["as_of_date"] = pd.to_datetime(df_wide["as_of_date"])
df_wide = df_wide[df_wide["as_of_date"] >= "2010-01-01"].copy()

meta = df_wide[["as_of_date", "ccy"]].reset_index(drop=True)
X = df_wide[list(TARGET_TENORS)].to_numpy(dtype=np.float32)
print("Wide shape:", X.shape)

# -----------------------------
# 2) Currency rename + colors
# -----------------------------
currency_rename_map = {
    "ad": "AUD", "AD": "AUD",
    "cd": "CAD", "CD": "CAD",
    "dk": "DKK", "DK": "DKK",
    "eu": "EUR", "EU": "EUR",
    "jy": "JPY", "JY": "JPY",
    "nk": "NOK", "NK": "NOK",
    "sk": "SEK", "SK": "SEK",
    "uk": "GBP", "UK": "GBP",
    "us": "USD", "US": "USD",
}
meta["ccy"] = meta["ccy"].map(lambda x: currency_rename_map.get(x, x))
df_wide["ccy"] = df_wide["ccy"].map(lambda x: currency_rename_map.get(x, x))


# Use your theme palette for consistent currency colors
ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
currency_color_map = {ccy: custom_palette[i % len(custom_palette)] for i, ccy in enumerate(ccy_order)}


# -----------------------------
# 3) Scale to decimals (auto-detect)
# -----------------------------
median_abs = float(np.nanmedian(np.abs(X)))
SCALE_IS_PERCENT = median_abs > 0.5
print("Median |swap|:", median_abs, "=> SCALE_IS_PERCENT =", SCALE_IS_PERCENT)

if SCALE_IS_PERCENT:
    X = X / 100.0

X_tensor = torch.from_numpy(X)  # (N,8) CPU
print("X_tensor:", tuple(X_tensor.shape))
print("First row TRUE:", X_tensor[0].numpy())

# -----------------------------
# 3b) Helper configs
# -----------------------------
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

# =============================
# PAPER PLOTS A + B (Observed only)
#   A) swap curves on one date
#   B) 10Y time series
# =============================

def plot_swap_curves_on_date_observed(df_wide_obs: pd.DataFrame,
                                      target_tenors,
                                      tenors_years: np.ndarray,
                                      currency_colors: dict,
                                      date_pick,
                                      plot_cfg: H.PlotConfig):
    date_pick = pd.to_datetime(date_pick)
    dfo = df_wide_obs.copy()
    dfo["as_of_date"] = pd.to_datetime(dfo["as_of_date"])

    sel = dfo[dfo["as_of_date"] == date_pick].copy()
    if sel.empty:
        raise ValueError(f"No rows found for date {date_pick.date()}")

    # one curve per currency
    sel = sel.sort_values(["ccy", "as_of_date"]).drop_duplicates(subset=["ccy"], keep="last")
    Y = sel[list(target_tenors)].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel["ccy"].values):
        color = currency_colors.get(ccy, None)
        ax.plot(
            tenors_years, Y[i],
            marker="o",
            color=color,
            label=ccy,
            alpha=0.9,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )

    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Swap rate (decimals)")
    ax.set_title(f"Observed swap curves on {date_pick.date()}")
    ax.grid(True)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    H.save_figure(fig, plot_cfg, f"paper_fig2a_observed_curves_{date_pick.date()}")


def plot_swap_timeseries_one_tenor_observed(df_wide_obs: pd.DataFrame,
                                           tenor_col,
                                           currency_colors: dict,
                                           plot_cfg: H.PlotConfig,
                                           title: str = None):
    dfo = df_wide_obs.copy()
    dfo["as_of_date"] = pd.to_datetime(dfo["as_of_date"])

    fig, ax = plt.subplots(figsize=(10, 4))
    for ccy, g in dfo.groupby("ccy"):
        g = g.sort_values("as_of_date")
        color = currency_colors.get(ccy, None)
        ax.plot(
            g["as_of_date"], g[tenor_col].astype(float),
            color=color,
            label=ccy,
            alpha=0.9,
            marker=None,  # time series usually no markers
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Swap rate (decimals)")
    ax.set_title(title if title is not None else f"Observed {tenor_col} swap rate over time")
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    H.save_figure(fig, plot_cfg, f"paper_fig2b_timeseries_{tenor_col}")


# Build decimals version of observed df (so plots match model scale)
df_wide_dec = df_wide.copy()
if SCALE_IS_PERCENT:
    for col in TARGET_TENORS:
        df_wide_dec[col] = df_wide_dec[col].astype(float) / 100.0

# A) Choose paper date if it exists, otherwise first available
paper_date = pd.to_datetime("2016-08-30")
date_pick_A = paper_date if (df_wide_dec["as_of_date"] == paper_date).any() else df_wide_dec["as_of_date"].iloc[0]

plot_swap_curves_on_date_observed(
    df_wide_obs=df_wide_dec,
    target_tenors=TARGET_TENORS,
    tenors_years=tenors,
    currency_colors=currency_color_map,
    date_pick=date_pick_A,
    plot_cfg=plot_cfg,
)

# B) 10Y time series (or closest tenor to 10)
TENOR_10Y = 10
if TENOR_10Y not in TARGET_TENORS:
    TENOR_10Y = min(TARGET_TENORS, key=lambda t: abs(float(t) - 10.0))

plot_swap_timeseries_one_tenor_observed(
    df_wide_obs=df_wide_dec,
    tenor_col=TENOR_10Y,
    currency_colors=currency_color_map,
    plot_cfg=plot_cfg,
    title=f"Observed {TENOR_10Y}Y swap rate over time (all currencies)",
)

# -----------------------------
# 4) DataLoader (paper settings)
# -----------------------------
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 1000

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

model.eval()

xb = X_tensor[:3].to(device)
out = model(xb)
z = out[1]  # (3,d)

# -----------------------------
# 6) Inference (in-sample)
# -----------------------------
@torch.no_grad()
def run_model_batches(model, X_tensor_cpu, batch_size=256, device="cpu"):
    model.eval()
    S_hats, zs = [], []
    N = X_tensor_cpu.shape[0]
    for i in range(0, N, batch_size):
        xb = X_tensor_cpu[i : i + batch_size].to(device)
        out = model(xb)
        S_hat, z = out[0], out[1]
        S_hats.append(S_hat.detach().cpu())
        zs.append(z.detach().cpu())
    return torch.cat(S_hats, dim=0), torch.cat(zs, dim=0)

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

plot_latents_over_time(z_all[mask], meta_eval, plot_cfg)

# -----------------------------
# 10) Parameter-model plots
# -----------------------------
@torch.no_grad()
def run_model_full_batches(model, X_tensor_cpu, batch_size=256, device="cpu"):
    model.eval()
    zs, mus, sigmas_or_Ls, rts = [], [], [], []
    N = X_tensor_cpu.shape[0]

    for i in range(0, N, batch_size):
        xb = X_tensor_cpu[i : i + batch_size].to(device)
        out = model(xb)

        z = out[1]
        mu = out[6]
        sigma_or_L = out[7]
        r_tilde = out[8]

        zs.append(z.detach().cpu())
        mus.append(mu.detach().cpu())
        sigmas_or_Ls.append(sigma_or_L.detach().cpu())
        rts.append(r_tilde.detach().cpu())

    return (
        torch.cat(zs, dim=0),
        torch.cat(mus, dim=0),
        torch.cat(sigmas_or_Ls, dim=0),
        torch.cat(rts, dim=0),
    )

z_all_full, mu_all_full, sigma_all_full, r_all_full = run_model_full_batches(
    model, X_tensor, batch_size=256, device=device
)

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

# ============================================================
# 11) Andreasen Sharpe-ratio diagnostics: N, LN, SR + plots
#    + extra sanity checks (mu norm, grad norm, drift term)
#    + extra plots (mu_norm over time + hist)
#    + L (Cholesky) plots (you already had)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ---- IMPORTANT: ensure you have this import somewhere in your file ----
from Code.utils.sharpe_ratio import SR_andreasen_reference


# ---------- helpers ----------
@torch.no_grad()
def pick_one_curve_per_currency_on_date(meta_df: pd.DataFrame, date_pick):
    m = meta_df.copy()
    m["as_of_date"] = pd.to_datetime(m["as_of_date"])
    date_pick = pd.to_datetime(date_pick)

    sel = m[m["as_of_date"] == date_pick].copy()
    if sel.empty:
        raise ValueError(f"No rows in meta_df for date {date_pick.date()}")

    sel = sel.sort_values(["ccy", "as_of_date"]).drop_duplicates(subset=["ccy"], keep="last")
    return sel.index.to_numpy(), sel


def plot_mu_norm_over_time_and_hist(params_df: pd.DataFrame, mu_cols, cfg):
    """
    Adds mu_norm = sqrt(sum_k mu_k^2) then plots over time and histogram.
    """
    # defensive: only keep mu cols that exist
    mu_cols = [c for c in mu_cols if c in params_df.columns]
    if len(mu_cols) == 0:
        print("No mu columns found for mu_norm plot. Skipping.")
        return

    mu_sq = None
    for c in mu_cols:
        v = params_df[c].astype(float).values
        mu_sq = v * v if mu_sq is None else (mu_sq + v * v)

    params_df = params_df.copy()
    params_df["mu_norm"] = np.sqrt(mu_sq)

    H.plot_param_over_time(params_df, "mu_norm", cfg=cfg, title="||mu(z)|| over time")
    H.hist_param(params_df, "mu_norm", cfg=cfg)
    print("Saved mu_norm plots.")


def plot_L_on_date(
    model,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    currency_colors: dict,
    date_pick,
    figures_dir=None,
    tag="L_cholesky",
):
    idxs, sel_meta = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    device = next(model.parameters()).device
    xb = X_tensor_cpu[idxs].to(device)

    model.eval()
    out = model(xb)
    sigma = out[7]  # your code returns sigma = L_from_sigmas_rhos(...) i.e. Cholesky L  (B,d,d)

    if sigma.ndim != 3:
        raise ValueError(f"Expected sigma/L to be (B,d,d), got {tuple(sigma.shape)}")

    B, d, _ = sigma.shape
    sig_np = sigma.detach().cpu().numpy()

    # --- 1) Frobenius norm of L per currency
    L_frob = np.linalg.norm(sig_np.reshape(B, -1), axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel_meta["ccy"].values):
        ax.scatter(i, L_frob[i], color=currency_colors.get(ccy, None), label=ccy)
    ax.set_xticks(range(B))
    ax.set_xticklabels(sel_meta["ccy"].values, rotation=45, ha="right")
    ax.set_ylabel("||L||_F")
    ax.set_title(f"Cholesky L magnitude on {pd.to_datetime(date_pick).date()}")
    ax.grid(True)
    fig.tight_layout()

    if figures_dir is not None:
        path = f"{figures_dir}/{tag}_frob_{pd.to_datetime(date_pick).date()}.png"
        fig.savefig(path, dpi=250)
        print("Saved:", path)

    # --- 2) Plot each entry of L across currencies
    fig2, axes = plt.subplots(nrows=d, ncols=d, figsize=(10, 7), sharex=True)
    if d == 1:
        axes = np.array([[axes]])

    x = np.arange(B)
    for i in range(d):
        for j in range(d):
            ax2 = axes[i, j]
            vals = sig_np[:, i, j]
            for k, ccy in enumerate(sel_meta["ccy"].values):
                ax2.scatter(x[k], vals[k], color=currency_colors.get(ccy, None), alpha=0.9)
            ax2.set_title(f"L[{i+1},{j+1}]")
            ax2.grid(True)

    for ax2 in axes[-1, :]:
        ax2.set_xticks(x)
        ax2.set_xticklabels(sel_meta["ccy"].values, rotation=45, ha="right")

    fig2.suptitle(f"Entries of Cholesky L on {pd.to_datetime(date_pick).date()}", y=1.02)
    fig2.tight_layout()

    if figures_dir is not None:
        path2 = f"{figures_dir}/{tag}_entries_{pd.to_datetime(date_pick).date()}.png"
        fig2.savefig(path2, dpi=250)
        print("Saved:", path2)

    # --- 3) Print matrices
    print("\nPer-currency L matrices:")
    for i, ccy in enumerate(sel_meta["ccy"].values):
        print(ccy, "\n", sig_np[i])

    return sigma, sel_meta


def LN_term_decomposition_with_sanity_prints(
    model,
    xb_one: torch.Tensor,          # shape (1,8)
    tau_max=30,
    print_taus=(5, 10, 30),        # maturities at which to print grad norms etc.
):
    """
    Computes LN decomposition for a single curve:
      LN = -dN/dtau - rN + mu·∇N + 0.5 Tr(Cov Hess N)
    and prints sanity checks:
      mu, ||mu||, r_tilde, L diag, ||∇N|| and drift at selected taus.
    """
    device = xb_one.device
    dtype = xb_one.dtype
    model.eval()

    # forward WITH graph
    S_hat, z, P_full, A_vals, B_vals, G_vals, mu, sigma, r_tilde = model(xb_one.requires_grad_(True))

    # Sanity: mu and r and L diag
    with torch.no_grad():
        mu_np = mu.detach().cpu().numpy()
        print("\n[Sanity] mu:", mu_np)
        print("[Sanity] ||mu||:", float(mu.norm().detach().cpu()))
        print("[Sanity] r_tilde:", float(r_tilde.detach().cpu().view(-1)[0]))

        # sigma is L (Cholesky) in your code
        L_np = sigma.detach().cpu().numpy()[0]
        print("[Sanity] L diag:", np.diag(L_np))

    # dN/dtau on 0..tau_max (dt=1 year)
    dP_dtau_full = torch.zeros_like(P_full)
    dP_dtau_full[:, 0] = (P_full[:, 1] - P_full[:, 0])
    dP_dtau_full[:, -1] = (P_full[:, -1] - P_full[:, -2])
    if tau_max >= 2:
        dP_dtau_full[:, 1:-1] = 0.5 * (P_full[:, 2:] - P_full[:, :-2])

    N_tau = P_full[:, 1:]            # (1, tau_max)
    dN_dtau = dP_dtau_full[:, 1:]    # (1, tau_max)

    r = r_tilde.view(-1, 1)          # (1,1)
    d = z.shape[1]
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    drift_term = torch.zeros(1, tau_max, device=device, dtype=dtype)
    trace_term = torch.zeros(1, tau_max, device=device, dtype=dtype)

    print_taus = set(int(t) for t in print_taus if 1 <= int(t) <= tau_max)

    for m in range(tau_max):
        Nm = N_tau[:, m]
        g = torch.autograd.grad(Nm.sum(), z, create_graph=True)[0]  # (1,d)

        drift_term[:, m] = (g * mu).sum(dim=1)

        hvp_sum = torch.zeros(1, device=device, dtype=dtype)
        for v in sigma_cols:
            gv = (g * v).sum()
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]
            hvp_sum += (Hg_v * v).sum(dim=1)

        trace_term[:, m] = 0.5 * hvp_sum

        tau_here = m + 1
        if tau_here in print_taus:
            with torch.no_grad():
                print(f"[Sanity] tau={tau_here:2d} ||grad N||:", float(g.norm().detach().cpu()))
                print(f"[Sanity] tau={tau_here:2d} drift term mu·∇N:", float(drift_term[:, m].detach().cpu()))

    term_dN = (-dN_dtau).detach().cpu().numpy().squeeze(0)
    term_rN = (-(r * N_tau)).detach().cpu().numpy().squeeze(0)
    term_mu = drift_term.detach().cpu().numpy().squeeze(0)
    term_tr = trace_term.detach().cpu().numpy().squeeze(0)
    LN = term_dN + term_rN + term_mu + term_tr

    return {
        "N_tau": N_tau.detach(),
        "LN": torch.from_numpy(LN).to(device=device, dtype=dtype).view(1, -1),
        "terms": {
            "minus_dN_dt": term_dN,
            "minus_rN": term_rN,
            "mu_gradN": term_mu,
            "half_trace": term_tr,
            "LN_sum": LN,
        },
    }


def plot_LN_terms_for_one_curve_from_terms(terms_dict, cfg: H.PlotConfig, tag="onecurve", tau_max=30):
    tau_np = np.arange(1, tau_max + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tau_np, terms_dict["minus_dN_dt"], label="-dN/dτ")
    ax.plot(tau_np, terms_dict["minus_rN"], label="-rN")
    ax.plot(tau_np, terms_dict["mu_gradN"], label="μ·∇N")
    ax.plot(tau_np, terms_dict["half_trace"], label="0.5 Tr")
    ax.plot(tau_np, terms_dict["LN_sum"], label="LN (sum)", linewidth=2.5)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Value")
    ax.set_title("LN term decomposition (single curve)")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()

    H.save_figure(fig, cfg, f"LN_terms_{tag}")
    print("Saved LN term decomposition plot.")


def plot_LN_on_date_reference(
    model,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    currency_colors: dict,
    date_pick,
    cfg: H.PlotConfig,
    tau_max=30,
    sigma_bar=0.006,
):
    idxs, sel_meta = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    xb = X_tensor_cpu[idxs].to(next(model.parameters()).device)

    # MUST have grads (SR_andreasen_reference uses derivatives)
    model.eval()
    N_tau, LN, SR, tau = SR_andreasen_reference(model, xb, tau_max=tau_max, sigma_bar=sigma_bar)

    tau_np = tau.detach().cpu().numpy()
    LN_np = LN.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel_meta["ccy"].values):
        ax.plot(tau_np, LN_np[i], label=ccy, color=currency_colors.get(ccy, None), alpha=0.9)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("LN residual")
    ax.set_title(f"LN(τ) on {pd.to_datetime(date_pick).date()} (reference)")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    H.save_figure(fig, cfg, f"LN_reference_{pd.to_datetime(date_pick).date()}")
    print("Saved LN reference plot.")

    return N_tau, LN, SR, tau, sel_meta


def plot_SR_on_date_reference(
    model,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    currency_colors: dict,
    date_pick,
    cfg: H.PlotConfig,
    tau_max=30,
    sigma_bar=0.006,
):
    idxs, sel_meta = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    xb = X_tensor_cpu[idxs].to(next(model.parameters()).device)

    model.eval()
    N_tau, LN, SR, tau = SR_andreasen_reference(model, xb, tau_max=tau_max, sigma_bar=sigma_bar)

    tau_np = tau.detach().cpu().numpy()
    SR_np = SR.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel_meta["ccy"].values):
        ax.plot(tau_np, SR_np[i], label=ccy, color=currency_colors.get(ccy, None), alpha=0.9)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Sharpe ratio (approx, reference)")
    ax.set_title(f"SR(τ) on {pd.to_datetime(date_pick).date()} (reference)")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    H.save_figure(fig, cfg, f"SR_reference_{pd.to_datetime(date_pick).date()}")
    print("Saved SR reference plot.")

    return N_tau, LN, SR, tau, sel_meta


# ============================================================
# RUN THE DIAGNOSTICS
# ============================================================

# 0) Extra plots: mu_norm over time + hist (needs params_df + mu_cols from section 10)
try:
    plot_mu_norm_over_time_and_hist(params_df, mu_cols, plot_cfg)
except Exception as e:
    print("mu_norm plots failed (continuing):", repr(e))

# 1) Choose a date
paper_date = pd.to_datetime("2016-08-30")
date_pick = paper_date if (meta_eval["as_of_date"] == paper_date).any() else meta_eval["as_of_date"].iloc[0]

# 2) Pick one finite curve index for “single curve” sanity/decomposition
with torch.no_grad():
    finite_mask = torch.isfinite(X_tensor).all(dim=1)
i0 = int(torch.nonzero(finite_mask, as_tuple=False)[0].item())
xb1 = X_tensor[i0:i0+1].to(device)

print("\nSingle curve index used:", i0)

# 3) Reference SR/LN quick numbers for that curve (NOT under no_grad)
model.eval()
N1, LN1, SR1, tau1 = SR_andreasen_reference(model, xb1, tau_max=30, sigma_bar=0.006)
print("SR min/max:", float(SR1.min().detach().cpu()), float(SR1.max().detach().cpu()))
print("N(30Y):", float(N1[0, -1].detach().cpu()))
print("LN(30Y):", float(LN1[0, -1].detach().cpu()))
print("SR(30Y):", float(SR1[0, -1].detach().cpu()))

# 4) LN decomposition + sanity prints (mu, ||mu||, L diag, ||grad N|| at taus)
decomp = LN_term_decomposition_with_sanity_prints(
    model=model,
    xb_one=xb1,
    tau_max=30,
    print_taus=(5, 10, 30),
)

# 5) Plot the decomposition from the computed terms
plot_LN_terms_for_one_curve_from_terms(
    decomp["terms"],
    cfg=plot_cfg,
    tag=f"idx{i0}",
    tau_max=30,
)

# 6) Plot LN curves across currencies (reference)
_ = plot_LN_on_date_reference(
    model=model,
    X_tensor_cpu=X_tensor,
    meta_df=meta_eval,
    currency_colors=currency_color_map,
    date_pick=date_pick,
    cfg=plot_cfg,
    tau_max=30,
    sigma_bar=0.006,
)

# 7) Plot SR curves across currencies (reference)
_ = plot_SR_on_date_reference(
    model=model,
    X_tensor_cpu=X_tensor,
    meta_df=meta_eval,
    currency_colors=currency_color_map,
    date_pick=date_pick,
    cfg=plot_cfg,
    tau_max=30,
    sigma_bar=0.006,
)

# 8) Plot L (Cholesky) on same date (your L check)
sigma_L, meta_used = plot_L_on_date(
    model=model,
    X_tensor_cpu=X_tensor,
    meta_df=meta_eval,
    currency_colors=currency_color_map,
    date_pick=date_pick,
    figures_dir=FIGURES_DIR,
    tag="L_cholesky",
)

print(f"\nSharpe/LN/L diagnostics saved to: {FIGURES_DIR}")