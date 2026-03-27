# Code/utils/helpers.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# -----------------------------
# Config objects
# -----------------------------
@dataclass(frozen=True)
class PlotConfig:
    figures_dir: Union[str, Path]
    use_tag: str = ""
    currency_colors: Optional[Dict[str, str]] = None
    dpi: int = 300

    @property
    def out_dir(self) -> Path:
        p = Path(self.figures_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass(frozen=True)
class DataConfig:
    target_tenors: Sequence[Union[int, str]]
    tenor_years: np.ndarray  # e.g. np.array([1,2,3,5,10,15,20,30], float)
    scale_is_percent: bool


# -----------------------------
# Matrix utilities
# -----------------------------
def cov_from_L(L: torch.Tensor) -> torch.Tensor:
    """
    L: (B,d,d) diffusion (Cholesky-like). Returns Sigma = L L^T: (B,d,d).
    """
    if L.ndim != 3 or L.shape[-1] != L.shape[-2]:
        raise ValueError(f"Expected L shape (B,d,d), got {tuple(L.shape)}")
    return L @ L.transpose(1, 2)


def vols_from_cov(Sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Sigma: (B,d,d) -> vols: (B,d)."""
    if Sigma.ndim != 3 or Sigma.shape[-1] != Sigma.shape[-2]:
        raise ValueError(f"Expected Sigma shape (B,d,d), got {tuple(Sigma.shape)}")
    diag = torch.diagonal(Sigma, dim1=1, dim2=2)
    # Clamp for numerical safety before sqrt (positive-definite assumption)
    return torch.sqrt(torch.clamp(diag, min=eps))


def corr_from_cov(Sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Sigma: (B,d,d) -> Corr: (B,d,d)."""
    vol = vols_from_cov(Sigma, eps=eps)  # (B,d)
    # Clamp for numerical safety to prevent division by zero
    denom = torch.clamp(vol.unsqueeze(2) * vol.unsqueeze(1), min=eps)  # (B,d,d)
    Corr = Sigma / denom
    d = Corr.shape[1]
    eye = torch.eye(d, device=Corr.device, dtype=Corr.dtype).unsqueeze(0)
    Corr = Corr * (1 - eye) + eye
    return Corr


# -----------------------------
# Dataframe builders
# -----------------------------
def build_params_df_from_L(
    meta: pd.DataFrame,
    mu: torch.Tensor,          # (B,d)
    L: torch.Tensor,           # (B,d,d)
    r_tilde: torch.Tensor,     # (B,) or (B,1)
    eps: float = 1e-12
) -> pd.DataFrame:
    """
    Use when your model returns a full diffusion matrix L(z).
    Produces mu_k, sigma_k (marginal vols), rho_ij, and r_tilde.
    """
    if r_tilde.ndim == 2 and r_tilde.shape[1] == 1:
        r_tilde = r_tilde.squeeze(1)

    if mu.ndim != 2:
        raise ValueError(f"mu must be (B,d), got {tuple(mu.shape)}")

    Sigma = cov_from_L(L)              # (B,d,d)
    vol = vols_from_cov(Sigma, eps)    # (B,d)
    Corr = corr_from_cov(Sigma, eps)   # (B,d,d)

    out = meta.copy()
    d = mu.shape[1]

    mu_np = mu.detach().cpu().numpy()
    vol_np = vol.detach().cpu().numpy()
    Corr_np = Corr.detach().cpu().numpy()
    r_np = r_tilde.detach().cpu().numpy()

    for k in range(d):
        out[f"mu{k+1}"] = mu_np[:, k]
        out[f"sigma{k+1}"] = vol_np[:, k]

    for i in range(d):
        for j in range(i + 1, d):
            out[f"rho{i+1}{j+1}"] = Corr_np[:, i, j]

    out["r_tilde"] = r_np
    return out.sort_values(["ccy", "as_of_date"]).reset_index(drop=True)


def build_params_df_from_diag_vol(
    meta: pd.DataFrame,
    mu: torch.Tensor,          # (B,d)
    sigma: torch.Tensor,       # (B,d)
    r_tilde: torch.Tensor      # (B,) or (B,1)
) -> pd.DataFrame:
    """
    Use when your model returns only diagonal vols sigma_k(z) (no correlations).
    Produces mu_k, sigma_k, r_tilde (no rho_ij columns).
    """
    if r_tilde.ndim == 2 and r_tilde.shape[1] == 1:
        r_tilde = r_tilde.squeeze(1)

    if mu.ndim != 2 or sigma.ndim != 2:
        raise ValueError(f"mu and sigma must be (B,d). Got mu={tuple(mu.shape)}, sigma={tuple(sigma.shape)}")
    if mu.shape != sigma.shape:
        raise ValueError(f"mu and sigma must match shapes, got {tuple(mu.shape)} vs {tuple(sigma.shape)}")

    out = meta.copy()
    d = mu.shape[1]

    mu_np = mu.detach().cpu().numpy()
    sig_np = sigma.detach().cpu().numpy()
    r_np = r_tilde.detach().cpu().numpy()

    for k in range(d):
        out[f"mu{k+1}"] = mu_np[:, k]
        out[f"sigma{k+1}"] = sig_np[:, k]

    out["r_tilde"] = r_np
    return out.sort_values(["ccy", "as_of_date"]).reset_index(drop=True)


def cols_matching(df: pd.DataFrame, pattern: str) -> List[str]:
    pat = re.compile(pattern)
    return [c for c in df.columns if pat.match(c)]


# -----------------------------
# Plot + save helpers (explicit fig)
# -----------------------------
def _safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))


def save_figure(fig: plt.Figure, cfg: PlotConfig, name: str) -> Tuple[Path, Path]:
    safe = _safe_name(name)
    tag = f"_{cfg.use_tag}" if cfg.use_tag else ""
    png_path = cfg.out_dir / f"{safe}{tag}.png"
    pdf_path = cfg.out_dir / f"{safe}{tag}.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {png_path}")
    return png_path, pdf_path


def plot_param_over_time(params_df: pd.DataFrame, col: str, cfg: PlotConfig, title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(11, 4))

    for ccy, g in params_df.groupby("ccy"):
        color = cfg.currency_colors.get(ccy) if cfg.currency_colors else None
        ax.plot(g["as_of_date"], g[col], color=color, label=ccy)

    ax.set_title(title if title is not None else col)
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save_figure(fig, cfg, f"{col}_over_time")
    #plt.show()


def hist_param(params_df: pd.DataFrame, col: str, cfg: PlotConfig, bins: int = 50):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(params_df[col].values, bins=bins)
    ax.set_title(f"Histogram of {col}")
    ax.grid(True)
    save_figure(fig, cfg, f"hist_{col}")
    #plt.show()


def plot_recon_on_date(
    df_wide_used: pd.DataFrame,
    S_hat_all_eval: torch.Tensor,     # (N,8)
    meta_eval_df: pd.DataFrame,
    date_pick: Union[str, pd.Timestamp],
    data_cfg: DataConfig,
    cfg: PlotConfig
):
    m = meta_eval_df.copy()
    date_pick = pd.to_datetime(date_pick)
    idx = (m["as_of_date"] == date_pick).values
    if idx.sum() == 0:
        raise ValueError(f"No rows found for date {date_pick.date()}.")

    X_true = df_wide_used.loc[idx, list(data_cfg.target_tenors)].to_numpy(dtype=np.float32)
    if data_cfg.scale_is_percent:
        X_true = X_true / 100.0

    X_pred = S_hat_all_eval[idx].detach().cpu().numpy()
    ccys = m.loc[idx, "ccy"].values

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, ccy in enumerate(ccys):
        color = cfg.currency_colors.get(ccy) if cfg.currency_colors else None
        ax.plot(data_cfg.tenor_years, X_true[i], marker="o", color=color, alpha=0.6)
        ax.plot(data_cfg.tenor_years, X_pred[i], marker="x", linestyle="--", color=color, alpha=0.9)

    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Swap rate (decimals)")
    ax.set_title(f"Actual (o) vs Reconstructed (x) on {date_pick.date()}")
    ax.grid(True)

    save_figure(fig, cfg, f"reconstruction_{date_pick.date()}")
    #plt.show()

def rmse_bps_per_currency_paper(S_true, S_pred, meta_df):
    """
    Computes in-sample RMSE in basis points per currency.
    """

    if torch.is_tensor(S_true):
        S_true = S_true.detach().cpu().numpy()
    if torch.is_tensor(S_pred):
        S_pred = S_pred.detach().cpu().numpy()

    err = S_pred - S_true  # decimals
    tmp = meta_df.copy()

    rmses = {}
    for ccy in tmp["ccy"].unique():
        idx = (tmp["ccy"].values == ccy)
        e = err[idx, :]
        rmses[ccy] = float(np.sqrt(np.mean(e**2)) * 10000.0)

    out = pd.Series(rmses).sort_values()
    out.loc["Average"] = out.mean()
    return out

def check_monotonicity(P):
    # P: (B,T)
    violations = (P[:, 1:] - P[:, :-1]) > 1e-8
    n_viol = violations.sum().item()
    return n_viol

def instantaneous_forward(P: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    # P: (B,T), tau: (T,)
    logP = torch.log(P)
    dlogP = torch.zeros_like(logP)

    # forward/backward at boundaries
    dlogP[:, 0] = (logP[:, 1] - logP[:, 0]) / (tau[1] - tau[0])
    dlogP[:, -1] = (logP[:, -1] - logP[:, -2]) / (tau[-1] - tau[-2])

    # central differences in interior
    denom = (tau[2:] - tau[:-2]).unsqueeze(0)  # (1,T-2)
    dlogP[:, 1:-1] = (logP[:, 2:] - logP[:, :-2]) / denom

    f = -dlogP
    return f

def finite_minmax(x: torch.Tensor):
    xf = x[torch.isfinite(x)]
    if xf.numel() == 0:
        return float("nan"), float("nan")
    return float(xf.min().detach().cpu()), float(xf.max().detach().cpu())

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

    save_figure(fig, plot_cfg, f"paper_fig2a_observed_curves_{date_pick.date()}")


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

    save_figure(fig, plot_cfg, f"paper_fig2b_timeseries_{tenor_col}")

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

        ax.set_title(f"Latent factors for {k+1}-factor model")
        ax.grid(True)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    H.save_figure(fig, cfg, f"latent_factors_over_time_{LATENT_DIM}_factor")