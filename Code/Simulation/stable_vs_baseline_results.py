"""
stable_vs_baseline_results.py
============================
Generate thesis-ready figures and tables comparing the baseline and stable
model variants under Euler--Maruyama simulation.

Outputs (saved to Figures/Simulation/):
  fig_eigenvalues.png      -- Eigenvalues of the drift matrix M
  fig_sigma_bounds.png     -- Volatility sigma_i(z) distributions
  fig_latent_fans.png      -- Percentile fan + sample paths for latent factors
  fig_short_rate_fan.png   -- Percentile fan + sample paths for short rate
  fig_terminal_dist.png    -- Terminal distributions of z and r
  fig_log_growth.png       -- Log10 mean|z| growth over time
  fig_yield_curves.png     -- Decoded swap / yield curves at t=1,5,10 yr
  sim_diagnostics.csv         -- Summary table (raw values)
  sim_diagnostics_table.tex   -- Summary table formatted as a LaTeX
                                 ``tabular`` fragment, ready to be
                                 ``\\input{}``-ed inside the thesis chapter.
"""

import importlib
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── path setup ───────────────────────────────────────────────────────────
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THESIS_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for p in [THESIS_ROOT, PROJECT_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Code import config
from Code.load_swapdata import my_data
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.utils.common import set_paper_theme
from Code.Simulation.simulate_model import simulate_latent_paths, compute_discount_paths

# =============================================================================
# USER SETTINGS
# =============================================================================
LATENT_DIM      = 2
EPOCHS          = 5000

BASELINE_CKPT = os.path.join(
    THESIS_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_baseline", f"ep{EPOCHS}",
    f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt",
)
STABLE_CKPT = os.path.join(
    THESIS_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_stable", f"ep{EPOCHS}",
    f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt",
)

USE             = "bbg"
CCY_FILTER      = "EUR"
AS_OF_DATE      = None          # None -> first row
N_PATHS         = 500
N_STEPS         = 120           # 10 yr at monthly dt
DT              = 1 / 12
DIFFUSION_SCALE = 1.0
SEED            = 1234
DEVICE          = "cpu"
DTYPE           = torch.float64

N_SIGMA_GRID    = 2000          # extra random states for sigma-bound test
N_SAMPLE_PATHS  = 8             # individual paths overlaid on fan charts

# Horizons at which to decode yield curves (in years)
CURVE_HORIZONS  = [1, 5, 10]

OUT_DIR = os.path.join(THESIS_ROOT, "Figures", "Simulation")
os.makedirs(OUT_DIR, exist_ok=True)

# Colours
C_BASE = "#2c4f8c"
C_STAB = "#c0392b"
C_GREY = "#888888"


# =============================================================================
# Model loader (variant-aware)
# =============================================================================
def _load_model(variant: str, ckpt_path: str, latent_dim: int, device: str, dtype):
    """Construct FullModel under *variant* and load checkpoint weights."""
    from Code import config as _cfg
    _orig = _cfg.VARIANT
    _cfg.VARIANT = variant
    try:
        if variant == "stable":
            import Code.model.full_model_stable as _m
            importlib.reload(_m)
            FullModel = _m.FullModel
        else:
            import Code.model.full_model as _m
            importlib.reload(_m)
            FullModel = _m.FullModel
        model = FullModel(latent_dim=latent_dim).to(device)
    finally:
        _cfg.VARIANT = _orig
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    result = model.load_state_dict(sd, strict=False)
    if result.unexpected_keys:
        print(f"  [{variant}] dropped old params: {result.unexpected_keys}")
    model = model.to(dtype=dtype)
    model.eval()
    print(f"  [{variant}] loaded  {ckpt_path}")
    print(f"  [{variant}] K={type(model.K).__name__}  H={type(model.H).__name__}")
    return model


def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


# =============================================================================
# Diagnostics helpers
# =============================================================================
def _eigenvalues_of_M(model, variant: str):
    if variant == "stable":
        M = model.K.stable_matrix().detach().cpu().numpy()
    else:
        M = model.K.lin.weight.detach().cpu().numpy()
    eigs = np.linalg.eigvals(M)
    return M, eigs


@torch.no_grad()
def _sigma_stats(model, z_cloud, z_extra):
    sig_train, _ = model.H(z_cloud)
    sig_extra, _ = model.H(z_extra)
    return _np(sig_train), _np(sig_extra)


@torch.no_grad()
def _insample_rmse(model, X_tensor, device, dtype):
    """Root-mean-square swap-rate reconstruction error on full training set (bps)."""
    X = X_tensor.to(device=device, dtype=dtype)
    z = model.encoder(X)
    _, aux = model.decode_from_z(z, tau=None, do_arb_checks=False, return_aux=True)
    S_hat = aux["S_hat"]
    if S_hat is None:
        raise ValueError("model did not return S_hat")
    rmse = float(torch.sqrt(((S_hat - X) ** 2).mean()).item()) * 1e4  # bps
    return rmse


@torch.no_grad()
def _decode_curves_at_horizons(model, z_paths_np, times, horizons, tenors_np, device, dtype):
    """
    For each requested horizon, pick the nearest time index, decode all paths'
    latent states to swap-rate curves, and return mean +/- percentile bands.

    tenors_np : 1-D array of market maturities (x-axis for S_hat), length n_tenors.

    Returns a dict: horizon -> {"tau", "mean", "p5", "p25", "p75", "p95",
                                "samples", "valid_frac"}
    """
    results = {}
    n_paths, n_times, d = z_paths_np.shape
    rng = np.random.default_rng(42)
    for h in horizons:
        t_idx = int(np.argmin(np.abs(times - h)))
        z_h = torch.tensor(z_paths_np[:, t_idx, :], dtype=dtype, device=device)
        _, aux = model.decode_from_z(z_h, tau=None, do_arb_checks=False, return_aux=True)
        S_hat = _np(aux["S_hat"])      # (n_paths, n_tenors)
        valid_frac = float(np.mean(np.isfinite(S_hat).all(axis=1)))
        print(f"    horizon {h} yr: {valid_frac*100:.1f}% finite paths")
        sample_idx = rng.choice(n_paths, size=min(5, n_paths), replace=False)
        results[h] = {
            "tau":        tenors_np,           # market maturities — same length as S_hat columns
            "mean":       np.nanmean(S_hat, axis=0),
            "p5":         np.nanpercentile(S_hat,  5, axis=0),
            "p25":        np.nanpercentile(S_hat, 25, axis=0),
            "p75":        np.nanpercentile(S_hat, 75, axis=0),
            "p95":        np.nanpercentile(S_hat, 95, axis=0),
            "samples":    S_hat[sample_idx],
            "valid_frac": valid_frac,
        }
    return results


def _sim_summary(z_np, r_np, disc_np, label):
    """Compute summary dict for one set of simulation results."""
    fin_z = np.isfinite(z_np).all(axis=(1, 2))
    fin_r = np.isfinite(r_np).all(axis=1)
    fin_all = fin_z & fin_r
    return {
        "max_abs_z":          f"{np.nanmax(np.abs(z_np)):.2f}",
        "terminal_r_mean":    f"{np.nanmean(r_np[:, -1]) * 100:.2f}%",
        "terminal_r_std":     f"{np.nanstd(r_np[:, -1]) * 100:.2f}%",
        "terminal_z_std":     "; ".join(
            f"z{d+1}={np.nanstd(z_np[:, -1, d]):.3f}"
            for d in range(z_np.shape[2])
        ),
        "terminal_D_median":  f"{np.nanmedian(disc_np[:, -1]):.4f}",
        "terminal_D_range":   f"[{np.nanmin(disc_np[:, -1]):.4f}, {np.nanmax(disc_np[:, -1]):.4f}]",
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_paper_theme()
    times = np.arange(N_STEPS + 1) * DT

    # ── load data ─────────────────────────────────────────────────────
    meta, X_tensor, _meta_full, _X_full, tenors, *_ = my_data(use=USE, ccy_filter=CCY_FILTER)
    X_tensor = X_tensor.to(dtype=DTYPE)
    tenors_np = np.asarray(tenors, dtype=float)

    # ── load both models ──────────────────────────────────────────────
    print("Loading models ...")
    model_base = _load_model("baseline", BASELINE_CKPT, LATENT_DIM, DEVICE, DTYPE)
    model_stab = _load_model("stable",   STABLE_CKPT,   LATENT_DIM, DEVICE, DTYPE)

    # ── z0 from each model's own encoder ──────────────────────────────
    start_idx = 0 if AS_OF_DATE is None else int(
        np.where(meta["as_of_date"].values == np.datetime64(AS_OF_DATE))[0][0]
    )
    S0 = X_tensor[start_idx:start_idx + 1].to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        z0_base = model_base.encoder(S0)
        z0_stab = model_stab.encoder(S0)
    print(f"z0 baseline : {_np(z0_base).flatten()}")
    print(f"z0 stable   : {_np(z0_stab).flatten()}")

    # ── training latent cloud ─────────────────────────────────────────
    with torch.no_grad():
        z_train_base = model_base.encoder(X_tensor.to(DEVICE))
        z_train_stab = model_stab.encoder(X_tensor.to(DEVICE))

    # ── expanded grid (3× std beyond training range) ──────────────────
    rng = np.random.default_rng(42)
    z_np_base = _np(z_train_base)
    z_np_stab = _np(z_train_stab)
    z_extra_base = torch.tensor(
        rng.normal(z_np_base.mean(0), 3 * z_np_base.std(0), (N_SIGMA_GRID, LATENT_DIM)),
        dtype=DTYPE, device=DEVICE,
    )
    z_extra_stab = torch.tensor(
        rng.normal(z_np_stab.mean(0), 3 * z_np_stab.std(0), (N_SIGMA_GRID, LATENT_DIM)),
        dtype=DTYPE, device=DEVICE,
    )

    # ── in-sample RMSE ────────────────────────────────────────────────
    print("Computing in-sample RMSE ...")
    try:
        rmse_base = _insample_rmse(model_base, X_tensor, DEVICE, DTYPE)
        rmse_stab = _insample_rmse(model_stab, X_tensor, DEVICE, DTYPE)
        print(f"  Baseline RMSE: {rmse_base:.2f} bps  |  Stable RMSE: {rmse_stab:.2f} bps")
    except Exception as e:
        print(f"  [RMSE] failed: {e}")
        rmse_base = rmse_stab = float("nan")

    # ── simulate ──────────────────────────────────────────────────────
    print("Simulating ...")
    t0 = time.time()
    torch.manual_seed(SEED)
    z_base, r_base, _, _ = simulate_latent_paths(
        model=model_base, z0=z0_base,
        n_paths=N_PATHS, n_steps=N_STEPS, dt=DT,
        device=DEVICE, diffusion_scale=DIFFUSION_SCALE,
    )
    torch.manual_seed(SEED)
    z_stab, r_stab, _, _ = simulate_latent_paths(
        model=model_stab, z0=z0_stab,
        n_paths=N_PATHS, n_steps=N_STEPS, dt=DT,
        device=DEVICE, diffusion_scale=DIFFUSION_SCALE,
    )
    print(f"Done in {time.time()-t0:.1f}s")

    zb = _np(z_base); zs = _np(z_stab)
    rb = _np(r_base); rs = _np(r_stab)

    disc_base = _np(compute_discount_paths(r_base, DT))
    disc_stab = _np(compute_discount_paths(r_stab, DT))

    # sample-path indices for overlay (fixed seed)
    sp_rng  = np.random.default_rng(99)
    sp_idx  = sp_rng.choice(N_PATHS, size=N_SAMPLE_PATHS, replace=False)

    # =================================================================
    # FIG 1: Eigenvalues of the drift matrix M
    # =================================================================
    print("\n-- Fig 1: Drift eigenvalues --")
    M_base, eigs_base = _eigenvalues_of_M(model_base, "baseline")
    M_stab, eigs_stab = _eigenvalues_of_M(model_stab, "stable")
    print(f"  Baseline eigenvalues: {eigs_base}")
    print(f"  Stable   eigenvalues: {eigs_stab}")

    fig, ax = plt.subplots(figsize=(6, 4))
    x_pos = np.arange(LATENT_DIM); w = 0.35
    ax.bar(x_pos - w/2, np.real(eigs_base), width=w, color=C_BASE, edgecolor="none", label="Baseline", zorder=3)
    ax.bar(x_pos + w/2, np.real(eigs_stab), width=w, color=C_STAB, edgecolor="none", label="Stable",   zorder=3)
    ax.axhline(0,     color="black", linewidth=0.8)
    ax.axhline(-1e-3, color=C_GREY,  linewidth=0.8, linestyle="--", label=r"$-\epsilon$" + f" = {-1e-3}")
    for i, (eb, es) in enumerate(zip(np.real(eigs_base), np.real(eigs_stab))):
        offset_b = max(0.05 * abs(eb), 0.02)
        offset_s = max(0.05 * abs(es), 0.02)
        yb = (eb + offset_b * np.sign(eb)) if eb != 0 else offset_b
        ys = (es + offset_s * np.sign(es)) if es != 0 else offset_s
        ax.text(i - w/2, yb, f"{eb:.4f}", ha="center",
                va="bottom" if eb >= 0 else "top", fontsize=8, color=C_BASE)
        ax.text(i + w/2, ys, f"{es:.4f}", ha="center",
                va="bottom" if es >= 0 else "top", fontsize=8, color=C_STAB)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$\\lambda_{{{d+1}}}$" for d in range(LATENT_DIM)])
    ax.set_ylabel("Re($\\lambda$)")
    ax.set_title(f"Eigenvalues of drift matrix $M$ ($\\ell = {LATENT_DIM}$)")
    ax.legend(fontsize=9, frameon=False); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_eigenvalues.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 2: Volatility sigma bounds
    # =================================================================
    print("\n-- Fig 2: Sigma bounds --")
    sig_train_base, sig_extra_base = _sigma_stats(model_base, z_train_base, z_extra_base)
    sig_train_stab, sig_extra_stab = _sigma_stats(model_stab, z_train_stab, z_extra_stab)

    fig, axes = plt.subplots(1, LATENT_DIM, figsize=(5*LATENT_DIM, 4), squeeze=False)
    for d in range(LATENT_DIM):
        ax = axes[0, d]
        ax.hist(sig_train_base[:, d], bins=60, density=True, alpha=0.5,  color=C_BASE, label="Baseline (train)")
        ax.hist(sig_train_stab[:, d], bins=60, density=True, alpha=0.5,  color=C_STAB, label="Stable (train)")
        ax.hist(sig_extra_base[:, d], bins=60, density=True, alpha=0.25, color=C_BASE,
                histtype="step", linewidth=1.5, linestyle="--", label="Baseline (expanded)")
        ax.hist(sig_extra_stab[:, d], bins=60, density=True, alpha=0.25, color=C_STAB,
                histtype="step", linewidth=1.5, linestyle="--", label="Stable (expanded)")
        if hasattr(model_stab.H, "sigma_min"):
            ax.axvline(model_stab.H.sigma_min, color=C_GREY, linewidth=1, linestyle=":",
                       label=f"$\\sigma_{{\\min}}={model_stab.H.sigma_min}$")
            ax.axvline(model_stab.H.sigma_max, color=C_GREY, linewidth=1, linestyle=":",
                       label=f"$\\sigma_{{\\max}}={model_stab.H.sigma_max}$")
        ax.set_xlabel(f"$\\sigma_{{{d+1}}}$"); ax.set_ylabel("Density")
        ax.set_title(f"Volatility $\\sigma_{{{d+1}}}(z)$")
        ax.legend(fontsize=7, frameon=False); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Volatility distributions ($\\ell = {LATENT_DIM}$)", fontsize=12, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_sigma_bounds.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 3: Latent-path percentile fans  WITH  sample-path overlay
    # =================================================================
    print("\n-- Fig 3: Latent fans + sample paths --")
    fig, axes = plt.subplots(2, LATENT_DIM, figsize=(6*LATENT_DIM, 8), squeeze=False)
    for row, (arr, col, lbl) in enumerate([(zb, C_BASE, "Baseline"), (zs, C_STAB, "Stable")]):
        for dd in range(LATENT_DIM):
            ax = axes[row, dd]
            med = np.nanmedian(arr[:, :, dd], axis=0)
            p5  = np.nanpercentile(arr[:, :, dd],  5, axis=0)
            p95 = np.nanpercentile(arr[:, :, dd], 95, axis=0)
            p25 = np.nanpercentile(arr[:, :, dd], 25, axis=0)
            p75 = np.nanpercentile(arr[:, :, dd], 75, axis=0)
            ax.fill_between(times, p5,  p95, color=col, alpha=0.12)
            ax.fill_between(times, p25, p75, color=col, alpha=0.28)
            ax.plot(times, med, color=col, linewidth=1.8, label="Median", zorder=3)
            # Overlay individual sample paths
            for k, idx in enumerate(sp_idx):
                ax.plot(times, arr[idx, :, dd], color=col, linewidth=0.5,
                        alpha=0.45, zorder=2,
                        label="Sample paths" if k == 0 else None)
            ax.set_xlabel("Time (years)"); ax.set_ylabel(f"$z_{{{dd+1}}}$")
            ax.set_title(f"{lbl} — $z_{{{dd+1}}}$")
            ax.legend(fontsize=8, frameon=False); ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Latent-factor fans (5/25/50/75/95) + {N_SAMPLE_PATHS} sample paths"
        f"  —  N={N_PATHS}, $\\ell={LATENT_DIM}$",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_latent_fans.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 4: Short-rate percentile fan  WITH  sample-path overlay
    # =================================================================
    print("\n-- Fig 4: Short-rate fan + sample paths --")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, arr, col, lbl in [
        (axes[0], rb, C_BASE, "Baseline"),
        (axes[1], rs, C_STAB, "Stable"),
    ]:
        med = np.nanmedian(arr, axis=0) * 100
        p5  = np.nanpercentile(arr,  5, axis=0) * 100
        p95 = np.nanpercentile(arr, 95, axis=0) * 100
        p25 = np.nanpercentile(arr, 25, axis=0) * 100
        p75 = np.nanpercentile(arr, 75, axis=0) * 100
        ax.fill_between(times, p5,  p95, color=col, alpha=0.10, label="5–95 %")
        ax.fill_between(times, p25, p75, color=col, alpha=0.22, label="25–75 %")
        ax.plot(times, med, color=col, linewidth=1.8, label="Median", zorder=3)
        for k, idx in enumerate(sp_idx):
            ax.plot(times, arr[idx, :] * 100, color=col, linewidth=0.5,
                    alpha=0.45, zorder=2,
                    label="Sample paths" if k == 0 else None)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_xlabel("Time (years)"); ax.set_ylabel("Short rate (%)")
        ax.set_title(f"{lbl} short-rate fan")
        ax.legend(fontsize=8, frameon=False); ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Short-rate fan (5/25/50/75/95) + {N_SAMPLE_PATHS} sample paths  —  N={N_PATHS}",
        fontsize=11,
    )
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_short_rate_fan.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 5: Terminal distributions
    # =================================================================
    print("\n-- Fig 5: Terminal distributions --")
    if LATENT_DIM >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        z_train_stab_np = _np(z_train_stab)
        zmean_s = z_train_stab_np.mean(axis=0); zstd_s = z_train_stab_np.std(axis=0)
        axes[0].axhspan(zmean_s[1] - 2*zstd_s[1], zmean_s[1] + 2*zstd_s[1],
                        color=C_GREY, alpha=0.12, zorder=0)
        axes[0].axvspan(zmean_s[0] - 2*zstd_s[0], zmean_s[0] + 2*zstd_s[0],
                        color=C_GREY, alpha=0.12, zorder=0, label="Training +/-2sigma")
        axes[0].scatter(zs[:, -1, 0], zs[:, -1, 1], s=8, alpha=0.5, color=C_STAB,
                        label="Stable", zorder=2)
        # NaN-safe annotation for baseline off-scale value
        _bz1_mean = np.nanmean(zb[:, -1, 0])
        if np.isfinite(_bz1_mean) and abs(_bz1_mean) > 1:
            _orders = f"~{np.log10(abs(_bz1_mean)):.0f} orders"
        elif np.isfinite(_bz1_mean):
            _orders = "within O(1)"
        else:
            _orders = "NaN"
        axes[0].annotate(
            f"Baseline z1 mean: {_bz1_mean:.1e}\n(off-scale by {_orders})",
            xy=(0.03, 0.97), xycoords="axes fraction",
            ha="left", va="top", fontsize=7, color=C_BASE,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )
        axes[0].set_xlabel("$z_1$"); axes[0].set_ylabel("$z_2$")
        axes[0].set_title(f"Stable terminal $(z_1,z_2)$ -- $t={times[-1]:.0f}$ yr")
        axes[0].legend(fontsize=8, frameon=False); axes[0].grid(True, alpha=0.3)

        axes[1].hist(rs[:, -1]*100, bins=40, alpha=0.75, color=C_STAB, label="Stable", density=True)
        axes[1].set_xlabel("Short rate (%)"); axes[1].set_title("Stable terminal $r(T)$")
        axes[1].legend(fontsize=8, frameon=False); axes[1].grid(True, alpha=0.3)

        axes[2].hist(rb[:, -1]*100, bins=40, alpha=0.75, color=C_BASE, label="Baseline", density=True)
        axes[2].set_xlabel("Short rate (%)")
        axes[2].set_title("Baseline terminal $r(T)$ -- saturation")
        axes[2].legend(fontsize=8, frameon=False); axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f"Terminal distributions -- baseline vs stable  (N={N_PATHS}, $\\ell={LATENT_DIM}$)",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(zb[:, -1, 0], bins=40, alpha=0.5, color=C_BASE, label="Baseline")
        axes[0].hist(zs[:, -1, 0], bins=40, alpha=0.5, color=C_STAB, label="Stable")
        axes[0].set_xlabel("$z_1$"); axes[0].set_title("Terminal $z_1$")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].hist(rb[:, -1]*100, bins=40, alpha=0.5, color=C_BASE, label="Baseline")
        axes[1].hist(rs[:, -1]*100, bins=40, alpha=0.5, color=C_STAB, label="Stable")
        axes[1].set_xlabel("Short rate (%)"); axes[1].set_title("Terminal r")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_terminal_dist.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 6: Log-scale |z| growth
    # =================================================================
    print("\n-- Fig 6: Log-growth --")
    # Need training cloud stats for both models
    with torch.no_grad():
        zmean_base = _np(z_train_base).mean(axis=0)
        zstd_base  = _np(z_train_base).std(axis=0)
        zmean_stab = _np(z_train_stab).mean(axis=0)
        zstd_stab  = _np(z_train_stab).std(axis=0)

    fig, axes = plt.subplots(1, LATENT_DIM, figsize=(6*LATENT_DIM, 4.5), squeeze=False)
    fig.suptitle(
        r"Mean $|z_d(t)|$ on $\log_{10}$ scale — exponential divergence vs bounded OU",
        fontsize=11,
    )
    for dd in range(LATENT_DIM):
        ax = axes[0, dd]
        mean_abs_base = np.maximum(np.nanmean(np.abs(zb[:, :, dd]), axis=0), 1e-12)
        mean_abs_stab = np.maximum(np.nanmean(np.abs(zs[:, :, dd]), axis=0), 1e-12)
        ax.plot(times, np.log10(mean_abs_base), color=C_BASE, linewidth=2.0, label="Baseline")
        ax.plot(times, np.log10(mean_abs_stab), color=C_STAB, linewidth=2.0, label="Stable")
        support_level = np.log10(2 * zstd_stab[dd] + abs(zmean_stab[dd]))
        ax.axhline(support_level, color=C_GREY, linestyle="--", linewidth=1.0,
                   label=r"Training $\pm 2\sigma$ level")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(r"$\log_{10}(\mathrm{mean}\;|z_d|)$")
        ax.set_title(f"$z_{{{dd+1}}}$ growth")
        ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.grid(True, alpha=0.3)
        final_dec_base = np.log10(max(mean_abs_base[-1], 1e-12))
        final_dec_stab = np.log10(max(mean_abs_stab[-1], 1e-12))
        ax.annotate(
            f"Baseline final: ~$10^{{{final_dec_base:.0f}}}$\n"
            f"Stable final:   ~$10^{{{final_dec_stab:.1f}}}$",
            xy=(0.03, 0.95), xycoords="axes fraction",
            ha="left", va="top", fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_log_growth.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 7: Simulated yield / swap-rate curves at t = 1, 5, 10 yr
    # =================================================================
    print("\n-- Fig 7: Simulated yield curves at select horizons --")
    try:
        curves_base = _decode_curves_at_horizons(model_base, zb, times, CURVE_HORIZONS, tenors_np, DEVICE, DTYPE)
        curves_stab = _decode_curves_at_horizons(model_stab, zs, times, CURVE_HORIZONS, tenors_np, DEVICE, DTYPE)

        n_h = len(CURVE_HORIZONS)
        fig, axes = plt.subplots(2, n_h, figsize=(5.5*n_h, 8), squeeze=False)
        YIELD_CLIP = 50.0   # clip y-axis to [-50%, 50%] so baseline panels are readable

        for row, (curves, col_model, lbl_model) in enumerate([
            (curves_base, C_BASE, "Baseline"),
            (curves_stab, C_STAB, "Stable"),
        ]):
            for ci, h in enumerate(CURVE_HORIZONS):
                ax  = axes[row, ci]
                cd  = curves[h]
                tau = cd["tau"]
                hc  = col_model   # colour follows model row (consistent with other figures)
                ax.fill_between(tau, cd["p25"]*100, cd["p75"]*100,
                                color=hc, alpha=0.25, label="25-75 %")
                ax.fill_between(tau, cd["p5"]*100,  cd["p95"]*100,
                                color=hc, alpha=0.10, label="5-95 %")
                ax.plot(tau, cd["mean"]*100, color=hc, linewidth=2.0, label="Mean", zorder=3)
                for k, samp in enumerate(cd["samples"]):
                    ax.plot(tau, samp*100, color=hc, linewidth=0.5, alpha=0.5,
                            zorder=2, label="Sample paths" if k == 0 else None)
                ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.35)
                ax.set_xlabel("Maturity $\\tau$ (years)")
                ax.set_ylabel("Swap rate (%)")
                # Clip y-axis if bands or mean exceed threshold
                extreme_pct = max(
                    np.nanmax(np.abs(cd["p95"] * 100)),
                    np.nanmax(np.abs(cd["p5"]  * 100)),
                    np.nanmax(np.abs(cd["mean"] * 100)),
                )
                if lbl_model == "Baseline" and extreme_pct > YIELD_CLIP:
                    ax.set_ylim(-YIELD_CLIP, YIELD_CLIP)
                    ax.annotate(
                        f"y-axis clipped to $\\pm${YIELD_CLIP:.0f}%\n"
                        f"(valid paths: {cd['valid_frac']*100:.0f}%)",
                        xy=(0.97, 0.97), xycoords="axes fraction",
                        ha="right", va="top", fontsize=7, color=C_BASE,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                    )
                ax.set_title(f"{lbl_model} -- $t = {h}$ yr")
                ax.legend(fontsize=7, frameon=False)
                ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Decoded swap-rate curves at simulation horizons  (N={N_PATHS}, $\\ell={LATENT_DIM}$)",
            fontsize=12, y=1.01,
        )
        fig.tight_layout()
        p = os.path.join(OUT_DIR, "fig_yield_curves.png")
        fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")
    except Exception as e:
        print(f"  [Fig 7 yield curves] failed: {e}")
        import traceback; traceback.print_exc()

    # =================================================================
    # TABLE: Simulation diagnostics CSV (for LaTeX)
    # =================================================================
    print("\n-- Diagnostics table --")
    sum_base = _sim_summary(zb, rb, disc_base, "Baseline")
    sum_stab = _sim_summary(zs, rs, disc_stab, "Stable")

    rows = []
    nice = {
        "max_abs_z":          "Max |z| across all paths",
        "terminal_r_mean":    "Terminal r mean",
        "terminal_r_std":     "Terminal r std",
        "terminal_z_std":     "Terminal z std",
        "terminal_D_median":  "Terminal discount D(T) median",
        "terminal_D_range":   "Terminal discount D(T) range",
    }
    for key in sum_base:
        rows.append({
            "Metric":   nice.get(key, key),
            "Baseline": sum_base[key],
            "Stable":   sum_stab[key],
        })

    # r range
    rows.append({
        "Metric":   "r range (%)",
        "Baseline": f"[{np.nanmin(rb)*100:.2f}, {np.nanmax(rb)*100:.2f}]",
        "Stable":   f"[{np.nanmin(rs)*100:.2f}, {np.nanmax(rs)*100:.2f}]",
    })

    # Fraction of paths that stayed bounded
    rows.append({
        "Metric":   "Fraction with |z_T| < 10",
        "Baseline": f"{np.mean(np.max(np.abs(zb[:, -1, :]), axis=1) < 10)*100:.1f}%",
        "Stable":   f"{np.mean(np.max(np.abs(zs[:, -1, :]), axis=1) < 10)*100:.1f}%",
    })

    # In-sample reconstruction RMSE (bps)
    rows.append({
        "Metric":   "In-sample reconstruction RMSE (bps)",
        "Baseline": f"{rmse_base:.2f}",
        "Stable":   f"{rmse_stab:.2f}",
    })

    # Eigenvalues
    for d in range(LATENT_DIM):
        rows.append({
            "Metric":   f"Re(lambda_{d+1}) of M",
            "Baseline": f"{np.real(eigs_base[d]):.6f}",
            "Stable":   f"{np.real(eigs_stab[d]):.6f}",
        })

    # Sigma ranges
    for d in range(LATENT_DIM):
        rows.append({
            "Metric":   f"sigma_{d+1} range (train cloud)",
            "Baseline": f"[{sig_train_base[:, d].min():.4f}, {sig_train_base[:, d].max():.4f}]",
            "Stable":   f"[{sig_train_stab[:, d].min():.4f}, {sig_train_stab[:, d].max():.4f}]",
        })
    for d in range(LATENT_DIM):
        rows.append({
            "Metric":   f"sigma_{d+1} range (expanded grid)",
            "Baseline": f"[{sig_extra_base[:, d].min():.4f}, {sig_extra_base[:, d].max():.4f}]",
            "Stable":   f"[{sig_extra_stab[:, d].min():.4f}, {sig_extra_stab[:, d].max():.4f}]",
        })

    df_diag = pd.DataFrame(rows)
    p = os.path.join(OUT_DIR, "sim_diagnostics.csv")
    df_diag.to_csv(p, index=False)
    print(f"  Saved {p}")

    # ── LaTeX tabular fragment for the thesis chapter ────────────────────
    # Built from the same ``rows`` we just wrote — no CSV round-trip needed.
    # If make_diagnostics_table.py is missing for any reason, fall through
    # gracefully so the rest of the simulation output is unaffected.
    try:
        from make_diagnostics import build_rows, render  # local module
        d = {r["Metric"]: (r["Baseline"], r["Stable"]) for r in rows}
        tex_path = os.path.join(OUT_DIR, "sim_diagnostics_table.tex")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(render(build_rows(d)))
        print(f"  Saved {tex_path}")
    except ImportError:
        print("  [tex] make_diagnostics.py not found — skipping .tex fragment")
    except Exception as e:
        print(f"  [tex] failed to write .tex fragment: {e}")

    print(df_diag.to_string(index=False))

    # -- console summary --------------------------------------------------
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  FULL SIMULATION SUMMARY")
    print(SEP)
    for label, z, r in [("baseline", zb, rb), ("stable", zs, rs)]:
        fin_z = np.isfinite(z).all()
        fin_r = np.isfinite(r).all()
        print(f"  [{label:>8}]  finite_z={fin_z}  finite_r={fin_r}")
        for dd in range(LATENT_DIM):
            vals = z[:, :, dd]
            print(f"    z[{dd}]  mean={np.nanmean(vals):.4f}  std={np.nanstd(vals):.4f}"
                  f"  min={np.nanmin(vals):.4f}  max={np.nanmax(vals):.4f}")
        print(f"    r      mean={np.nanmean(r)*100:.2f}%  std={np.nanstd(r)*100:.2f}%"
              f"  min={np.nanmin(r)*100:.2f}%  max={np.nanmax(r)*100:.2f}%")

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()