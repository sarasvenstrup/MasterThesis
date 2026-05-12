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

    print(f"Initial short rate — Baseline: {rb[0,0]*100:.2f}%  Stable: {rs[0,0]*100:.2f}%")

    disc_base = _np(compute_discount_paths(r_base, DT))
    disc_stab = _np(compute_discount_paths(r_stab, DT))

    # sample-path indices for overlay (fixed seed)
    sp_rng  = np.random.default_rng(99)
    sp_idx  = sp_rng.choice(N_PATHS, size=N_SAMPLE_PATHS, replace=False)

    # =================================================================
    # FIG 0: Training latent cloud — time series per factor
    # =================================================================
    print("\n-- Fig 0: Training latent cloud (time series) --")
    if LATENT_DIM >= 2:
        dates = pd.to_datetime(meta["as_of_date"].values)
        init_date = pd.Timestamp("2010-01-29")
        init_mask = dates == init_date

        fig, axes = plt.subplots(LATENT_DIM, 1, figsize=(12, 4 * LATENT_DIM), sharex=True)
        if LATENT_DIM == 1:
            axes = [axes]

        h_base = h_stab = h_band_base = h_band_stab = h_star_base = h_star_stab = None

        for dd in range(LATENT_DIM):
            ax = axes[dd]

            # ±2σ horizontal bands
            for z_train, c, alpha in [
                (z_np_base, C_BASE, 0.12),
                (z_np_stab, C_STAB, 0.12),
            ]:
                mean_d = z_train[:, dd].mean()
                std_d  = z_train[:, dd].std()
                h = ax.axhspan(mean_d - 2*std_d, mean_d + 2*std_d,
                               color=c, alpha=alpha, zorder=0)
                if dd == 0:
                    if c == C_BASE:
                        h_band_base = h
                    else:
                        h_band_stab = h

            # Training dots
            sc_base = ax.scatter(dates, z_np_base[:, dd],
                                 color=C_BASE, alpha=0.45, s=12, zorder=2)
            sc_stab = ax.scatter(dates, z_np_stab[:, dd],
                                 color=C_STAB, alpha=0.45, s=12, zorder=2)
            if dd == 0:
                h_base = sc_base
                h_stab = sc_stab

            # Initial state stars
            if init_mask.any():
                idx = np.where(init_mask)[0][0]
                st_base = ax.scatter(dates[idx], z_np_base[idx, dd],
                                     color=C_BASE, s=150, marker="*", zorder=5,
                                     edgecolors="black", linewidths=0.5)
                st_stab = ax.scatter(dates[idx], z_np_stab[idx, dd],
                                     color=C_STAB, s=150, marker="*", zorder=5,
                                     edgecolors="black", linewidths=0.5)
                if dd == 0:
                    h_star_base = st_base
                    h_star_stab = st_stab

            ax.set_ylabel(f"$z_{{{dd+1}}}$")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("")

        # single legend below, moved slightly closer to the x-axis
        fig.legend(
            [h_base, h_stab, h_band_base, h_band_stab, h_star_base, h_star_stab],
            ["Baseline", "Stable",
             r"$\pm 2\sigma$ region (Baseline)", r"$\pm 2\sigma$ region (Stable)",
             "Initial state, Baseline (29 Jan 2010)",
             "Initial state, Stable (29 Jan 2010)"],
            loc="upper center", bbox_to_anchor=(0.5, 0.08),
            frameon=False, fontsize=8, ncol=3,
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.14)
        p = os.path.join(OUT_DIR, "fig_training_cloud.png")
        fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")

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
        ax.legend(fontsize=7, frameon=False); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_sigma_bounds.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # Small LaTeX table: volatility ranges
    # =================================================================
    print("\n-- Table: Sigma ranges --")

    def _sigma_range_tex(arr, d):
        lo = float(np.nanmin(arr[:, d]))
        hi = float(np.nanmax(arr[:, d]))
        return rf"\([{lo:.4f},\,{hi:.4f}]\)"

    lines = []
    lines.append("% Auto-generated by stable_vs_baseline_results.py — do not edit by hand.")
    lines.append(r"\begin{tabular}{@{}llcc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Factor} & \textbf{Evaluation set} & \textbf{Baseline} & \textbf{Stable} \\")
    lines.append(r"\midrule")

    for d in range(LATENT_DIM):
        factor = rf"\(\sigma_{d + 1}(z)\)"

        lines.append(
            f"{factor} & Encoded training cloud & "
            f"{_sigma_range_tex(sig_train_base, d)} & "
            f"{_sigma_range_tex(sig_train_stab, d)} \\\\"
        )

        lines.append(
            f"{factor} & Expanded diagnostic set & "
            f"{_sigma_range_tex(sig_extra_base, d)} & "
            f"{_sigma_range_tex(sig_extra_stab, d)} \\\\"
        )

        if d < LATENT_DIM - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    sigma_table_path = os.path.join(OUT_DIR, "sigma_ranges_table.tex")
    with open(sigma_table_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  Saved {sigma_table_path}")

    # =================================================================
    # FIG 3: Latent-path percentile fans  WITH  sample-path overlay
    # =================================================================
    print("\n-- Fig 3: Latent fans + sample paths --")
    fig, axes = plt.subplots(LATENT_DIM, 2, figsize=(12, 4 * LATENT_DIM), squeeze=False)
    col_data_fig3    = [(zb, C_BASE, "Baseline"), (zs, C_STAB, "Stable")]
    col_handles_fig3 = [None, None]

    for col_idx, (arr, c, lbl) in enumerate(col_data_fig3):
        for dd in range(LATENT_DIM):
            ax = axes[dd, col_idx]
            med = np.nanmedian(arr[:, :, dd], axis=0)
            p5  = np.nanpercentile(arr[:, :, dd],  5, axis=0)
            p95 = np.nanpercentile(arr[:, :, dd], 95, axis=0)
            p25 = np.nanpercentile(arr[:, :, dd], 25, axis=0)
            p75 = np.nanpercentile(arr[:, :, dd], 75, axis=0)
            h_outer = ax.fill_between(times, p5,  p95, color=c, alpha=0.12, label="5–95 %")
            h_inner = ax.fill_between(times, p25, p75, color=c, alpha=0.28, label="25–75 %")
            h_med,  = ax.plot(times, med, color=c, linewidth=1.8, label="Median", zorder=3)
            h_samp  = None
            for k, idx in enumerate(sp_idx):
                line, = ax.plot(times, arr[idx, :, dd], color=c, linewidth=0.5,
                                alpha=0.45, zorder=2,
                                label="Sample paths" if k == 0 else None)
                if k == 0:
                    h_samp = line
            if col_idx == 0:
                ax.set_ylabel(f"$z_{{{dd+1}}}$")
            if dd == LATENT_DIM - 1:
                ax.set_xlabel("Years")
            if dd == 0:
                axes[0, col_idx].set_title(lbl)
                col_handles_fig3[col_idx] = [h_med, h_inner, h_outer, h_samp]
            ax.grid(True, alpha=0.3)

    for col_idx in range(2):
        axes[LATENT_DIM - 1, col_idx].legend(
            col_handles_fig3[col_idx],
            ["Median", "25–75 %", "5–95 %", "Sample paths"],
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            frameon=False, fontsize=8, ncol=4,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    p = os.path.join(OUT_DIR, "fig_latent_fans.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 4: Short-rate percentile fan  WITH  sample-path overlay
    # =================================================================
    print("\n-- Fig 4: Short-rate fan + sample paths --")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    col_handles_fig4 = None
    for ax_idx, (ax, arr, col, lbl) in enumerate([
        (axes[0], rb, C_BASE, "Baseline"),
        (axes[1], rs, C_STAB, "Stable"),
    ]):
        med = np.nanmedian(arr, axis=0) * 100
        p5  = np.nanpercentile(arr,  5, axis=0) * 100
        p95 = np.nanpercentile(arr, 95, axis=0) * 100
        p25 = np.nanpercentile(arr, 25, axis=0) * 100
        p75 = np.nanpercentile(arr, 75, axis=0) * 100
        h_outer = ax.fill_between(times, p5,  p95, color=col, alpha=0.10, label="5–95 %")
        h_inner = ax.fill_between(times, p25, p75, color=col, alpha=0.22, label="25–75 %")
        h_med,  = ax.plot(times, med, color=col, linewidth=1.8, label="Median", zorder=3)
        h_samp  = None
        for k, idx in enumerate(sp_idx):
            line, = ax.plot(times, arr[idx, :] * 100, color=col, linewidth=0.5,
                    alpha=0.45, zorder=2,
                    label="Sample paths" if k == 0 else None)
            if k == 0:
                h_samp = line
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_title(lbl)
        ax.set_xlabel("Years")
        if ax_idx == 0:
            ax.set_ylabel("Short rate (%)")
            col_handles_fig4 = [h_med, h_inner, h_outer, h_samp]
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.legend(
            col_handles_fig4,
            ["Median", "25–75 %", "5–95 %", "Sample paths"],
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            frameon=False, fontsize=8, ncol=4,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    p = os.path.join(OUT_DIR, "fig_short_rate_fan.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 5: Terminal distributions  (2x2)
    # rows = r(T) histogram / z scatter,  cols = Baseline / Stable
    # =================================================================
    print("\n-- Fig 5: Terminal distributions --")
    if LATENT_DIM >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
        col_data_fig5 = [
            (zb, rb, z_train_base, M_base, C_BASE, "Baseline"),
            (zs, rs, z_train_stab, M_stab, C_STAB, "Stable"),
        ]

        for col_idx, (z_arr, r_arr, z_train, M_curr, c, lbl) in enumerate(col_data_fig5):
            z_train_np = _np(z_train)
            zmean = z_train_np.mean(axis=0)
            zstd  = z_train_np.std(axis=0)

            # Row 0: terminal r(T) histogram
            ax0 = axes[0, col_idx]
            ax0.hist(r_arr[:, -1] * 100, bins=40, alpha=0.75, color=c, density=True)
            ax0.set_title(lbl)
            ax0.set_xlabel("Short rate (%)")
            if col_idx == 0:
                ax0.set_ylabel("Density")
            ax0.grid(True, alpha=0.3)

            # Row 1: terminal latent scatter
            ax1 = axes[1, col_idx]
            ax1.axhspan(zmean[1] - 2*zstd[1], zmean[1] + 2*zstd[1],
                        color=C_GREY, alpha=0.12, zorder=0)
            ax1.axvspan(zmean[0] - 2*zstd[0], zmean[0] + 2*zstd[0],
                        color=C_GREY, alpha=0.12, zorder=0)
            for val in [zmean[0] - 2*zstd[0], zmean[0] + 2*zstd[0]]:
                ax1.axvline(val, color=C_GREY, linewidth=1.2, linestyle="--", zorder=1)
            for val in [zmean[1] - 2*zstd[1], zmean[1] + 2*zstd[1]]:
                ax1.axhline(val, color=C_GREY, linewidth=1.2, linestyle="--", zorder=1)
            h_band = ax1.axvline(zmean[0], color=C_GREY, linewidth=0, label="Training ±2σ")
            h_scat = ax1.scatter(z_arr[:, -1, 0], z_arr[:, -1, 1],
                                 s=8, alpha=0.5, color=c, zorder=2, label="Terminal $z$")

            # Eigenvector arrows scaled to axis range
            import matplotlib.colors as mcolors
            darken = 0.38 if c == C_BASE else 0.6
            arrow_c = tuple(x * darken for x in mcolors.to_rgb(c))
            eig_vals_f, eig_vecs_f = np.linalg.eig(M_curr)
            xlim = ax1.get_xlim(); ylim = ax1.get_ylim()
            scale = 0.25 * min(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
            ox, oy = zmean[0], zmean[1]
            dom_i = np.argmax(np.abs(np.real(eig_vals_f)))
            for i in [dom_i]:
                ev  = np.real(eig_vecs_f[:, i])
                ev  = ev / (np.linalg.norm(ev) + 1e-12)
                lam = np.real(eig_vals_f[i])
                dx, dy = ev * scale
                ax1.annotate("", xy=(ox + dx, oy + dy), xytext=(ox, oy),
                             arrowprops=dict(arrowstyle="-|>", color=arrow_c, lw=1.8),
                             zorder=5)
                right_perp = np.array([dy, -dx])
                right_perp = right_perp / (np.linalg.norm(right_perp) + 1e-12) * scale * 0.25
                ax1.text(ox + dx + right_perp[0], oy + dy + right_perp[1],
                         f"$\\lambda_{{{i+1}}}={lam:+.3f}$",
                         fontsize=7, color=arrow_c, ha="center", va="center", zorder=5,
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

            ax1.set_xlabel("$z_1$")
            if col_idx == 0:
                ax1.set_ylabel("$z_2$")
            ax1.grid(True, alpha=0.3)

            # Shared legend below bottom subplot
            ax1.legend(
                [h_band, h_scat],
                ["Training ±2σ", "Terminal $z$"],
                loc="upper center", bbox_to_anchor=(0.5, -0.18),
                frameon=False, fontsize=8, ncol=2,
            )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(zb[:, -1, 0], bins=40, alpha=0.5, color=C_BASE, label="Baseline")
        axes[0].hist(zs[:, -1, 0], bins=40, alpha=0.5, color=C_STAB, label="Stable")
        axes[0].set_xlabel("$z_1$")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].hist(rb[:, -1]*100, bins=40, alpha=0.5, color=C_BASE, label="Baseline")
        axes[1].hist(rs[:, -1]*100, bins=40, alpha=0.5, color=C_STAB, label="Stable")
        axes[1].set_xlabel("Short rate (%)")
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
    for dd in range(LATENT_DIM):
        mean_abs_base = np.maximum(np.nanmean(np.abs(zb[:, :, dd]), axis=0), 1e-12)
        mean_abs_stab = np.maximum(np.nanmean(np.abs(zs[:, :, dd]), axis=0), 1e-12)
        support_level = np.log10(2 * zstd_stab[dd] + abs(zmean_stab[dd]))

        ax = axes[0, dd]
        ax.plot(times, np.log10(mean_abs_base), color=C_BASE, linewidth=2.0, label="Baseline")
        ax.plot(times, np.log10(mean_abs_stab), color=C_STAB, linewidth=2.0, label="Stable")
        ax.axhline(support_level, color=C_GREY, linestyle="--", linewidth=1.0,
                   label="Training boundary")
        ax.set_title(f"$z_{{{dd+1}}}$ growth")
        ax.set_xlabel("Years")
        if dd == 0:
            ax.set_ylabel(r"$\log_{10}(\mathrm{mean}\;|z_d|)$")
        ax.grid(True, alpha=0.3)

    # shared legend below the figures
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
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
        YIELD_CLIP = 50.0   # currently unused — clipping disabled

        horizon_labels = {h: f"{h}Y" for h in CURVE_HORIZONS}
        row_handles = [None, None]

        for row, (curves, col_model, lbl_model) in enumerate([
            (curves_base, C_BASE, "Baseline"),
            (curves_stab, C_STAB, "Stable"),
        ]):
            for ci, h in enumerate(CURVE_HORIZONS):
                ax  = axes[row, ci]
                cd  = curves[h]
                tau = cd["tau"]
                hc  = col_model
                h_inner = ax.fill_between(tau, cd["p25"]*100, cd["p75"]*100,
                                color=hc, alpha=0.25, label="25-75 %")
                h_outer = ax.fill_between(tau, cd["p5"]*100,  cd["p95"]*100,
                                color=hc, alpha=0.10, label="5-95 %")
                h_mean, = ax.plot(tau, cd["mean"]*100, color=hc, linewidth=2.0, label="Mean", zorder=3)
                h_samp = None
                for k, samp in enumerate(cd["samples"]):
                    line, = ax.plot(tau, samp*100, color=hc, linewidth=0.5, alpha=0.5,
                            zorder=2, label="Sample paths" if k == 0 else None)
                    if k == 0:
                        h_samp = line
                ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.35)
                if row == 1:
                    ax.set_xlabel("Maturity (years)")
                if ci == 0:
                    ax.set_ylabel("Swap rate (%)")
                if row == 0:
                    ax.set_title(horizon_labels[h])
                if ci == 0:
                    row_handles[row] = [h_mean, h_inner, h_outer, h_samp]
                ax.grid(True, alpha=0.3)

        for row, lbl_model in enumerate(["Baseline", "Stable"]):
            axes[row, 0].annotate(
                lbl_model,
                xy=(0, 0.5), xycoords="axes fraction",
                xytext=(-0.18, 0.5), textcoords="axes fraction",
                ha="center", va="center", fontsize=11, fontweight="bold",
                rotation=90,
            )
            axes[row, -1].legend(
                row_handles[row],
                ["Mean", "25-75 %", "5-95 %", "Sample paths"],
                loc="center left", bbox_to_anchor=(1.02, 0.5),
                frameon=True, facecolor="white", edgecolor="none",
                fontsize=7,
            )

        # Clip baseline row to ±50% and annotate
        for ci in range(n_h):
            axes[0, ci].set_ylim(-50, 50)
            axes[0, ci].annotate(
                "Distribution truncated at ±50%",
                xy=(0.5, 0.97), xycoords="axes fraction",
                ha="center", va="top", fontsize=7, color=C_BASE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"),
            )

        # Shared y-axis across the Stable row
        ylims_stab = [axes[1, ci].get_ylim() for ci in range(n_h)]
        ymin_stab  = min(yl[0] for yl in ylims_stab)
        ymax_stab  = max(yl[1] for yl in ylims_stab)
        for ci in range(n_h):
            axes[1, ci].set_ylim(ymin_stab, ymax_stab)

        fig.tight_layout()
        fig.subplots_adjust(right=0.85)
        p = os.path.join(OUT_DIR, "fig_yield_curves.png")
        fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")
    except Exception as e:
        print(f"  [Fig 7 yield curves] failed: {e}")
        import traceback; traceback.print_exc()

    # =================================================================
    # FIG 8: Rolling OOS RMSE over time — Baseline (dashed) vs Stable (solid)
    # =================================================================
    print("\n-- Fig 8: Rolling OOS RMSE over time --")
    try:
        import warnings
        import matplotlib.dates as mdates
        from Code.load_swapdata import custom_palette

        ROLL_SUBDIR  = "train5Y_test6M_step6M"
        ROLL_EPOCHS  = 3500
        ROLL_CLIP    = 100
        DIM_COLORS   = {2: custom_palette[4], 3: custom_palette[0], 4: custom_palette[6]}
        EVENTS       = {
            "GFC\n(15 Sep 2008)":       "2008-09-15",
            "ECB QE\n(22 Jan 2015)":    "2015-01-22",
            "COVID\n(1 Mar 2020)":      "2020-03-01",
            "Rate hikes\n(1 Mar 2022)": "2022-03-01",
        }

        def _load_roll(dim, variant):
            fname = f"oos_rolling_bbg_dim{dim}_train5Y_test6M_step6M.csv"
            path  = os.path.join(
                THESIS_ROOT, "Figures", "OOSResults", "Roll",
                f"OOS_roll_dim{dim}_{variant}", ROLL_SUBDIR,
                f"ep{ROLL_EPOCHS}", fname,
            )
            if not os.path.exists(path):
                warnings.warn(f"Rolling CSV not found: {path}")
                return None
            df = pd.read_csv(path)
            df["test_start"] = pd.to_datetime(df["test_start"])
            return df

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        _any = False
        _x_min = _x_max = None

        for ax, dim in zip(axes, [2, 3, 4]):
            col = DIM_COLORS[dim]
            for variant, ls, alpha, lbl in [
                ("baseline", "--", 0.45, "Baseline"),
                ("stable",   "-",  1.0,  "Stable"),
            ]:
                df = _load_roll(dim, variant)
                if df is None:
                    continue
                _any = True
                # track x range across all panels
                _x_min = df["test_start"].min() if _x_min is None else min(_x_min, df["test_start"].min())
                _x_max = df["test_start"].max() if _x_max is None else max(_x_max, df["test_start"].max())

                avg_clipped = df["avg_rmse_bps"].clip(upper=ROLL_CLIP)
                ax.plot(df["test_start"], avg_clipped,
                        linewidth=1.8, color=col, linestyle=ls, alpha=alpha, label=lbl, zorder=5)

                # spike value annotations
                for _, row in df[df["avg_rmse_bps"] > ROLL_CLIP].iterrows():
                    ax.text(row["test_start"], ROLL_CLIP - 3,
                            f"{row['avg_rmse_bps']:.0f}",
                            fontsize=6, ha="center", va="top",
                            color=col, alpha=alpha, fontweight="bold", zorder=6)

            ax.set_ylim(0, ROLL_CLIP)
            ax.set_ylabel("RMSE (bps)")
            ax.annotate(f"$\\ell={dim}$", xy=(0.99, 0.97), xycoords="axes fraction",
                        ha="right", va="top", fontsize=10, fontweight="bold", color=col)
            ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="none",
                      loc="center left", bbox_to_anchor=(1.02, 0.5))
            ax.grid(True, alpha=0.3)

            # event markers per panel (skip GFC — outside data range)
            for ev_label, ev_date in EVENTS.items():
                if "GFC" in ev_label:
                    continue
                d = pd.Timestamp(ev_date)
                ax.axvline(d, color="0.5", linewidth=1.0, linestyle="--")
                if dim == 2:
                    ax.text(d, ROLL_CLIP, ev_label, fontsize=7,
                            ha="center", va="bottom", color="0.4")

        if _any:
            axes[-1].set_xlim(_x_min - pd.Timedelta(days=120), _x_max)
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.subplots_adjust(right=0.88)
            p = os.path.join(OUT_DIR, "fig_rolling_rmse.png")
            fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
            print(f"  Saved {p}")
        else:
            print("  SKIPPED — no rolling CSVs found")
            plt.close(fig)
    except Exception as e:
        print(f"  [Fig 8 rolling RMSE] failed: {e}")
        import traceback; traceback.print_exc()

    # =================================================================
    # FIG 9: Per-observation IS RMSE scatter over time, coloured by regime
    #        ℓ=4 only — top panel = Stable, bottom panel = Baseline
    #        Shared y-axis scale
    # =================================================================
    print("\n-- Fig 9: Regime scatter OOS (dim=4, stable vs baseline) --")
    try:
        from Code.load_swapdata import custom_palette as _cp
        import matplotlib.dates as _mdates

        _REGIME_GROUPS = [
            ("Normal, Non-negative",   False, False, _cp[2]),
            ("Inverted, Non-negative", True,  False, "black"),
            ("Normal, Negative",       False, True,  "indianred"),
            ("Inverted, Negative",     True,  True,  _cp[8]),
        ]

        _DIM9 = 4

        def _load_oos_scatter(variant):
            fname = "predictions_test_all.csv"
            path  = os.path.join(
                THESIS_ROOT, "Figures", "OOSResults", "Roll",
                f"OOS_roll_dim{_DIM9}_{variant}", ROLL_SUBDIR,
                f"ep{ROLL_EPOCHS}", fname,
            )
            if not os.path.exists(path):
                print(f"  SKIPPED {variant} dim={_DIM9} — OOS predictions not found: {path}")
                return None
            df = pd.read_csv(path)
            df["as_of_date"] = pd.to_datetime(df["as_of_date"])
            actual_cols = sorted([c for c in df.columns if c.startswith("actual_")])
            fitted_cols = sorted([c for c in df.columns if c.startswith("fitted_")])
            actual = df[actual_cols].values
            fitted = df[fitted_cols].values
            df["rmse_bps"] = np.sqrt(np.mean((actual - fitted) ** 2, axis=1)) * 1e4
            df["inv_flag"] = actual[:, 0] > actual[:, -1]
            df["neg_flag"] = (actual < 0).any(axis=1)
            return df

        fig9, axes9 = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)

        _legend_handles9 = []
        for ax9, variant, lbl_model in zip(axes9,
                                           ["baseline", "stable"],
                                           ["Baseline", "Stable"]):
            df9 = _load_oos_scatter(variant)
            if df9 is None:
                continue

            dates9   = df9["as_of_date"].values
            rmse9    = df9["rmse_bps"].values
            inv_all9 = df9["inv_flag"].values
            neg_all9 = df9["neg_flag"].values
            print(f"  {variant} dim={_DIM9}: avg OOS RMSE = {rmse9.mean():.2f} bps")

            for lbl, inv_flag, neg_flag, col in _REGIME_GROUPS:
                mask = (inv_all9 == inv_flag) & (neg_all9 == neg_flag)
                if not mask.any():
                    continue
                sc = ax9.scatter(dates9[mask], rmse9[mask],
                                 s=3, alpha=0.35, color=col, marker="o",
                                 label=lbl, zorder=3)
                if not _legend_handles9:
                    _legend_handles9.append(sc)
                elif lbl not in [h.get_label() for h in _legend_handles9]:
                    _legend_handles9.append(sc)

            ax9.set_ylabel("RMSE (bps)")
            ax9.annotate(lbl_model, xy=(0.99, 0.97), xycoords="axes fraction",
                         ha="right", va="top", fontsize=10, fontweight="bold",
                         color=C_STAB if variant == "stable" else C_BASE)
            ax9.grid(True, alpha=0.3)

        # single legend on the top panel
        _all_handles9 = []
        _all_labels9  = []
        for lbl, inv_flag, neg_flag, col in _REGIME_GROUPS:
            import matplotlib.lines as _mlines
            _all_handles9.append(_mlines.Line2D([], [], marker="o", color=col,
                                                linestyle="None", markersize=5))
            _all_labels9.append(lbl)
        axes9[0].legend(_all_handles9, _all_labels9,
                        fontsize=7, frameon=True, facecolor="white", edgecolor="none",
                        loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=1)

        axes9[-1].xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
        fig9.autofmt_xdate()
        fig9.tight_layout()
        fig9.subplots_adjust(right=0.88)
        p = os.path.join(OUT_DIR, "fig_regime_scatter.png")
        fig9.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig9)
        print(f"  Saved {p}")
    except Exception as e:
        print(f"  [Fig 9 regime scatter] failed: {e}")
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

    # =================================================================
    # TABLE: OOS summary across latent dimensions
    # =================================================================
    print("\n-- OOS summary table --")
    try:
        _roll_subdir = "train5Y_test6M_step6M"
        _roll_epochs = 3500
        _dims        = [2, 3, 4]
        _variants    = [("baseline", "Baseline"), ("stable", "Stable")]
        _tex_rows    = []

        for i, dim in enumerate(_dims):
            for variant, label in _variants:
                fname    = f"oos_rolling_bbg_dim{dim}_{_roll_subdir}.csv"
                csv_path = os.path.join(
                    THESIS_ROOT, "Figures", "OOSResults", "Roll",
                    f"OOS_roll_dim{dim}_{variant}",
                    _roll_subdir, f"ep{_roll_epochs}", fname,
                )
                if not os.path.exists(csv_path):
                    print(f"  [OOS table] missing: {csv_path}")
                    _tex_rows.append(f"    {dim} & {label} & -- & -- & -- \\\\")
                    continue
                _df      = pd.read_csv(csv_path)
                mean_is  = _df["avg_in_rmse_bps"].mean()
                mean_oos = _df["avg_rmse_bps"].mean()
                max_oos  = _df["avg_rmse_bps"].max()
                _tex_rows.append(
                    f"    {dim} & {label} & {mean_is:.1f} & {mean_oos:.1f} & {max_oos:.1f} \\\\"
                )
            if i < len(_dims) - 1:
                _tex_rows.append(r"    \midrule")

        oos_tex_path = os.path.join(OUT_DIR, "oos_summary_table.tex")
        with open(oos_tex_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_tex_rows) + "\n")
        print(f"  Saved {oos_tex_path}")
    except Exception as e:
        print(f"  [OOS table] failed: {e}")
        import traceback; traceback.print_exc()

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