"""
stable_vs_baseline_results.py
============================
Generate thesis-ready figures and tables comparing the baseline and stable
model variants under Euler--Maruyama simulation.

Outputs (saved to Figures/Pricing/):
  fig_eigenvalues.png     — Eigenvalues of the drift matrix M
  fig_sigma_bounds.png    — Volatility σ_i(z) distributions (training cloud + expanded grid)
  fig_latent_fans.png     — Percentile fan charts for latent factors z_1, z_2
  fig_short_rate_fan.png  — Percentile fan chart for the short rate
  fig_terminal_dist.png   — Terminal distributions of z and r
  sim_diagnostics.csv     — Summary table loaded by LaTeX
"""

import importlib
import math
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
    import importlib
    _orig = _cfg.VARIANT
    _cfg.VARIANT = variant          # must be set BEFORE FullModel is constructed
    try:
        if variant == "stable":
            import Code.model.full_model_stable as _m
            importlib.reload(_m)
            FullModel = _m.FullModel
        else:
            import Code.model.full_model as _m
            importlib.reload(_m)
            FullModel = _m.FullModel
        model = FullModel(latent_dim=latent_dim).to(device)  # build INSIDE try, while VARIANT is set
    finally:
        _cfg.VARIANT = _orig        # restore after construction
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


# =============================================================================
# Simulation (shared Brownian increments)
# =============================================================================
@torch.no_grad()
def _simulate(model, z0, n_paths, n_steps, dt, device, dtype, dW_all):
    d = z0.shape[1]
    sqrt_dt = math.sqrt(dt)
    z = z0.repeat(n_paths, 1).to(device=device, dtype=dtype)

    z_paths = torch.empty(n_paths, n_steps + 1, d, device=device, dtype=dtype)
    r_paths = torch.empty(n_paths, n_steps + 1, device=device, dtype=dtype)

    z_paths[:, 0, :] = z
    r_val = model.R(z)
    if r_val.ndim == 2 and r_val.shape[-1] == 1:
        r_val = r_val.squeeze(-1)
    r_paths[:, 0] = r_val

    for t in range(n_steps):
        mu = model.K(z)
        sigmas, rhos = model.H(z)
        L = L_from_sigmas_rhos(sigmas, rhos)

        dW = dW_all[:, t, :].to(device=device, dtype=dtype) * sqrt_dt
        shock = torch.bmm(L, dW.unsqueeze(-1)).squeeze(-1)
        z = z + mu * dt + shock

        z_paths[:, t + 1, :] = z
        r_val = model.R(z)
        if r_val.ndim == 2 and r_val.shape[-1] == 1:
            r_val = r_val.squeeze(-1)
        r_paths[:, t + 1] = r_val

    return z_paths, r_paths


def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


# =============================================================================
# Diagnostics helpers
# =============================================================================
def _eigenvalues_of_M(model, variant: str):
    """Extract eigenvalues of the drift matrix M."""
    if variant == "stable":
        M = model.K.stable_matrix().detach().cpu().numpy()
    else:
        # baseline: M is just the weight matrix of the linear layer
        M = model.K.lin.weight.detach().cpu().numpy()
    eigs = np.linalg.eigvals(M)
    return M, eigs


@torch.no_grad()
def _sigma_stats(model, z_cloud, z_extra):
    """Evaluate H on training cloud and expanded grid, return sigma arrays."""
    sig_train, _ = model.H(z_cloud)
    sig_extra, _ = model.H(z_extra)
    return _np(sig_train), _np(sig_extra)


def _sim_summary(z_np, r_np, label):
    """Compute summary dict for one set of simulation results."""
    fin_z = np.isfinite(z_np).all(axis=(1, 2))
    fin_r = np.isfinite(r_np).all(axis=1)
    fin_all = fin_z & fin_r
    return {
        "pct_finite": f"{100 * fin_all.mean():.1f}%",
        "max_abs_z":  f"{np.nanmax(np.abs(z_np)):.2f}",
        "terminal_r_mean": f"{np.nanmean(r_np[:, -1]) * 100:.2f}%",
        "terminal_r_std":  f"{np.nanstd(r_np[:, -1]) * 100:.2f}%",
        "terminal_z_std":  "; ".join(
            f"z{d+1}={np.nanstd(z_np[:, -1, d]):.3f}"
            for d in range(z_np.shape[2])
        ),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_paper_theme()
    times = np.arange(N_STEPS + 1) * DT

    # ── load data ─────────────────────────────────────────────────────
    meta, X_tensor, *_ = my_data(use=USE, ccy_filter=CCY_FILTER)
    X_tensor = X_tensor.to(dtype=DTYPE)

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

    # ── training latent cloud (for sigma-bound analysis) ──────────────
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

    # ── shared Brownian increments ────────────────────────────────────
    torch.manual_seed(SEED)
    dW_all = torch.randn(N_PATHS, N_STEPS, LATENT_DIM)

    # ── simulate ──────────────────────────────────────────────────────
    print("Simulating ...")
    t0 = time.time()
    z_base, r_base = _simulate(model_base, z0_base, N_PATHS, N_STEPS, DT, DEVICE, DTYPE, dW_all)
    z_stab, r_stab = _simulate(model_stab, z0_stab, N_PATHS, N_STEPS, DT, DEVICE, DTYPE, dW_all)
    print(f"Done in {time.time()-t0:.1f}s")

    zb = _np(z_base)
    zs = _np(z_stab)
    rb = _np(r_base)
    rs = _np(r_stab)

    # =================================================================
    # FIG 1: Eigenvalues of the drift matrix M
    # =================================================================
    print("\n── Fig 1: Drift eigenvalues ──")
    M_base, eigs_base = _eigenvalues_of_M(model_base, "baseline")
    M_stab, eigs_stab = _eigenvalues_of_M(model_stab, "stable")

    print(f"  Baseline eigenvalues: {eigs_base}")
    print(f"  Stable   eigenvalues: {eigs_stab}")

    fig, ax = plt.subplots(figsize=(6, 4))
    x_pos = np.arange(LATENT_DIM)
    w = 0.35
    ax.bar(x_pos - w / 2, np.real(eigs_base), width=w, color=C_BASE,
           edgecolor="none", label="Baseline", zorder=3)
    ax.bar(x_pos + w / 2, np.real(eigs_stab), width=w, color=C_STAB,
           edgecolor="none", label="Stable", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline(-1e-3, color=C_GREY, linewidth=0.8, linestyle="--",
               label=r"$-\epsilon$" + f" = {-1e-3}")

    # Annotate values
    for i, (eb, es) in enumerate(zip(np.real(eigs_base), np.real(eigs_stab))):
        ax.text(i - w / 2, eb - 0.02 * np.sign(eb), f"{eb:.4f}",
                ha="center", va="top" if eb < 0 else "bottom", fontsize=8, color=C_BASE)
        ax.text(i + w / 2, es - 0.02 * np.sign(es), f"{es:.4f}",
                ha="center", va="top", fontsize=8, color=C_STAB)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$\\lambda_{{{d+1}}}$" for d in range(LATENT_DIM)])
    ax.set_ylabel("Re($\\lambda$)")
    ax.set_title(f"Eigenvalues of drift matrix $M$ ($\\ell = {LATENT_DIM}$)")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_eigenvalues.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 2: Volatility σ bounds
    # =================================================================
    print("\n── Fig 2: Sigma bounds ──")
    sig_train_base, sig_extra_base = _sigma_stats(model_base, z_train_base, z_extra_base)
    sig_train_stab, sig_extra_stab = _sigma_stats(model_stab, z_train_stab, z_extra_stab)

    fig, axes = plt.subplots(1, LATENT_DIM, figsize=(5 * LATENT_DIM, 4), squeeze=False)
    for d in range(LATENT_DIM):
        ax = axes[0, d]
        # training cloud
        ax.hist(sig_train_base[:, d], bins=60, density=True, alpha=0.5,
                color=C_BASE, label="Baseline (train)")
        ax.hist(sig_train_stab[:, d], bins=60, density=True, alpha=0.5,
                color=C_STAB, label="Stable (train)")
        # expanded grid
        ax.hist(sig_extra_base[:, d], bins=60, density=True, alpha=0.25,
                color=C_BASE, histtype="step", linewidth=1.5, linestyle="--",
                label="Baseline (expanded)")
        ax.hist(sig_extra_stab[:, d], bins=60, density=True, alpha=0.25,
                color=C_STAB, histtype="step", linewidth=1.5, linestyle="--",
                label="Stable (expanded)")
        # Stable bounds
        if hasattr(model_stab.H, "sigma_min"):
            ax.axvline(model_stab.H.sigma_min, color=C_GREY, linewidth=1, linestyle=":",
                       label=f"$\\sigma_{{\\min}}={model_stab.H.sigma_min}$")
            ax.axvline(model_stab.H.sigma_max, color=C_GREY, linewidth=1, linestyle=":",
                       label=f"$\\sigma_{{\\max}}={model_stab.H.sigma_max}$")
        ax.set_xlabel(f"$\\sigma_{{{d+1}}}$")
        ax.set_ylabel("Density")
        ax.set_title(f"Volatility $\\sigma_{{{d+1}}}(z)$")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Volatility distributions ($\\ell = {LATENT_DIM}$)", fontsize=12, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_sigma_bounds.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 3: Latent-path percentile fans (side by side)
    # =================================================================
    print("\n── Fig 3: Latent fans ──")
    fig, axes = plt.subplots(2, LATENT_DIM, figsize=(6 * LATENT_DIM, 8), squeeze=False)
    for row, (arr, col, lbl) in enumerate([(zb, C_BASE, "Baseline"), (zs, C_STAB, "Stable")]):
        for dd in range(LATENT_DIM):
            ax = axes[row, dd]
            med = np.nanmedian(arr[:, :, dd], axis=0)
            p5  = np.nanpercentile(arr[:, :, dd], 5, axis=0)
            p95 = np.nanpercentile(arr[:, :, dd], 95, axis=0)
            p25 = np.nanpercentile(arr[:, :, dd], 25, axis=0)
            p75 = np.nanpercentile(arr[:, :, dd], 75, axis=0)
            ax.fill_between(times, p5, p95, color=col, alpha=0.15)
            ax.fill_between(times, p25, p75, color=col, alpha=0.30)
            ax.plot(times, med, color=col, linewidth=1.8, label=f"{lbl} median")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel(f"$z_{{{dd+1}}}$")
            ax.set_title(f"{lbl} — $z_{{{dd+1}}}$")
            ax.legend(fontsize=8, frameon=False)
            ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Latent-factor percentile fans (5/25/50/75/95)  —  N={N_PATHS}, $\\ell={LATENT_DIM}$",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_latent_fans.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 4: Short-rate percentile fan — SEPARATE panels per model
    # Overlaying both on one axis makes the stable model invisible when
    # the baseline saturates at large values. Separate panels allow each
    # model to use its own y-scale so both are actually readable.
    # =================================================================
    print("\n── Fig 4: Short-rate fan ──")
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
        ax.fill_between(times, p5, p95, color=col, alpha=0.12, label="5–95%")
        ax.fill_between(times, p25, p75, color=col, alpha=0.25, label="25–75%")
        ax.plot(times, med, color=col, linewidth=1.8, label="Median")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Short rate (%)")
        ax.set_title(f"{lbl} short-rate fan")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Short-rate percentile fan  (5/25/50/75/95)  —  N={N_PATHS}",
        fontsize=11,
    )
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_short_rate_fan.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # FIG 5: Terminal distributions
    # The baseline z is at ~1e20 so cannot be shown on the same scatter
    # as the stable model. Instead:
    #   - Left panel:  stable terminal (z1, z2) scatter with training cloud
    #   - Middle panel: stable terminal r histogram (economically realistic)
    #   - Right panel:  baseline terminal r histogram (bimodal saturation)
    # The baseline z scale is reported as a text annotation.
    # =================================================================
    print("\n── Fig 5: Terminal distributions ──")
    if LATENT_DIM >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Left: stable z scatter only — baseline is off-scale
        with torch.no_grad():
            z_train_stab_np = _np(z_train_stab)
        zmean_s = z_train_stab_np.mean(axis=0)
        zstd_s  = z_train_stab_np.std(axis=0)
        axes[0].axhspan(zmean_s[1] - 2*zstd_s[1], zmean_s[1] + 2*zstd_s[1],
                        color=C_GREY, alpha=0.12, zorder=0)
        axes[0].axvspan(zmean_s[0] - 2*zstd_s[0], zmean_s[0] + 2*zstd_s[0],
                        color=C_GREY, alpha=0.12, zorder=0, label="Training ±2σ")
        axes[0].scatter(zs[:, -1, 0], zs[:, -1, 1], s=8, alpha=0.5,
                        color=C_STAB, label="Stable", zorder=2)
        axes[0].annotate(
            f"Baseline z₁ mean: {np.nanmean(zb[:,-1,0]):.1e}\n"
            f"(off-scale by ~{max(np.log10(abs(np.nanmean(zb[:,-1,0]))+1),0):.0f} orders)",
            xy=(0.03, 0.97), xycoords="axes fraction",
            ha="left", va="top", fontsize=7, color=C_BASE,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )
        axes[0].set_xlabel("$z_1$"); axes[0].set_ylabel("$z_2$")
        axes[0].set_title(f"Stable terminal $(z_1, z_2)$ — $t={times[-1]:.0f}$ yr")
        axes[0].legend(fontsize=8, frameon=False)
        axes[0].grid(True, alpha=0.3)

        # Middle: stable r histogram
        axes[1].hist(rs[:, -1] * 100, bins=40, alpha=0.75, color=C_STAB,
                     label="Stable", density=True)
        axes[1].set_xlabel("Short rate (%)")
        axes[1].set_title("Stable terminal $r(T)$")
        axes[1].legend(fontsize=8, frameon=False)
        axes[1].grid(True, alpha=0.3)

        # Right: baseline r histogram (bimodal saturation)
        axes[2].hist(rb[:, -1] * 100, bins=40, alpha=0.75, color=C_BASE,
                     label="Baseline", density=True)
        axes[2].set_xlabel("Short rate (%)")
        axes[2].set_title("Baseline terminal $r(T)$ — saturation")
        axes[2].legend(fontsize=8, frameon=False)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f"Terminal distributions — baseline vs stable  (N={N_PATHS}, $\\ell={LATENT_DIM}$)",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(zb[:, -1, 0], bins=40, alpha=0.5, color=C_BASE, label="Baseline")
        axes[0].hist(zs[:, -1, 0], bins=40, alpha=0.5, color=C_STAB, label="Stable")
        axes[0].set_xlabel("$z_1$"); axes[0].set_title("Terminal $z_1$")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].hist(rb[:, -1] * 100, bins=40, alpha=0.5, color=C_BASE, label="Baseline")
        axes[1].hist(rs[:, -1] * 100, bins=40, alpha=0.5, color=C_STAB, label="Stable")
        axes[1].set_xlabel("Short rate (%)"); axes[1].set_title("Terminal r")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.tight_layout()

    p = os.path.join(OUT_DIR, "fig_terminal_dist.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # =================================================================
    # TABLE: Simulation diagnostics CSV (for LaTeX)
    # =================================================================
    print("\n── Diagnostics table ──")
    sum_base = _sim_summary(zb, rb, "Baseline")
    sum_stab = _sim_summary(zs, rs, "Stable")

    rows = []
    nice = {
        "pct_finite":     "Paths finite (%)",
        "max_abs_z":      "Max |z| across all paths",
        "terminal_r_mean": "Terminal r mean",
        "terminal_r_std":  "Terminal r std",
        "terminal_z_std":  "Terminal z std",
    }
    for key in sum_base:
        rows.append({
            "Metric":   nice.get(key, key),
            "Baseline": sum_base[key],
            "Stable":   sum_stab[key],
        })

    # Add eigenvalue info
    for d in range(LATENT_DIM):
        rows.append({
            "Metric":   f"Re(lambda_{d+1}) of M",
            "Baseline": f"{np.real(eigs_base[d]):.6f}",
            "Stable":   f"{np.real(eigs_stab[d]):.6f}",
        })

    # Add sigma range info
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
    print(df_diag.to_string(index=False))

    # ── console summary ───────────────────────────────────────────────
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

