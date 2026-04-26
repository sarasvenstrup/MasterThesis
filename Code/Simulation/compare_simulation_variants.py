"""
Compare Euler-Maruyama simulations from the baseline and stable model variants.

Both models are loaded from their respective checkpoints, given the **same**
initial latent state z0 and the **same** Brownian increments, so any difference
in the simulated paths is purely due to the learned K and H networks.

Outputs (saved to Figures/Simulation/):
  fig1  - Latent-state paths  z[d](t)  — SEPARATE panels, own y-scales
  fig2  - Short-rate  r(t)  — SEPARATE panels, own y-scales
  fig3  - Terminal distributions (histograms, sensible x-scales)
  fig4  - Percentile fan charts for z  — SEPARATE rows per model
  fig5  - Percentile fan chart for r  — SEPARATE panels per model
  fig6  - Log-scale |z| growth: exponential divergence vs bounded OU
"""

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── paths ────────────────────────────────────────────────────────────────
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
LATENT_DIM     = 2
EPOCHS = 5000
BASELINE_CKPT  = os.path.join(
    THESIS_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_baseline", f"ep{EPOCHS}",
    f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt",
)
STABLE_CKPT    = os.path.join(
    THESIS_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_stable", f"ep{EPOCHS}",
    f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt",
)

USE            = "bbg"
CCY_FILTER     = "EUR"
AS_OF_DATE     = None          # None -> first row
N_PATHS        = 500
N_STEPS        = 120           # 10 yr at monthly dt
DT             = 1 / 12
DIFFUSION_SCALE = 1.0
SEED           = 1234
DEVICE         = "cpu"
DTYPE          = torch.float64

N_PATHS_PLOT   = 30
OUT_DIR        = os.path.join(THESIS_ROOT, "Figures", "Simulation")
os.makedirs(OUT_DIR, exist_ok=True)

# Colours
C_BASE = "#2c4f8c"
C_STAB = "#c0392b"
C_TRAIN = "#aaaaaa"   # training cloud shading


# =============================================================================
# Helpers
# =============================================================================
def _load_model(variant, ckpt_path, latent_dim, device, dtype):
    """Construct FullModel under *variant* and load checkpoint weights."""
    import importlib
    from Code import config as _cfg
    _orig = _cfg.VARIANT
    _cfg.VARIANT = variant
    try:
        if variant == "stable":
            import Code.model.full_model_stable as _m
            importlib.reload(_m)
        else:
            import Code.model.full_model as _m
            importlib.reload(_m)
        model = _m.FullModel(latent_dim=latent_dim).to(device)
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


@torch.no_grad()
def _simulate(model, z0, n_paths, n_steps, dt, device, dtype, dW_all):
    """Euler-Maruyama with pre-generated Brownian increments *dW_all*."""
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
        L = L_from_sigmas_rhos(sigmas, rhos, validate=False)

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


def _train_cloud_stats(model, X_tensor, device, dtype):
    """Return (mean, std) vectors of z over the training set."""
    with torch.no_grad():
        z_train = model.encoder(X_tensor.to(device=device, dtype=dtype))
    z_np = _np(z_train)
    return z_np.mean(axis=0), z_np.std(axis=0)


# =============================================================================
# Main
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

    # ── training cloud stats for each model ───────────────────────────
    zmean_base, zstd_base = _train_cloud_stats(model_base, X_tensor, DEVICE, DTYPE)
    zmean_stab, zstd_stab = _train_cloud_stats(model_stab, X_tensor, DEVICE, DTYPE)

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

    # ── shared Brownian increments ────────────────────────────────────
    torch.manual_seed(SEED)
    dW_all = torch.randn(N_PATHS, N_STEPS, LATENT_DIM)

    # ── simulate ──────────────────────────────────────────────────────
    print("Simulating ...")
    t0 = time.time()
    z_base, r_base = _simulate(model_base, z0_base, N_PATHS, N_STEPS, DT, DEVICE, DTYPE, dW_all)
    z_stab, r_stab = _simulate(model_stab, z0_stab, N_PATHS, N_STEPS, DT, DEVICE, DTYPE, dW_all)
    print(f"Done in {time.time()-t0:.1f}s")

    zb = _np(z_base)   # (N, T+1, d)
    zs = _np(z_stab)
    rb = _np(r_base)    # (N, T+1)
    rs = _np(r_stab)

    # ── console summary ───────────────────────────────────────────────
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  SIMULATION SUMMARY")
    print(SEP)
    for label, z, r in [("baseline", zb, rb), ("stable", zs, rs)]:
        fin_z = np.isfinite(z).all()
        fin_r = np.isfinite(r).all()
        print(f"  [{label:>8}]  finite_z={fin_z}  finite_r={fin_r}")
        for dd in range(LATENT_DIM):
            vals = z[:, :, dd]
            print(f"    z[{dd}]  mean={np.nanmean(vals):.4f}  std={np.nanstd(vals):.4f}"
                  f"  min={np.nanmin(vals):.4f}  max={np.nanmax(vals):.4f}")
        print(f"    r      mean={np.nanmean(r)*100:.2f}bp  std={np.nanstd(r)*100:.2f}bp"
              f"  min={np.nanmin(r)*100:.2f}bp  max={np.nanmax(r)*100:.2f}bp")

    # =================================================================
    # FIG 1: Latent paths — separate panels, own y-scales per model
    # Baseline row: shows divergence on its own (possibly huge) scale
    # Stable row: shows bounded OU with ±2σ training cloud shaded
    # =================================================================
    fig, axes = plt.subplots(2, LATENT_DIM,
                              figsize=(6 * LATENT_DIM, 7), squeeze=False)
    fig.suptitle(
        f"Simulated latent paths — Baseline (top) vs Stable (bottom)\n"
        f"N={N_PATHS}, {N_PATHS_PLOT} shown, dim={LATENT_DIM}, ep={EPOCHS}",
        fontsize=11,
    )

    for dd in range(LATENT_DIM):
        # --- top row: baseline ---
        ax = axes[0, dd]
        for i in range(min(N_PATHS_PLOT, N_PATHS)):
            ax.plot(times, zb[i, :, dd], color=C_BASE, alpha=0.20, linewidth=0.5)
        ax.plot(times, np.nanmean(zb[:, :, dd], axis=0),
                color=C_BASE, linewidth=2.0, label="Mean")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(f"$z_{{{dd+1}}}$")
        ax.set_title(f"Baseline  $z_{{{dd+1}}}(t)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Annotate the final scale so the divergence is obvious
        max_abs = np.nanmax(np.abs(zb[:, :, dd]))
        ax.annotate(
            f"max|z| ≈ {max_abs:.1e}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            color=C_BASE,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

        # --- bottom row: stable ---
        ax = axes[1, dd]
        # training cloud ±2σ shading
        lo = zmean_stab[dd] - 2 * zstd_stab[dd]
        hi = zmean_stab[dd] + 2 * zstd_stab[dd]
        ax.axhspan(lo, hi, color=C_TRAIN, alpha=0.18,
                   label="Training ±2σ")
        ax.axhline(zmean_stab[dd], color=C_TRAIN, linewidth=1.0,
                   linestyle="--", alpha=0.7)
        for i in range(min(N_PATHS_PLOT, N_PATHS)):
            ax.plot(times, zs[i, :, dd], color=C_STAB, alpha=0.20, linewidth=0.5)
        ax.plot(times, np.nanmean(zs[:, :, dd], axis=0),
                color=C_STAB, linewidth=2.0, label="Mean")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(f"$z_{{{dd+1}}}$")
        ax.set_title(f"Stable  $z_{{{dd+1}}}(t)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig1_latent_paths.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # =================================================================
    # FIG 2: Short rate — separate panels per model
    # Left: Baseline (showing saturation at hard limits)
    # Right: Stable (economically realistic range)
    # =================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    for ax, r_np, col, lbl in [
        (axes[0], rb, C_BASE, "Baseline"),
        (axes[1], rs, C_STAB, "Stable"),
    ]:
        for i in range(min(N_PATHS_PLOT, N_PATHS)):
            ax.plot(times, r_np[i, :] * 100, color=col, alpha=0.20, linewidth=0.5)
        ax.plot(times, np.nanmean(r_np, axis=0) * 100,
                color=col, linewidth=2.0, label="Mean")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Short rate (%)")
        ax.set_title(f"{lbl} short rate $r(t)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # annotate range
        rng = (np.nanmin(r_np) * 100, np.nanmax(r_np) * 100)
        ax.annotate(
            f"range [{rng[0]:.1f}%, {rng[1]:.1f}%]",
            xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    fig.suptitle(
        f"Short rate $r(t)$ — Baseline (left) vs Stable (right)  (N={N_PATHS}, ep={EPOCHS})",
        fontsize=11,
    )
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_short_rate.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # =================================================================
    # FIG 3: Terminal distributions
    # Panels: (a) stable terminal z scatter  (b) r histograms for both
    # The baseline z scatter is on order 1e20, so we skip it and only
    # show the r histograms where the saturation is clearly visible.
    # =================================================================
    fig = plt.figure(figsize=(14, 4.5))

    # Left: stable terminal z scatter (baseline would be off any axis)
    ax0 = fig.add_subplot(1, 3, 1)
    if LATENT_DIM >= 2:
        ax0.scatter(zs[:, -1, 0], zs[:, -1, 1], s=8, alpha=0.5,
                    color=C_STAB, label="Stable")
        # add baseline as single text annotation showing its actual location
        ax0.annotate(
            f"Baseline terminal z₁: {np.nanmean(zb[:,-1,0]):.1e}\n"
            f"Baseline terminal z₂: {np.nanmean(zb[:,-1,1]):.1e}\n"
            f"(off-scale by ~{np.log10(max(abs(np.nanmean(zb[:,-1,0])),1)):.0f} orders)",
            xy=(0.03, 0.97), xycoords="axes fraction",
            ha="left", va="top", fontsize=6.5,
            color=C_BASE,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )
        # shade training region
        ax0.axhspan(zmean_stab[1] - 2*zstd_stab[1],
                    zmean_stab[1] + 2*zstd_stab[1],
                    color=C_TRAIN, alpha=0.12, zorder=0)
        ax0.axvspan(zmean_stab[0] - 2*zstd_stab[0],
                    zmean_stab[0] + 2*zstd_stab[0],
                    color=C_TRAIN, alpha=0.12, zorder=0, label="Train ±2σ")
        ax0.set_xlabel("$z_1$"); ax0.set_ylabel("$z_2$")
    ax0.set_title(f"Terminal $(z_1, z_2)$ at $t={times[-1]:.0f}$ yr\n(stable only — baseline off-scale)")
    ax0.legend(fontsize=7, frameon=False)
    ax0.grid(True, alpha=0.3)

    # Middle: terminal r histogram for stable only (proper scale)
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.hist(rs[:, -1] * 100, bins=40, alpha=0.75, color=C_STAB,
             label="Stable", density=True)
    ax1.set_xlabel("Short rate (%)")
    ax1.set_title(f"Stable terminal $r(T)$  — $t={times[-1]:.0f}$ yr")
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(True, alpha=0.3)

    # Right: terminal r histogram for baseline (showing bimodal saturation)
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.hist(rb[:, -1] * 100, bins=40, alpha=0.75, color=C_BASE,
             label="Baseline", density=True)
    ax2.set_xlabel("Short rate (%)")
    ax2.set_title(f"Baseline terminal $r(T)$  — bimodal saturation")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Terminal distributions at $t = {times[-1]:.0f}$ yr  (N={N_PATHS}, ep={EPOCHS})",
        fontsize=11,
    )
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig3_terminal_distributions.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # =================================================================
    # FIG 4: Percentile fans — 2 rows (baseline / stable), own y-scales
    # =================================================================
    fig, axes = plt.subplots(2, LATENT_DIM,
                              figsize=(6 * LATENT_DIM, 8), squeeze=False)
    fig.suptitle(
        f"Latent-factor percentile fans (5/25/50/75/95)  —  N={N_PATHS}, ep={EPOCHS}",
        fontsize=11,
    )

    for row, (arr, col, lbl, zmean, zstd) in enumerate([
        (zb, C_BASE, "Baseline", zmean_base, zstd_base),
        (zs, C_STAB, "Stable",   zmean_stab, zstd_stab),
    ]):
        for dd in range(LATENT_DIM):
            ax = axes[row, dd]
            med = np.nanmedian(arr[:, :, dd], axis=0)
            p5  = np.nanpercentile(arr[:, :, dd],  5, axis=0)
            p95 = np.nanpercentile(arr[:, :, dd], 95, axis=0)
            p25 = np.nanpercentile(arr[:, :, dd], 25, axis=0)
            p75 = np.nanpercentile(arr[:, :, dd], 75, axis=0)
            ax.fill_between(times, p5, p95, color=col, alpha=0.10, label="5–95%")
            ax.fill_between(times, p25, p75, color=col, alpha=0.22, label="25–75%")
            ax.plot(times, med, color=col, linewidth=1.8, label="Median")
            if lbl == "Stable":
                lo = zmean[dd] - 2 * zstd[dd]
                hi = zmean[dd] + 2 * zstd[dd]
                ax.axhspan(lo, hi, color=C_TRAIN, alpha=0.15, zorder=0,
                           label="Training ±2σ")
                ax.axhline(zmean[dd], color=C_TRAIN, linewidth=1.0,
                           linestyle="--", alpha=0.6)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel(f"$z_{{{dd+1}}}$")
            ax.set_title(f"{lbl} — $z_{{{dd+1}}}$ fan")
            ax.legend(fontsize=7, frameon=False)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig4_percentile_fans.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # =================================================================
    # FIG 5: Short-rate percentile fans — two separate panels
    # =================================================================
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
        f"Short-rate percentile fan  (5/25/50/75/95)  —  N={N_PATHS}, ep={EPOCHS}",
        fontsize=11,
    )
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig5_short_rate_fan.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # =================================================================
    # FIG 6 (NEW): Log-scale |z| growth — clearest divergence diagnostic
    # y-axis: log₁₀(mean |z_d| across paths)
    # Baseline → straight line (exponential growth)
    # Stable   → flattens (bounded OU)
    # This is the single most informative diagnostic plot.
    # =================================================================
    fig, axes = plt.subplots(1, LATENT_DIM,
                              figsize=(6 * LATENT_DIM, 4.5), squeeze=False)
    fig.suptitle(
        r"Mean $|z_d(t)|$ on log$_{10}$ scale — exponential divergence vs bounded OU",
        fontsize=11,
    )

    for dd in range(LATENT_DIM):
        ax = axes[0, dd]
        # Baseline: mean |z| per time step
        mean_abs_base = np.nanmean(np.abs(zb[:, :, dd]), axis=0)
        mean_abs_stab = np.nanmean(np.abs(zs[:, :, dd]), axis=0)

        # guard log(0)
        mean_abs_base = np.maximum(mean_abs_base, 1e-12)
        mean_abs_stab = np.maximum(mean_abs_stab, 1e-12)

        ax.plot(times, np.log10(mean_abs_base), color=C_BASE,
                linewidth=2.0, label="Baseline")
        ax.plot(times, np.log10(mean_abs_stab), color=C_STAB,
                linewidth=2.0, label="Stable")

        # Shade training support level (log scale of 2σ)
        support_level = np.log10(2 * zstd_stab[dd] + abs(zmean_stab[dd]))
        ax.axhline(support_level, color=C_TRAIN, linestyle="--",
                   linewidth=1.0, label="Training ±2σ level")

        ax.set_xlabel("Time (years)")
        ax.set_ylabel(r"$\log_{10}(\mathrm{mean}\;|z_d|)$")
        ax.set_title(f"$z_{{{dd+1}}}$ growth")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)

        # Annotate final decade for baseline
        final_decades = np.log10(max(mean_abs_base[-1], 1))
        ax.annotate(
            f"Baseline final: ~10^{final_decades:.0f}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=8, color=C_BASE,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig6_log_growth.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

