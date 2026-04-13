"""
Compare Euler-Maruyama simulations from the baseline and stable model variants.

Both models are loaded from their respective checkpoints, given the **same**
initial latent state z0 and the **same** Brownian increments, so any difference
in the simulated paths is purely due to the learned K and H networks.

Outputs (saved to Code/Pricing/compare_out/):
  fig1  - Latent-state paths  z[d](t)  side-by-side
  fig2  - Short-rate  r(t)  comparison
  fig3  - Terminal distributions of z and r
  fig4  - Percentile fan charts for z
  fig5  - Percentile fan chart for r
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
EPOCHS = 3500
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
OUT_DIR        = os.path.join(SCRIPT_DIR, "compare_out", f"dim{LATENT_DIM}_ep{EPOCHS}")
os.makedirs(OUT_DIR, exist_ok=True)

# Colours
C_BASE = "#2c4f8c"
C_STAB = "#c0392b"


# =============================================================================
# Helpers
# =============================================================================
def _load_model(variant, ckpt_path, latent_dim, device, dtype):
    """Construct FullModel under *variant* and load checkpoint weights."""
    config.VARIANT = variant
    import Code.model.full_model as fm
    importlib.reload(fm)

    model = fm.FullModel(latent_dim=latent_dim).to(device)
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
            print(f"    z[{dd}]  mean={vals.mean():.4f}  std={vals.std():.4f}"
                  f"  min={vals.min():.4f}  max={vals.max():.4f}")
        print(f"    r      mean={r.mean()*100:.2f}bp  std={r.std()*100:.2f}bp"
              f"  min={r.min()*100:.2f}bp  max={r.max()*100:.2f}bp")

    # =================================================================
    # FIG 1: Latent paths z[d](t)
    # =================================================================
    fig, axes = plt.subplots(1, LATENT_DIM, figsize=(6 * LATENT_DIM, 5), squeeze=False)
    for dd in range(LATENT_DIM):
        ax = axes[0, dd]
        for i in range(min(N_PATHS_PLOT, N_PATHS)):
            ax.plot(times, zb[i, :, dd], color=C_BASE, alpha=0.25, linewidth=0.6)
            ax.plot(times, zs[i, :, dd], color=C_STAB, alpha=0.25, linewidth=0.6)
        ax.plot(times, zb[:, :, dd].mean(axis=0), color=C_BASE, linewidth=2.0, label="baseline mean")
        ax.plot(times, zs[:, :, dd].mean(axis=0), color=C_STAB, linewidth=2.0, label="stable mean")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(f"$z_{{{dd+1}}}$")
        ax.set_title(f"Latent factor $z_{{{dd+1}}}(t)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Simulated latent paths — baseline vs stable  (N={N_PATHS}, dim={LATENT_DIM})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig1_latent_paths.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # =================================================================
    # FIG 2: Short rate r(t)
    # =================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(N_PATHS_PLOT, N_PATHS)):
        ax.plot(times, rb[i, :] * 100, color=C_BASE, alpha=0.20, linewidth=0.6)
        ax.plot(times, rs[i, :] * 100, color=C_STAB, alpha=0.20, linewidth=0.6)
    ax.plot(times, rb.mean(axis=0) * 100, color=C_BASE, linewidth=2.0, label="baseline mean")
    ax.plot(times, rs.mean(axis=0) * 100, color=C_STAB, linewidth=2.0, label="stable mean")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Short rate (%)")
    ax.set_title(f"Short rate r(t) — baseline vs stable  (N={N_PATHS})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_short_rate.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # =================================================================
    # FIG 3: Terminal distributions
    # =================================================================
    if LATENT_DIM >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(zb[:, -1, 0], zb[:, -1, 1], s=6, alpha=0.4, color=C_BASE, label="baseline")
        axes[0].scatter(zs[:, -1, 0], zs[:, -1, 1], s=6, alpha=0.4, color=C_STAB, label="stable")
        axes[0].set_xlabel("$z_1$")
        axes[0].set_ylabel("$z_2$")
        axes[0].set_title(f"Terminal latent state  t = {times[-1]:.1f} yr")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(rb[:, -1] * 100, bins=40, alpha=0.5, color=C_BASE, label="baseline")
        axes[1].hist(rs[:, -1] * 100, bins=40, alpha=0.5, color=C_STAB, label="stable")
        axes[1].set_xlabel("Short rate (%)")
        axes[1].set_title(f"Terminal short-rate distribution  t = {times[-1]:.1f} yr")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(zb[:, -1, 0], bins=40, alpha=0.5, color=C_BASE, label="baseline")
        ax.hist(zs[:, -1, 0], bins=40, alpha=0.5, color=C_STAB, label="stable")
        ax.set_xlabel("$z_1$")
        ax.set_title(f"Terminal $z_1$ distribution  t = {times[-1]:.1f} yr")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    p = os.path.join(OUT_DIR, "fig3_terminal_distributions.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # =================================================================
    # FIG 4: z-path percentile fans
    # =================================================================
    fig, axes = plt.subplots(1, LATENT_DIM, figsize=(6 * LATENT_DIM, 5), squeeze=False)
    for dd in range(LATENT_DIM):
        ax = axes[0, dd]
        for arr, col, lbl in [(zb, C_BASE, "baseline"), (zs, C_STAB, "stable")]:
            med = np.median(arr[:, :, dd], axis=0)
            p5  = np.percentile(arr[:, :, dd], 5, axis=0)
            p95 = np.percentile(arr[:, :, dd], 95, axis=0)
            p25 = np.percentile(arr[:, :, dd], 25, axis=0)
            p75 = np.percentile(arr[:, :, dd], 75, axis=0)
            ax.fill_between(times, p5, p95, color=col, alpha=0.10)
            ax.fill_between(times, p25, p75, color=col, alpha=0.20)
            ax.plot(times, med, color=col, linewidth=1.8, label=f"{lbl} median")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(f"$z_{{{dd+1}}}$")
        ax.set_title(f"$z_{{{dd+1}}}$ fan  (5/25/50/75/95 pctl)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Percentile fans — baseline vs stable", fontsize=12, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig4_percentile_fans.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # =================================================================
    # FIG 5: Short-rate percentile fan
    # =================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    for arr, col, lbl in [(rb, C_BASE, "baseline"), (rs, C_STAB, "stable")]:
        med = np.median(arr, axis=0) * 100
        p5  = np.percentile(arr, 5, axis=0) * 100
        p95 = np.percentile(arr, 95, axis=0) * 100
        p25 = np.percentile(arr, 25, axis=0) * 100
        p75 = np.percentile(arr, 75, axis=0) * 100
        ax.fill_between(times, p5, p95, color=col, alpha=0.10)
        ax.fill_between(times, p25, p75, color=col, alpha=0.20)
        ax.plot(times, med, color=col, linewidth=1.8, label=f"{lbl} median")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Short rate (%)")
    ax.set_title("Short-rate fan  (5/25/50/75/95 pctl)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig5_short_rate_fan.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

