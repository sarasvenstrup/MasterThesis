# =============================================================================
# plot_ood_drift_comparison.py
#
# Compares OOD latent drift between dim=2 and dim=4 stable models.
# Produces one figure with two panels:
#
#   Left  — OOD fraction vs simulation horizon for both models
#   Right — Non-finite decoder fraction vs simulation horizon for both models
#
# Usage:
#   python Code/Pricing/plot_ood_drift_comparison.py
# =============================================================================

import os
import sys
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "..")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.Simulation.simulate_model import (
    load_and_setup_model,
    simulate_latent_paths,
    compute_latent_statistics,
)
from Code.load_swapdata import my_data

# ── Configuration ─────────────────────────────────────────────────────────
MODELS = [
    {
        "label":      r"$\ell=2$ stable",
        "checkpoint": r"Figures\TrainingResults\dim2_stable\ep5000\checkpoint_dim2_ep5000.pt",
        "latent_dim": 2,
        "color":      "steelblue",
    },
    {
        "label":      r"$\ell=4$ stable",
        "checkpoint": r"Figures\TrainingResults\dim4_stable\ep5000\checkpoint_dim4_ep5000.pt",
        "latent_dim": 4,
        "color":      "crimson",
    },
]

N_PATHS  = 5000
N_STEPS  = 120        # 10 years monthly
DT       = 1 / 12
SEED     = 1234

OUT_DIR  = r"Figures\PricingResults\Diagnostics\EUR_dim_comparison\plots"
DECODE_TAUS = torch.linspace(0.5, 15.0, 30)
# ──────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def decode_finite_fraction(model, z_snapshot, taus):
    try:
        P = model.decode_from_z(z_snapshot, tau=taus.to(z_snapshot.device, dtype=z_snapshot.dtype))
        return torch.isfinite(P).all(dim=1).float().mean().item()
    except Exception:
        return float("nan")


def run_model(cfg: dict, device, X_tensor):
    ckpt = os.path.join(REPO_ROOT, cfg["checkpoint"])
    print(f"\nLoading {cfg['label']} from {ckpt}")
    model = load_and_setup_model(device, ckpt, latent_dim=cfg["latent_dim"], use_double=True)
    model.eval()

    dtype = next(model.parameters()).dtype
    X = X_tensor.to(device=device, dtype=dtype)

    # Training distribution
    z_all = []
    for i in range(0, X.shape[0], 256):
        z_all.append(model.encoder(X[i:i+256]).detach())
    z_all = torch.cat(z_all, dim=0)
    z_min = z_all.cpu().numpy().min(axis=0)
    z_max = z_all.cpu().numpy().max(axis=0)
    print(f"  Training range: min={z_min.round(3)}  max={z_max.round(3)}")

    # Initial state: mid-sample observation
    mid = X.shape[0] // 2
    z0 = model.encoder(X[mid:mid+1]).detach()

    # Simulate
    print(f"  Simulating {N_PATHS} paths x {N_STEPS} steps ...")
    z_paths, _, _, _ = simulate_latent_paths(
        model, z0, N_PATHS, N_STEPS, DT, device,
        diffusion_scale=1.0, use_antithetic=False,
    )
    z_np = z_paths.detach().cpu().numpy()   # (N, T+1, d)

    # OOD fraction per step
    ood = np.zeros(N_STEPS + 1)
    for t in range(N_STEPS + 1):
        z_t = z_np[:, t, :]
        ood[t] = ((z_t < z_min) | (z_t > z_max)).any(axis=1).mean()

    # Non-finite fraction per step
    taus = DECODE_TAUS.to(device=device, dtype=z_paths.dtype)
    nonfinite = np.zeros(N_STEPS + 1)
    print("  Computing non-finite fraction ...")
    for t in range(N_STEPS + 1):
        ff = decode_finite_fraction(model, z_paths[:, t, :], taus)
        nonfinite[t] = 1.0 - ff if not math.isnan(ff) else float("nan")

    return ood, nonfinite


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    out_dir = os.path.join(REPO_ROOT, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    meta, X_tensor, *_ = my_data(use="bbg")

    times = np.arange(N_STEPS + 1) * DT
    results = []
    for cfg in MODELS:
        ood, nonfinite = run_model(cfg, device, X_tensor)
        results.append((cfg, ood, nonfinite))

    # ── Figure: side-by-side OOD and non-finite ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    for cfg, ood, nonfinite in results:
        ax1.plot(times, ood * 100, color=cfg["color"], lw=2.0, label=cfg["label"])
        if not np.all(np.isnan(nonfinite)):
            ax2.plot(times, nonfinite * 100, color=cfg["color"], lw=2.0,
                     ls="--", label=cfg["label"])

    for ax, title in [
        (ax1, "Out-of-distribution drift"),
        (ax2, "Paths with non-finite decoder output"),
    ]:
        ax.set_xlabel("Simulation horizon (years)", fontsize=11)
        ax.set_ylabel("Fraction of paths (%)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"OOD drift and decoder failure: $\\ell=2$ vs $\\ell=4$ stable model\n"
        f"({N_PATHS:,} paths, monthly steps, OOD = outside training min/max)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "ood_drift_dim2_vs_dim4.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\nSummary:")
    for cfg, ood, nonfinite in results:
        print(f"\n  {cfg['label']}:")
        for yr in [1, 5, 10]:
            t_idx = min(int(round(yr / DT)), N_STEPS)
            print(f"    t={yr}Y: OOD={ood[t_idx]*100:.1f}%  "
                  f"non-finite={nonfinite[t_idx]*100:.1f}%")


if __name__ == "__main__":
    main()
