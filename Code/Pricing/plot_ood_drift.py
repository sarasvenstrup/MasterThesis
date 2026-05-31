# =============================================================================
# plot_ood_drift.py
#
# Diagnostic figure showing that simulated latent states drift outside the
# training distribution, and that non-finite decoder outputs are directly
# correlated with out-of-distribution (OOD) states.
#
# Produces two figures:
#
#   Fig 1 — "latent_ood_drift.png"
#       4 panels (one per latent dimension) showing:
#         - Shaded band: training distribution ±2 std
#         - Dark band: 10th–90th percentile of simulated paths over time
#         - Light band: 5th–95th percentile of simulated paths
#       Directly shows paths spreading beyond the training range with time.
#
#   Fig 2 — "ood_vs_nonfinite.png"
#       Single panel showing two lines vs simulation time:
#         - Fraction of paths with any z dimension outside training ±2 std
#         - Fraction of paths with non-finite decoder output at that time step
#       Directly shows the correlation between OOD states and decoder failure.
#
# Usage:
#   python Code/Pricing/plot_ood_drift.py
#   python Code/Pricing/plot_ood_drift.py --checkpoint <path> --out-dir <dir>
# =============================================================================

import argparse
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

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    r"Figures\TrainingResults\dim4_stable\ep5000\checkpoint_dim4_ep5000.pt"
)
DEFAULT_OUT_DIR = r"Figures\PricingResults\Diagnostics\EUR_dim4_stable\plots"
N_PATHS   = 5000          # MC paths — enough for stable percentiles
N_STEPS   = 120           # monthly steps → 10-year horizon
DT        = 1 / 12
LATENT_DIM = 4
OOD_SIGMA  = 2.0          # threshold: outside ± OOD_SIGMA * std_train = OOD
SEED       = 1234
# ─────────────────────────────────────────────────────────────────────────────

DECODE_TAUS = torch.linspace(0.5, 15.0, 30)   # maturities used to probe the decoder


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--out-dir",    default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-paths",    type=int, default=N_PATHS)
    ap.add_argument("--n-steps",    type=int, default=N_STEPS)
    ap.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    return ap.parse_args()


@torch.no_grad()
def decode_finite_fraction(model, z_snapshot: torch.Tensor, taus: torch.Tensor) -> float:
    """
    Solve the full ODE bond-price computation P(z, tau) for a batch of z
    states. Returns the fraction of paths whose decoded bond prices are all
    finite across the probed maturities.

    z_snapshot : (n_paths, d)
    taus       : (T,)
    """
    try:
        taus_in = taus.to(z_snapshot.device, dtype=z_snapshot.dtype)
        # decode_from_z runs the full ODE: G → alpha/beta/gamma → A/B → exp(A-BG)
        # This is the actual computation that produces inf in the pricing chapter.
        P_full = model.decode_from_z(z_snapshot, tau=taus_in)   # (B, T)
        # A path is "finite" only if all its bond prices are finite
        path_finite = torch.isfinite(P_full).all(dim=1)          # (B,)
        finite_frac = path_finite.float().mean().item()
    except Exception as e:
        finite_frac = float("nan")
    return finite_frac


def run(args):
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ckpt = os.path.join(REPO_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    out_dir = os.path.join(REPO_ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model from {ckpt}")
    model = load_and_setup_model(device, ckpt, latent_dim=args.latent_dim, use_double=True)
    model.eval()

    # ── Training distribution ─────────────────────────────────────────────
    meta, X_tensor, *_ = my_data(use="bbg")
    X_tensor = X_tensor.to(device=device, dtype=next(model.parameters()).dtype)
    z_mean, z_cov, z_std = compute_latent_statistics(model, X_tensor, device, args.latent_dim)
    z_mean_np = z_mean.cpu().numpy()
    z_std_np  = z_std.cpu().numpy()

    # Training min/max: the actual interpolation region seen during training
    with torch.no_grad():
        z_all = []
        for i in range(0, X_tensor.shape[0], 256):
            z_all.append(model.encoder(X_tensor[i:i+256]).detach())
        z_all = torch.cat(z_all, dim=0)
    z_min_np = z_all.cpu().numpy().min(axis=0)
    z_max_np = z_all.cpu().numpy().max(axis=0)
    print("Training z min:", z_min_np)
    print("Training z max:", z_max_np)

    # ── Initial state: use a representative mid-sample observation ─────────
    # Pick the observation closest to the training mean to start near the
    # centre of the distribution, giving a fairer OOD trajectory picture.
    mid_idx = int(X_tensor.shape[0] // 2)
    z0 = model.encoder(X_tensor[mid_idx:mid_idx+1]).detach()
    print(f"Initial z0: {z0.detach().cpu().numpy().flatten()}")

    # ── Simulate ──────────────────────────────────────────────────────────
    print(f"Simulating {args.n_paths} paths x {args.n_steps} steps ...")
    z_paths, r_paths, _, _ = simulate_latent_paths(
        model, z0, args.n_paths, args.n_steps, DT, device,
        diffusion_scale=1.0, use_antithetic=False,
    )
    # z_paths: (n_paths, n_steps+1, d)
    z_np = z_paths.detach().cpu().numpy()     # (N, T+1, d)
    times = np.arange(args.n_steps + 1) * DT

    # ── Per-step OOD fraction ─────────────────────────────────────────────
    # A path/step is OOD if any z dimension falls outside the training
    # min/max range — i.e. the model is required to extrapolate.
    lo = z_min_np   # (d,)
    hi = z_max_np   # (d,)
    ood_per_step = np.zeros(args.n_steps + 1)
    for t in range(args.n_steps + 1):
        z_t = z_np[:, t, :]                               # (N, d)
        outside = ((z_t < lo) | (z_t > hi)).any(axis=1)  # (N,) bool
        ood_per_step[t] = outside.mean()

    # ── Per-step non-finite decoder fraction ──────────────────────────────
    taus_probe = DECODE_TAUS.to(device=device, dtype=z_paths.dtype)
    nonfinite_per_step = np.zeros(args.n_steps + 1)
    print("Computing non-finite fraction per time step ...")
    for t in range(args.n_steps + 1):
        z_t = z_paths[:, t, :]    # (N, d)
        ff = decode_finite_fraction(model, z_t, taus_probe)
        nonfinite_per_step[t] = 1.0 - ff if not math.isnan(ff) else float("nan")

    # ── Figure 1: Latent drift per dimension ─────────────────────────────
    d = args.latent_dim
    ncols = 2
    nrows = math.ceil(d / ncols)
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows), dpi=150)
    axes1 = np.array(axes1).flatten()

    pcts = [5, 10, 25, 50, 75, 90, 95]
    z_pct = np.percentile(z_np, pcts, axis=0)   # (7, T+1, d)

    for di in range(d):
        ax = axes1[di]
        # Training min/max band (actual interpolation region)
        ax.axhspan(lo[di], hi[di], color="green", alpha=0.15,
                   label="Training range [min, max]")
        ax.axhline(z_mean_np[di], color="green", lw=1.0, ls="--", alpha=0.7)
        # Simulated percentile bands
        ax.fill_between(times, z_pct[1, :, di], z_pct[5, :, di],
                        color="steelblue", alpha=0.25, label="10th–90th pct")
        ax.fill_between(times, z_pct[0, :, di], z_pct[6, :, di],
                        color="steelblue", alpha=0.12, label="5th–95th pct")
        ax.plot(times, z_pct[3, :, di], color="steelblue", lw=1.2, label="Median")
        ax.set_title(f"$z_{{{di+1}}}$", fontsize=11)
        ax.set_xlabel("Simulation horizon (years)")
        ax.set_ylabel("Latent state value")
        ax.grid(True, alpha=0.25)
        if di == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Hide unused panels
    for di in range(d, len(axes1)):
        axes1[di].set_visible(False)

    fig1.suptitle(
        "Simulated latent-state spread vs. training distribution\n"
        f"({args.n_paths:,} paths, monthly steps, $\\ell={d}$)",
        fontsize=11,
    )
    fig1.tight_layout()
    fig1_path = os.path.join(out_dir, "latent_ood_drift.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {fig1_path}")

    # ── Figure 2: OOD fraction vs non-finite fraction ────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=150)
    ax2.plot(times, ood_per_step * 100, color="darkorange", lw=1.8,
             label="OOD paths (any $z_d$ outside training range)")
    if not np.all(np.isnan(nonfinite_per_step)):
        ax2.plot(times, nonfinite_per_step * 100, color="crimson", lw=1.8, ls="--",
                 label="Paths with non-finite decoder output")
    ax2.set_xlabel("Simulation horizon (years)", fontsize=11)
    ax2.set_ylabel("Fraction of paths (%)", fontsize=11)
    ax2.set_title(
        "Out-of-distribution drift and decoder failure vs. simulation horizon\n"
        f"({args.n_paths:,} paths, $\\ell={d}$, OOD defined as outside training min/max)",
        fontsize=10,
    )
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, "ood_vs_nonfinite.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    for yr in [1, 5, 10]:
        t_idx = min(int(round(yr / DT)), args.n_steps)
        print(f"  t={yr}Y: OOD={ood_per_step[t_idx]*100:.1f}%  "
              f"non-finite={nonfinite_per_step[t_idx]*100:.1f}%")


if __name__ == "__main__":
    run(parse_args())
