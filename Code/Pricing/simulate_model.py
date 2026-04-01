# ============================= Import Packages ===============================
import argparse
import math
import os
import sys
import time
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ============================= Environment Setup ===============================
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
THESIS_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# IMPORTANT:
# config.py is the single source of truth for the active variant.
from Code import config
from Code.load_swapdata import my_data
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.utils.ode import (
    d_tau_autograd_nodewise,
    grad_and_trace_cov_hess_G,
    paper_alpha_beta_gamma_trace,
    solve_AB,
)
from Code.utils.rates import par_swap_from_discount

print(f"Repo root: {REPO_ROOT}")
print(f"Active model variant from config.py: {config.VARIANT}")


# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Safer deterministic behavior for research/debugging
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Argument Parsing
# ==========================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Simulate latent Q-paths and decode arbitrage-free curves")

    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension (must be 2)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--use", type=str, default="bbg", help="Data source")
    parser.add_argument("--n_paths", type=int, default=100, help="Number of simulation paths")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=24,
        help="Number of time steps (e.g. 24 for 2 years monthly, 120 for 10 years monthly)",
    )
    parser.add_argument("--dt", type=float, default=1 / 12, help="Time step size")
    parser.add_argument(
        "--idx_choice",
        type=int,
        default=1390,
        help="Index of initial curve (-1 for latest, 1390 = USD Nov 2016, most central USD curve)",
    )
    parser.add_argument(
        "--discretization",
        type=str,
        default="euler",
        choices=["euler", "milstein", "milstein_pc", "second_order_milstein"],
        help=(
            "Latent SDE discretization. "
            "'milstein_pc' is a predictor-corrector Milstein-style scheme. "
            "'second_order_milstein' is kept only as a backward-compatible alias."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--show_plots", action="store_true", help="Show plots interactively")
    parser.add_argument("--no_plots", action="store_true", help="Disable all plotting")
    parser.add_argument(
        "--max_mahal",
        type=float,
        default=4.0,
        help="Mahalanobis threshold for decoder extrapolation warning",
    )
    parser.add_argument(
        "--early_stop_fraction",
        type=float,
        default=0.90,
        help="Stop decoding if this fraction of paths leaves training region",
    )
    parser.add_argument(
        "--tau_fine_step",
        type=float,
        default=1 / 52,
        help="Fine maturity step near tau=0 for decoder diagnostics (years)",
    )
    parser.add_argument(
        "--tau_fine_horizon",
        type=float,
        default=1.0,
        help="Use fine tau spacing on [0, tau_fine_horizon] before switching to annual grid",
    )
    parser.add_argument(
        "--martingale_dates",
        type=str,
        default="5,10,20,30",
        help="Comma-separated fixed maturity dates U (years from today) for discounted-bond martingale diagnostics",
    )
    parser.add_argument(
        "--martingale_tol",
        type=float,
        default=0.02,
        help="Relative tolerance for martingale-diagnostic warning",
    )
    return parser


# ==========================================================
# Checkpoint & Model Loading
# ==========================================================

def resolve_checkpoint_path(repo_root: str, use: str, latent_dim: int, epochs: int) -> str:
    variant = config.VARIANT  # e.g. "baseline" or "stable"

    # New canonical location: Figures/TrainingResults/dim{d}_{variant}/ep{epochs}/checkpoint_dim{d}_ep{epochs}.pt
    new_filename = f"checkpoint_dim{latent_dim}_ep{epochs}.pt"
    new_path = os.path.join(
        THESIS_ROOT,
        "Figures",
        "TrainingResults",
        f"dim{latent_dim}_{variant}",
        f"ep{epochs}",
        new_filename,
    )

    # Legacy locations kept as fallbacks
    old_filename = f"fullmodel_{use}_dim{latent_dim}_ep{epochs}.pt"
    candidates = [
        new_path,
        os.path.join(repo_root, "..", "checkpoints", old_filename),
        os.path.join(repo_root, "checkpoints", old_filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    searched = "\n".join(f"  - {os.path.abspath(p)}" for p in candidates)

    # Show what IS available under TrainingResults to help the user pick a valid --epochs
    available_hint = ""
    tr_root = os.path.join(THESIS_ROOT, "Figures", "TrainingResults")
    if os.path.isdir(tr_root):
        import glob

        found = []
        for pt in glob.glob(os.path.join(tr_root, "*", "*", "checkpoint_*.pt")):
            found.append(f"  - {pt}")
        if found:
            available_hint = "\nAvailable checkpoints in TrainingResults:\n" + "\n".join(sorted(found))

    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}{available_hint}")


def safe_load_state_dict(model, state_dict):
    incompat = model.load_state_dict(state_dict, strict=False)

    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)

    allowed_missing = set()
    allowed_unexpected = set()

    real_missing = [k for k in missing if k not in allowed_missing]
    real_unexpected = [k for k in unexpected if k not in allowed_unexpected]

    if missing:
        print(f"[load_state_dict] Missing keys: {missing}")
    if unexpected:
        print(f"[load_state_dict] Unexpected keys: {unexpected}")

    if real_missing or real_unexpected:
        raise RuntimeError(
            "Non-benign checkpoint/model mismatch detected.\n"
            f"Real missing keys: {real_missing}\n"
            f"Real unexpected keys: {real_unexpected}"
        )


def load_and_setup_model(device, use, latent_dim, epochs):
    if latent_dim != 2:
        raise ValueError("This simulation script currently supports only the 2-factor model (latent_dim=2).")

    checkpoint_path = resolve_checkpoint_path(REPO_ROOT, use, latent_dim, epochs)
    raw = torch.load(checkpoint_path, map_location=device)

    from Code.model.full_model import FullModel

    # Plain state dict (saved via torch.save(model.state_dict(), ...))
    # vs wrapped dict (saved via torch.save({"model_state_dict": ..., ...}, ...))
    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        saved_variant = raw.get("variant", "unknown")
        if saved_variant != "unknown" and saved_variant != config.VARIANT:
            raise ValueError(
                f"Checkpoint variant '{saved_variant}' does not match active "
                f"config.VARIANT '{config.VARIANT}'. Update Code/config.py."
            )
    else:
        # Plain OrderedDict — the Figures/TrainingResults checkpoints
        state_dict = raw

    model = FullModel(latent_dim=latent_dim)
    safe_load_state_dict(model, state_dict)

    model.to(device).double()
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Active config variant: {config.VARIANT}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    return model


# ==========================================================
# Small Helpers
# ==========================================================

@torch.no_grad()
def get_mu(model, z):
    return model.K(z)


@torch.no_grad()
def get_L(model, z):
    H_out = model.H(z)

    # old H API: returns (sigmas, rhos)
    if isinstance(H_out, tuple) and len(H_out) == 2:
        sigmas, rhos = H_out
        return L_from_sigmas_rhos(sigmas, rhos)

    # new H API: returns lower-triangular matrix L directly
    if torch.is_tensor(H_out) and H_out.ndim == 3:
        return H_out

    raise TypeError(
        "Unsupported model.H(z) output. Expected either "
        "(sigmas, rhos) or a tensor L of shape (B,d,d)."
    )


@torch.no_grad()
def get_r(model, z):
    r = model.R(z)
    if r.ndim == 2 and r.shape[-1] == 1:
        r = r.squeeze(-1)
    return r


def tenor_label(tenor_value):
    return f"{int(tenor_value)}Y"


def normalize_discretization_name(name: str) -> str:
    name = name.lower()
    if name == "second_order_milstein":
        warnings.warn(
            "'second_order_milstein' is deprecated and renamed to 'milstein_pc'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "milstein_pc"
    return name


def parse_float_list(text: str):
    if text is None or str(text).strip() == "":
        return []
    vals = []
    for chunk in str(text).split(","):
        s = chunk.strip()
        if s == "":
            continue
        vals.append(float(s))
    return vals


def build_decoder_tau_grid(model, device, dtype, fine_step=1 / 52, fine_horizon=1.0):
    if fine_step <= 0:
        raise ValueError("tau_fine_step must be positive")
    if fine_horizon < 0:
        raise ValueError("tau_fine_horizon must be non-negative")

    tau_max = float(model.tau_max)
    fine_horizon = min(float(fine_horizon), tau_max)

    fine_tau = torch.arange(
        0.0,
        fine_horizon + 0.5 * fine_step,
        fine_step,
        device=device,
        dtype=dtype,
    )
    annual_tau = torch.arange(1.0, tau_max + 1.0, 1.0, device=device, dtype=dtype)
    tau_grid = torch.unique(torch.cat([fine_tau, annual_tau]), sorted=True)

    if tau_grid[0].item() != 0.0:
        tau_grid = torch.cat([torch.zeros(1, device=device, dtype=dtype), tau_grid])
        tau_grid = torch.unique(tau_grid, sorted=True)

    return tau_grid


def get_grid_indices_for_values(grid: torch.Tensor, values: torch.Tensor, tol: float = 1e-10):
    idx_list = []
    for v in values:
        diffs = torch.abs(grid - v)
        idx = torch.argmin(diffs)
        if diffs[idx].item() > tol:
            raise RuntimeError(
                f"Requested tau={float(v):.12f} not found on decoder grid within tolerance {tol:.1e}."
            )
        idx_list.append(int(idx.item()))
    return idx_list


def load_data_and_initial_curve(use, idx_choice, device):
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=use)
    X_tensor = X_tensor.double()

    if idx_choice < 0:
        idx_choice = X_tensor.shape[0] + idx_choice

    if idx_choice < 0 or idx_choice >= X_tensor.shape[0]:
        raise IndexError(f"idx_choice={idx_choice} out of bounds for X_tensor of length {X_tensor.shape[0]}")

    S0 = X_tensor[idx_choice : idx_choice + 1].to(device)
    meta_row = meta.iloc[idx_choice] if hasattr(meta, "iloc") else None

    return {
        "meta": meta,
        "X_tensor": X_tensor,
        "meta_full": meta_full,
        "X_tensor_full": X_tensor_full,
        "tenors": tenors,
        "df_wide": df_wide,
        "df_wide_all": df_wide_all,
        "SCALE_IS_PERCENT": SCALE_IS_PERCENT,
        "S0": S0,
        "meta_row": meta_row,
    }


# ==========================================================
# Latent Statistics
# ==========================================================

def compute_latent_statistics(model, X_tensor, device, latent_dim):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Computing training latent region statistics")
    print("=" * 60)

    z_train_list = []
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], 256):
            batch = X_tensor[i : min(i + 256, X_tensor.shape[0])].to(device)
            z_batch = model.encoder(batch)
            z_train_list.append(z_batch)

    z_train = torch.cat(z_train_list, dim=0)
    z_train_mean = z_train.mean(dim=0).detach()
    z_train_cov = torch.cov(z_train.t()).detach()
    z_train_std = z_train.std(dim=0).detach()

    print(f"Training latent cloud mean: {z_train_mean.cpu().numpy()}")
    print(f"Training latent cloud std:  {z_train_std.cpu().numpy()}")
    print("Training latent cloud range:")
    for d in range(latent_dim):
        print(f"  z[{d}]: [{z_train[:, d].min().item():.6f}, {z_train[:, d].max().item():.6f}]")
    print("=" * 60 + "\n")

    return z_train_mean, z_train_cov, z_train_std


# ==========================================================
# SDE Discretization Helpers
# ==========================================================

def _finite_diff_diffusion_jacobian(model, z, eps=1e-4):
    """
    Finite-difference Jacobian of B(z)=L(z): dB_ij / dz_k
    Returns:
        B0:    (n, d, d)
        jac_B: (n, d, d, d)
    """
    B0 = get_L(model, z)
    n, d, m = B0.shape
    jac_B = torch.empty((n, d, m, d), device=z.device, dtype=z.dtype)

    for k in range(d):
        perturb = torch.zeros_like(z)
        step = eps * torch.maximum(torch.ones_like(z[:, k]), z[:, k].abs())
        perturb[:, k] = step

        B_plus = get_L(model, z + perturb)
        B_minus = get_L(model, z - perturb)

        denom = (2.0 * step).view(-1, 1, 1)
        jac_B[:, :, :, k] = (B_plus - B_minus) / denom

    return B0, jac_B


def _milstein_correction(B, jac_B, dW, dt):
    """
    Commutative-noise Milstein correction:
        0.5 * sum_j (B_.j · grad B_ij) * (dW_j^2 - dt)
    """
    directional_deriv = torch.einsum("nkj,nijk->nij", B, jac_B)
    return 0.5 * torch.sum(directional_deriv * ((dW**2 - dt).unsqueeze(1)), dim=2)


def _stable_drift_step(model, z: torch.Tensor, shock: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Implicit Euler step for linear stable drift when available:
        (I - dt M) z_{t+1} = z_t + dt N + shock
    Fallback: explicit Euler drift + shock
    """
    if hasattr(model.K, "stable_matrix"):
        M = model.K.stable_matrix()  # (d,d)
        N = getattr(model.K, "N", None)

        d = z.shape[1]
        I = torch.eye(d, device=z.device, dtype=z.dtype)
        A = I - dt * M

        rhs = z + shock
        if N is not None:
            rhs = rhs + dt * N.unsqueeze(0)

        A_batch = A.unsqueeze(0).expand(rhs.shape[0], -1, -1)
        return torch.linalg.solve(A_batch, rhs.unsqueeze(-1)).squeeze(-1)

    return z + model.K(z) * dt + shock


# ==========================================================
# Latent Path Simulation
# ==========================================================

def simulate_latent_paths(
    model,
    z0,
    n_paths,
    n_steps,
    dt,
    device,
    discretization="euler",
):
    if z0.dim() != 2 or z0.shape[0] != 1:
        raise ValueError(f"Expected z0 shape (1,d), got {tuple(z0.shape)}")

    discretization = normalize_discretization_name(discretization)
    valid_discretizations = {"euler", "milstein", "milstein_pc"}
    if discretization not in valid_discretizations:
        raise ValueError(f"Unknown discretization='{discretization}'. Choose from {sorted(valid_discretizations)}")

    d = z0.shape[1]
    sqrt_dt = math.sqrt(dt)

    if discretization in {"milstein", "milstein_pc"} and d > 1:
        warnings.warn(
            "Milstein-style updates use a commutative-noise approximation and ignore Lévy-area terms "
            "for multidimensional latent diffusion.",
            RuntimeWarning,
            stacklevel=2,
        )

    z = z0.repeat(n_paths, 1).to(device)

    z_paths = torch.empty((n_paths, n_steps + 1, d), device=device, dtype=z.dtype)
    r_paths = torch.empty((n_paths, n_steps + 1), device=device, dtype=z.dtype)
    mu_paths = torch.empty((n_paths, n_steps + 1, d), device=device, dtype=z.dtype)
    L_paths = torch.empty((n_paths, n_steps + 1, d, d), device=device, dtype=z.dtype)

    z_paths[:, 0, :] = z
    r_paths[:, 0] = get_r(model, z)
    mu_paths[:, 0, :] = get_mu(model, z)
    L_paths[:, 0, :, :] = get_L(model, z)

    for t in range(n_steps):
        if discretization == "euler":
            B = get_L(model, z)
            dW = torch.randn(n_paths, B.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
            shock = torch.bmm(B, dW.unsqueeze(-1)).squeeze(-1)
            z = _stable_drift_step(model, z, shock, dt)

        elif discretization == "milstein":
            B, jac_B = _finite_diff_diffusion_jacobian(model, z)
            dW = torch.randn(n_paths, B.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
            shock = torch.bmm(B, dW.unsqueeze(-1)).squeeze(-1)
            corr = _milstein_correction(B, jac_B, dW, dt)
            z = _stable_drift_step(model, z, shock + corr, dt)

        else:  # milstein_pc
            B0, jac_B0 = _finite_diff_diffusion_jacobian(model, z)
            dW = torch.randn(n_paths, B0.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
            shock0 = torch.bmm(B0, dW.unsqueeze(-1)).squeeze(-1)
            corr0 = _milstein_correction(B0, jac_B0, dW, dt)

            z_pred = _stable_drift_step(model, z, shock0 + corr0, dt)

            B1, jac_B1 = _finite_diff_diffusion_jacobian(model, z_pred)
            shock1 = torch.bmm(B1, dW.unsqueeze(-1)).squeeze(-1)
            corr1 = _milstein_correction(B1, jac_B1, dW, dt)

            avg_shock = 0.5 * (shock0 + shock1)
            avg_corr = 0.5 * (corr0 + corr1)
            z = _stable_drift_step(model, z, avg_shock + avg_corr, dt)

        if not torch.isfinite(z).all():
            raise RuntimeError(f"Non-finite latent state encountered at step {t + 1}")

        z_paths[:, t + 1, :] = z
        r_paths[:, t + 1] = get_r(model, z)
        mu_paths[:, t + 1, :] = get_mu(model, z)
        L_paths[:, t + 1, :, :] = get_L(model, z)

    return z_paths, r_paths, mu_paths, L_paths


# ==========================================================
# Discounting Helpers
# ==========================================================

def compute_discount_paths(r_paths: torch.Tensor, dt: float, method: str = "trapezoid") -> torch.Tensor:
    if dt <= 0:
        raise ValueError("dt must be positive")
    if r_paths.ndim != 2:
        raise ValueError(f"Expected r_paths to have shape (n_paths, n_steps+1), got {tuple(r_paths.shape)}")

    n_paths, n_times = r_paths.shape
    if n_times < 2:
        return torch.ones_like(r_paths)

    if method == "left":
        increments = r_paths[:, :-1] * dt
    elif method == "trapezoid":
        increments = 0.5 * (r_paths[:, :-1] + r_paths[:, 1:]) * dt
    else:
        raise ValueError("method must be 'left' or 'trapezoid'")

    int_r = torch.cumsum(increments, dim=1)
    disc = torch.ones((n_paths, n_times), device=r_paths.device, dtype=r_paths.dtype)
    disc[:, 1:] = torch.exp(-int_r)
    return disc


# ==========================================================
# Decoder / Arbitrage-Free Reconstruction
# ==========================================================

def decode_from_latent_script(model, z, tau, G_floor=1e-5, check_short_rate=True):
    """
    Decode latent states to discount-factor curves on an arbitrary tau grid.

    Args:
        model: trained FullModel
        z:     (B,d) or (d,)
        tau:   strictly increasing 1D tensor including 0 if decoder invariants are to be checked

    Returns:
        P_full: discount factors on tau-grid, shape (B, len(tau))
        A_vals, B_vals, G_vals, mu, sigma, r_tilde, diagnostics
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)

    device = z.device
    dtype = z.dtype
    tau = tau.to(device=device, dtype=dtype)

    if tau.ndim != 1 or tau.numel() < 2:
        raise RuntimeError("tau grid must be 1D and contain at least two points")
    if not torch.all(tau[1:] > tau[:-1]):
        raise RuntimeError("tau grid must be strictly increasing")
    if abs(float(tau[0].item())) > 1e-12:
        raise RuntimeError("tau grid must start at 0 to enforce decoder boundary conditions")

    G_vals = model.G(z, tau)
    if G_vals.dim() == 1:
        G_vals = G_vals.unsqueeze(0)

    if not torch.isfinite(G_vals).all():
        raise RuntimeError("Non-finite G_vals encountered")

    # Explicitly protect the hard-constraint anchor near tau=0
    G0 = G_vals[:, 0]
    min_abs_G0 = G0.abs().min().item()
    if min_abs_G0 < G_floor:
        raise RuntimeError(
            f"G(z,0) too close to zero: min |G(z,0)| = {min_abs_G0:.3e}. "
            "Decoder ODE becomes ill-conditioned."
        )

    mu = model.K(z)

    sigma = get_L(model, z)

    r_tilde = model.R(z)
    if r_tilde.ndim == 2 and r_tilde.shape[-1] == 1:
        r_tilde = r_tilde.squeeze(-1)

    def G_single(z_single):
        return model.G(z_single.unsqueeze(0), tau).squeeze(0)

    dG_dtau = d_tau_autograd_nodewise(model.G, z, tau)
    grad_z_G, trace_cov_hess = grad_and_trace_cov_hess_G(G_single, z, sigma)

    alpha, beta, gamma = paper_alpha_beta_gamma_trace(
        G=G_vals,
        dG_dtau=dG_dtau,
        grad_z_G=grad_z_G,
        trace_cov_hess=trace_cov_hess,
        mu=mu,
        sigma=sigma,
        r_tilde=r_tilde,
    )

    if not torch.isfinite(alpha).all():
        raise RuntimeError("Non-finite alpha encountered")
    if not torch.isfinite(beta).all():
        raise RuntimeError("Non-finite beta encountered")
    if not torch.isfinite(gamma).all():
        raise RuntimeError("Non-finite gamma encountered")

    A_vals, B_vals = solve_AB(tau, alpha, beta, gamma)

    if not torch.isfinite(A_vals).all():
        raise RuntimeError("Non-finite A_vals encountered")
    if not torch.isfinite(B_vals).all():
        raise RuntimeError("Non-finite B_vals encountered")

    # Hard decoder invariants at tau=0
    if not torch.allclose(A_vals[:, 0], torch.zeros_like(A_vals[:, 0]), atol=1e-8, rtol=0.0):
        raise RuntimeError("Decoder invariant violated: A(z,0) != 0")
    if not torch.allclose(B_vals[:, 0], torch.zeros_like(B_vals[:, 0]), atol=1e-8, rtol=0.0):
        raise RuntimeError("Decoder invariant violated: B(z,0) != 0")

    expo = A_vals - B_vals * G_vals
    if not torch.isfinite(expo).all():
        raise RuntimeError("Non-finite exponent encountered in bond pricing")

    # Clamp only at the exponentiation stage for numerical overflow protection
    expo = torch.clamp(expo, min=-80.0, max=20.0)
    P_full = torch.exp(expo)

    if not torch.isfinite(P_full).all():
        raise RuntimeError("Non-finite discount factors encountered")
    if (P_full <= 0).any():
        raise RuntimeError("Non-positive discount factors encountered")

    if not torch.allclose(P_full[:, 0], torch.ones_like(P_full[:, 0]), atol=1e-8, rtol=0.0):
        raise RuntimeError("Decoder invariant violated: P(z,0) != 1")

    short_rate_tau_used = None
    max_short_rate_err = float("nan")
    if check_short_rate and tau.numel() >= 2:
        tau1 = tau[1] - tau[0]
        short_rate_tau_used = float(tau1.item())
        f0_approx = -(torch.log(P_full[:, 1]) - torch.log(P_full[:, 0])) / tau1
        short_rate_err = (f0_approx - r_tilde).abs()
        max_short_rate_err = short_rate_err.max().item()
    else:
        short_rate_err = torch.zeros_like(r_tilde)

    diagnostics = {
        "G_range": (G_vals.min().item(), G_vals.max().item()),
        "P_range": (P_full[:, 1:].min().item(), P_full[:, 1:].max().item()),
        "min_abs_G0": min_abs_G0,
        "short_rate_tau_used": short_rate_tau_used,
        "max_short_rate_err": max_short_rate_err,
    }

    return P_full, A_vals, B_vals, G_vals, mu, sigma, r_tilde, diagnostics


# ==========================================================
# Diagnostics on Simulated Paths
# ==========================================================

def analyze_paths(z_paths, r_paths, mu_paths, L_paths, latent_dim):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Analyzing mu and L")
    print("=" * 60)

    mu_np = mu_paths.detach().cpu().numpy()
    L_np = L_paths.detach().cpu().numpy()
    z_np = z_paths.detach().cpu().numpy()

    print("\n--- MU (Drift) Statistics ---")
    for d in range(latent_dim):
        mu_d = mu_np[:, :, d]
        print(f"mu[{d}]: mean={mu_d.mean():.6f}, std={mu_d.std():.6f}, min={mu_d.min():.6f}, max={mu_d.max():.6f}")

    print("\n--- L (Diffusion) Statistics ---")
    for i in range(latent_dim):
        for j in range(latent_dim):
            L_ij = L_np[:, :, i, j]
            print(
                f"L[{i},{j}]: mean={L_ij.mean():.6f}, std={L_ij.std():.6f}, min={L_ij.min():.6f}, max={L_ij.max():.6f}"
            )

    print("\n--- MU Variance Analysis ---")
    mu_var_time = mu_np.var(axis=0)
    mu_var_path = mu_np.var(axis=1)
    print(f"Mean variance of mu across paths at each time step: {mu_var_time.mean():.6e}")
    print(f"Mean variance of mu across time for each path: {mu_var_path.mean():.6e}")

    print("\n--- L Frobenius Norm Analysis ---")
    L_norms = np.linalg.norm(L_np, axis=(2, 3))
    print(f"L Frobenius norm: mean={L_norms.mean():.6f}, std={L_norms.std():.6f}, min={L_norms.min():.6f}, max={L_norms.max():.6f}")

    print("\n--- Sample mu values at t=0 (first 3 paths) ---")
    for p in range(min(3, z_np.shape[0])):
        print(f"Path {p}: mu = {mu_np[p, 0, :]}")

    print("\n--- Sample mu values at final time (first 3 paths) ---")
    for p in range(min(3, z_np.shape[0])):
        print(f"Path {p}: mu = {mu_np[p, -1, :]}")

    print("\n--- Sample covariance eigenvalues Sigma=L@L.T at t=0 (first 3 paths) ---")
    for p in range(min(3, z_np.shape[0])):
        L_matrix = L_np[p, 0, :, :]
        Sigma = L_matrix @ L_matrix.T
        eigvals = np.linalg.eigvalsh(Sigma)
        print(f"Path {p}: eigenvalues = {eigvals}\nSigma =\n{Sigma}")

    mu_range = mu_np.max() - mu_np.min()
    print("\n--- MU Range Check ---")
    print(f"Overall mu range (max-min): {mu_range:.6e}")
    print(f"Is mu nearly constant? {mu_range < 1e-4}")

    print("\n--- Z-mu Correlation Analysis ---")
    for d in range(latent_dim):
        z_d_flat = z_np[:, :, d].flatten()
        mu_d_flat = mu_np[:, :, d].flatten()
        corr = np.corrcoef(z_d_flat, mu_d_flat)[0, 1]
        print(f"Correlation between z[{d}] and mu[{d}]: {corr:.6f}")

    print("=" * 60 + "\n")


# ==========================================================
# Decode, Save, and Track Latent-Region Diagnostics
# ==========================================================

def decode_and_save_results(
    model,
    z_paths,
    r_paths,
    z_train_mean,
    z_train_cov,
    decoder_tau_grid,
    annual_indices,
    device,
    n_steps,
    n_paths,
    dt,
    tenors,
    use,
    latent_dim,
    epochs,
    max_mahal,
    early_stop_fraction,
):
    eps_reg = 1e-8
    I_reg = torch.eye(z_train_cov.shape[0], device=device, dtype=z_train_cov.dtype)
    z_cov_inv = torch.linalg.inv(z_train_cov + eps_reg * I_reg)

    tenor_cols = [tenor_label(ten) for ten in tenors]
    times = np.arange(n_steps + 1) * dt

    swap_df_list = []
    latent_df_list = []
    mahal_df_list = []
    decoder_diag_df_list = []

    early_stop_time = None

    print("Decoding simulated curves...")
    t0 = time.time()

    for t in range(n_steps + 1):
        z_t = z_paths[:, t, :]

        z_centered = z_t - z_train_mean
        quad = torch.sum((z_centered @ z_cov_inv) * z_centered, dim=1)
        mahal_dist = torch.sqrt(torch.clamp(quad, min=0.0))

        out_of_region = (mahal_dist > max_mahal).sum().item()
        out_of_region_frac = out_of_region / n_paths

        if out_of_region_frac >= 0.10:
            warnings.warn(
                f"At time t={times[t]:.3f}: {out_of_region}/{n_paths} paths ({out_of_region_frac:.1%}) "
                f"exceed {max_mahal:.2f} Mahalanobis distance from training region",
                RuntimeWarning,
            )

        if out_of_region_frac >= early_stop_fraction:
            warnings.warn(
                f"EARLY STOPPING: {out_of_region_frac:.1%} of paths exceed Mahalanobis threshold "
                f"at time t={times[t]:.3f}. Decoder extrapolation is unreliable.",
                RuntimeWarning,
            )
            early_stop_time = t
            break

        P_full, _, _, _, _, _, _, dec_diag = decode_from_latent_script(
            model,
            z_t,
            decoder_tau_grid,
            check_short_rate=True,
        )
        P_annual = P_full[:, annual_indices]
        S_sim = par_swap_from_discount(P_annual, tenors)

        decoder_diag_df_list.append(
            {
                "time": times[t],
                "max_mahal_dist": float(mahal_dist.max().item()),
                "mean_mahal_dist": float(mahal_dist.mean().item()),
                "frac_mahal_gt_threshold": float(out_of_region_frac),
                "decoder_min_G_abs0": float(dec_diag["min_abs_G0"]),
                "decoder_max_short_rate_err": float(dec_diag["max_short_rate_err"]),
                "decoder_short_rate_tau_used": dec_diag["short_rate_tau_used"],
                "decoder_P_min": float(dec_diag["P_range"][0]),
                "decoder_P_max": float(dec_diag["P_range"][1]),
            }
        )

        for p in range(n_paths):
            swap_row = {"time": times[t], "path_id": p}
            for i, col in enumerate(tenor_cols):
                swap_row[f"swap_{col}"] = S_sim[p, i].item()
            swap_df_list.append(swap_row)

            latent_row = {"time": times[t], "path_id": p, "r": r_paths[p, t].detach().item()}
            for d in range(latent_dim):
                latent_row[f"z{d}"] = z_paths[p, t, d].detach().item()
            latent_df_list.append(latent_row)

            mahal_df_list.append(
                {
                    "time": times[t],
                    "path_id": p,
                    "mahal_dist": mahal_dist[p].detach().item(),
                }
            )

        if t == 0 or t == n_steps or t % max(1, n_steps // 10) == 0:
            tau_used = dec_diag["short_rate_tau_used"]
            tau_str = f"{tau_used:.6f}" if tau_used is not None else "NA"
            print(
                f"  t={times[t]:.3f} | "
                f"max Mahalanobis={mahal_dist.max().item():.3f} | "
                f"short-rate tau={tau_str} | "
                f"max short-rate err={dec_diag['max_short_rate_err']:.3e}"
            )

    elapsed = time.time() - t0
    print(f"Decoding finished in {elapsed:.2f}s")

    out_dir = os.path.join(THESIS_ROOT, "Figures", "Pricing", "simulations")
    os.makedirs(out_dir, exist_ok=True)

    swap_df = pd.DataFrame(swap_df_list)
    latent_df = pd.DataFrame(latent_df_list)
    mahal_df = pd.DataFrame(mahal_df_list)
    decoder_diag_df = pd.DataFrame(decoder_diag_df_list)

    suffix = f"{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}"
    swap_csv_path = os.path.join(out_dir, f"simulated_swap_curves_{suffix}.csv")
    latent_csv_path = os.path.join(out_dir, f"simulated_latent_{suffix}.csv")
    mahal_csv_path = os.path.join(out_dir, f"simulated_mahal_{suffix}.csv")
    decoder_diag_csv_path = os.path.join(out_dir, f"decoder_diagnostics_{suffix}.csv")

    swap_df.to_csv(swap_csv_path, index=False)
    latent_df.to_csv(latent_csv_path, index=False)
    mahal_df.to_csv(mahal_csv_path, index=False)
    decoder_diag_df.to_csv(decoder_diag_csv_path, index=False)

    print(f"Saved simulated swap curves to {swap_csv_path}")
    print(f"Saved simulated latent paths to {latent_csv_path}")
    print(f"Saved Mahalanobis diagnostics to {mahal_csv_path}")
    print(f"Saved decoder diagnostics to {decoder_diag_csv_path}")

    if early_stop_time is not None:
        print(
            f"\n[WARNING] Simulation stopped early at t={times[early_stop_time]:.3f} "
            f"due to excessive latent-region violation."
        )
        print(f"Data saved contains {len(swap_df_list) // n_paths} time steps instead of {n_steps + 1}.")

    print("Simulation and saving completed successfully.")

    return swap_df, latent_df, mahal_df, decoder_diag_df, out_dir, times, early_stop_time


# ==========================================================
# Martingale Diagnostics for Discounted Bond Prices
# ==========================================================

def martingale_diagnostics(
    model,
    z_paths,
    discount_paths,
    times,
    maturity_dates,
    out_dir,
    use,
    latent_dim,
    epochs,
    n_paths,
    n_steps,
    martingale_tol=0.02,
):
    if len(maturity_dates) == 0:
        print("No martingale dates requested; skipping discounted-bond martingale diagnostics.")
        return pd.DataFrame()

    maturity_dates = sorted(float(u) for u in maturity_dates)
    max_time = float(times[-1])
    if all(u <= 0 for u in maturity_dates):
        print("All martingale dates are non-positive; skipping diagnostics.")
        return pd.DataFrame()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Discounted-bond martingale check")
    print("=" * 60)

    device = z_paths.device
    dtype = z_paths.dtype
    out_rows = []
    rel_err_tracker = {u: [] for u in maturity_dates}

    for t_idx, t_now in enumerate(times):
        valid_U = [u for u in maturity_dates if u > float(t_now) + 1e-12]
        if len(valid_U) == 0:
            continue

        tau_remaining = torch.tensor([u - float(t_now) for u in valid_U], device=device, dtype=dtype)
        tau_grid = torch.cat([torch.zeros(1, device=device, dtype=dtype), tau_remaining])
        tau_grid = torch.unique(tau_grid, sorted=True)

        P_full, _, _, _, _, _, _, _ = decode_from_latent_script(
            model,
            z_paths[:, t_idx, :],
            tau_grid,
            check_short_rate=False,
        )

        for u in valid_U:
            tau_u = torch.tensor(float(u - float(t_now)), device=device, dtype=dtype)
            idx = torch.argmin(torch.abs(tau_grid - tau_u))
            if torch.abs(tau_grid[idx] - tau_u).item() > 1e-10:
                raise RuntimeError(f"Could not locate tau={float(tau_u):.12f} on diagnostic grid")

            discounted_bond = discount_paths[:, t_idx] * P_full[:, idx]
            mean_val = discounted_bond.mean().item()
            std_val = discounted_bond.std(unbiased=False).item()
            sem_val = std_val / math.sqrt(discounted_bond.shape[0])

            # At t=0, D_0 = 1 and all paths are identical, so this is the reference value
            initial_tau_grid = torch.tensor([0.0, u], device=device, dtype=dtype)
            if t_idx == 0:
                P0_full, _, _, _, _, _, _, _ = decode_from_latent_script(
                    model,
                    z_paths[:1, 0, :],
                    initial_tau_grid,
                    check_short_rate=False,
                )
                initial_val = P0_full[0, 1].item()
            else:
                # Read back initial value consistently from the same model+z0
                P0_full, _, _, _, _, _, _, _ = decode_from_latent_script(
                    model,
                    z_paths[:1, 0, :],
                    initial_tau_grid,
                    check_short_rate=False,
                )
                initial_val = P0_full[0, 1].item()

            rel_err = abs(mean_val - initial_val) / max(abs(initial_val), 1e-12)
            rel_err_tracker[u].append(rel_err)

            out_rows.append(
                {
                    "time": float(t_now),
                    "U": float(u),
                    "tau_remaining": float(u - float(t_now)),
                    "disc_bond_mean": float(mean_val),
                    "disc_bond_std": float(std_val),
                    "disc_bond_sem": float(sem_val),
                    "initial_disc_bond_value": float(initial_val),
                    "relative_mean_error": float(rel_err),
                }
            )

    mart_df = pd.DataFrame(out_rows)
    suffix = f"{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}"
    mart_csv_path = os.path.join(out_dir, f"martingale_diagnostics_{suffix}.csv")
    mart_df.to_csv(mart_csv_path, index=False)
    print(f"Saved martingale diagnostics to {mart_csv_path}")

    for u in maturity_dates:
        errs = rel_err_tracker.get(u, [])
        if len(errs) == 0:
            print(f"  U={u:.2f}: no valid times in simulation window")
            continue
        max_err = max(errs)
        print(f"  U={u:.2f}: max relative mean error = {max_err:.3%}")
        if max_err > martingale_tol:
            warnings.warn(
                f"Discounted-bond martingale diagnostic exceeded tolerance for U={u:.2f}: "
                f"max relative mean error {max_err:.3%} > tol {martingale_tol:.3%}",
                RuntimeWarning,
            )

    print("=" * 60 + "\n")
    return mart_df


# ==========================================================
# Plotting
# ==========================================================

def generate_plots(
    model,
    z_paths,
    r_paths,
    mu_paths,
    L_paths,
    discount_paths,
    swap_df,
    mart_df,
    tenors,
    out_dir,
    times,
    n_paths,
    n_steps,
    use,
    latent_dim,
    epochs,
    show_plots=False,
):
    print("Generating plots...")

    n_plot = min(5, n_paths)

    # ----------------------------------------------------------
    # Drift plot
    # ----------------------------------------------------------
    fig_mu, axes_mu = plt.subplots(latent_dim, 1, figsize=(10, 6), sharex=True)
    axes_mu = np.atleast_1d(axes_mu)

    for d in range(latent_dim):
        ax = axes_mu[d]
        for p in range(n_plot):
            ax.plot(times, mu_paths[p, :, d].detach().cpu().numpy(), alpha=0.7)
        ax.set_ylabel(f"mu{d}")
        ax.grid(True, alpha=0.3)

    axes_mu[-1].set_xlabel("Time (years)")
    fig_mu.suptitle("Drift along simulated paths")
    plt.tight_layout()

    mu_plot_path = os.path.join(
        out_dir,
        f"simulated_mu_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
    )
    fig_mu.savefig(mu_plot_path, dpi=300)
    print(f"Saved mu plot to {mu_plot_path}")
    if show_plots:
        plt.show()
    plt.close(fig_mu)

    # ----------------------------------------------------------
    # Diffusion factor L(z) element plot
    # ----------------------------------------------------------
    fig_L, axes_L = plt.subplots(latent_dim, latent_dim, figsize=(12, 8), sharex=True)
    if latent_dim == 1:
        axes_L = np.atleast_2d(axes_L)

    for i in range(latent_dim):
        for j in range(latent_dim):
            ax = axes_L[i, j]
            for p in range(n_plot):
                ax.plot(times, L_paths[p, :, i, j].detach().cpu().numpy(), alpha=0.7)
            ax.set_ylabel(f"L[{i},{j}]")
            ax.grid(True, alpha=0.3)

    for ax in axes_L[-1, :]:
        ax.set_xlabel("Time (years)")

    fig_L.suptitle("Diffusion-factor elements along simulated paths")
    plt.tight_layout()

    L_plot_path = os.path.join(
        out_dir,
        f"simulated_L_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
    )
    fig_L.savefig(L_plot_path, dpi=300)
    print(f"Saved L plot to {L_plot_path}")
    if show_plots:
        plt.show()
    plt.close(fig_L)

    # ----------------------------------------------------------
    # Diagonal/off-diagonal decomposition of L(z)
    # ----------------------------------------------------------
    n_off = latent_dim * (latent_dim - 1) // 2
    n_rows = latent_dim + (n_off if n_off > 0 else 0)

    fig_ld, axes_ld = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), sharex=True)
    axes_ld = np.atleast_1d(axes_ld)

    # diagonal entries
    for i in range(latent_dim):
        ax = axes_ld[i]
        for p in range(n_plot):
            ax.plot(times, L_paths[p, :, i, i].detach().cpu().numpy(), alpha=0.7)
        ax.set_ylabel(f"L[{i},{i}]")
        ax.grid(True, alpha=0.3)

    # strict lower-triangular entries
    row_idx = latent_dim
    for i in range(1, latent_dim):
        for j in range(i):
            ax = axes_ld[row_idx]
            for p in range(n_plot):
                ax.plot(times, L_paths[p, :, i, j].detach().cpu().numpy(), alpha=0.7)
            ax.set_ylabel(f"L[{i},{j}]")
            ax.grid(True, alpha=0.3)
            row_idx += 1

    axes_ld[-1].set_xlabel("Time (years)")
    fig_ld.suptitle("Diagonal and lower-triangular diffusion-factor entries")
    plt.tight_layout()

    ld_plot_path = os.path.join(
        out_dir,
        f"simulated_L_diag_offdiag_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
    )
    fig_ld.savefig(ld_plot_path, dpi=300)
    print(f"Saved L diag/offdiag plot to {ld_plot_path}")
    if show_plots:
        plt.show()
    plt.close(fig_ld)

    # ----------------------------------------------------------
    # Instantaneous covariance Sigma = L L^T
    # ----------------------------------------------------------
    sigma_paths = torch.matmul(L_paths, L_paths.transpose(-1, -2))

    fig_S, axes_S = plt.subplots(latent_dim, latent_dim, figsize=(12, 8), sharex=True)
    if latent_dim == 1:
        axes_S = np.atleast_2d(axes_S)

    for i in range(latent_dim):
        for j in range(latent_dim):
            ax = axes_S[i, j]
            for p in range(n_plot):
                ax.plot(times, sigma_paths[p, :, i, j].detach().cpu().numpy(), alpha=0.7)
            ax.set_ylabel(f"Sigma[{i},{j}]")
            ax.grid(True, alpha=0.3)

    for ax in axes_S[-1, :]:
        ax.set_xlabel("Time (years)")

    fig_S.suptitle("Instantaneous covariance elements along simulated paths")
    plt.tight_layout()

    S_plot_path = os.path.join(
        out_dir,
        f"simulated_Sigma_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
    )
    fig_S.savefig(S_plot_path, dpi=300)
    print(f"Saved Sigma plot to {S_plot_path}")
    if show_plots:
        plt.show()
    plt.close(fig_S)

    # ----------------------------------------------------------
    # Latent paths + short rate + discount factor
    # ----------------------------------------------------------
    fig_lat, axes = plt.subplots(latent_dim + 2, 1, figsize=(10, 8), sharex=True)
    axes = np.atleast_1d(axes)

    for d in range(latent_dim):
        ax = axes[d]
        for p in range(min(10, n_paths)):
            ax.plot(times, z_paths[p, :, d].detach().cpu().numpy(), alpha=0.7)
        ax.set_ylabel(f"z{d}")
        ax.grid(True, alpha=0.3)

    ax_r = axes[latent_dim]
    for p in range(min(10, n_paths)):
        ax_r.plot(times, r_paths[p, :].detach().cpu().numpy(), alpha=0.7)
    ax_r.set_ylabel("r")
    ax_r.grid(True, alpha=0.3)

    ax_d = axes[latent_dim + 1]
    for p in range(min(10, n_paths)):
        ax_d.plot(times, discount_paths[p, :].detach().cpu().numpy(), alpha=0.7)
    ax_d.set_ylabel("D_t")
    ax_d.set_xlabel("Time (years)")
    ax_d.grid(True, alpha=0.3)

    fig_lat.suptitle("Simulated latent paths, short rate, and discount factor")
    plt.tight_layout()

    latent_plot_path = os.path.join(
        out_dir,
        f"simulated_latent_paths_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
    )
    fig_lat.savefig(latent_plot_path, dpi=300)
    print(f"Saved latent paths plot to {latent_plot_path}")
    if show_plots:
        plt.show()
    plt.close(fig_lat)

    # ----------------------------------------------------------
    # Mean swap rates
    # ----------------------------------------------------------
    if not swap_df.empty:
        fig_swap, ax = plt.subplots(figsize=(10, 6))
        tenors_to_plot = [1.0, 5.0, 10.0, 30.0]

        for ten in tenors_to_plot:
            if float(ten) in tenors:
                col = f"swap_{tenor_label(ten)}"
                if col in swap_df.columns:
                    mean_curve = swap_df.groupby("time")[col].mean()
                    ax.plot(mean_curve.index, mean_curve.values, linewidth=2, label=f"{int(ten)}Y")

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Swap Rate")
        ax.set_title("Mean simulated swap rates over time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        swap_plot_path = os.path.join(
            out_dir,
            f"simulated_swap_rates_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
        )
        fig_swap.savefig(swap_plot_path, dpi=300)
        print(f"Saved swap rates plot to {swap_plot_path}")
        if show_plots:
            plt.show()
        plt.close(fig_swap)
    else:
        print("No decoded swap curves available to plot.")

    # ----------------------------------------------------------
    # Martingale diagnostic plot
    # ----------------------------------------------------------
    if mart_df is not None and not mart_df.empty:
        fig_m, ax = plt.subplots(figsize=(10, 6))
        for U in sorted(mart_df["U"].unique()):
            sub = mart_df[mart_df["U"] == U].sort_values("time")
            ax.plot(sub["time"], sub["disc_bond_mean"], linewidth=2, label=f"U={U:g}")
            ax.axhline(sub["initial_disc_bond_value"].iloc[0], linewidth=1, linestyle="--", alpha=0.6)

        ax.set_xlabel("Time (years)")
        ax.set_ylabel(r"Mean of $D_t P(t,U)$")
        ax.set_title("Discounted-bond martingale diagnostic")
        ax.legend()
        ax.grid(True, alpha=0.3)

        mart_plot_path = os.path.join(
            out_dir,
            f"martingale_diagnostics_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
        )
        fig_m.savefig(mart_plot_path, dpi=300)
        print(f"Saved martingale plot to {mart_plot_path}")
        if show_plots:
            plt.show()
        plt.close(fig_m)

        fig_merr, ax = plt.subplots(figsize=(10, 6))
        for U in sorted(mart_df["U"].unique()):
            sub = mart_df[mart_df["U"] == U].sort_values("time")
            ax.plot(sub["time"], sub["relative_mean_error"], linewidth=2, label=f"U={U:g}")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Relative mean error")
        ax.set_title("Discounted-bond martingale relative error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        merr_plot_path = os.path.join(
            out_dir,
            f"martingale_relative_error_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
        )
        fig_merr.savefig(merr_plot_path, dpi=300)
        print(f"Saved martingale relative-error plot to {merr_plot_path}")
        if show_plots:
            plt.show()
        plt.close(fig_merr)

    print("Plotting completed.")

# ==========================================================
# Main
# ==========================================================

def main(argv=None):
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    LATENT_DIM = args.latent_dim
    EPOCHS = args.epochs
    USE = args.use
    N_PATHS = args.n_paths
    N_STEPS = args.n_steps
    DT = args.dt
    IDX_CHOICE = args.idx_choice
    DISCRETIZATION = args.discretization
    SEED = args.seed
    SHOW_PLOTS = bool(args.show_plots and not args.no_plots)

    set_seed(SEED)

    if LATENT_DIM != 2:
        raise ValueError("This script is for the 2-factor model only.")

    if DT <= 0:
        raise ValueError("dt must be positive")
    if N_PATHS <= 0:
        raise ValueError("n_paths must be positive")
    if N_STEPS <= 0:
        raise ValueError("n_steps must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {SEED}")

    # ========== Load data once ==========
    data = load_data_and_initial_curve(USE, IDX_CHOICE, device)
    X_tensor = data["X_tensor"]
    tenors = data["tenors"]
    S0 = data["S0"]
    meta_row = data["meta_row"]
    SCALE_IS_PERCENT = data["SCALE_IS_PERCENT"]

    print(f"SCALE_IS_PERCENT from my_data(): {SCALE_IS_PERCENT}")
    if meta_row is not None:
        print(f"Initial curve metadata row:\n{meta_row}")

    # ========== Load model ==========
    model = load_and_setup_model(device, USE, LATENT_DIM, EPOCHS)

    # ========== Decoder grid ==========
    decoder_tau_grid = build_decoder_tau_grid(
        model,
        device=device,
        dtype=torch.float64,
        fine_step=args.tau_fine_step,
        fine_horizon=args.tau_fine_horizon,
    )
    annual_tau = torch.arange(1.0, float(model.tau_max) + 1.0, 1.0, device=device, dtype=torch.float64)
    annual_indices = get_grid_indices_for_values(decoder_tau_grid, annual_tau)
    print(
        "Decoder tau grid built with "
        f"{decoder_tau_grid.numel()} points; first positive tau = {decoder_tau_grid[1].item():.6f}, "
        f"tau_max = {decoder_tau_grid[-1].item():.6f}"
    )

    # ========== Training latent statistics ==========
    z_train_mean, z_train_cov, z_train_std = compute_latent_statistics(model, X_tensor, device, LATENT_DIM)

    # ========== Initial latent state ==========
    with torch.no_grad():
        z0 = model.encoder(S0)
    print(f"Initial latent state z0: {z0.detach().cpu().numpy().flatten()}")

    # ========== Simulate ==========
    print(f"Simulating {N_PATHS} paths with {N_STEPS} steps (dt={DT}, scheme={DISCRETIZATION})...")
    z_paths, r_paths, mu_paths, L_paths = simulate_latent_paths(
        model=model,
        z0=z0,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        dt=DT,
        device=device,
        discretization=DISCRETIZATION,
    )
    print("Simulation completed.")

    # ========== Discount factors along paths ==========
    discount_paths = compute_discount_paths(r_paths, dt=DT, method="trapezoid")
    print(
        f"Built path discount factors: D_t range = "
        f"[{discount_paths.min().item():.6f}, {discount_paths.max().item():.6f}]"
    )

    # ========== Analyze latent dynamics ==========
    analyze_paths(z_paths, r_paths, mu_paths, L_paths, LATENT_DIM)

    # ========== Decode and save ==========
    swap_df, latent_df, mahal_df, decoder_diag_df, out_dir, times, early_stop_time = decode_and_save_results(
        model=model,
        z_paths=z_paths,
        r_paths=r_paths,
        z_train_mean=z_train_mean,
        z_train_cov=z_train_cov,
        decoder_tau_grid=decoder_tau_grid,
        annual_indices=annual_indices,
        device=device,
        n_steps=N_STEPS,
        n_paths=N_PATHS,
        dt=DT,
        tenors=tenors,
        use=USE,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        max_mahal=args.max_mahal,
        early_stop_fraction=args.early_stop_fraction,
    )

    # ========== Martingale diagnostics ==========
    maturity_dates = parse_float_list(args.martingale_dates)
    mart_df = martingale_diagnostics(
        model=model,
        z_paths=z_paths,
        discount_paths=discount_paths,
        times=times,
        maturity_dates=maturity_dates,
        out_dir=out_dir,
        use=USE,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        martingale_tol=args.martingale_tol,
    )

    # ========== Plot ==========
    if not args.no_plots:
        generate_plots(
            model=model,
            z_paths=z_paths,
            r_paths=r_paths,
            mu_paths=mu_paths,
            L_paths=L_paths,
            discount_paths=discount_paths,
            swap_df=swap_df,
            mart_df=mart_df,
            tenors=tenors,
            out_dir=out_dir,
            times=times,
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            use=USE,
            latent_dim=LATENT_DIM,
            epochs=EPOCHS,
            show_plots=SHOW_PLOTS,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
