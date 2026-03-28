# ============================= Import Packages ===============================
import argparse
import math
import os
import sys
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ============================= Environment Setup & Imports ===============================
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# IMPORTANT:
# config.py is the single source of truth for the active variant.
# Do NOT overwrite config.VARIANT here.
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
# Settings
# ==========================================================

SHOW_PLOTS = True  # Set to False to only save plots

# ==========================================================
# Helper Functions: Argument Parsing
# ==========================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Simulate swap curves from trained FullModel")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--use", type=str, default="bbg", help="Data source")
    parser.add_argument("--n_paths", type=int, default=100, help="Number of simulation paths")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=120,
        help="Number of time steps (e.g., 120 for 10 years monthly)",
    )
    parser.add_argument("--dt", type=float, default=1 / 12, help="Time step size")
    parser.add_argument("--idx_choice", type=int, default=-1, help="Index of initial curve (-1 for latest)")
    parser.add_argument(
        "--discretization",
        type=str,
        default="euler",
        choices=["euler", "milstein", "second_order_milstein"],
        help="Discretization scheme for latent SDE",
    )
    return parser


# ==========================================================
# Helper Functions: Model Loading & Setup
# ==========================================================


def resolve_checkpoint_path(repo_root: str, use: str, latent_dim: int, epochs: int) -> str:
    filename = f"fullmodel_{use}_dim{latent_dim}_ep{epochs}.pt"
    candidates = [
        os.path.join(repo_root, "..", "checkpoints", filename),  # matches training script
        os.path.join(repo_root, "checkpoints", filename),        # fallback
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    searched = "\n".join(f"  - {os.path.abspath(p)}" for p in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}")


@torch.no_grad()
def get_mu(model, z):
    return model.K(z)


@torch.no_grad()
def get_L(model, z):
    sigmas, rhos = model.H(z)
    return L_from_sigmas_rhos(sigmas, rhos)


@torch.no_grad()
def get_r(model, z):
    r = model.R(z)
    if r.ndim == 2 and r.shape[-1] == 1:
        r = r.squeeze(-1)
    return r


def tenor_label(tenor_value):
    """Normalize tenor to consistent label format (e.g., '1Y', '5Y', etc.)."""
    return f"{int(tenor_value)}Y"


def _finite_diff_diffusion_jacobian(model, z, eps=1e-4):
    """Finite-difference Jacobian of diffusion matrix B(z)=L(z): dB_ij/dz_k."""
    B0 = get_L(model, z)
    n, d, m = B0.shape
    jac_B = torch.empty((n, d, m, d), device=z.device, dtype=z.dtype)

    for k in range(d):
        perturb = torch.zeros_like(z)
        perturb[:, k] = eps
        B_plus = get_L(model, z + perturb)
        B_minus = get_L(model, z - perturb)
        jac_B[:, :, :, k] = (B_plus - B_minus) / (2.0 * eps)

    return B0, jac_B


def _milstein_correction(B, jac_B, dW, dt):
    """Commutative-noise Milstein correction: 0.5*sum_j (B_.j · grad B_ij) * (dW_j^2 - dt)."""
    directional_deriv = torch.einsum("nkj,nijk->nij", B, jac_B)
    return 0.5 * torch.sum(directional_deriv * ((dW ** 2 - dt).unsqueeze(1)), dim=2)


def _stable_drift_step(model, z: torch.Tensor, shock: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Drift update that uses an implicit (backward-Euler) step when the model's K
    exposes a stable_matrix(), and falls back to explicit Euler otherwise.

    For the stable linear drift mu(z) = M z + N the exact solution of the
    drift ODE over [t, t+dt] is z(t+dt) = exp(M dt) z(t) + M^{-1}(exp(M dt)-I) N,
    but that requires a matrix exponential. We instead use the implicit Euler
    approximation, which is unconditionally stable for stable M:

        (I - dt * M) z_{t+1} = z_t + dt * N + shock
        z_{t+1} = (I - dt * M)^{-1} (z_t + dt * N + shock)

    This is O(d^3) per step but d is tiny (2–4), so it is negligible.
    """
    if hasattr(model.K, "stable_matrix"):
        M = model.K.stable_matrix()  # (d, d)
        N = model.K.N  # (d,) or None
        d = z.shape[1]
        I = torch.eye(d, device=z.device, dtype=z.dtype)
        A = I - dt * M  # (d, d)
        rhs = z + shock  # (n_paths, d)
        if N is not None:
            rhs = rhs + dt * N.unsqueeze(0)

        A_batch = A.unsqueeze(0).expand(rhs.shape[0], -1, -1)  # (n_paths, d, d)
        return torch.linalg.solve(A_batch, rhs.unsqueeze(-1)).squeeze(-1)

    return z + model.K(z) * dt + shock


def load_initial_curve(use, idx_choice, device):
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=use)
    X_tensor = X_tensor.double()

    if idx_choice < 0:
        idx_choice = X_tensor.shape[0] + idx_choice

    if idx_choice < 0 or idx_choice >= X_tensor.shape[0]:
        raise IndexError(f"idx_choice={idx_choice} out of bounds")

    S0 = X_tensor[idx_choice:idx_choice + 1].to(device)
    meta_row = meta.iloc[idx_choice] if hasattr(meta, "iloc") else None
    return S0, meta_row, X_tensor, meta


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

    d = z0.shape[1]
    sqrt_dt = math.sqrt(dt)
    discretization = discretization.lower()
    valid_discretizations = {"euler", "milstein", "second_order_milstein"}
    if discretization not in valid_discretizations:
        raise ValueError(f"Unknown discretization='{discretization}'. Choose from {sorted(valid_discretizations)}")

    if discretization in {"milstein", "second_order_milstein"} and d > 1:
        warnings.warn(
            "Milstein updates use a commutative-noise approximation and ignore Levy-area terms for "
            "multidimensional latent diffusion.",
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

        else:  # second_order_milstein
            B0, jac_B0 = _finite_diff_diffusion_jacobian(model, z)
            dW = torch.randn(n_paths, B0.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
            shock0 = torch.bmm(B0, dW.unsqueeze(-1)).squeeze(-1)
            corr0 = _milstein_correction(B0, jac_B0, dW, dt)
            z_pred = _stable_drift_step(model, z, shock0 + corr0, dt)

            B1, jac_B1 = _finite_diff_diffusion_jacobian(model, z_pred)
            shock1 = torch.bmm(B1, dW.unsqueeze(-1)).squeeze(-1)
            corr1 = _milstein_correction(B1, jac_B1, dW, dt)
            z = _stable_drift_step(model, z, 0.5 * (shock0 + shock1) + 0.5 * (corr0 + corr1), dt)

        if not torch.isfinite(z).all():
            raise RuntimeError(f"Non-finite z encountered at step {t + 1}")

        z_paths[:, t + 1, :] = z
        r_paths[:, t + 1] = get_r(model, z)
        mu_paths[:, t + 1, :] = get_mu(model, z)
        L_paths[:, t + 1, :, :] = get_L(model, z)

    return z_paths, r_paths, mu_paths, L_paths


def decode_from_latent_script(model, z):
    """
    Decode latent states to discount factor curves with comprehensive numerical stability checks.

    Returns:
        P_mkt: Market discount factors (excluding τ=0)
        A_vals, B_vals, G_vals: ODE solution components
        mu, sigma, r_tilde: Model parameters
        diagnostics: Dict with warnings/info about potential issues
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)

    device = z.device
    dtype = z.dtype
    tau = torch.arange(0, model.tau_max + 1, device=device, dtype=dtype)

    G_vals = model.G(z, tau)
    if G_vals.dim() == 1:
        G_vals = G_vals.unsqueeze(0)

    if not torch.isfinite(G_vals).all():
        raise RuntimeError("Non-finite G_vals encountered")

    G_min_abs = G_vals.abs().min().item()
    if G_min_abs < 1e-6:
        warnings.warn(f"Very small |G| encountered: {G_min_abs:.3e}", RuntimeWarning)

    mu = model.K(z)
    sigmas, rhos = model.H(z)
    r_tilde = model.R(z)
    if r_tilde.ndim == 2 and r_tilde.shape[-1] == 1:
        r_tilde = r_tilde.squeeze(-1)

    sigma = L_from_sigmas_rhos(sigmas, rhos)

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

    expo = A_vals - B_vals * G_vals
    if not torch.isfinite(expo).all():
        raise RuntimeError("Non-finite exponent in bond pricing")

    expo = torch.clamp(expo, min=-80.0, max=20.0)
    P_full = torch.exp(expo)

    if (P_full <= 0).any():
        raise RuntimeError("Non-positive discount factors encountered")
    if not torch.isfinite(P_full).all():
        raise RuntimeError("Non-finite discount factors encountered")

    P_diffs = P_full[:, 1:] - P_full[:, :-1]
    increasing = (P_diffs > 1e-3).sum().item()
    if increasing > 0:
        frac_incr = increasing / P_diffs.numel()
        if frac_incr > 0.1:
            warnings.warn(
                f"Discount curve increasing in {frac_incr:.1%} of maturities (should decrease)",
                RuntimeWarning,
            )

    P_mkt = P_full[:, 1:]
    diagnostics = {
        "G_range": (G_vals.min().item(), G_vals.max().item()),
        "P_range": (P_mkt.min().item(), P_mkt.max().item()),
    }

    return P_mkt, A_vals, B_vals, G_vals, mu, sigma, r_tilde, diagnostics


# ==========================================================
# Stage 1: Load and Setup Model
# ==========================================================

def load_and_setup_model(device, use, latent_dim, epochs):
    """Load checkpoint and model, verify variant consistency."""
    checkpoint_path = resolve_checkpoint_path(REPO_ROOT, use, latent_dim, epochs)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    saved_variant = checkpoint.get("variant", "unknown")
    print(f"Checkpoint variant: {saved_variant}")

    if saved_variant != "unknown" and saved_variant != config.VARIANT:
        raise ValueError(
            f"Checkpoint variant '{saved_variant}' does not match active config.VARIANT '{config.VARIANT}'. "
            f"Update Code/config.py so training and simulation use the same variant, then rerun."
        )

    # Import FullModel only after variant consistency has been checked.
    from Code.model.full_model import FullModel

    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        print(f"Loading model with saved configuration: {model_config}")
        model = FullModel(**model_config)
    else:
        print("[WARNING] Checkpoint missing 'model_config'. Using fallback constructor.")
        model = FullModel(latent_dim=latent_dim)

    # Handle checkpoint compatibility: convert old H_sigma parameters to new format
    state_dict = checkpoint["model_state_dict"]
    if config.VARIANT == "stable":
        # Check if this is an old H_sigma_stable checkpoint (raw_sigma_amps -> raw_logsigma_offset)
        if "H.raw_sigma_amps" in state_dict and "H.raw_logsigma_offset" not in state_dict:
            print("[INFO] Converting old H_sigma_stable checkpoint parameters (raw_sigma_amps -> raw_logsigma_offset)")
            state_dict["H.raw_logsigma_offset"] = state_dict.pop("H.raw_sigma_amps")

    model.load_state_dict(state_dict, strict=False)
    model.to(device).double()
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Active config variant: {config.VARIANT}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    return model


def compute_latent_statistics(model, X_tensor, device, latent_dim):
    """Compute training latent region statistics."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Computing training latent region statistics")
    print("=" * 60)

    z_train_list = []
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], 100):
            batch = X_tensor[i:min(i + 100, X_tensor.shape[0])].to(device)
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
        print(f"  z[{d}]: [{z_train[:, d].min().item():.4f}, {z_train[:, d].max().item():.4f}]")
    print("=" * 60 + "\n")

    return z_train_mean, z_train_cov, z_train_std


# ==========================================================
# Stage 2: Run Simulation
# ==========================================================

def run_simulation(model, z0, n_paths, n_steps, dt, device, latent_dim, discretization):
    """Run latent path simulation."""
    print(f"Simulating {n_paths} paths with {n_steps} steps (dt={dt}, scheme={discretization})...")
    z_paths, r_paths, mu_paths, L_paths = simulate_latent_paths(
        model=model,
        z0=z0,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        device=device,
        discretization=discretization,
    )
    print("Simulation completed.")
    return z_paths, r_paths, mu_paths, L_paths


def analyze_paths(z_paths, r_paths, mu_paths, L_paths, latent_dim):
    """Analyze and print diagnostics for simulated paths."""
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
            print(f"L[{i},{j}]: mean={L_ij.mean():.6f}, std={L_ij.std():.6f}, min={L_ij.min():.6f}, max={L_ij.max():.6f}")

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

    print("\n--- Sample covariance (Sigma=L@L.T) eigenvalues at t=0 (first 3 paths) ---")
    for p in range(min(3, z_np.shape[0])):
        L_matrix = L_np[p, 0, :, :]
        Sigma = L_matrix @ L_matrix.T
        eigvals = np.linalg.eigvalsh(Sigma)
        print(f"Path {p}: Sigma eigenvalues = {eigvals}, Sigma:\n{Sigma}")

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
# Stage 3: Decode and Save Results
# ==========================================================

def decode_and_save_results(model, z_paths, r_paths, z_train_mean, z_train_cov, device, n_steps, n_paths, dt, tenors, use, latent_dim, epochs):
    """Decode latent paths to swap curves and save results."""
    eps_reg = 1e-8
    I_reg = torch.eye(z_train_cov.shape[0], device=device, dtype=z_train_cov.dtype)
    z_cov_inv = torch.linalg.inv(z_train_cov + eps_reg * I_reg)

    MAX_MAHAL_DISTANCE = 4.0
    EARLY_STOP_FRACTION = 0.9

    tenor_cols = [tenor_label(ten) for ten in tenors]
    times = np.arange(n_steps + 1) * dt

    swap_df_list = []
    latent_df_list = []
    z_mahal_list = []
    early_stop_time = None

    print("Decoding simulated curves...")
    for t in range(n_steps + 1):
        z_t = z_paths[:, t, :]

        z_centered = z_t - z_train_mean
        quad = torch.sum((z_centered @ z_cov_inv) * z_centered, dim=1)
        mahal_dist = torch.sqrt(torch.clamp(quad, min=0.0))
        z_mahal_list.append(mahal_dist.detach().cpu().numpy())

        out_of_region = (mahal_dist > MAX_MAHAL_DISTANCE).sum().item()
        out_of_region_frac = out_of_region / n_paths

        if out_of_region > 0:
            warnings.warn(
                f"At time t={times[t]:.3f}: {out_of_region}/{n_paths} paths ({out_of_region_frac:.1%}) "
                f"exceed {MAX_MAHAL_DISTANCE} sigma Mahalanobis distance from training region",
                RuntimeWarning,
            )

        if out_of_region_frac >= EARLY_STOP_FRACTION:
            warnings.warn(
                f"EARLY STOPPING: {out_of_region_frac:.1%} of paths exceed {MAX_MAHAL_DISTANCE} sigma threshold "
                f"at time t={times[t]:.3f}. Decoder extrapolation is unreliable. "
                f"Consider using a different initial curve or reducing n_steps.",
                RuntimeWarning,
            )
            early_stop_time = t
            break

        P_mkt, _, _, _, _, _, _, _ = decode_from_latent_script(model, z_t)
        S_sim = par_swap_from_discount(P_mkt, tenors)

        for p in range(n_paths):
            row = {"time": times[t], "path_id": p}
            for i, col in enumerate(tenor_cols):
                row[f"swap_{col}"] = S_sim[p, i].item()
            swap_df_list.append(row)

            latent_row = {"time": times[t], "path_id": p, "r": r_paths[p, t].detach().item()}
            for d in range(latent_dim):
                latent_row[f"z{d}"] = z_paths[p, t, d].detach().item()
            latent_df_list.append(latent_row)

    out_dir = os.path.join(REPO_ROOT, "Figures", "simulations")
    os.makedirs(out_dir, exist_ok=True)

    swap_df = pd.DataFrame(swap_df_list)
    swap_csv_path = os.path.join(
        out_dir,
        f"simulated_swap_curves_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.csv",
    )
    swap_df.to_csv(swap_csv_path, index=False)
    print(f"Saved simulated swap curves to {swap_csv_path}")

    latent_df = pd.DataFrame(latent_df_list)
    latent_csv_path = os.path.join(
        out_dir,
        f"simulated_latent_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.csv",
    )
    latent_df.to_csv(latent_csv_path, index=False)
    print(f"Saved simulated latent paths to {latent_csv_path}")

    if early_stop_time is not None:
        print(f"\n[WARNING] Simulation stopped early at t={times[early_stop_time]:.3f} due to excessive latent region violation.")
        print(f"  Data saved contains {len(swap_df_list) // n_paths} time steps instead of {n_steps + 1}.")

    print("Simulation and saving completed successfully.")

    return swap_df, latent_df, out_dir, times, early_stop_time


# ==========================================================
# Stage 4: Generate Plots
# ==========================================================

def generate_plots(z_paths, r_paths, mu_paths, L_paths, swap_df, tenors, out_dir, times, n_paths, n_steps, use, latent_dim, epochs, dt):
    """Generate and save all plots."""
    print("Generating plots...")

    # Plot mu
    n_plot = min(5, n_paths)
    fig_mu, axes_mu = plt.subplots(latent_dim, 1, figsize=(10, 6), sharex=True)
    axes_mu = np.atleast_1d(axes_mu)
    for d in range(latent_dim):
        ax = axes_mu[d]
        for p in range(n_plot):
            ax.plot(times, mu_paths[p, :, d].detach().cpu().numpy(), alpha=0.7)
        ax.set_ylabel(f"mu{d}")
        ax.grid(True, alpha=0.3)
    axes_mu[-1].set_xlabel("Time (years)")
    fig_mu.suptitle("Drift (mu) along simulated paths")
    plt.tight_layout()

    mu_plot_path = os.path.join(
        out_dir,
        f"simulated_mu_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
    )
    fig_mu.savefig(mu_plot_path, dpi=300)
    print(f"Saved mu plot to {mu_plot_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig_mu)

    # Plot L
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
    fig_L.suptitle("Diffusion matrix (L) elements along simulated paths")
    plt.tight_layout()

    L_plot_path = os.path.join(
        out_dir,
        f"simulated_L_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
    )
    fig_L.savefig(L_plot_path, dpi=300)
    print(f"Saved L plot to {L_plot_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig_L)

    # Plot latent paths
    fig, axes = plt.subplots(latent_dim + 1, 1, figsize=(10, 6), sharex=True)
    axes = np.atleast_1d(axes)

    for d in range(latent_dim):
        ax = axes[d]
        for p in range(min(10, n_paths)):
            ax.plot(times, z_paths[p, :, d].detach().cpu().numpy(), alpha=0.7)
        ax.set_ylabel(f"z{d}")
        ax.grid(True, alpha=0.3)

    ax = axes[-1]
    for p in range(min(10, n_paths)):
        ax.plot(times, r_paths[p, :].detach().cpu().numpy(), alpha=0.7)
    ax.set_ylabel("r")
    ax.set_xlabel("Time (years)")
    fig.suptitle("Simulated Latent Paths and Short Rate")
    plt.tight_layout()

    latent_plot_path = os.path.join(
        out_dir,
        f"simulated_latent_paths_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
    )
    fig.savefig(latent_plot_path, dpi=300)
    print(f"Saved latent paths plot to {latent_plot_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # Plot swap rates
    if not swap_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        tenors_to_plot = [1.0, 5.0, 10.0, 30.0]
        colors = ["blue", "green", "red", "orange"]

        for i, ten in enumerate(tenors_to_plot):
            if float(ten) in tenors:
                ten_label_str = tenor_label(ten)
                ten_col = f"swap_{ten_label_str}"
                if ten_col in swap_df.columns:
                    mean_curve = swap_df.groupby("time")[ten_col].mean()
                    ax.plot(mean_curve.index, mean_curve.values, label=f"{int(ten)}Y", color=colors[i], linewidth=2)

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Swap Rate")
        ax.set_title("Mean Simulated Swap Rates Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        swap_plot_path = os.path.join(
            out_dir,
            f"simulated_swap_rates_{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}.png",
        )
        fig.savefig(swap_plot_path, dpi=300)
        print(f"Saved swap rates plot to {swap_plot_path}")
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)
    else:
        print("No decoded swap curves available to plot.")

    print("Plotting completed.")


# ==========================================================
# Main Entry Point
# ==========================================================

def main(argv=None):
    """
    Main entry point supporting both script and console modes.
    
    Run from script:
        python this_file.py --n_paths 200

    Run from console:
        main([])
        main(["--n_paths", "10", "--n_steps", "12"])
    """
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
    X_tensor = X_tensor.double()

    # ========== Stage 1: Load Model ==========
    model = load_and_setup_model(device, USE, LATENT_DIM, EPOCHS)

    # ========== Stage 2: Compute Latent Statistics ==========
    z_train_mean, z_train_cov, z_train_std = compute_latent_statistics(model, X_tensor, device, LATENT_DIM)

    # ========== Stage 3: Load Initial Curve ==========
    S0, meta_row, X_tensor, meta = load_initial_curve(USE, IDX_CHOICE, device)
    with torch.no_grad():
        z0 = model.encoder(S0)
    print(f"Initial latent state z0: {z0.cpu().numpy().flatten()}")

    # ========== Stage 4: Run Simulation ==========
    z_paths, r_paths, mu_paths, L_paths = run_simulation(
        model, z0, N_PATHS, N_STEPS, DT, device, LATENT_DIM, DISCRETIZATION
    )

    # ========== Stage 5: Analyze Paths ==========
    analyze_paths(z_paths, r_paths, mu_paths, L_paths, LATENT_DIM)

    # ========== Stage 6: Decode and Save ==========
    swap_df, latent_df, out_dir, times, early_stop_time = decode_and_save_results(
        model, z_paths, r_paths, z_train_mean, z_train_cov, device,
        N_STEPS, N_PATHS, DT, tenors, USE, LATENT_DIM, EPOCHS
    )

    # ========== Stage 7: Generate Plots ==========
    generate_plots(z_paths, r_paths, mu_paths, L_paths, swap_df, tenors, out_dir, times, N_PATHS, N_STEPS, USE, LATENT_DIM, EPOCHS, DT)


if __name__ == "__main__":
    main()