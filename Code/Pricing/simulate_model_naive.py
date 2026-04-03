import argparse
import math
import os
import sys
import time
import random
import warnings

import matplotlib
# Will set backend dynamically based on --show_plots argument
# matplotlib.use("Agg")
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

if THESIS_ROOT not in sys.path:
    sys.path.insert(0, THESIS_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    parser = argparse.ArgumentParser(description="Naive latent simulation and soft-fail decoding experiment")

    parser.add_argument("--stage", type=str, default="all", choices=["simulate", "decode", "martingale", "all"])
    parser.add_argument("--bundle_path", type=str, default="", help="Path to saved simulation bundle .pt")

    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension (must be 2)")
    parser.add_argument("--epochs", type=int, default=3500, help="Training epochs")
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

    parser.add_argument(
        "--max_mahal",
        type=float,
        default=4.0,
        help="Mahalanobis threshold used only for diagnostics in the naive experiment",
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
    parser.add_argument(
        "--g0_floor",
        type=float,
        default=1e-5,
        help="Decoder safety floor for |G(z,0)|",
    )

    # Plot controls
    parser.add_argument("--plot_n_paths", type=int, default=20, help="How many sample paths to show in path plots")
    parser.add_argument(
        "--plot_curve_times",
        type=str,
        default="0,0.5,1.0,2.0",
        help="Comma-separated times for swap-curve snapshot plots",
    )
    parser.add_argument(
        "--plot_tenors",
        type=str,
        default="1,5,10,30",
        help="Comma-separated tenors for time-series swap plots",
    )
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=200,
        help="DPI for saved PNG figures",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively instead of just saving them",
    )
    return parser


def resolve_checkpoint_path(repo_root: str, use: str, latent_dim: int, epochs: int) -> str:
    variant = config.VARIANT

    new_filename = f"checkpoint_dim{latent_dim}_ep{epochs}.pt"
    new_path = os.path.join(
        THESIS_ROOT,
        "Figures",
        "TrainingResults",
        f"dim{latent_dim}_{variant}",
        f"ep{epochs}",
        new_filename,
    )

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
    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}")


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
        raise ValueError("This script currently supports only the 2-factor model (latent_dim=2).")

    checkpoint_path = resolve_checkpoint_path(REPO_ROOT, use, latent_dim, epochs)
    raw = torch.load(checkpoint_path, map_location=device)

    from Code.model.full_model import FullModel

    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        saved_variant = raw.get("variant", "unknown")
        if saved_variant != "unknown" and saved_variant != config.VARIANT:
            raise ValueError(
                f"Checkpoint variant '{saved_variant}' does not match active "
                f"config.VARIANT '{config.VARIANT}'. Update Code/config.py."
            )
    else:
        state_dict = raw

    model = FullModel(latent_dim=latent_dim)
    safe_load_state_dict(model, state_dict)

    model.to(device).double()
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Active config variant: {config.VARIANT}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    return model


@torch.no_grad()
def get_mu(model, z):
    return model.K(z)


@torch.no_grad()
def get_L(model, z):
    H_out = model.H(z)

    if isinstance(H_out, tuple) and len(H_out) == 2:
        sigmas, rhos = H_out
        return L_from_sigmas_rhos(sigmas, rhos)

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
    return f"{int(float(tenor_value))}Y"


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


def make_experiment_suffix(use, latent_dim, epochs, n_paths, n_steps, seed, discretization):
    disc = normalize_discretization_name(discretization)
    return f"{use}_dim{latent_dim}_ep{epochs}_paths{n_paths}_steps{n_steps}_seed{seed}_{disc}"


def get_simulation_out_dir():
    out_dir = os.path.join(THESIS_ROOT, "Figures", "Pricing", "simulations")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_simulation_bundle(
    out_dir,
    suffix,
    z_paths,
    r_paths,
    mu_paths,
    L_paths,
    discount_paths,
    times,
    z_train_mean,
    z_train_cov,
    tenors,
    decoder_tau_grid,
    annual_indices,
):
    bundle = {
        "z_paths": z_paths.detach().cpu(),
        "r_paths": r_paths.detach().cpu(),
        "mu_paths": mu_paths.detach().cpu(),
        "L_paths": L_paths.detach().cpu(),
        "discount_paths": discount_paths.detach().cpu(),
        "times": np.asarray(times),
        "z_train_mean": z_train_mean.detach().cpu(),
        "z_train_cov": z_train_cov.detach().cpu(),
        "tenors": np.asarray(tenors),
        "decoder_tau_grid": decoder_tau_grid.detach().cpu(),
        "annual_indices": list(annual_indices),
    }
    bundle_path = os.path.join(out_dir, f"simulation_bundle_{suffix}.pt")
    torch.save(bundle, bundle_path)
    print(f"Saved simulation bundle to {bundle_path}")
    return bundle_path


def load_simulation_bundle(bundle_path, device):
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)

    bundle["z_paths"] = bundle["z_paths"].to(device)
    bundle["r_paths"] = bundle["r_paths"].to(device)
    bundle["mu_paths"] = bundle["mu_paths"].to(device)
    bundle["L_paths"] = bundle["L_paths"].to(device)
    bundle["discount_paths"] = bundle["discount_paths"].to(device)
    bundle["z_train_mean"] = bundle["z_train_mean"].to(device)
    bundle["z_train_cov"] = bundle["z_train_cov"].to(device)
    bundle["decoder_tau_grid"] = bundle["decoder_tau_grid"].to(device)
    return bundle


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


def diagnose_G0_on_training_cloud(model, X_tensor, device, batch_size=256):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Checking G(z,0) on training latent cloud")
    print("=" * 60)

    g0_list = []

    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], batch_size):
            batch = X_tensor[i : min(i + batch_size, X_tensor.shape[0])].to(device)
            z_batch = model.encoder(batch)

            tau0 = torch.zeros(1, device=device, dtype=z_batch.dtype)
            G0_batch = model.G(z_batch, tau0)

            if G0_batch.ndim == 2:
                G0_batch = G0_batch[:, 0]
            elif G0_batch.ndim == 1:
                pass
            else:
                raise RuntimeError(f"Unexpected shape for G(z,0): {tuple(G0_batch.shape)}")

            g0_list.append(G0_batch)

    G0 = torch.cat(g0_list, dim=0)
    absG0 = G0.abs()

    print(f"G(z,0) raw min        : {G0.min().item():.6e}")
    print(f"G(z,0) raw max        : {G0.max().item():.6e}")
    print(f"|G(z,0)| min          : {absG0.min().item():.6e}")
    print(f"|G(z,0)| mean         : {absG0.mean().item():.6e}")
    print(f"|G(z,0)| median       : {absG0.median().item():.6e}")
    print(f"|G(z,0)| 1% quantile  : {torch.quantile(absG0, 0.01).item():.6e}")
    print(f"|G(z,0)| 5% quantile  : {torch.quantile(absG0, 0.05).item():.6e}")
    print(f"count(|G0| < 1e-2)    : {(absG0 < 1e-2).sum().item()}")
    print(f"count(|G0| < 1e-3)    : {(absG0 < 1e-3).sum().item()}")
    print(f"count(|G0| < 1e-4)    : {(absG0 < 1e-4).sum().item()}")
    print("=" * 60 + "\n")

    return G0


def diagnose_G0_on_simulated_paths(model, z_paths):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Checking G(z_t,0) on simulated latent paths")
    print("=" * 60)

    n_paths, n_times, _ = z_paths.shape
    device = z_paths.device
    dtype = z_paths.dtype

    rows = []

    with torch.no_grad():
        tau0 = torch.zeros(1, device=device, dtype=dtype)

        for t in range(n_times):
            z_t = z_paths[:, t, :]
            G0_t = model.G(z_t, tau0)

            if G0_t.ndim == 2:
                G0_t = G0_t[:, 0]
            elif G0_t.ndim == 1:
                pass
            else:
                raise RuntimeError(f"Unexpected shape for G(z_t,0): {tuple(G0_t.shape)}")

            absG0_t = G0_t.abs()

            row = {
                "time_index": t,
                "G0_min": G0_t.min().item(),
                "G0_max": G0_t.max().item(),
                "absG0_min": absG0_t.min().item(),
                "absG0_mean": absG0_t.mean().item(),
                "absG0_median": absG0_t.median().item(),
                "absG0_1pct": torch.quantile(absG0_t, 0.01).item(),
                "absG0_5pct": torch.quantile(absG0_t, 0.05).item(),
                "count_absG0_lt_1e_2": (absG0_t < 1e-2).sum().item(),
                "count_absG0_lt_1e_3": (absG0_t < 1e-3).sum().item(),
                "count_absG0_lt_1e_4": (absG0_t < 1e-4).sum().item(),
                "count_absG0_lt_1e_5": (absG0_t < 1e-5).sum().item(),
            }
            rows.append(row)

            print(
                f"t_idx={t:2d} | "
                f"min |G0|={row['absG0_min']:.3e} | "
                f"1%={row['absG0_1pct']:.3e} | "
                f"5%={row['absG0_5pct']:.3e} | "
                f"<1e-3: {row['count_absG0_lt_1e_3']:3d} | "
                f"<1e-4: {row['count_absG0_lt_1e_4']:3d}"
            )

    print("=" * 60 + "\n")
    return pd.DataFrame(rows)


def _finite_diff_diffusion_jacobian(model, z, eps=1e-4):
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
    directional_deriv = torch.einsum("nkj,nijk->nij", B, jac_B)
    return 0.5 * torch.sum(directional_deriv * ((dW**2 - dt).unsqueeze(1)), dim=2)


def _stable_drift_step(model, z: torch.Tensor, shock: torch.Tensor, dt: float) -> torch.Tensor:
    if hasattr(model.K, "stable_matrix"):
        M = model.K.stable_matrix()
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


def decode_from_latent_script(model, z, tau, G_floor=1e-5, check_short_rate=True):
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

    if not torch.allclose(A_vals[:, 0], torch.zeros_like(A_vals[:, 0]), atol=1e-8, rtol=0.0):
        raise RuntimeError("Decoder invariant violated: A(z,0) != 0")
    if not torch.allclose(B_vals[:, 0], torch.zeros_like(B_vals[:, 0]), atol=1e-8, rtol=0.0):
        raise RuntimeError("Decoder invariant violated: B(z,0) != 0")

    expo = A_vals - B_vals * G_vals
    if not torch.isfinite(expo).all():
        raise RuntimeError("Non-finite exponent encountered in bond pricing")

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

    diagnostics = {
        "G_range": (G_vals.min().item(), G_vals.max().item()),
        "P_range": (P_full[:, 1:].min().item(), P_full[:, 1:].max().item()),
        "min_abs_G0": min_abs_G0,
        "short_rate_tau_used": short_rate_tau_used,
        "max_short_rate_err": max_short_rate_err,
    }

    return P_full, A_vals, B_vals, G_vals, mu, sigma, r_tilde, diagnostics


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


def decode_and_save_results_naive(
    model,
    z_paths,
    r_paths,
    z_train_mean,
    z_train_cov,
    decoder_tau_grid,
    annual_indices,
    device,
    times,
    tenors,
    use,
    latent_dim,
    epochs,
    max_mahal,
    g0_floor,
    suffix,
):
    eps_reg = 1e-8
    I_reg = torch.eye(z_train_cov.shape[0], device=device, dtype=z_train_cov.dtype)
    z_cov_inv = torch.linalg.inv(z_train_cov + eps_reg * I_reg)

    n_paths, n_times, _ = z_paths.shape
    tenor_cols = [tenor_label(ten) for ten in tenors]

    swap_df_list = []
    latent_df_list = []
    mahal_df_list = []
    decoder_diag_df_list = []
    decoder_failures_list = []

    print("Naive decoding experiment...")
    t0 = time.time()

    for t in range(n_times):
        z_t = z_paths[:, t, :]
        r_t = r_paths[:, t]

        z_centered = z_t - z_train_mean
        quad = torch.sum((z_centered @ z_cov_inv) * z_centered, dim=1)
        mahal_dist = torch.sqrt(torch.clamp(quad, min=0.0))

        tau0 = torch.zeros(1, device=device, dtype=z_t.dtype)
        with torch.no_grad():
            G0_t = model.G(z_t, tau0)
            if G0_t.ndim == 2:
                G0_t = G0_t[:, 0]
            absG0_t = G0_t.abs()

        out_of_region = int((mahal_dist > max_mahal).sum().item())
        n_g0_bad_1e3 = int((absG0_t < 1e-3).sum().item())
        n_g0_bad_1e4 = int((absG0_t < 1e-4).sum().item())
        n_g0_bad_floor = int((absG0_t < g0_floor).sum().item())

        if out_of_region > 0:
            warnings.warn(
                f"At time t={times[t]:.3f}: {out_of_region}/{n_paths} paths exceed Mahalanobis {max_mahal:.2f}",
                RuntimeWarning,
            )

        batch_error = ""
        valid_decode = np.zeros(n_paths, dtype=bool)
        S_sim_np = np.full((n_paths, len(tenors)), np.nan, dtype=float)
        path_reasons = [""] * n_paths

        decoder_min_abs_G0_valid = np.nan
        decoder_max_short_rate_err_valid = np.nan

        try:
            P_full, _, _, _, _, _, _, dec_diag = decode_from_latent_script(
                model,
                z_t,
                decoder_tau_grid,
                G_floor=g0_floor,
                check_short_rate=True,
            )
            P_annual = P_full[:, annual_indices]
            S_sim = par_swap_from_discount(P_annual, tenors)

            S_sim_np = S_sim.detach().cpu().numpy()
            valid_decode[:] = True
            decoder_min_abs_G0_valid = dec_diag["min_abs_G0"]
            decoder_max_short_rate_err_valid = dec_diag["max_short_rate_err"]

        except RuntimeError as e:
            batch_error = str(e)

            min_abs_valid = np.inf
            max_sr_err_valid = -np.inf

            for p in range(n_paths):
                try:
                    P_full_p, _, _, _, _, _, _, dec_diag_p = decode_from_latent_script(
                        model,
                        z_t[p : p + 1],
                        decoder_tau_grid,
                        G_floor=g0_floor,
                        check_short_rate=True,
                    )
                    P_annual_p = P_full_p[:, annual_indices]
                    S_sim_p = par_swap_from_discount(P_annual_p, tenors)

                    S_sim_np[p, :] = S_sim_p[0].detach().cpu().numpy()
                    valid_decode[p] = True

                    min_abs_valid = min(min_abs_valid, float(dec_diag_p["min_abs_G0"]))
                    max_sr_err_valid = max(max_sr_err_valid, float(dec_diag_p["max_short_rate_err"]))

                except RuntimeError as e_p:
                    path_reasons[p] = str(e_p)
                    decoder_failures_list.append(
                        {
                            "time": float(times[t]),
                            "path_id": p,
                            "reason": str(e_p),
                            "mahal_dist": float(mahal_dist[p].item()),
                            "absG0": float(absG0_t[p].item()),
                        }
                    )

            if np.isfinite(min_abs_valid):
                decoder_min_abs_G0_valid = min_abs_valid
            if np.isfinite(max_sr_err_valid) and max_sr_err_valid > -np.inf:
                decoder_max_short_rate_err_valid = max_sr_err_valid

        n_valid = int(valid_decode.sum())
        frac_valid = n_valid / n_paths

        decoder_diag_df_list.append(
            {
                "time": float(times[t]),
                "max_mahal_dist": float(mahal_dist.max().item()),
                "mean_mahal_dist": float(mahal_dist.mean().item()),
                "frac_mahal_gt_threshold": float(out_of_region / n_paths),
                "n_paths_absG0_lt_1e_3": n_g0_bad_1e3,
                "n_paths_absG0_lt_1e_4": n_g0_bad_1e4,
                "n_paths_absG0_lt_floor": n_g0_bad_floor,
                "min_absG0_all_paths": float(absG0_t.min().item()),
                "decoder_min_G_abs0_valid_paths": decoder_min_abs_G0_valid,
                "decoder_max_short_rate_err_valid_paths": decoder_max_short_rate_err_valid,
                "n_valid_decode": n_valid,
                "frac_valid_decode": frac_valid,
                "batch_error": batch_error,
            }
        )

        for p in range(n_paths):
            swap_row = {
                "time": float(times[t]),
                "path_id": p,
                "valid_decode": bool(valid_decode[p]),
                "failure_reason": path_reasons[p],
            }
            for i, col in enumerate(tenor_cols):
                swap_row[f"swap_{col}"] = float(S_sim_np[p, i]) if np.isfinite(S_sim_np[p, i]) else np.nan
            swap_df_list.append(swap_row)

            latent_row = {
                "time": float(times[t]),
                "path_id": p,
                "r": float(r_t[p].detach().item()),
                "absG0": float(absG0_t[p].item()),
                "valid_decode": bool(valid_decode[p]),
            }
            for d in range(latent_dim):
                latent_row[f"z{d}"] = float(z_t[p, d].detach().item())
            latent_df_list.append(latent_row)

            mahal_df_list.append(
                {
                    "time": float(times[t]),
                    "path_id": p,
                    "mahal_dist": float(mahal_dist[p].detach().item()),
                    "absG0": float(absG0_t[p].item()),
                }
            )

        if t == 0 or t == n_times - 1 or t % max(1, max(1, n_times - 1) // 10) == 0:
            print(
                f"  t={times[t]:.3f} | "
                f"valid decode={n_valid}/{n_paths} | "
                f"min |G0|={absG0_t.min().item():.3e} | "
                f"max Mahalanobis={mahal_dist.max().item():.3f}"
            )

    elapsed = time.time() - t0
    print(f"Naive decoding finished in {elapsed:.2f}s")

    out_dir = get_simulation_out_dir()

    swap_df = pd.DataFrame(swap_df_list)
    latent_df = pd.DataFrame(latent_df_list)
    mahal_df = pd.DataFrame(mahal_df_list)
    decoder_diag_df = pd.DataFrame(decoder_diag_df_list)
    decoder_failures_df = pd.DataFrame(decoder_failures_list)

    swap_csv_path = os.path.join(out_dir, f"simulated_swap_curves_{suffix}.csv")
    latent_csv_path = os.path.join(out_dir, f"simulated_latent_{suffix}.csv")
    mahal_csv_path = os.path.join(out_dir, f"simulated_mahal_{suffix}.csv")
    decoder_diag_csv_path = os.path.join(out_dir, f"decoder_diagnostics_{suffix}.csv")
    decoder_failures_csv_path = os.path.join(out_dir, f"decoder_failures_{suffix}.csv")

    swap_df.to_csv(swap_csv_path, index=False)
    latent_df.to_csv(latent_csv_path, index=False)
    mahal_df.to_csv(mahal_csv_path, index=False)
    decoder_diag_df.to_csv(decoder_diag_csv_path, index=False)
    decoder_failures_df.to_csv(decoder_failures_csv_path, index=False)

    print(f"Saved simulated swap curves to {swap_csv_path}")
    print(f"Saved simulated latent paths to {latent_csv_path}")
    print(f"Saved Mahalanobis diagnostics to {mahal_csv_path}")
    print(f"Saved decoder diagnostics to {decoder_diag_csv_path}")
    print(f"Saved decoder failures to {decoder_failures_csv_path}")

    return swap_df, latent_df, mahal_df, decoder_diag_df, decoder_failures_df, out_dir


def build_tau_grid_to_maturity(decoder_tau_grid, tau_end, device, dtype):
    """Build a dense tau grid that ALWAYS includes the exact endpoint tau_end.
    
    Args:
        decoder_tau_grid: Base dense tau grid (typically weekly near zero, then annual)
        tau_end: The exact maturity endpoint that must be included
        device: Torch device
        dtype: Torch dtype
        
    Returns:
        tau_grid: Sorted unique grid with 0, interior points, and exact tau_end
    """
    tau_end = float(tau_end)
    if tau_end <= 0:
        raise ValueError(f"tau_end must be positive, got {tau_end}")

    base = decoder_tau_grid.to(device=device, dtype=dtype)

    # Keep only interior points strictly between 0 and tau_end
    interior = base[(base > 0.0) & (base < tau_end - 1e-12)]

    tau_grid = torch.cat(
        [
            torch.zeros(1, device=device, dtype=dtype),
            interior,
            torch.tensor([tau_end], device=device, dtype=dtype),
        ]
    )
    tau_grid = torch.unique(tau_grid, sorted=True)

    if tau_grid.numel() < 2:
        tau_grid = torch.tensor([0.0, tau_end], device=device, dtype=dtype)

    return tau_grid


def martingale_diagnostics_naive(
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
    g0_floor,
    suffix,
    decoder_tau_grid=None,
    annual_indices=None,
    martingale_tol=0.02,
):
    if len(maturity_dates) == 0:
        print("No martingale dates requested; skipping discounted-bond martingale diagnostics.")
        return pd.DataFrame()

    maturity_dates = sorted(float(u) for u in maturity_dates)
    if all(u <= 0 for u in maturity_dates):
        print("All martingale dates are non-positive; skipping diagnostics.")
        return pd.DataFrame()

    if decoder_tau_grid is None:
        raise ValueError("decoder_tau_grid must be provided for martingale diagnostics.")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Discounted-bond martingale check (exact endpoint on dense tau grid)")
    print("=" * 60)

    device = z_paths.device
    dtype = z_paths.dtype
    out_rows = []

    n_paths_local, n_times, _ = z_paths.shape

    for t_idx, t_now in enumerate(times):
        t_now = float(t_now)
        valid_U = [u for u in maturity_dates if u > t_now + 1e-12]
        if len(valid_U) == 0:
            continue

        for u in valid_U:
            tau_remaining = float(u - t_now)

            # Build a dense grid that ALWAYS includes the exact endpoint tau_remaining
            tau_grid = build_tau_grid_to_maturity(
                decoder_tau_grid=decoder_tau_grid,
                tau_end=tau_remaining,
                device=device,
                dtype=dtype,
            )
            tau_idx = get_grid_indices_for_values(
                tau_grid,
                torch.tensor([tau_remaining], device=device, dtype=dtype),
            )[0]

            vals = []
            for p in range(n_paths_local):
                try:
                    P_full, _, _, _, _, _, _, _ = decode_from_latent_script(
                        model,
                        z_paths[p : p + 1, t_idx, :],
                        tau_grid,
                        G_floor=g0_floor,
                        check_short_rate=False,
                    )
                    disc_val = discount_paths[p, t_idx] * P_full[0, tau_idx]
                    vals.append(float(disc_val.item()))
                except RuntimeError:
                    pass

            # Initial value P(0,U), again with exact endpoint U included
            try:
                tau_grid_init = build_tau_grid_to_maturity(
                    decoder_tau_grid=decoder_tau_grid,
                    tau_end=float(u),
                    device=device,
                    dtype=dtype,
                )
                tau_idx_init = get_grid_indices_for_values(
                    tau_grid_init,
                    torch.tensor([float(u)], device=device, dtype=dtype),
                )[0]

                P0_full, _, _, _, _, _, _, _ = decode_from_latent_script(
                    model,
                    z_paths[:1, 0, :],
                    tau_grid_init,
                    G_floor=g0_floor,
                    check_short_rate=False,
                )
                initial_val = float(P0_full[0, tau_idx_init].item())
            except RuntimeError:
                initial_val = np.nan

            if len(vals) == 0:
                mean_val = np.nan
                std_val = np.nan
                sem_val = np.nan
                rel_err = np.nan
                n_valid = 0
            else:
                arr = np.asarray(vals, dtype=float)
                mean_val = float(arr.mean())
                std_val = float(arr.std(ddof=0))
                sem_val = float(std_val / math.sqrt(len(arr)))
                rel_err = (
                    abs(mean_val - initial_val) / max(abs(initial_val), 1e-12)
                    if np.isfinite(initial_val)
                    else np.nan
                )
                n_valid = len(vals)

            out_rows.append(
                {
                    "time": t_now,
                    "U": float(u),
                    "tau_remaining": tau_remaining,
                    "disc_bond_mean": mean_val,
                    "disc_bond_std": std_val,
                    "disc_bond_sem": sem_val,
                    "initial_disc_bond_value": initial_val,
                    "relative_mean_error": rel_err,
                    "n_valid_paths": n_valid,
                    "frac_valid_paths": n_valid / n_paths_local,
                }
            )

    mart_df = pd.DataFrame(out_rows)
    mart_csv_path = os.path.join(out_dir, f"martingale_diagnostics_{suffix}.csv")
    mart_df.to_csv(mart_csv_path, index=False)
    print(f"Saved martingale diagnostics to {mart_csv_path}")

    if not mart_df.empty:
        for u in sorted(mart_df["U"].unique()):
            sub = mart_df[mart_df["U"] == u]
            finite_errs = sub["relative_mean_error"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite_errs) == 0:
                print(f"  U={u:.2f}: no valid diagnostic points")
                continue
            max_err = float(finite_errs.max())
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
# Plot helpers
# ==========================================================

def _save_close(fig, path, dpi=200, show=False):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _sample_path_indices(n_paths, max_paths):
    if n_paths <= max_paths:
        return np.arange(n_paths)
    return np.linspace(0, n_paths - 1, max_paths, dtype=int)


def _nearest_values(base_values, requested_values):
    base_values = np.asarray(base_values, dtype=float)
    out = []
    for v in requested_values:
        out.append(base_values[np.argmin(np.abs(base_values - v))])
    return list(dict.fromkeys([float(x) for x in out]))


def _mean_std_bands(arr_2d):
    mean = np.nanmean(arr_2d, axis=0)
    std = np.nanstd(arr_2d, axis=0)
    return mean, mean - std, mean + std


def plot_latent_2d_paths(z_paths, times, out_dir, suffix, max_paths=20, dpi=200, show=False):
    z_np = z_paths.detach().cpu().numpy()
    idx = _sample_path_indices(z_np.shape[0], max_paths)

    fig, ax = plt.subplots(figsize=(7, 6))
    for p in idx:
        ax.plot(z_np[p, :, 0], z_np[p, :, 1], alpha=0.8)
        ax.scatter(z_np[p, 0, 0], z_np[p, 0, 1], s=15)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("Latent trajectories in (z1, z2)")
    ax.grid(True, alpha=0.3)
    _save_close(fig, os.path.join(out_dir, f"plot_latent_2d_{suffix}.png"), dpi=dpi, show=show)


def plot_latent_components(z_paths, times, out_dir, suffix, max_paths=20, dpi=200, show=False):
    z_np = z_paths.detach().cpu().numpy()
    idx = _sample_path_indices(z_np.shape[0], max_paths)
    t = np.asarray(times)

    for d in range(z_np.shape[2]):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for p in idx:
            ax.plot(t, z_np[p, :, d], alpha=0.8)
        ax.set_xlabel("time")
        ax.set_ylabel(f"z{d}")
        ax.set_title(f"Latent component z{d} across sample paths")
        ax.grid(True, alpha=0.3)
        _save_close(fig, os.path.join(out_dir, f"plot_latent_z{d}_{suffix}.png"), dpi=dpi, show=show)


def plot_short_rate_paths(r_paths, times, out_dir, suffix, max_paths=20, dpi=200, show=False):
    r_np = r_paths.detach().cpu().numpy()
    idx = _sample_path_indices(r_np.shape[0], max_paths)
    t = np.asarray(times)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in idx:
        ax.plot(t, r_np[p], alpha=0.8)
    mean, lo, hi = _mean_std_bands(r_np)
    ax.plot(t, mean, linewidth=2.5, label="mean")
    ax.fill_between(t, lo, hi, alpha=0.2, label="mean ± 1 std")
    ax.set_xlabel("time")
    ax.set_ylabel("short rate")
    ax.set_title("Short-rate paths")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_short_rate_{suffix}.png"), dpi=dpi, show=show)


def plot_discount_paths(discount_paths, times, out_dir, suffix, max_paths=20, dpi=200, show=False):
    d_np = discount_paths.detach().cpu().numpy()
    idx = _sample_path_indices(d_np.shape[0], max_paths)
    t = np.asarray(times)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in idx:
        ax.plot(t, d_np[p], alpha=0.8)
    mean, lo, hi = _mean_std_bands(d_np)
    ax.plot(t, mean, linewidth=2.5, label="mean")
    ax.fill_between(t, lo, hi, alpha=0.2, label="mean ± 1 std")
    ax.set_xlabel("time")
    ax.set_ylabel("discount factor")
    ax.set_title("Pathwise discount factors")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_discount_{suffix}.png"), dpi=dpi, show=show)


def plot_mu_summary(mu_paths, times, out_dir, suffix, dpi=200, show=False):
    mu_np = mu_paths.detach().cpu().numpy()
    t = np.asarray(times)

    for d in range(mu_np.shape[2]):
        mean = mu_np[:, :, d].mean(axis=0)
        std = mu_np[:, :, d].std(axis=0)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(t, mean, linewidth=2.5, label="mean")
        ax.fill_between(t, mean - std, mean + std, alpha=0.2, label="mean ± 1 std")
        ax.set_xlabel("time")
        ax.set_ylabel(f"mu[{d}]")
        ax.set_title(f"Drift summary for mu[{d}]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save_close(fig, os.path.join(out_dir, f"plot_mu_{d}_{suffix}.png"), dpi=dpi, show=show)


def plot_diffusion_summary(L_paths, times, out_dir, suffix, dpi=200, show=False):
    L_np = L_paths.detach().cpu().numpy()
    t = np.asarray(times)
    norms = np.linalg.norm(L_np, axis=(2, 3))
    mean = norms.mean(axis=0)
    std = norms.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, mean, linewidth=2.5, label="mean Frobenius norm")
    ax.fill_between(t, mean - std, mean + std, alpha=0.2, label="mean ± 1 std")
    ax.set_xlabel("time")
    ax.set_ylabel("||L||_F")
    ax.set_title("Diffusion magnitude summary")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_diffusion_norm_{suffix}.png"), dpi=dpi, show=show)


def plot_diffusion_entries(L_paths, times, out_dir, suffix, dpi=200, show=False):
    L_np = L_paths.detach().cpu().numpy()
    t = np.asarray(times)
    d = L_np.shape[2]

    for i in range(d):
        for j in range(d):
            arr = L_np[:, :, i, j]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(t, mean, linewidth=2.5, label="mean")
            ax.fill_between(t, mean - std, mean + std, alpha=0.2, label="mean ± 1 std")
            ax.set_xlabel("time")
            ax.set_ylabel(f"L[{i},{j}]")
            ax.set_title(f"Diffusion entry summary for L[{i},{j}]")
            ax.grid(True, alpha=0.3)
            ax.legend()
            _save_close(fig, os.path.join(out_dir, f"plot_diffusion_L_{i}{j}_{suffix}.png"), dpi=dpi, show=show)


def plot_g0_diagnostics(g0_df, times, out_dir, suffix, dpi=200, show=False):
    if g0_df.empty:
        return
    t = np.asarray(times)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, g0_df["absG0_min"].values, label="min |G0|")
    ax.plot(t, g0_df["absG0_1pct"].values, label="1% quantile")
    ax.plot(t, g0_df["absG0_5pct"].values, label="5% quantile")
    ax.set_xlabel("time")
    ax.set_ylabel("|G(z_t,0)|")
    ax.set_title("Simulated-path G(z,0) diagnostics")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_g0_diagnostics_{suffix}.png"), dpi=dpi, show=show)


def plot_decoder_diagnostics(decoder_diag_df, out_dir, suffix, dpi=200, show=False):
    if decoder_diag_df.empty:
        return

    t = decoder_diag_df["time"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, decoder_diag_df["frac_valid_decode"].to_numpy(), label="frac valid decode")
    ax.plot(t, decoder_diag_df["frac_mahal_gt_threshold"].to_numpy(), label="frac mahal > threshold")
    ax.set_xlabel("time")
    ax.set_ylabel("fraction")
    ax.set_title("Decoder success and region diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_decoder_validity_{suffix}.png"), dpi=dpi, show=show)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, decoder_diag_df["min_absG0_all_paths"].to_numpy(), label="min |G0| all paths")
    valid_col = "decoder_min_G_abs0_valid_paths"
    if valid_col in decoder_diag_df.columns:
        ax.plot(t, decoder_diag_df[valid_col].to_numpy(), label="min |G0| valid paths")
    ax.set_xlabel("time")
    ax.set_ylabel("|G0|")
    ax.set_title("Decoder G(z,0) safety diagnostics")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_decoder_g0_{suffix}.png"), dpi=dpi, show=show)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, decoder_diag_df["mean_mahal_dist"].to_numpy(), label="mean Mahalanobis")
    ax.plot(t, decoder_diag_df["max_mahal_dist"].to_numpy(), label="max Mahalanobis")
    ax.set_xlabel("time")
    ax.set_ylabel("distance")
    ax.set_title("Mahalanobis distance diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_mahalanobis_{suffix}.png"), dpi=dpi, show=show)


def plot_swap_curve_snapshots(swap_df, tenors, requested_times, out_dir, suffix, dpi=200, show=False):
    if swap_df.empty:
        return

    swap_cols = [f"swap_{tenor_label(t)}" for t in tenors]
    all_times = np.sort(swap_df["time"].unique())
    plot_times = _nearest_values(all_times, requested_times)
    tenor_vals = np.asarray(tenors, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    for t_sel in plot_times:
        sub = swap_df[(swap_df["time"] == t_sel) & (swap_df["valid_decode"])]
        if sub.empty:
            continue
        vals = sub[swap_cols].to_numpy(dtype=float)
        median = np.nanmedian(vals, axis=0)
        lo = np.nanquantile(vals, 0.05, axis=0)
        hi = np.nanquantile(vals, 0.95, axis=0)
        ax.plot(tenor_vals, median, label=f"t={t_sel:g} median")
        ax.fill_between(tenor_vals, lo, hi, alpha=0.12)

    ax.set_xlabel("tenor")
    ax.set_ylabel("swap rate")
    ax.set_title("Swap-curve snapshots across simulated paths")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(
        fig,
        os.path.join(out_dir, f"plot_swap_snapshots_{suffix}.png"),
        dpi=dpi,
        show=show,
    )


def plot_selected_tenor_timeseries(swap_df, selected_tenors, out_dir, suffix, dpi=200, show=False):
    if swap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    times = np.sort(swap_df["time"].unique())

    for ten in selected_tenors:
        col = f"swap_{tenor_label(ten)}"
        if col not in swap_df.columns:
            continue
        med = []
        lo = []
        hi = []
        for t in times:
            vals = swap_df.loc[(swap_df["time"] == t) & (swap_df["valid_decode"]), col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                med.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
            else:
                med.append(np.nanmedian(vals))
                lo.append(np.nanquantile(vals, 0.05))
                hi.append(np.nanquantile(vals, 0.95))
        med = np.asarray(med)
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        ax.plot(times, med, label=tenor_label(ten))
        ax.fill_between(times, lo, hi, alpha=0.12)

    ax.set_xlabel("time")
    ax.set_ylabel("swap rate")
    ax.set_title("Selected swap tenors through time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_swap_tenor_timeseries_{suffix}.png"), dpi=dpi, show=show)


def plot_failure_reasons(decoder_failures_df, out_dir, suffix, dpi=200, show=False):
    if decoder_failures_df.empty or "reason" not in decoder_failures_df.columns:
        return
    counts = decoder_failures_df["reason"].fillna("unknown").value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(np.arange(len(counts)), counts.values)
    ax.set_yticks(np.arange(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title("Top decoder failure reasons")
    ax.grid(True, axis="x", alpha=0.3)
    _save_close(fig, os.path.join(out_dir, f"plot_decoder_failures_{suffix}.png"), dpi=dpi, show=show)


def plot_martingale_diagnostics(mart_df, out_dir, suffix, dpi=200, show=False):
    if mart_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for U in sorted(mart_df["U"].dropna().unique()):
        sub = mart_df[mart_df["U"] == U].sort_values("time")
        ax.plot(sub["time"], sub["relative_mean_error"], label=f"U={U:g}")
    ax.set_xlabel("time")
    ax.set_ylabel("relative mean error")
    ax.set_title("Discounted-bond martingale diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_close(fig, os.path.join(out_dir, f"plot_martingale_error_{suffix}.png"), dpi=dpi, show=show)

    fig, ax = plt.subplots(figsize=(8, 5))
    for U in sorted(mart_df["U"].dropna().unique()):
        sub = mart_df[mart_df["U"] == U].sort_values("time")
        ax.plot(sub["time"], sub["disc_bond_mean"], label=f"U={U:g} mean")
        ax.plot(sub["time"], sub["initial_disc_bond_value"], linestyle="--", label=f"U={U:g} initial")
    ax.set_xlabel("time")
    ax.set_ylabel("discounted bond value")
    ax.set_title("Discounted-bond mean vs initial value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    _save_close(fig, os.path.join(out_dir, f"plot_martingale_levels_{suffix}.png"), dpi=dpi, show=show)


def main(argv=None):
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    # Set matplotlib backend after pyplot is already imported
    if args.show_plots:
        plt.switch_backend("TkAgg")  # Interactive backend
    else:
        plt.switch_backend("Agg")    # Non-interactive backend

    LATENT_DIM = args.latent_dim
    EPOCHS = args.epochs
    USE = args.use
    N_PATHS = args.n_paths
    N_STEPS = args.n_steps
    DT = args.dt
    IDX_CHOICE = args.idx_choice
    DISCRETIZATION = args.discretization
    SEED = args.seed
    G0_FLOOR = args.g0_floor

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

    suffix = make_experiment_suffix(USE, LATENT_DIM, EPOCHS, N_PATHS, N_STEPS, SEED, DISCRETIZATION)
    out_dir = get_simulation_out_dir()

    model = load_and_setup_model(device, USE, LATENT_DIM, EPOCHS)

    if args.stage in {"simulate", "all"}:
        data = load_data_and_initial_curve(USE, IDX_CHOICE, device)
        X_tensor = data["X_tensor"]
        tenors = data["tenors"]
        S0 = data["S0"]
        meta_row = data["meta_row"]
        SCALE_IS_PERCENT = data["SCALE_IS_PERCENT"]

        print(f"SCALE_IS_PERCENT from my_data(): {SCALE_IS_PERCENT}")
        if meta_row is not None:
            print(f"Initial curve metadata row:\n{meta_row}")

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

        z_train_mean, z_train_cov, z_train_std = compute_latent_statistics(model, X_tensor, device, LATENT_DIM)
        diagnose_G0_on_training_cloud(model, X_tensor, device)

        with torch.no_grad():
            z0 = model.encoder(S0)
        print(f"Initial latent state z0: {z0.detach().cpu().numpy().flatten()}")

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

        discount_paths = compute_discount_paths(r_paths, dt=DT, method="trapezoid")
        print(
            f"Built path discount factors: D_t range = "
            f"[{discount_paths.min().item():.6f}, {discount_paths.max().item():.6f}]"
        )

        analyze_paths(z_paths, r_paths, mu_paths, L_paths, LATENT_DIM)

        g0_sim_df = diagnose_G0_on_simulated_paths(model, z_paths)
        g0_csv_path = os.path.join(out_dir, f"simulated_G0_diagnostics_{suffix}.csv")
        g0_sim_df.to_csv(g0_csv_path, index=False)
        print(f"Saved simulated G0 diagnostics to {g0_csv_path}")

        times = np.arange(N_STEPS + 1) * DT
        bundle_path = save_simulation_bundle(
            out_dir=out_dir,
            suffix=suffix,
            z_paths=z_paths,
            r_paths=r_paths,
            mu_paths=mu_paths,
            L_paths=L_paths,
            discount_paths=discount_paths,
            times=times,
            z_train_mean=z_train_mean,
            z_train_cov=z_train_cov,
            tenors=tenors,
            decoder_tau_grid=decoder_tau_grid,
            annual_indices=annual_indices,
        )

        # Plots from simulation stage
        plot_latent_2d_paths(z_paths, times, out_dir, suffix, max_paths=args.plot_n_paths, dpi=args.plot_dpi, show=args.show_plots)
        plot_latent_components(z_paths, times, out_dir, suffix, max_paths=args.plot_n_paths, dpi=args.plot_dpi, show=args.show_plots)
        plot_short_rate_paths(r_paths, times, out_dir, suffix, max_paths=args.plot_n_paths, dpi=args.plot_dpi, show=args.show_plots)
        plot_discount_paths(discount_paths, times, out_dir, suffix, max_paths=args.plot_n_paths, dpi=args.plot_dpi, show=args.show_plots)
        plot_mu_summary(mu_paths, times, out_dir, suffix, dpi=args.plot_dpi, show=args.show_plots)
        plot_diffusion_summary(L_paths, times, out_dir, suffix, dpi=args.plot_dpi, show=args.show_plots)
        plot_diffusion_entries(L_paths, times, out_dir, suffix, dpi=args.plot_dpi, show=args.show_plots)
        plot_g0_diagnostics(g0_sim_df, times, out_dir, suffix, dpi=args.plot_dpi, show=args.show_plots)

        if args.stage == "simulate":
            print("\nDone.")
            return

    if args.stage in {"decode", "martingale"}:
        if not args.bundle_path:
            raise ValueError("For stage='decode' or 'martingale', provide --bundle_path")
        bundle_path = args.bundle_path
    elif args.stage == "all":
        bundle_path = os.path.join(out_dir, f"simulation_bundle_{suffix}.pt")
    else:
        bundle_path = None

    if args.stage in {"decode", "martingale", "all"}:
        bundle = load_simulation_bundle(bundle_path, device=device)
        z_paths = bundle["z_paths"]
        r_paths = bundle["r_paths"]
        discount_paths = bundle["discount_paths"]
        times = bundle["times"]
        z_train_mean = bundle["z_train_mean"]
        z_train_cov = bundle["z_train_cov"]
        tenors = bundle["tenors"]
        decoder_tau_grid = bundle["decoder_tau_grid"]
        annual_indices = bundle["annual_indices"]

    if args.stage in {"decode", "all"}:
        swap_df, latent_df, mahal_df, decoder_diag_df, decoder_failures_df, _ = decode_and_save_results_naive(
            model=model,
            z_paths=z_paths,
            r_paths=r_paths,
            z_train_mean=z_train_mean,
            z_train_cov=z_train_cov,
            decoder_tau_grid=decoder_tau_grid,
            annual_indices=annual_indices,
            device=device,
            times=times,
            tenors=tenors,
            use=USE,
            latent_dim=LATENT_DIM,
            epochs=EPOCHS,
            max_mahal=args.max_mahal,
            g0_floor=G0_FLOOR,
            suffix=suffix,
        )

        plot_decoder_diagnostics(decoder_diag_df, out_dir, suffix, dpi=args.plot_dpi, show=args.show_plots)
        plot_swap_curve_snapshots(
            swap_df,
            tenors,
            requested_times=parse_float_list(args.plot_curve_times),
            out_dir=out_dir,
            suffix=suffix,
            dpi=args.plot_dpi,
            show=args.show_plots,
        )
        plot_selected_tenor_timeseries(
            swap_df,
            selected_tenors=parse_float_list(args.plot_tenors),
            out_dir=out_dir,
            suffix=suffix,
            dpi=args.plot_dpi,
            show=args.show_plots,
        )
        plot_failure_reasons(decoder_failures_df, out_dir, suffix, dpi=args.plot_dpi, show=args.show_plots)

        if args.stage == "decode":
            print("\nDone.")
            return

    if args.stage in {"martingale", "all"}:
        maturity_dates = parse_float_list(args.martingale_dates)
        mart_df = martingale_diagnostics_naive(
            model=model,
            z_paths=z_paths,
            discount_paths=discount_paths,
            times=times,
            maturity_dates=maturity_dates,
            out_dir=out_dir,
            use=USE,
            latent_dim=LATENT_DIM,
            epochs=EPOCHS,
            n_paths=z_paths.shape[0],
            n_steps=z_paths.shape[1] - 1,
            g0_floor=G0_FLOOR,
            suffix=suffix,
            decoder_tau_grid=decoder_tau_grid,
            annual_indices=annual_indices,
            martingale_tol=args.martingale_tol,
        )
        plot_martingale_diagnostics(mart_df, out_dir, suffix, dpi=args.plot_dpi, show=args.show_plots)

    print("\nDone.")


if __name__ == "__main__":
    main()
