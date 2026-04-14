import math
import os
import sys
import time
import pandas as pd

import numpy as np
import torch
import matplotlib.pyplot as plt

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

from Code import config
from Code.model.full_model import FullModel
from Code.load_swapdata import my_data
from Code.model.sigma_matrix import L_from_sigmas_rhos

# ==========================================================
# Checkpoint switch
# ==========================================================
checkpoint_path = r"C:\Users\Bruger\PycharmProjects\MasterThesis\Figures\TrainingResults\dim2_stable\ep200\checkpoint_dim2_ep200.pt"


def load_and_setup_model(device, checkpoint_path, latent_dim=2, use_double=True):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = FullModel(latent_dim=latent_dim).to(device)

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  [load] missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  [load] unexpected keys (dropped): {result.unexpected_keys}")

    if use_double:
        model = model.double()

    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    return model


@torch.no_grad()
def get_mu(model, z):
    return model.K(z)


@torch.no_grad()
def get_L(model, z):
    sigmas, rhos = model.H(z)
    return L_from_sigmas_rhos(sigmas, rhos, validate=False)


@torch.no_grad()
def get_r(model, z):
    r = model.R(z)
    if r.ndim == 2 and r.shape[-1] == 1:
        r = r.squeeze(-1)
    return r


def compute_latent_statistics(model, X_tensor, device, latent_dim):
    z_train_list = []

    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], 256):
            batch = X_tensor[i:min(i + 256, X_tensor.shape[0])].to(device=device, dtype=next(model.parameters()).dtype)
            z_batch = model.encoder(batch)
            z_train_list.append(z_batch)

    z_train = torch.cat(z_train_list, dim=0)
    z_train_mean = z_train.mean(dim=0).detach()
    z_train_cov = torch.cov(z_train.t()).detach()
    z_train_std = z_train.std(dim=0).detach()

    print("Training latent cloud mean:", z_train_mean.cpu().numpy())
    print("Training latent cloud std: ", z_train_std.cpu().numpy())
    for d in range(latent_dim):
        print(f"  z[{d}] range = [{z_train[:, d].min().item():.6f}, {z_train[:, d].max().item():.6f}]")

    return z_train_mean, z_train_cov, z_train_std


def simulate_latent_paths(
        model,
        z0,
        n_paths,
        n_steps,
        dt,
        device,
        diffusion_scale=1.0,
):
    if z0.dim() != 2 or z0.shape[0] != 1:
        raise ValueError(f"Expected z0 shape (1,d), got {tuple(z0.shape)}")

    if diffusion_scale < 0:
        raise ValueError("diffusion_scale must be non-negative")

    d = z0.shape[1]
    sqrt_dt = math.sqrt(dt)

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
        B = get_L(model, z)
        dW = torch.randn(n_paths, B.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
        shock = diffusion_scale * torch.bmm(B, dW.unsqueeze(-1)).squeeze(-1)
        drift = get_mu(model, z) * dt

        z = z + drift + shock

        z_paths[:, t + 1, :] = z
        r_paths[:, t + 1] = get_r(model, z)
        mu_paths[:, t + 1, :] = get_mu(model, z)
        L_paths[:, t + 1, :, :] = get_L(model, z)

    return z_paths, r_paths, mu_paths, L_paths


def compute_discount_paths(r_paths: torch.Tensor, dt: float) -> torch.Tensor:
    if dt <= 0:
        raise ValueError("dt must be positive")
    if r_paths.ndim != 2:
        raise ValueError(f"Expected r_paths to have shape (n_paths, n_steps+1), got {tuple(r_paths.shape)}")

    n_paths, n_times = r_paths.shape
    if n_times < 2:
        return torch.ones_like(r_paths)

    increments = 0.5 * (r_paths[:, :-1] + r_paths[:, 1:]) * dt

    int_r = torch.cumsum(increments, dim=1)
    # Clamp the integrated rate to [-30, 30] so exp never over/underflows.
    # This corresponds to discount factors in [exp(-30), exp(30)] ~ [1e-13, 1e13],
    # which is physically unreachable but prevents NaN propagation when simulated
    # paths stray outside the training support.
    int_r = int_r.clamp(min=-30.0, max=30.0)
    disc = torch.ones((n_paths, n_times), device=r_paths.device, dtype=r_paths.dtype)
    disc[:, 1:] = torch.exp(-int_r)
    return disc


def plot_simulation_results(results, n_paths_to_plot=20):
    """
    Plot the simulation results including latent paths, interest rates, and discount curves.

    Args:
        results: Dictionary returned from run_simulation
        n_paths_to_plot: Number of paths to plot (default: 20)
    """
    z_paths = results["z_paths"].cpu().numpy()
    r_paths = results["r_paths"].cpu().numpy()
    discount_paths = results["discount_paths"].cpu().numpy()
    P_full_paths = results["P_full_paths"].cpu().numpy()
    times = results["times"]
    z0 = results["z0"].cpu().numpy().flatten()
    z_train_mean = results["z_train_mean"].cpu().numpy()
    z_train_std = results["z_train_std"].cpu().numpy()

    n_paths_plot = min(n_paths_to_plot, z_paths.shape[0])

    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Latent state paths
    n_latent = z_paths.shape[2]
    for d in range(n_latent):
        ax = plt.subplot(2, 2, d + 1)
        for i in range(n_paths_plot):
            ax.plot(times, z_paths[i, :, d], alpha=0.5, linewidth=0.8)
        ax.axhline(z0[d], color='red', linestyle='--', linewidth=2, label='z0')
        ax.axhline(z_train_mean[d], color='green', linestyle='--', linewidth=2, label='Train mean')
        ax.fill_between(
            times,
            z_train_mean[d] - z_train_std[d],
            z_train_mean[d] + z_train_std[d],
            alpha=0.2,
            color='green',
            label='Train ±1 std'
        )
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(f"z[{d}]")
        ax.set_title(f"Latent State z[{d}]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot 2: Short rate paths
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n_paths_plot):
        ax.plot(times, r_paths[i, :] * 100, alpha=0.5, linewidth=0.8)
    ax.axhline(r_paths[0, 0] * 100, color='red', linestyle='--', linewidth=2, label='Initial r')
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Short Rate (%)")
    ax.set_title("Short Rate Paths")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 3: Discount factor paths
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n_paths_plot):
        ax.plot(times, discount_paths[i, :], alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Discount Factor")
    ax.set_title("Stochastic Discount Paths")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 4: Discount curve term structure at different times
    fig, ax = plt.subplots(figsize=(12, 6))
    tau_grid = results["tau_grid"].cpu().numpy()
    time_indices = np.linspace(0, P_full_paths.shape[1] - 1, min(5, P_full_paths.shape[1]), dtype=int)

    for t_idx in time_indices:
        time_val = times[t_idx]
        paths_at_t = P_full_paths[:n_paths_plot, t_idx, :]
        mean_at_t = paths_at_t.mean(axis=0)
        std_at_t = paths_at_t.std(axis=0)

        ax.plot(tau_grid, mean_at_t, marker='o', label=f't={time_val:.2f}', linewidth=2)
        ax.fill_between(
            tau_grid,
            mean_at_t - std_at_t,
            mean_at_t + std_at_t,
            alpha=0.2
        )

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Discount Factor")
    ax.set_title("Term Structure of Discount Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def resolve_curve_index(meta, as_of_date=0):
    if as_of_date == 0 or as_of_date is None:
        return 0

    if "as_of_date" not in meta.columns:
        raise KeyError("meta does not contain column 'as_of_date'")

    target_date = pd.Timestamp(as_of_date).normalize()
    meta_dates = pd.to_datetime(meta["as_of_date"]).dt.normalize()

    matches = np.where(meta_dates.values == target_date.to_datetime64())[0]

    if len(matches) == 0:
        raise ValueError(f"No row found for as_of_date={target_date.date()}")

    if len(matches) > 1:
        print(f"Found {len(matches)} rows for {target_date.date()}; using first match.")

    return int(matches[0])


def run_simulation(
        use="bbg",
        latent_dim=2,
        checkpoint_path=None,
        n_paths=500,
        n_steps=24,
        dt=1 / 12,
        as_of_date=None,
        ccy_filter="",
        diffusion_scale=1.0,
        seed=1234,
        device=None,
        show_plot=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Seed: {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_and_setup_model(
        device=device,
        checkpoint_path=checkpoint_path,
        latent_dim=latent_dim,
        use_double=True,
    )

    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(
        use=use,
        ccy_filter=ccy_filter,
    )

    model_dtype = next(model.parameters()).dtype
    X_tensor = X_tensor.to(dtype=model_dtype)
    X_tensor_full = X_tensor_full.to(dtype=model_dtype)

    print(f"SCALE_IS_PERCENT from my_data(): {SCALE_IS_PERCENT}")

    start_idx = resolve_curve_index(meta, as_of_date=as_of_date)

    S0 = X_tensor[start_idx:start_idx + 1].to(device=device, dtype=model_dtype)
    meta_row = meta.iloc[start_idx]
    print(f"Initial curve metadata row:\n{meta_row}")

    # Training latent statistics
    z_train_mean, z_train_cov, z_train_std = compute_latent_statistics(model, X_tensor, device, latent_dim)

    # Initial latent state
    with torch.no_grad():
        z0 = model.encoder(S0)

    print(f"Initial latent state z0: {z0.detach().cpu().numpy().flatten()}")

    # Decode initial curve directly from latent state using FullModel decoder
    with torch.no_grad():
        _, aux0 = model.decode_from_z(
            z0,
            tau=None,
            do_arb_checks=False,
            return_aux=True,
        )

    P_full_0 = aux0["P_full"]  # shape (1, tau_max+1)
    P_mkt_0 = aux0["P_mkt"]  # shape (1, tau_max)
    S_hat_0 = aux0["S_hat"]  # shape (1, len(tenors))
    tau_grid = aux0["tau_grid"]

    print(
        f"Initial decoded discount curve range: "
        f"[{P_full_0.min().item():.6f}, {P_full_0.max().item():.6f}]"
    )

    print(
        f"Simulating {n_paths} paths with {n_steps} steps "
    )

    t0 = time.time()
    z_paths, r_paths, mu_paths, L_paths = simulate_latent_paths(
        model=model,
        z0=z0,
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        device=device,
        diffusion_scale=diffusion_scale,
    )
    print(f"Simulation completed in {time.time() - t0:.2f}s.")

    discount_paths = compute_discount_paths(r_paths, dt=dt)
    print(
        f"Built path discount factors: D_t range = "
        f"[{discount_paths.min().item():.6f}, {discount_paths.max().item():.6f}]"
    )

    n_paths, n_times, d = z_paths.shape
    z_flat = z_paths.reshape(-1, d)

    with torch.no_grad():
        _, aux = model.decode_from_z(
            z_flat,
            tau=None,  # <- exact same grid as FullModel.forward()
            do_arb_checks=False,
            return_aux=True,
        )

    P_full_flat = aux["P_full"]  # shape: (n_paths*n_times, tau_max+1)
    P_mkt_flat = aux["P_mkt"]  # shape: (n_paths*n_times, tau_max)
    S_hat_flat = aux["S_hat"]  # shape: (n_paths*n_times, len(tenors)) or None
    tau_grid = aux["tau_grid"]

    P_full_paths = P_full_flat.reshape(n_paths, n_times, -1)
    P_mkt_paths = P_mkt_flat.reshape(n_paths, n_times, -1)

    if S_hat_flat is not None:
        S_hat_paths = S_hat_flat.reshape(n_paths, n_times, -1)
    else:
        S_hat_paths = None

    print(
        f"Decoded latent paths to discount curves: "
        f"P_full range = [{P_full_paths.min().item():.6f}, {P_full_paths.max().item():.6f}]"
    )

    times = np.arange(n_steps + 1) * dt

    # ==========================================================
    # Save simulation latent summary (mean + percentiles + subset)
    # ==========================================================
    z_np = z_paths.cpu().numpy()  # (n_paths, n_times, 2)

    # --- Summary statistics across paths, per timestep ---
    percentiles = [5, 25, 50, 75, 95]
    rows = []
    for t_idx, t_val in enumerate(times):
        z1 = z_np[:, t_idx, 0]
        z2 = z_np[:, t_idx, 1]
        row = {"time": t_val}
        for p in percentiles:
            row[f"z1_p{p}"] = np.percentile(z1, p)
            row[f"z2_p{p}"] = np.percentile(z2, p)
        row["z1_mean"] = z1.mean()
        row["z2_mean"] = z2.mean()
        rows.append(row)

    df_sim_summary = pd.DataFrame(rows)

    # --- Small subset of raw paths for trajectory plotting ---
    n_subset = min(50, n_paths)
    rng = np.random.default_rng(seed)
    subset_idx = rng.choice(n_paths, size=n_subset, replace=False)
    z_subset = z_np[subset_idx]  # (50, n_times, 2)

    path_ids = np.repeat(np.arange(n_subset), len(times))
    df_sim_subset = pd.DataFrame({
        "path": path_ids,
        "time": np.tile(times, n_subset),
        "z_1": z_subset[:, :, 0].flatten(),
        "z_2": z_subset[:, :, 1].flatten(),
    })

    sim_csv_dir = os.path.dirname(checkpoint_path)
    ccy_tag = ccy_filter if ccy_filter else "all"

    summary_path = os.path.join(sim_csv_dir, f"latent_sim_summary_{ccy_tag}_npaths{n_paths}_nsteps{n_steps}.csv")
    subset_path = os.path.join(sim_csv_dir, f"latent_sim_subset_{ccy_tag}_npaths{n_paths}_nsteps{n_steps}.csv")

    df_sim_summary.to_csv(summary_path, index=False)
    df_sim_subset.to_csv(subset_path, index=False)
    print("Saved simulation summary:", summary_path)
    print("Saved simulation subset: ", subset_path)

    annual_indices = list(range(1, P_full_paths.shape[-1]))  # tau = 1,2,...,tau_max
    bundle_path = None

    results_dict = {
        "model": model,
        "meta": meta,
        "meta_full": meta_full,
        "S0": S0,
        "meta_row": meta_row,
        "z0": z0,
        "z_paths": z_paths,
        "r_paths": r_paths,
        "mu_paths": mu_paths,
        "L_paths": L_paths,
        "discount_paths": discount_paths,  # pathwise money-market discounting
        "P_full_0": P_full_0,  # initial cross-sectional curve incl tau=0
        "P_mkt_0": P_mkt_0,  # initial cross-sectional curve at 1..tau_max
        "S_hat_0": S_hat_0,  # reconstructed swaps at t=0
        "P_full_paths": P_full_paths,  # decoded from simulated z_paths
        "P_mkt_paths": P_mkt_paths,  # decoded from simulated z_paths
        "S_hat_paths": S_hat_paths,  # swap curves decoded from simulated z_paths
        "times": times,
        "z_train_mean": z_train_mean,
        "z_train_cov": z_train_cov,
        "z_train_std": z_train_std,
        "tenors": tenors,
        "tau_grid": tau_grid,
        "annual_indices": annual_indices,
        "bundle_path": bundle_path,
    }

    if show_plot:
        plot_simulation_results(results_dict)

    return results_dict


# =============================================================================
# DIFFERENTIABLE SIMULATION TO EXPIRY
# =============================================================================

def simulate_to_expiry_differentiable(
    model,
    z0      : torch.Tensor,    # (1, d)  — initial latent state, detached
    n_steps : int,
    dt      : float,
    n_paths : int,
    eps     : torch.Tensor,    # (n_paths, n_steps, d) — PRE-DRAWN, no grad
    freeze_K: bool = True,     # If False, K.N grad flows; K.V always frozen
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable Euler-Maruyama simulation to a fixed horizon.

    Returns
    -------
    z_T : (n_paths, d)  — terminal latent state WITH grad w.r.t. H
                          (and K.N if freeze_K=False)
    D_T : (n_paths,)    — pathwise money-market discount factor, always
                          DETACHED (R is frozen throughout)

    The noise eps must be pre-drawn and passed in as a fixed tensor (no grad).
    This is the reparameterization trick: randomness is decoupled from the
    parameters so autograd flows cleanly through L(z_t ; H) at every step.

    Discount uses the trapezoid rule:
        log D_T = -Σ_t  ½ (r_t + r_{t+1}) · dt
    evaluated under torch.no_grad() via model.R — R is never trained here.

    When freeze_K=False:
        K.N has requires_grad=True → grad flows to K.N via the drift
        K.V always stays frozen   → negative-definiteness is preserved
    """
    sqrt_dt = math.sqrt(dt)

    z = z0.expand(n_paths, -1).clone()          # (n_paths, d)

    with torch.no_grad():
        r_prev = model.R(z).squeeze(-1)         # (n_paths,)

    log_D = torch.zeros(n_paths, device=z.device, dtype=z.dtype)

    for t in range(n_steps):
        # Volatility — gradient flows through H
        sigmas, rhos = model.H(z)
        L = L_from_sigmas_rhos(sigmas, rhos, validate=False)    # (n_paths, d, d)

        # Drift — freeze K entirely, or let K.N receive gradients
        if freeze_K:
            with torch.no_grad():
                mu = model.K(z)
            drift = mu.detach() * dt
        else:
            # K.V.requires_grad = False → no grad flows to K.V
            # K.N.requires_grad = True  → grad flows to K.N
            drift = model.K(z) * dt

        # Euler step with fixed noise
        dW    = eps[:, t, :] * sqrt_dt          # (n_paths, d)
        shock = torch.bmm(L, dW.unsqueeze(-1)).squeeze(-1)
        z     = z + drift + shock

        # Trapezoid discount (detached — R is frozen)
        with torch.no_grad():
            r_next = model.R(z.detach()).squeeze(-1)
            log_D  = log_D - 0.5 * (r_prev + r_next) * dt
            r_prev = r_next

    D_T = log_D.exp().detach()                  # (n_paths,)
    return z, D_T


if __name__ == "__main__":
    run_simulation(checkpoint_path=checkpoint_path, ccy_filter="EUR", show_plot=True)