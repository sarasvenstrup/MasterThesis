
import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
import math
import warnings
import matplotlib.pyplot as plt

# Add current directory to path for imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- User option: show plots interactively? ---
SHOW_PLOTS = True  # Set to False to only save plots

from Code.load_swapdata import my_data
from Code.model.full_model import FullModel
from Code.utils.rates import par_swap_from_discount
from Code.model.sigma_matrix import L_from_sigmas_rhos
from Code.utils.ode import (
    d_tau_autograd_nodewise,
    grad_and_trace_cov_hess_G,
    paper_alpha_beta_gamma_trace,
    solve_AB,
)

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

def load_initial_curve(use, idx_choice, device):
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=use)
    X_tensor = X_tensor.float()

    if idx_choice < 0:
        idx_choice = X_tensor.shape[0] + idx_choice

    if idx_choice < 0 or idx_choice >= X_tensor.shape[0]:
        raise IndexError(f"idx_choice={idx_choice} out of bounds")

    S0 = X_tensor[idx_choice:idx_choice + 1].to(device)
    meta_row = meta.iloc[idx_choice] if hasattr(meta, "iloc") else None
    return S0, meta_row, X_tensor, meta

def simulate_latent_paths(model, z0, n_paths, n_steps, dt, device, simple_diffusion=False,
                          kappa=0.5, theta=0.0, sigma_simple=0.1,
                          discretization="euler"):

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
        if simple_diffusion:
            # Additive-noise OU: Milstein corrections are zero, so all schemes coincide with Euler here.
            dW = torch.randn(n_paths, d, device=device, dtype=z.dtype) * sqrt_dt
            z = z + kappa * (theta - z) * dt + sigma_simple * dW
        else:
            mu = get_mu(model, z)

            if discretization == "euler":
                B = get_L(model, z)
                dW = torch.randn(n_paths, B.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
                shock = torch.bmm(B, dW.unsqueeze(-1)).squeeze(-1)
                z = z + mu * dt + shock

            elif discretization == "milstein":
                B, jac_B = _finite_diff_diffusion_jacobian(model, z)
                dW = torch.randn(n_paths, B.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
                shock = torch.bmm(B, dW.unsqueeze(-1)).squeeze(-1)
                corr = _milstein_correction(B, jac_B, dW, dt)
                z = z + mu * dt + shock + corr

            else:  # second_order_milstein (predictor-corrector Milstein approximation)
                B0, jac_B0 = _finite_diff_diffusion_jacobian(model, z)
                dW = torch.randn(n_paths, B0.shape[-1], device=device, dtype=z.dtype) * sqrt_dt
                shock0 = torch.bmm(B0, dW.unsqueeze(-1)).squeeze(-1)
                corr0 = _milstein_correction(B0, jac_B0, dW, dt)
                z_pred = z + mu * dt + shock0 + corr0

                mu_pred = get_mu(model, z_pred)
                B1, jac_B1 = _finite_diff_diffusion_jacobian(model, z_pred)
                shock1 = torch.bmm(B1, dW.unsqueeze(-1)).squeeze(-1)
                corr1 = _milstein_correction(B1, jac_B1, dW, dt)

                z = z + 0.5 * (mu + mu_pred) * dt + 0.5 * (shock0 + shock1) + 0.5 * (corr0 + corr1)

        if not torch.isfinite(z).all():
            raise RuntimeError(f"Non-finite z encountered at step {t+1}")

        z_paths[:, t + 1, :] = z
        r_paths[:, t + 1] = get_r(model, z)
        mu_paths[:, t + 1, :] = get_mu(model, z)
        L_paths[:, t + 1, :, :] = get_L(model, z)

    return z_paths, r_paths, mu_paths, L_paths

def decode_from_latent_script(model, z):
    squeeze_back = False

    if z.dim() == 1:
        z = z.unsqueeze(0)
        squeeze_back = True

    device = z.device
    dtype = z.dtype

    tau = torch.arange(0, model.tau_max + 1, device=device, dtype=dtype)

    G_vals = model.G(z, tau)
    if G_vals.dim() == 1:
        G_vals = G_vals.unsqueeze(0)

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

    A_vals, B_vals = solve_AB(tau, alpha, beta, gamma)

    P_full = torch.exp(A_vals - B_vals * G_vals)
    P_mkt = P_full[:, 1:]

    return P_mkt, A_vals, B_vals, G_vals, mu, sigma, r_tilde, None

def main():
    parser = argparse.ArgumentParser(description="Simulate swap curves from trained FullModel")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--use", type=str, default="bbg", help="Data source")
    parser.add_argument("--n_paths", type=int, default=100, help="Number of simulation paths")
    parser.add_argument("--n_steps", type=int, default=120, help="Number of time steps (e.g., 120 for 10 years monthly)")
    parser.add_argument("--dt", type=float, default=1/12, help="Time step size")
    parser.add_argument("--idx_choice", type=int, default=-1, help="Index of initial curve (-1 for latest)")
    parser.add_argument("--simple_diffusion", action="store_true", help="Use simple OU diffusion for z instead of model dynamics")
    parser.add_argument("--kappa", type=float, default=0.5, help="Mean reversion speed for simple diffusion")
    parser.add_argument("--theta", type=float, default=0.0, help="Long-run mean for simple diffusion")
    parser.add_argument("--sigma_simple", type=float, default=0.1, help="Volatility for simple diffusion")
    parser.add_argument(
        "--discretization",
        type=str,
        default="euler",
        choices=["euler", "milstein", "second_order_milstein"],
        help="Discretization scheme for latent SDE",
    )
    args = parser.parse_args()

    LATENT_DIM = args.latent_dim
    EPOCHS = args.epochs
    USE = args.use
    N_PATHS = args.n_paths
    N_STEPS = args.n_steps
    DT = args.dt
    IDX_CHOICE = args.idx_choice
    SIMPLE_DIFFUSION = args.simple_diffusion
    KAPPA = args.kappa
    THETA = args.theta
    SIGMA_SIMPLE = args.sigma_simple
    DISCRETIZATION = args.discretization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data to get tenors
    meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)

    # Load trained model
    checkpoint_path = os.path.join(REPO_ROOT, "..", "checkpoints", f"fullmodel_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FullModel(latent_dim=checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # Load initial curve
    S0, meta_row, X_tensor, meta = load_initial_curve(USE, IDX_CHOICE, device)
    with torch.no_grad():
        z0 = model.encoder(S0)
    print(f"Initial latent state z0: {z0.cpu().numpy().flatten()}")

    # Simulate latent paths
    print(f"Simulating {N_PATHS} paths with {N_STEPS} steps (dt={DT}, scheme={DISCRETIZATION})...")
    z_paths, r_paths, mu_paths, L_paths = simulate_latent_paths(
        model=model,
        z0=z0,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        dt=DT,
        device=device,
        simple_diffusion=SIMPLE_DIFFUSION,
        kappa=KAPPA,
        theta=THETA,
        sigma_simple=SIGMA_SIMPLE,
        discretization=DISCRETIZATION,
    )
    print("Simulation completed.")

    # ===== DIAGNOSTIC: Analyze mu and L =====
    print("\n" + "="*60)
    print("DIAGNOSTIC: Analyzing mu and L")
    print("="*60)
    
    # Convert to numpy for analysis
    mu_np = mu_paths.cpu().numpy()
    L_np = L_paths.cpu().numpy()
    z_np = z_paths.cpu().numpy()
    
    # Statistics for mu
    print("\n--- MU (Drift) Statistics ---")
    for d in range(LATENT_DIM):
        mu_d = mu_np[:, :, d]
        print(f"mu[{d}]: mean={mu_d.mean():.6f}, std={mu_d.std():.6f}, min={mu_d.min():.6f}, max={mu_d.max():.6f}")
    
    # Statistics for L
    print("\n--- L (Diffusion) Statistics ---")
    for i in range(LATENT_DIM):
        for j in range(LATENT_DIM):
            L_ij = L_np[:, :, i, j]
            print(f"L[{i},{j}]: mean={L_ij.mean():.6f}, std={L_ij.std():.6f}, min={L_ij.min():.6f}, max={L_ij.max():.6f}")
    
    # Check mu variance across paths and time
    print("\n--- MU Variance Analysis ---")
    mu_var_time = mu_np.var(axis=0)  # Variance across paths at each time
    mu_var_path = mu_np.var(axis=1)  # Variance across time for each path
    print(f"Mean variance of mu across paths at each time step: {mu_var_time.mean():.6e}")
    print(f"Mean variance of mu across time for each path: {mu_var_path.mean():.6e}")
    
    # Check L norm along paths
    print("\n--- L Frobenius Norm Analysis ---")
    L_norms = np.linalg.norm(L_np, axis=(2, 3))  # Frobenius norm for each (path, time)
    print(f"L Frobenius norm: mean={L_norms.mean():.6f}, std={L_norms.std():.6f}, min={L_norms.min():.6f}, max={L_norms.max():.6f}")
    
    # Sample values at initial and final time steps
    print("\n--- Sample mu values at t=0 (first 3 paths) ---")
    for p in range(min(3, N_PATHS)):
        print(f"Path {p}: mu = {mu_np[p, 0, :]}")
    
    print("\n--- Sample mu values at final time (first 3 paths) ---")
    for p in range(min(3, N_PATHS)):
        print(f"Path {p}: mu = {mu_np[p, -1, :]}")
    
    print("\n--- Sample L eigenvalues at t=0 (first 3 paths) ---")
    for p in range(min(3, N_PATHS)):
        L_matrix = L_np[p, 0, :, :]
        eigvals = np.linalg.eigvals(L_matrix)
        print(f"Path {p}: L eigenvalues = {eigvals}, L matrix:\n{L_matrix}")
    
    # Check if mu is essentially constant (should vary if diffusion is working)
    mu_range = mu_np.max() - mu_np.min()
    print(f"\n--- MU Range Check ---")
    print(f"Overall mu range (max-min): {mu_range:.6e}")
    print(f"Is mu nearly constant? {mu_range < 1e-4}")
    
    # Check correlation between z and mu
    print("\n--- Z-mu Correlation Analysis ---")
    for d in range(LATENT_DIM):
        z_d_flat = z_np[:, :, d].flatten()
        mu_d_flat = mu_np[:, :, d].flatten()
        corr = np.corrcoef(z_d_flat, mu_d_flat)[0, 1]
        print(f"Correlation between z[{d}] and mu[{d}]: {corr:.6f}")
    
    print("="*60 + "\n")

    # Prepare output directory
    out_dir = os.path.join(REPO_ROOT, "..", "Figures", "simulations")
    os.makedirs(out_dir, exist_ok=True)

    # Plot mu and L for first 5 paths
    print("Plotting mu and L for first 5 paths...")
    n_plot = min(5, N_PATHS)
    times = np.arange(N_STEPS + 1) * DT
    # Plot mu
    fig_mu, axes_mu = plt.subplots(LATENT_DIM, 1, figsize=(10, 6), sharex=True)
    if LATENT_DIM == 1:
        axes_mu = [axes_mu]
    for d in range(LATENT_DIM):
        ax = axes_mu[d]
        for p in range(n_plot):
            ax.plot(times, mu_paths[p, :, d].cpu().numpy(), alpha=0.7)
        ax.set_ylabel(f'mu{d}')
        ax.grid(True, alpha=0.3)
    axes_mu[-1].set_xlabel('Time (years)')
    fig_mu.suptitle('Drift (mu) along simulated paths')
    plt.tight_layout()
    plt.show()
    mu_plot_path = os.path.join(out_dir, f"simulated_mu_bbg_dim{LATENT_DIM}_ep{EPOCHS}_paths{N_PATHS}_steps{N_STEPS}.png")
    fig_mu.savefig(mu_plot_path, dpi=300)
    plt.close(fig_mu)
    print(f"Saved mu plot to {mu_plot_path}")

    # Plot L
    fig_L, axes_L = plt.subplots(LATENT_DIM, LATENT_DIM, figsize=(12, 8), sharex=True)
    if LATENT_DIM == 1:
        axes_L = np.array([[axes_L]])
    for i in range(LATENT_DIM):
        for j in range(LATENT_DIM):
            ax = axes_L[i, j]
            for p in range(n_plot):
                ax.plot(times, L_paths[p, :, i, j].cpu().numpy(), alpha=0.7)
            ax.set_ylabel(f'L[{i},{j}]')
            ax.grid(True, alpha=0.3)
    for ax in axes_L[-1, :]:
        ax.set_xlabel('Time (years)')
    fig_L.suptitle('Diffusion matrix (L) elements along simulated paths')
    plt.tight_layout()
    plt.show()
    L_plot_path = os.path.join(out_dir, f"simulated_L_bbg_dim{LATENT_DIM}_ep{EPOCHS}_paths{N_PATHS}_steps{N_STEPS}.png")
    fig_L.savefig(L_plot_path, dpi=300)
    plt.close(fig_L)
    print(f"Saved L plot to {L_plot_path}")

    # Prepare output directory
    out_dir = os.path.join(REPO_ROOT, "..", "Figures", "simulations")
    os.makedirs(out_dir, exist_ok=True)

    # Decode and collect swap curves
    times = np.arange(N_STEPS + 1) * DT
    swap_df_list = []
    latent_df_list = []

    print("Decoding simulated curves...")
    for t in range(N_STEPS + 1):
        z_t = z_paths[:, t, :]
        P_mkt, _, _, _, _, _, _, _ = decode_from_latent_script(model, z_t)
        S_sim = par_swap_from_discount(P_mkt, tenors)
        
        for p in range(N_PATHS):
            # Swap curves
            row = {'time': times[t], 'path_id': p}
            for i, ten in enumerate(tenors):
                row[f'swap_{ten}Y'] = S_sim[p, i].item()
            swap_df_list.append(row)
            
            # Latent and short rate
            latent_row = {'time': times[t], 'path_id': p, 'r': r_paths[p, t].item()}
            for d in range(LATENT_DIM):
                latent_row[f'z{d}'] = z_paths[p, t, d].item()
            latent_df_list.append(latent_row)

    # Save to CSV
    swap_df = pd.DataFrame(swap_df_list)
    swap_csv_path = os.path.join(out_dir, f"simulated_swap_curves_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_paths{N_PATHS}_steps{N_STEPS}.csv")
    swap_df.to_csv(swap_csv_path, index=False)
    print(f"Saved simulated swap curves to {swap_csv_path}")

    latent_df = pd.DataFrame(latent_df_list)
    latent_csv_path = os.path.join(out_dir, f"simulated_latent_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_paths{N_PATHS}_steps{N_STEPS}.csv")
    latent_df.to_csv(latent_csv_path, index=False)
    print(f"Saved simulated latent paths to {latent_csv_path}")

    print("Simulation and saving completed successfully.")

    # Plotting
    print("Generating plots...")

    # Plot latent paths and short rate
    fig, axes = plt.subplots(LATENT_DIM + 1, 1, figsize=(10, 6), sharex=True)
    if LATENT_DIM == 1:
        axes = [axes]

    for d in range(LATENT_DIM):
        ax = axes[d]
        for p in range(min(10, N_PATHS)):  # plot first 10 paths
            ax.plot(times, z_paths[p, :, d].cpu().numpy(), alpha=0.7)
        ax.set_ylabel(f'z{d}')
        ax.grid(True, alpha=0.3)

    ax = axes[-1]
    for p in range(min(10, N_PATHS)):
        ax.plot(times, r_paths[p, :].cpu().numpy(), alpha=0.7)
    ax.set_ylabel('r')
    ax.set_xlabel('Time (years)')
    fig.suptitle('Simulated Latent Paths and Short Rate')
    plt.tight_layout()
    latent_plot_path = os.path.join(out_dir, f"simulated_latent_paths_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_paths{N_PATHS}_steps{N_STEPS}.png")
    fig.savefig(latent_plot_path, dpi=300)
    print(f"Saved latent paths plot to {latent_plot_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # Plot mean swap rates over time for selected tenors
    fig, ax = plt.subplots(figsize=(10, 6))
    tenors_to_plot = [1.0, 5.0, 10.0, 30.0]
    colors = ['blue', 'green', 'red', 'orange']
    for i, ten in enumerate(tenors_to_plot):
        if float(ten) in tenors:
            ten_col = f'swap_{ten}Y'
            mean_curve = swap_df.groupby('time')[ten_col].mean()
            ax.plot(mean_curve.index, mean_curve.values, label=f'{int(ten)}Y', color=colors[i], linewidth=2)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Swap Rate')
    ax.set_title('Mean Simulated Swap Rates Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    swap_plot_path = os.path.join(out_dir, f"simulated_swap_rates_{USE}_dim{LATENT_DIM}_ep{EPOCHS}_paths{N_PATHS}_steps{N_STEPS}.png")
    fig.savefig(swap_plot_path, dpi=300)
    print(f"Saved swap rates plot to {swap_plot_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    print("Plotting completed.")

if __name__ == "__main__":
    main()

