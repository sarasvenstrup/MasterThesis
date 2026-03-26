import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
import math
import matplotlib.pyplot as plt

# Add current directory to path for imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.load_swapdata import my_data
from Code.model.full_model import FullModel
from Code.utils.rates import par_swap_from_discount
from Code.utils.sigma_matrix import L_from_sigmas_rhos
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

def simulate_latent_paths(model, z0, n_paths, n_steps, dt, device, simple_diffusion=False, kappa=0.5, theta=0.0, sigma_simple=0.1):
    if z0.dim() != 2 or z0.shape[0] != 1:
        raise ValueError(f"Expected z0 shape (1,d), got {tuple(z0.shape)}")

    d = z0.shape[1]
    sqrt_dt = math.sqrt(dt)

    z = z0.repeat(n_paths, 1).to(device)

    z_paths = torch.empty((n_paths, n_steps + 1, d), device=device, dtype=z.dtype)
    r_paths = torch.empty((n_paths, n_steps + 1), device=device, dtype=z.dtype)

    z_paths[:, 0, :] = z
    r_paths[:, 0] = get_r(model, z)

    for t in range(n_steps):
        mu = get_mu(model, z)
        L = get_L(model, z)

        if simple_diffusion:
            # Simple OU diffusion
            eps = torch.randn(n_paths, d, device=device, dtype=z.dtype)
            shock = sigma_simple * eps
            z = theta + (z - theta) * torch.exp(-kappa * dt) + shock * sqrt_dt
        else:
            eps = torch.randn(n_paths, d, device=device, dtype=z.dtype)
            shock = torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
            z = z + mu * dt + shock * sqrt_dt

        if not torch.isfinite(z).all():
            raise RuntimeError(f"Non-finite z encountered at step {t+1}")

        z_paths[:, t + 1, :] = z
        r_paths[:, t + 1] = get_r(model, z)

    return z_paths, r_paths

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
    print(f"Simulating {N_PATHS} paths with {N_STEPS} steps (dt={DT})...")
    with torch.no_grad():
        z_paths, r_paths = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            dt=DT,
            device=device,
            simple_diffusion=SIMPLE_DIFFUSION,
            kappa=KAPPA,
            theta=THETA,
            sigma_simple=SIGMA_SIMPLE
        )
    print("Simulation completed.")

    # Prepare output directory
    out_dir = os.path.join(REPO_ROOT, "..", "Figures", "simulations")
    os.makedirs(out_dir, exist_ok=True)

    # Decode and collect swap curves
    times = np.arange(0, (N_STEPS + 1) * DT, DT)
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
    plt.show()
    plt.close(fig)
    print(f"Saved latent paths plot to {latent_plot_path}")

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
    plt.show()
    plt.close(fig)
    print(f"Saved swap rates plot to {swap_plot_path}")

    print("Plotting completed.")

if __name__ == "__main__":
    main()

