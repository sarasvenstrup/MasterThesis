import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar

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

def price_cap(r_paths, dt, strike, notional=1.0):
    """
    Price a cap on the short rate.
    Assumes annual cap with quarterly resets (simplified).
    Cap rate: strike
    Notional: notional
    """
    n_paths, n_steps1 = r_paths.shape
    n_steps = n_steps1 - 1  # number of intervals

    # Assume cap resets every dt (e.g., monthly)
    # Payoff at each reset: max(r(t) - strike, 0) * dt * notional
    # Discounted back to t=0

    cap_values = torch.zeros(n_paths, device=r_paths.device, dtype=r_paths.dtype)

    for t in range(1, n_steps1):  # start from t=1
        time_to_reset = t * dt
        r_at_reset = r_paths[:, t]
        payoff = torch.clamp(r_at_reset - strike, min=0.0) * dt * notional
        # Discount factor approximation: exp(-integral r du) ≈ exp(-r * time_to_reset)
        # For simplicity, use the short rate at reset for discounting
        df = torch.exp(-r_at_reset * time_to_reset)
        cap_values += payoff * df

    cap_price = cap_values.mean().item()
    return cap_price

def price_swaption(z_paths, r_paths, model, dt, strike, expiry, tenor, notional=1.0):
    """
    Price a European swaption (call on swap rate).
    At expiry, compute swap rate, payoff max(S - strike, 0) * annuity, discounted back.
    """
    n_paths = z_paths.shape[0]
    n_steps = z_paths.shape[1] - 1
    total_time = n_steps * dt
    
    # Calculate index at expiry
    if expiry > total_time:
        print(f"Warning: expiry {expiry} > total simulated time {total_time}, using last step")
        expiry_idx = n_steps
    else:
        expiry_idx = min(int(round(expiry / dt)), n_steps)

    swaption_values = []

    for p in range(n_paths):
        z_at_expiry = z_paths[p, expiry_idx, :]
        
        # Decode to get discount factors and swap rate at expiry
        with torch.no_grad():
            P_mkt, _, _, _, _, _, _, _ = decode_from_latent_script(model, z_at_expiry.unsqueeze(0))
            S_all = par_swap_from_discount(P_mkt, model.tenors)
        
        # Use 5Y tenor for swap rate (index 3 in model.tenors which is [1,2,3,5,10,15,20,30])
        swap_idx = min(3, P_mkt.shape[1] - 1)
        S_at_expiry = S_all[0, swap_idx].item()
        
        # Compute annuity as sum of discount factors from expiry onwards
        # For simplicity: sum of first 'tenor' discount factors
        annuity = 0.0
        for tau_idx in range(min(tenor, P_mkt.shape[1])):
            annuity += P_mkt[0, tau_idx].item()
        
        # Swaption payoff at expiry
        payoff_at_expiry = max(S_at_expiry - strike, 0.0) * annuity * notional
        
        # Discount back to t=0 using the simulated short rates
        # Simple approximation: discount using average short rate
        avg_rate = r_paths[p, :expiry_idx+1].mean().item() if expiry_idx > 0 else r_paths[p, 0].item()
        df_to_expiry = np.exp(-avg_rate * expiry) if np.isfinite(avg_rate) else 1.0
        
        swaption_value = payoff_at_expiry * df_to_expiry
        
        if np.isfinite(swaption_value):
            swaption_values.append(swaption_value)

    if len(swaption_values) == 0:
        print("Warning: All swaption values are non-finite!")
        return 0.0
    
    swaption_price = np.mean(swaption_values)
    return swaption_price

def bachelier_price(forward, strike, sigma, expiry, annuity, notional, is_call=True):
    """
    Bachelier model for swaption pricing (normal volatility).
    Price = annuity * notional * [-(K-F) * N(-d) + sigma * sqrt(T) * n(d)]
    where d = (F - K) / (sigma * sqrt(T))
    """
    if sigma <= 0 or expiry <= 0:
        return 0.0
    
    intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    
    if sigma * np.sqrt(expiry) < 1e-10:
        return annuity * notional * intrinsic
    
    d = (forward - strike) / (sigma * np.sqrt(expiry))
    
    if is_call:
        price = annuity * notional * ((forward - strike) * norm.cdf(d) + sigma * np.sqrt(expiry) * norm.pdf(d))
    else:
        price = annuity * notional * ((strike - forward) * norm.cdf(-d) + sigma * np.sqrt(expiry) * norm.pdf(d))
    
    return price

def implied_normal_vol(market_price, forward, strike, expiry, annuity, notional, is_call=True, initial_guess=0.01):
    """
    Calculate implied normal volatility from market price using Bachelier model.
    """
    def objective(sigma):
        theoretical = bachelier_price(forward, strike, sigma, expiry, annuity, notional, is_call)
        return (theoretical - market_price) ** 2
    
    result = minimize_scalar(objective, bounds=(1e-6, 1.0), method='bounded')
    
    return result.x if result.success else np.nan

def extract_market_params_at_expiry(z_paths, model, device, dt, expiry, tenor):
    """
    Extract forward swap rate and annuity at swaption expiry from model.
    
    Args:
        z_paths: Simulated latent paths (n_paths, n_steps+1, latent_dim)
        model: FullModel instance
        device: torch device
        dt: time step size
        expiry: expiry time
        tenor: swaption tenor
    
    Returns:
        forward_swap, annuity, z_at_expiry (dict with path-averaged values)
    """
    n_paths = z_paths.shape[0]
    n_steps = z_paths.shape[1] - 1
    expiry_idx = min(int(round(expiry / dt)), n_steps)
    
    forward_swaps = []
    annuities = []
    
    for p in range(n_paths):
        z_at_expiry = z_paths[p, expiry_idx, :]
        
        with torch.no_grad():
            P_mkt, _, _, _, _, _, _, _ = decode_from_latent_script(model, z_at_expiry.unsqueeze(0))
            S_all = par_swap_from_discount(P_mkt, model.tenors)
        
        swap_idx = min(3, P_mkt.shape[1] - 1)
        forward_swap = S_all[0, swap_idx].item()
        forward_swaps.append(forward_swap)
        
        annuity = sum(P_mkt[0, tau_idx].item() for tau_idx in range(min(tenor, P_mkt.shape[1])))
        annuities.append(annuity)
    
    avg_forward = np.mean(forward_swaps)
    avg_annuity = np.mean(annuities)
    
    return {
        'forward_swap': avg_forward,
        'annuity': avg_annuity,
        'forward_swaps': forward_swaps,
        'annuities': annuities
    }

def price_swaption_from_norm_vol(forward, strike, norm_vol, expiry, annuity, notional=1.0, is_call=True):
    """
    Price a swaption using normal (Bachelier) model with quoted normal volatility.
    
    Args:
        forward: Forward swap rate
        strike: Strike rate
        norm_vol: Normal volatility (in decimal, e.g., 0.01 for 1%)
        expiry: Time to expiry
        annuity: Annuity factor (sum of discount factors)
        notional: Notional amount
        is_call: True for call (payer swaption), False for put (receiver swaption)
    
    Returns:
        Price of the swaption
    """
    return bachelier_price(forward, strike, norm_vol, expiry, annuity, notional, is_call)

def main():
    parser = argparse.ArgumentParser(description="Price options using simulated paths from FullModel")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--use", type=str, default="bbg", help="Data source")
    parser.add_argument("--n_paths", type=int, default=1000, help="Number of simulation paths")
    parser.add_argument("--n_steps", type=int, default=120, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=1/12, help="Time step size")
    parser.add_argument("--idx_choice", type=int, default=-1, help="Index of initial curve")
    parser.add_argument("--option_type", type=str, choices=["cap", "swaption"], default="cap", help="Type of option to price")
    parser.add_argument("--strike", type=float, default=0.03, help="Strike rate")
    parser.add_argument("--notional", type=float, default=1.0, help="Notional amount")
    parser.add_argument("--simple_diffusion", action="store_true", help="Use simple OU diffusion")
    parser.add_argument("--kappa", type=float, default=0.5, help="Mean reversion for simple diffusion")
    parser.add_argument("--theta", type=float, default=0.0, help="Long-run mean for simple diffusion")
    parser.add_argument("--sigma_simple", type=float, default=0.1, help="Volatility for simple diffusion")
    parser.add_argument("--expiry", type=float, default=1.0, help="Expiry for swaption")
    parser.add_argument("--tenor", type=int, default=5, help="Tenor for swaption")
    parser.add_argument("--output_norm_vol", action="store_true", help="Output implied normal volatility for swaptions")
    parser.add_argument("--norm_vol", type=float, default=None, help="Input normal volatility for direct swaption pricing (in decimal, e.g., 0.01 for 1%)")
    parser.add_argument("--pricing_mode", type=str, choices=["monte_carlo", "norm_vol_quote"], default="monte_carlo", 
                       help="Pricing mode: 'monte_carlo' for simulation-based, 'norm_vol_quote' for direct Bachelier pricing from norm vol")
    parser.add_argument("--is_receiver", action="store_true", help="Price receiver swaption (put) instead of payer swaption (call)")

    args = parser.parse_args()

    LATENT_DIM = args.latent_dim
    EPOCHS = args.epochs
    USE = args.use
    N_PATHS = args.n_paths
    N_STEPS = args.n_steps
    DT = args.dt
    IDX_CHOICE = args.idx_choice
    OPTION_TYPE = args.option_type
    STRIKE = args.strike
    NOTIONAL = args.notional
    SIMPLE_DIFFUSION = args.simple_diffusion
    KAPPA = args.kappa
    THETA = args.theta
    SIGMA_SIMPLE = args.sigma_simple
    EXPIRY = args.expiry
    TENOR = args.tenor
    OUTPUT_NORM_VOL = args.output_norm_vol

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

    # Determine call/put direction
    IS_CALL = not args.is_receiver
    PRICING_MODE = args.pricing_mode
    NORM_VOL_INPUT = args.norm_vol

    # Price the option
    if OPTION_TYPE == "cap":
        price = price_cap(r_paths, DT, STRIKE, NOTIONAL)
        print(f"Cap price: {price:.6f} (strike={STRIKE}, notional={NOTIONAL})")
    elif OPTION_TYPE == "swaption":
        if PRICING_MODE == "norm_vol_quote":
            # Price directly from normal volatility quote
            if NORM_VOL_INPUT is None:
                raise ValueError("--norm_vol required when using norm_vol_quote pricing mode")
            
            # Extract market parameters at expiry
            print("Extracting market parameters at swaption expiry...")
            market_params = extract_market_params_at_expiry(z_paths, model, device, DT, EXPIRY, TENOR)
            forward_swap = market_params['forward_swap']
            annuity = market_params['annuity']
            
            print(f"  Forward swap rate: {forward_swap:.6f}")
            print(f"  Annuity factor: {annuity:.6f}")
            print(f"  Input normal volatility: {NORM_VOL_INPUT:.6f} ({NORM_VOL_INPUT*10000:.2f} bp)")
            
            # Price swaption using Bachelier model
            price = price_swaption_from_norm_vol(
                forward=forward_swap,
                strike=STRIKE,
                norm_vol=NORM_VOL_INPUT,
                expiry=EXPIRY,
                annuity=annuity,
                notional=NOTIONAL,
                is_call=IS_CALL
            )
            swaption_type = "Payer" if IS_CALL else "Receiver"
            print(f"{swaption_type} swaption price (from norm vol): {price:.6f}")
            print(f"  (strike={STRIKE}, expiry={EXPIRY}, tenor={TENOR}, notional={NOTIONAL})")
            
        else:  # monte_carlo mode (default)
            # Price using Monte Carlo simulation
            price = price_swaption(z_paths, r_paths, model, DT, STRIKE, EXPIRY, TENOR, NOTIONAL)
            print(f"Swaption price (Monte Carlo): {price:.6f} (strike={STRIKE}, expiry={EXPIRY}, tenor={TENOR}, notional={NOTIONAL})")
            
            # If requested, compute implied normal volatility
            if OUTPUT_NORM_VOL or NORM_VOL_INPUT is not None:
                # Extract market parameters
                market_params = extract_market_params_at_expiry(z_paths, model, device, DT, EXPIRY, TENOR)
                forward_swap = market_params['forward_swap']
                annuity = market_params['annuity']
                
                # Calculate implied normal vol
                norm_vol = implied_normal_vol(
                    market_price=price,
                    forward=forward_swap,
                    strike=STRIKE,
                    expiry=EXPIRY,
                    annuity=annuity,
                    notional=NOTIONAL,
                    is_call=IS_CALL
                )
                
                if np.isfinite(norm_vol):
                    print(f"Implied normal volatility: {norm_vol:.6f} (basis points: {norm_vol*10000:.2f})")
                    
                    # If norm_vol was provided as input, compare
                    if NORM_VOL_INPUT is not None:
                        print(f"Input normal volatility: {NORM_VOL_INPUT:.6f} ({NORM_VOL_INPUT*10000:.2f} bp)")
                        print(f"Difference: {abs(norm_vol - NORM_VOL_INPUT):.6f} ({abs(norm_vol - NORM_VOL_INPUT)*10000:.2f} bp)")
                else:
                    print("Could not compute implied normal volatility (optimization failed)")

    print("Pricing completed.")

if __name__ == "__main__":
    main()

