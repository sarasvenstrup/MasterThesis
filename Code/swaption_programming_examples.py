"""
Programmatic Examples for Swaption Pricing with Normal Volatility

This module shows how to use the pricing functions directly in Python
without going through the command-line interface.
"""

import torch
import numpy as np
from Code.price_options import (
    load_initial_curve,
    simulate_latent_paths,
    extract_market_params_at_expiry,
    price_swaption_from_norm_vol,
    implied_normal_vol,
    decode_from_latent_script,
)
from Code.model.full_model import FullModel
from Code.load_swapdata import my_data
from Code.utils.rates import par_swap_from_discount
import os


def example_1_direct_norm_vol_pricing():
    """
    Example 1: Price a swaption directly from a normal volatility quote.
    
    Workflow:
    1. Load model and initial curve
    2. Simulate a few paths to extract market parameters at expiry
    3. Use Bachelier model to price from normal volatility quote
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Direct Norm Vol Quote Pricing")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    LATENT_DIM = 2
    EPOCHS = 100
    checkpoint_path = f"../checkpoints/fullmodel_bbg_dim{LATENT_DIM}_ep{EPOCHS}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FullModel(latent_dim=checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load initial curve
    S0, meta_row, X_tensor, meta = load_initial_curve("bbg", -1, device)
    with torch.no_grad():
        z0 = model.encoder(S0)
    
    # Simulate paths
    print("Simulating 100 paths...")
    with torch.no_grad():
        z_paths, r_paths = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=100,
            n_steps=120,
            dt=1/12,
            device=device,
            simple_diffusion=False,
        )
    
    # Extract market parameters at expiry
    market_params = extract_market_params_at_expiry(
        z_paths=z_paths,
        model=model,
        device=device,
        dt=1/12,
        expiry=1.0,
        tenor=5
    )
    
    forward = market_params['forward_swap']
    annuity = market_params['annuity']
    
    print(f"\nMarket parameters at 1Y expiry:")
    print(f"  Forward swap rate (5Y): {forward:.6f}")
    print(f"  Annuity factor: {annuity:.6f}")
    
    # Price swaptions at different strikes with 50 bp normal volatility
    norm_vol = 0.005  # 50 basis points
    strikes = [0.02, 0.025, 0.03, 0.035, 0.04]
    
    print(f"\nSwaption prices with {norm_vol*10000:.0f} bp normal volatility:")
    print("Strike  | Payer Price | Receiver Price")
    print("--------|-------------|----------------")
    
    for strike in strikes:
        payer_price = price_swaption_from_norm_vol(
            forward=forward,
            strike=strike,
            norm_vol=norm_vol,
            expiry=1.0,
            annuity=annuity,
            notional=1.0,
            is_call=True
        )
        
        receiver_price = price_swaption_from_norm_vol(
            forward=forward,
            strike=strike,
            norm_vol=norm_vol,
            expiry=1.0,
            annuity=annuity,
            notional=1.0,
            is_call=False
        )
        
        print(f"{strike:6.3f}  | {payer_price:11.6f} | {receiver_price:14.6f}")


def example_2_implied_norm_vol_from_monte_carlo():
    """
    Example 2: Generate Monte Carlo price and derive implied normal volatility.
    
    Workflow:
    1. Load model and initial curve
    2. Simulate many paths
    3. Price a swaption via Monte Carlo
    4. Calculate implied normal volatility
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Implied Normal Volatility from Monte Carlo")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    LATENT_DIM = 2
    EPOCHS = 100
    checkpoint_path = f"../checkpoints/fullmodel_bbg_dim{LATENT_DIM}_ep{EPOCHS}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FullModel(latent_dim=checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load initial curve
    S0, meta_row, X_tensor, meta = load_initial_curve("bbg", -1, device)
    with torch.no_grad():
        z0 = model.encoder(S0)
    
    # Simulate many paths
    print("Simulating 1000 paths...")
    with torch.no_grad():
        z_paths, r_paths = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=1000,
            n_steps=120,
            dt=1/12,
            device=device,
            simple_diffusion=False,
        )
    
    # Extract market parameters
    market_params = extract_market_params_at_expiry(
        z_paths=z_paths,
        model=model,
        device=device,
        dt=1/12,
        expiry=1.0,
        tenor=5
    )
    
    forward = market_params['forward_swap']
    annuity = market_params['annuity']
    
    print(f"\nForward swap: {forward:.6f}")
    print(f"Annuity: {annuity:.6f}")
    
    # Example: Price a swaption at strike = forward rate (ATM)
    strike = forward
    
    # Simulate a Monte Carlo price (simple example using payout)
    # In practice, you'd use the full price_swaption function
    mc_price = 0.002  # Placeholder - in real use, compute from full simulation
    
    # Derive implied normal volatility
    implied_vol = implied_normal_vol(
        market_price=mc_price,
        forward=forward,
        strike=strike,
        expiry=1.0,
        annuity=annuity,
        notional=1.0,
        is_call=True
    )
    
    print(f"\nAt-the-money (ATM) swaption:")
    print(f"  Strike: {strike:.6f}")
    print(f"  Monte Carlo Price: {mc_price:.6f}")
    print(f"  Implied Normal Vol: {implied_vol:.6f} ({implied_vol*10000:.2f} bp)")


def example_3_term_structure_of_norm_vol():
    """
    Example 3: Compute swaption prices for different expiries.
    
    Shows how pricing varies with time to expiry while keeping
    other parameters constant.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Term Structure of Normal Volatility")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    LATENT_DIM = 2
    EPOCHS = 100
    checkpoint_path = f"../checkpoints/fullmodel_bbg_dim{LATENT_DIM}_ep{EPOCHS}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FullModel(latent_dim=checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load initial curve
    S0, meta_row, X_tensor, meta = load_initial_curve("bbg", -1, device)
    with torch.no_grad():
        z0 = model.encoder(S0)
    
    # Simulate paths (large enough to cover all expiries)
    print("Simulating 500 paths with 240 steps (20 years)...")
    with torch.no_grad():
        z_paths, r_paths = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=500,
            n_steps=240,
            dt=1/12,
            device=device,
            simple_diffusion=False,
        )
    
    # Extract market parameters at different expiries
    expiries = [0.5, 1.0, 2.0, 5.0, 10.0]  # 6M, 1Y, 2Y, 5Y, 10Y
    norm_vol = 0.005  # 50 bp
    strike = 0.03
    tenor = 5
    
    print(f"\nPayer swaption prices (strike={strike}, norm_vol={norm_vol*10000:.0f} bp, tenor={tenor}Y):")
    print("Expiry | Forward  | Annuity | Price")
    print("-------|----------|---------|-------")
    
    for expiry in expiries:
        market_params = extract_market_params_at_expiry(
            z_paths=z_paths,
            model=model,
            device=device,
            dt=1/12,
            expiry=expiry,
            tenor=tenor
        )
        
        forward = market_params['forward_swap']
        annuity = market_params['annuity']
        
        price = price_swaption_from_norm_vol(
            forward=forward,
            strike=strike,
            norm_vol=norm_vol,
            expiry=expiry,
            annuity=annuity,
            notional=1.0,
            is_call=True
        )
        
        print(f"{expiry:6.1f} | {forward:8.6f} | {annuity:7.4f} | {price:7.6f}")


def example_4_volatility_smile():
    """
    Example 4: Analyze volatility smile (norm vol as function of strike).
    
    In the Bachelier model with constant volatility, the "smile" is flat,
    but this shows how prices vary with strike.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Volatility Smile Analysis")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    LATENT_DIM = 2
    EPOCHS = 100
    checkpoint_path = f"../checkpoints/fullmodel_bbg_dim{LATENT_DIM}_ep{EPOCHS}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FullModel(latent_dim=checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load initial curve
    S0, meta_row, X_tensor, meta = load_initial_curve("bbg", -1, device)
    with torch.no_grad():
        z0 = model.encoder(S0)
    
    # Simulate paths
    print("Simulating 200 paths...")
    with torch.no_grad():
        z_paths, r_paths = simulate_latent_paths(
            model=model,
            z0=z0,
            n_paths=200,
            n_steps=120,
            dt=1/12,
            device=device,
        )
    
    # Get market parameters
    market_params = extract_market_params_at_expiry(
        z_paths=z_paths,
        model=model,
        device=device,
        dt=1/12,
        expiry=1.0,
        tenor=5
    )
    
    forward = market_params['forward_swap']
    annuity = market_params['annuity']
    
    # Fixed parameters
    norm_vol = 0.005
    expiry = 1.0
    
    # Generate strikes around ATM
    atm_offset = np.linspace(-0.02, 0.02, 9)  # ±200 bp around ATM
    strikes = forward + atm_offset
    
    print(f"\nForward: {forward:.6f}, Annuity: {annuity:.6f}")
    print(f"Normal volatility: {norm_vol*10000:.0f} bp")
    print("\nStrike  | Moneyness | Payer Price | Receiver Price")
    print("--------|-----------|-------------|----------------")
    
    for strike in strikes:
        payer_price = price_swaption_from_norm_vol(
            forward=forward,
            strike=strike,
            norm_vol=norm_vol,
            expiry=expiry,
            annuity=annuity,
            notional=1.0,
            is_call=True
        )
        
        receiver_price = price_swaption_from_norm_vol(
            forward=forward,
            strike=strike,
            norm_vol=norm_vol,
            expiry=expiry,
            annuity=annuity,
            notional=1.0,
            is_call=False
        )
        
        moneyness = (strike - forward) * 10000  # in basis points
        print(f"{strike:6.4f} | {moneyness:9.0f} bp | {payer_price:11.6f} | {receiver_price:14.6f}")


if __name__ == "__main__":
    # Run examples
    example_1_direct_norm_vol_pricing()
    # example_2_implied_norm_vol_from_monte_carlo()
    example_3_term_structure_of_norm_vol()
    example_4_volatility_smile()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)

