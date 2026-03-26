#!/usr/bin/env python
"""
Quick Reference: Swaption Pricing Examples

This file contains example command-line invocations for pricing swaptions
with normal volatility using the extended price_options.py script.
"""

# ============================================================================
# EXAMPLE 1: Simple Norm Vol Quote Pricing (1Y × 5Y Payer Swaption)
# ============================================================================
# Price a payer swaption with market normal volatility quote (50 bp)
# Command line:
"""
python price_options.py \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.005 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5
"""
# Output: Payer swaption price based on 50 bp normal volatility
# Market parameters (forward swap rate, annuity) computed from model


# ============================================================================
# EXAMPLE 2: Monte Carlo Pricing with Implied Norm Vol Output
# ============================================================================
# Simulate swaption price and compute implied normal volatility
# Command line:
"""
python price_options.py \
  --option_type swaption \
  --pricing_mode monte_carlo \
  --n_paths 5000 \
  --n_steps 240 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5 \
  --output_norm_vol
"""
# Output: Monte Carlo price + Implied normal volatility


# ============================================================================
# EXAMPLE 3: Receiver Swaption (Put) from Norm Vol
# ============================================================================
# Price a 2Y×10Y receiver swaption with 45 bp normal volatility
# Command line:
"""
python price_options.py \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.0045 \
  --strike 0.025 \
  --expiry 2.0 \
  --tenor 10 \
  --is_receiver \
  --notional 10000000
"""
# Output: Receiver swaption price for 10M notional


# ============================================================================
# EXAMPLE 4: Compare Model Price to Market Norm Vol Quote
# ============================================================================
# Monte Carlo simulate a price, then compute what norm vol that implies
# Then compare to a market quote
# Command line:
"""
python price_options.py \
  --option_type swaption \
  --pricing_mode monte_carlo \
  --n_paths 1000 \
  --n_steps 120 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5 \
  --output_norm_vol \
  --norm_vol 0.005
"""
# Output: Monte Carlo price, implied norm vol, and comparison to market quote


# ============================================================================
# EXAMPLE 5: Different Latent Dimensions and Epochs
# ============================================================================
# Price using a different trained model (e.g., 3D latent space, 5000 epochs)
# Command line:
"""
python price_options.py \
  --latent_dim 3 \
  --epochs 5000 \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.0055 \
  --strike 0.035 \
  --expiry 5.0 \
  --tenor 10
"""
# Loads: checkpoints/fullmodel_bbg_dim3_ep5000.pt


# ============================================================================
# EXAMPLE 6: Range of Strikes (Script-based)
# ============================================================================
# Python code to price swaptions at multiple strikes from norm vol
"""
import subprocess
import json

strikes = [0.01, 0.02, 0.03, 0.04, 0.05]
norm_vol = 0.005  # 50 bp
results = {}

for strike in strikes:
    cmd = [
        "python", "price_options.py",
        "--option_type", "swaption",
        "--pricing_mode", "norm_vol_quote",
        "--norm_vol", str(norm_vol),
        "--strike", str(strike),
        "--expiry", "1.0",
        "--tenor", "5"
    ]
    # Parse output and store results
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Extract price from output
    results[strike] = result.stdout

print(json.dumps(results, indent=2))
"""


# ============================================================================
# EXAMPLE 7: Different Initial Curves
# ============================================================================
# Price using different points on the initial yield curve
# Command line:
"""
python price_options.py \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --idx_choice 10 \
  --norm_vol 0.005 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5
"""
# Uses the 10th observation from the data set


# ============================================================================
# COMMAND LINE REFERENCE
# ============================================================================
"""
PRICING MODE:
  --pricing_mode monte_carlo      (default) Simulate paths, price via MC
  --pricing_mode norm_vol_quote   Use Bachelier with norm vol quote

NORMAL VOLATILITY:
  --norm_vol FLOAT                Normal volatility in decimal
                                  Required for norm_vol_quote mode
                                  Example: 0.005 means 50 basis points

SWAPTION SPECS:
  --strike FLOAT                  Strike rate (default: 0.03)
  --expiry FLOAT                  Time to expiry in years (default: 1.0)
  --tenor INT                     Swap tenor in years (default: 5)
  --notional FLOAT                Notional amount (default: 1.0)
  --is_receiver                   Price receiver (put) not payer (call)

MONTE CARLO ONLY:
  --n_paths INT                   Number of simulation paths (default: 1000)
  --n_steps INT                   Number of time steps (default: 120)
  --dt FLOAT                      Time step size (default: 1/12)
  --output_norm_vol               Output implied normal volatility

MODEL SELECTION:
  --latent_dim INT                Latent dimension (default: 2)
  --epochs INT                    Training epochs (default: 100)
  --use STR                       Data source: bbg, testdata (default: bbg)
  --idx_choice INT                Index of initial curve (default: -1, last)

DIFFUSION DYNAMICS:
  --simple_diffusion              Use simple OU instead of model dynamics
  --kappa FLOAT                   Mean reversion (simple OU)
  --theta FLOAT                   Long-run mean (simple OU)
  --sigma_simple FLOAT            Volatility (simple OU)
"""


# ============================================================================
# TYPICAL WORKFLOW
# ============================================================================
"""
1. EXPLORATORY ANALYSIS (Monte Carlo)
   - Simulate model dynamics
   - Calculate Monte Carlo prices
   - Extract implied normal volatility
   → Discover model prices in normal vol units

2. MARKET PRICING (Norm Vol Quote)
   - Get market normal volatility quotes
   - Apply Bachelier model with forward/annuity from model
   - Price swaptions consistent with market quotes
   → No simulation needed for pricing step

3. CALIBRATION/COMPARISON
   - Run both modes on same initial conditions
   - Compare model implied norm vol to market quotes
   - Identify mis-pricings and arbitrage opportunities
"""

