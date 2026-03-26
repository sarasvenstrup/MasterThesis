# Swaption Pricing with Normal Volatility (Norm Vol)

## Overview

Your code has been extended to support pricing swaptions quoted on normal (Bachelier) volatility. This guide explains the new capabilities and how to use them.

## New Features

### 1. **Price Swaptions from Normal Volatility Quotes**
   - **Function**: `price_swaption_from_norm_vol()`
   - **Mode**: `--pricing_mode norm_vol_quote`
   - Price a swaption given a normal volatility quote directly without Monte Carlo simulation

### 2. **Extract Market Parameters**
   - **Function**: `extract_market_params_at_expiry()`
   - Extracts forward swap rate and annuity factor at swaption expiry across all simulation paths
   - Returns both individual path values and averages

### 3. **Implied Normal Volatility Calculation**
   - **Function**: `implied_normal_vol()`
   - Converts Monte Carlo prices to implied normal volatility
   - Uses Bachelier model for inversion

## Usage Examples

### Example 1: Price Swaption from Normal Volatility Quote (50 bp)

```bash
python price_options.py \
  --option_type swaption \
  --latent_dim 2 \
  --epochs 100 \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.005 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5
```

This prices a 1Y×5Y payer swaption with:
- Strike: 3%
- Normal volatility: 50 basis points (0.005)
- Notional: 1.0 (default)

### Example 2: Price Swaption with Monte Carlo and Output Implied Norm Vol

```bash
python price_options.py \
  --option_type swaption \
  --latent_dim 2 \
  --epochs 100 \
  --pricing_mode monte_carlo \
  --n_paths 1000 \
  --n_steps 120 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5 \
  --output_norm_vol
```

This:
- Simulates 1000 paths with 120 time steps
- Prices via Monte Carlo
- Computes implied normal volatility from the Monte Carlo price

### Example 3: Price Receiver Swaption from Norm Vol

```bash
python price_options.py \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.0045 \
  --strike 0.03 \
  --expiry 2.0 \
  --tenor 10 \
  --is_receiver \
  --notional 10000000
```

This prices a 2Y×10Y receiver swaption with:
- Strike: 3%
- Normal volatility: 45 basis points
- Notional: 10 million

## Command-Line Arguments

### Pricing Mode Selection
- `--pricing_mode monte_carlo` (default): Uses simulated paths to price the option
- `--pricing_mode norm_vol_quote`: Uses Bachelier model with provided normal volatility

### Normal Volatility Input
- `--norm_vol FLOAT`: Normal volatility in decimal (e.g., 0.005 for 50 bp)
  - Required when using `--pricing_mode norm_vol_quote`
  - Optional in `monte_carlo` mode (skips implied vol calculation if not provided)

### Swaption Specifications
- `--strike FLOAT`: Strike rate in decimal (default: 0.03)
- `--expiry FLOAT`: Time to expiry in years (default: 1.0)
- `--tenor INT`: Tenor of underlying swap in years (default: 5)
- `--is_receiver`: Price receiver swaption (default: payer/call)
- `--notional FLOAT`: Notional amount (default: 1.0)

### Output Options
- `--output_norm_vol`: Calculate and display implied normal volatility
  - Only applies in `monte_carlo` mode
  - Useful for comparing model prices to market norms

## Pricing Models

### Bachelier (Normal) Model

The implementation uses the Bachelier model for normal volatility pricing:

```
Price = Annuity × Notional × [-(K-F) × N(-d) + σ√T × n(d)]
```

Where:
- F = Forward swap rate
- K = Strike rate
- σ = Normal volatility
- T = Time to expiry
- Annuity = Sum of discount factors over the tenor
- N(·) = Cumulative normal distribution
- n(·) = Normal probability density

### Implied Normal Volatility

The implied normal volatility is computed by inverting the Bachelier formula using constrained optimization (scipy.optimize.minimize_scalar).

## Key Functions

### `price_swaption_from_norm_vol(forward, strike, norm_vol, expiry, annuity, notional, is_call)`
**Purpose**: Price a swaption using Bachelier model with normal volatility

**Parameters**:
- `forward`: Forward swap rate
- `strike`: Strike rate
- `norm_vol`: Normal volatility (decimal)
- `expiry`: Time to expiry (years)
- `annuity`: Annuity factor (sum of discount factors)
- `notional`: Notional amount
- `is_call`: True for payer, False for receiver

**Returns**: Swaption price

### `extract_market_params_at_expiry(z_paths, model, device, dt, expiry, tenor)`
**Purpose**: Extract forward swap rate and annuity across all paths at expiry

**Parameters**:
- `z_paths`: Simulated latent factor paths (n_paths, n_steps+1, latent_dim)
- `model`: FullModel instance
- `device`: torch device
- `dt`: Time step size
- `expiry`: Swaption expiry (years)
- `tenor`: Swap tenor (years)

**Returns**: Dictionary with:
- `forward_swap`: Average forward swap rate
- `annuity`: Average annuity factor
- `forward_swaps`: List of forward rates per path
- `annuities`: List of annuity factors per path

### `implied_normal_vol(market_price, forward, strike, expiry, annuity, notional, is_call)`
**Purpose**: Calculate implied normal volatility from a market price

**Parameters**:
- `market_price`: Observed swaption price
- `forward`: Forward swap rate
- `strike`: Strike rate
- `expiry`: Time to expiry
- `annuity`: Annuity factor
- `notional`: Notional amount
- `is_call`: True for payer, False for receiver

**Returns**: Implied normal volatility (decimal), or NaN if optimization fails

## Workflow: Pricing from Market Norm Vol Quotes

1. **Load a trained model** with checkpoint
2. **Encode an initial yield curve** to get latent state z0
3. **Simulate latent factor paths** (quick if only using norm vol mode)
4. **Extract market parameters** at swaption expiry:
   - Forward swap rate
   - Annuity factor
5. **Apply Bachelier formula** with market normal volatility quote
6. **Get swaption price** directly

Example output:
```
Extracting market parameters at swaption expiry...
  Forward swap rate: 0.025000
  Annuity factor: 4.500000
  Input normal volatility: 0.005000 (50.00 bp)
Payer swaption price (from norm vol): 0.002250
  (strike=0.03, expiry=1.0, tenor=5, notional=1.0)
```

## Comparison: Monte Carlo vs Norm Vol Quote

| Feature | Monte Carlo | Norm Vol Quote |
|---------|-------------|---|
| Simulation Required | Yes | No* |
| Computational Cost | Higher | Lower |
| Best for | Model calibration, discovering prices | Market quote pricing |
| Output | Price + Implied Norm Vol | Price directly |
| Inputs | Model parameters | Market norm vol |

*Can still run simulation but only uses market params at expiry; not needed for pricing step.

## Tips

1. **Use norm_vol_quote mode** when you have market normal volatility quotes and want quick pricing
2. **Use monte_carlo mode** when you want to:
   - Discover prices from model dynamics
   - Compare model prices to market quotes
   - Calculate implied normal volatility
3. **Monitor annuity and forward rates** to ensure they're reasonable before pricing
4. **Basis points conversion**: Multiply by 10,000 to convert decimal to bp (e.g., 0.005 → 50 bp)

## Troubleshooting

### Error: "norm_vol required when using norm_vol_quote pricing mode"
- Solution: Add `--norm_vol 0.005` (or your desired normal volatility value)

### Non-finite implied volatility
- This occurs when optimization fails
- Check that forward rate is not too close to strike
- Try different initial path count or time discretization

### Annuity factor too small/large
- Verify that tenor and expiry parameters are reasonable
- Check that model.tenors includes the required maturities

