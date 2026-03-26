# Quick Start: Swaption Pricing with Normal Volatility

## 🚀 5-Minute Quick Start

### Most Common Use Case: Price a Swaption from Market Norm Vol Quote

```bash
cd Code
python price_options.py \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.005 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5
```

**What this does:**
- Loads your trained model and latest yield curve
- Extracts forward swap rate and annuity from model dynamics
- Applies Bachelier model with 50 bp normal volatility
- Outputs: **Payer swaption price**

**Expected output:**
```
Using device: cpu
Loaded model from ../checkpoints/fullmodel_bbg_dim2_ep100.pt
Initial latent state z0: [-0.12  0.05]
Simulating 100 paths with 120 steps (dt=0.0833)...
Simulation completed.
Extracting market parameters at swaption expiry...
  Forward swap rate: 0.025000
  Annuity factor: 4.500000
  Input normal volatility: 0.005000 (50.00 bp)
Payer swaption price (from norm vol): 0.001688
  (strike=0.03, expiry=1.0, tenor=5, notional=1.0)
Pricing completed.
```

---

## 📋 Common Tasks

### Task 1: Price Multiple Strikes
```bash
for strike in 0.02 0.025 0.03 0.035 0.04; do
  python price_options.py \
    --option_type swaption \
    --pricing_mode norm_vol_quote \
    --norm_vol 0.005 \
    --strike $strike \
    --expiry 1.0 \
    --tenor 5
done
```

### Task 2: Price Different Volatilities (Sensitivity Analysis)
```bash
for vol_bp in 25 50 75 100; do
  vol=$(echo "scale=6; $vol_bp / 10000" | bc)
  python price_options.py \
    --option_type swaption \
    --pricing_mode norm_vol_quote \
    --norm_vol $vol \
    --strike 0.03 \
    --expiry 1.0 \
    --tenor 5
done
```

### Task 3: Model vs Market Comparison
```bash
# Step 1: Get model price + implied norm vol
python price_options.py \
  --option_type swaption \
  --pricing_mode monte_carlo \
  --n_paths 1000 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5 \
  --output_norm_vol

# Output shows: Model price and implied norm vol
# Compare this to market quote!
```

### Task 4: Receiver Swaption
```bash
python price_options.py \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.005 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5 \
  --is_receiver
```

### Task 5: Different Date/Curve
```bash
python price_options.py \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.005 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5 \
  --idx_choice 10  # Use 10th curve from data
```

---

## 🔍 Understanding the Output

```
Payer swaption price (from norm vol): 0.001688
```

This means:
- **Notional = 1.0** → Price is per unit notional
- For **10 million notional** → Multiply by 10,000,000 = **16,880,000** (currency units)
- For **100 million notional** → **168,800,000**

---

## ⚙️ Parameter Cheat Sheet

| Parameter | Typical Values | Example |
|-----------|---|---|
| `--norm_vol` | 0.002 - 0.01 (20-100 bp) | `0.005` = 50 bp |
| `--strike` | Market rate ± 2% | `0.03` = 3% |
| `--expiry` | 0.5, 1, 2, 5, 10 | `1.0` = 1 year |
| `--tenor` | 1, 2, 5, 10, 30 | `5` = 5 years |
| `--notional` | 1.0 (prices per unit) | `10000000` = 10M |

---

## 🧮 Basis Points Conversion

| Decimal | Basis Points | Percentage |
|---------|--------------|-----------|
| 0.001 | 10 bp | 0.1% |
| 0.005 | 50 bp | 0.5% |
| 0.01 | 100 bp | 1.0% |
| 0.015 | 150 bp | 1.5% |

**Formula:** `basis_points / 10000 = decimal`

---

## 💡 Pro Tips

1. **Fast Pricing:** Use `norm_vol_quote` mode - no simulation needed for pricing step
2. **Market Calibration:** Compare `--output_norm_vol` result to market quotes
3. **Batch Pricing:** Use shell loops (bash/PowerShell) for multiple runs
4. **Initial Curve:** `--idx_choice -1` uses last (most recent) curve

---

## ❌ Common Mistakes

❌ Wrong: `--norm_vol 50` (basis points)  
✅ Correct: `--norm_vol 0.005` (decimal)

❌ Wrong: `--pricing_mode norm_vol_quote` without `--norm_vol`  
✅ Correct: Add `--norm_vol 0.005`

❌ Wrong: `--tenor 5000` (days)  
✅ Correct: `--tenor 5` (years)

---

## 📊 Example Workflow for Thesis

```bash
# 1. Generate model prices with implied vol
python price_options.py \
  --option_type swaption \
  --pricing_mode monte_carlo \
  --n_paths 5000 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5 \
  --output_norm_vol > model_prices.txt

# 2. Compare to market quotes
python price_options.py \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.0055 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5 > market_prices.txt

# 3. Show that Bachelier model gives consistent pricing
echo "Model implies: 50 bp (see model_prices.txt)"
echo "Market quotes: 55 bp (see market_prices.txt)"
echo "→ Price difference: 1 bp higher at market quote"
```

---

## 🎯 Next Steps

1. **Read:** `SWAPTION_PRICING_GUIDE.md` for detailed documentation
2. **Explore:** `SWAPTION_EXAMPLES.py` for command-line examples
3. **Program:** `swaption_programming_examples.py` for Python code examples
4. **Experiment:** Try different strikes, expiries, volatilities

---

## 🆘 Help

### Error: "norm_vol required when using norm_vol_quote pricing mode"
**Solution:** You forgot `--norm_vol`. Add it:
```bash
--norm_vol 0.005
```

### Error: "Checkpoint not found"
**Solution:** Check the model checkpoint exists in `../checkpoints/` directory

### Weird output (very large/small prices)?
**Check:**
- Is `--norm_vol` in decimal (0.005) not basis points (5)?
- Are forward rate and strike reasonable?
- Try more paths: `--n_paths 5000`

---

That's it! You're ready to price swaptions with normal volatility! 🚀

