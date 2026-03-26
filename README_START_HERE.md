# Swaption Pricing - Start Here

## What to use

Primary entrypoints:
- `Code/price_options.py` - single swaption/cap pricing CLI
- `Code/swaption_market_workflow.py` - batch pricing from a swaption quote CSV
- `Code/swaption_quotes_template.csv` - template for your own quote file

## Recommended workflow

### 1. Price one swaption from a quoted normal volatility
```powershell
cd "C:\Users\Bruger\PycharmProjects\MasterThesis\Code"
python price_options.py --option_type swaption --pricing_mode norm_vol_quote --norm_vol 0.005 --strike 0.03 --expiry 1.0 --tenor 5
```

### 2. Price a whole swaption quote sheet from CSV
```powershell
cd "C:\Users\Bruger\PycharmProjects\MasterThesis"
python Code\swaption_market_workflow.py --quotes_csv Code\swaption_quotes_template.csv
```

### 3. Add Monte Carlo comparison to the quote workflow
```powershell
cd "C:\Users\Bruger\PycharmProjects\MasterThesis"
python Code\swaption_market_workflow.py --quotes_csv Code\swaption_quotes_template.csv --run_mc --n_paths 500 --dt 0.0833333333
```

## Quote CSV format

Required columns:
- `expiry` - years
- `tenor` - years
- `strike` - decimal swap rate
- `option_type` - `payer` or `receiver`
- `norm_vol` or `norm_vol_bp`

Optional columns:
- `notional`
- `market_price` or `market_price_bp`
- `label`

## What the model is doing

The model:
1. encodes the current swap curve into latent factors,
2. simulates latent paths,
3. decodes discount factors,
4. computes forward swap rates and annuities at expiry,
5. prices swaptions with Bachelier normal-vol quotes,
6. optionally compares quote-based prices to Monte Carlo model prices.

## Files worth reading

- `Code/QUICKSTART.md`
- `Code/SWAPTION_PRICING_GUIDE.md`
- `Code/SWAPTION_EXAMPLES.py`
- `Code/swaption_programming_examples.py`

## Notes

- The swaption workflow is aligned with what the model naturally outputs: discount factors, forward swap rates, and annuities.
- The old SOFR option experiment is no longer part of the active workflow.
