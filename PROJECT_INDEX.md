# Project Index - Swaption Workflow

## Active workflow

### Core pricing code
- `Code/price_options.py`
  - Monte Carlo swaption pricing
  - Direct normal-vol quote pricing
  - Cap pricing prototype
- `Code/swaption_market_workflow.py`
  - Batch swaption pricing from a CSV file
  - Forward/annuity extraction from model paths
  - Optional Monte Carlo vs quote comparison
- `Code/swaption_quotes_template.csv`
  - Template market quote file

### Model and utilities
- `Code/model/full_model.py` - latent term-structure model
- `Code/utils/rates.py` - swap-rate helper utilities
- `Code/simulate_model.py` - model simulation utilities
- `checkpoints/` - trained model weights

### Documentation
- `README_START_HERE.md` - short practical start guide
- `Code/INDEX.md` - documentation hub under `Code/`
- `Code/QUICKSTART.md` - quick start for swaption pricing
- `Code/SWAPTION_PRICING_GUIDE.md` - detailed guide
- `Code/SWAPTION_EXAMPLES.py` - CLI examples
- `Code/swaption_programming_examples.py` - Python examples

## Typical usage

### Single quote pricing
```powershell
cd "C:\Users\Bruger\PycharmProjects\MasterThesis\Code"
python price_options.py --option_type swaption --pricing_mode norm_vol_quote --norm_vol 0.005 --strike 0.03 --expiry 1.0 --tenor 5
```

### Batch CSV pricing
```powershell
cd "C:\Users\Bruger\PycharmProjects\MasterThesis"
python Code\swaption_market_workflow.py --quotes_csv Code\swaption_quotes_template.csv
```

### Batch CSV pricing with Monte Carlo comparison
```powershell
cd "C:\Users\Bruger\PycharmProjects\MasterThesis"
python Code\swaption_market_workflow.py --quotes_csv Code\swaption_quotes_template.csv --run_mc --n_paths 500
```

## Quote CSV schema

Required:
- `expiry`
- `tenor`
- `strike`
- `option_type`
- `norm_vol` or `norm_vol_bp`

Optional:
- `notional`
- `market_price` or `market_price_bp`
- `label`

## Notes

- The active path is now swaption pricing, not SOFR futures option analysis.
- The swaption flow is the better fit for the model because the model naturally produces discount factors, forward swap rates, and annuities.
