"""
VISUAL QUICK REFERENCE: Swaption Pricing with Normal Volatility

This file provides visual diagrams and quick lookup tables.
"""

# ==============================================================================
# DECISION TREE: Which Pricing Mode Should I Use?
# ==============================================================================
"""
                        Do I have market
                      normal vol quotes?
                            |
                    ________|________
                   |                 |
                  YES               NO
                   |                 |
         Use: norm_vol_quote    Do I want to
                   |           discover model
              Price from       prices?
              Bachelier           |
                   |         _____|_____
                   |        |           |
         (Fast, direct)   YES          NO
                           |            |
                      Use: monte_carlo  Use existing
                      --pricing_mode    code or
                      monte_carlo       --option_type cap
                      --output_norm_vol
                           |
                      (Slower, but
                       gives implied vol)
"""

# ==============================================================================
# COMMAND BUILDER: Step-by-Step
# ==============================================================================
"""
Basic template:
    python price_options.py \\
        --option_type swaption \\
        --pricing_mode [MODE] \\
        [MODE_SPECIFIC_ARGS] \\
        --strike [STRIKE] \\
        --expiry [YEARS] \\
        --tenor [YEARS]

Step 1: Choose pricing mode
    ├─ norm_vol_quote  → Add: --norm_vol 0.005
    └─ monte_carlo     → Add: --n_paths 1000 --n_steps 120

Step 2: Choose option type
    ├─ payer (default)
    └─ receiver        → Add: --is_receiver

Step 3: Set swaption parameters
    ├─ --strike 0.03    (3%)
    ├─ --expiry 1.0     (1 year)
    └─ --tenor 5        (5 years)

Step 4: (Optional) Advanced parameters
    ├─ --notional 10000000        (set notional)
    ├─ --idx_choice 10            (different curve)
    ├─ --latent_dim 3             (different model)
    └─ --output_norm_vol          (for monte_carlo mode)
"""

# ==============================================================================
# PARAMETER REFERENCE TABLE
# ==============================================================================
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        PARAMETER REFERENCE TABLE                              ║
╠═════════════════════╦═══════════════════╦═════════════════╦═══════════════════╣
║     Parameter       ║     Default       ║   Data Type     ║   Typical Range   ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --norm_vol          │     None          │     float       │  0.002 - 0.01     ║
║                     │  (required!)      │                 │ (20-100 bps)      ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --strike            │      0.03         │     float       │  0.01 - 0.05      ║
║                     │                   │                 │  (1% - 5%)        ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --expiry            │      1.0          │     float       │  0.5 - 10.0       ║
║                     │                   │     (years)     │  (6M - 10Y)       ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --tenor             │      5            │     int         │  1, 2, 3, 5, 10   ║
║                     │                   │     (years)     │  (1Y - 30Y)       ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --n_paths           │      1000         │     int         │  100 - 10000      ║
║                     │ (MC only)         │                 │                   ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --n_steps           │      120          │     int         │  60 - 240         ║
║                     │ (MC only)         │                 │  (5Y - 20Y)       ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --dt                │     1/12          │     float       │  1/12, 1/4, 1/2   ║
║                     │  (MC only)        │     (monthly)   │                   ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --notional          │      1.0          │     float       │  1.0 - 1e9        ║
║                     │                   │     (units)     │                   ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --latent_dim        │      2            │     int         │  1, 2, 3, 4       ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --epochs            │      100          │     int         │  100, 1000, 5000  ║
╠═════════════════════╬═══════════════════╬═════════════════╬═══════════════════╣
║ --idx_choice        │      -1 (last)    │     int         │  0 - N_obs        ║
╚═════════════════════╩═══════════════════╩═════════════════╩═══════════════════╝

Flags (True/False):
  --is_receiver               (default: False) → Price receiver (put)
  --output_norm_vol           (default: False) → Output implied vol (MC only)
  --simple_diffusion          (default: False) → Use simple OU dynamics

Choices:
  --pricing_mode              [monte_carlo | norm_vol_quote]
  --option_type               [cap | swaption]
  --use                       [bbg | testdata]
"""

# ==============================================================================
# BASIS POINTS / DECIMAL CONVERSION
# ==============================================================================
"""
Quick conversion for normal volatility:

Basis Points  │  Decimal    │  Percentage
──────────────┼─────────────┼────────────
     10 bp    │   0.001     │    0.1%
     25 bp    │  0.0025     │   0.25%
     50 bp    │   0.005     │    0.5%
     75 bp    │  0.0075     │   0.75%
    100 bp    │    0.01     │    1.0%
    150 bp    │   0.015     │    1.5%
    200 bp    │    0.02     │    2.0%

RULE: bp / 10000 = decimal
      decimal × 10000 = bp
"""

# ==============================================================================
# COMMON SWAPTION SPECIFICATIONS
# ==============================================================================
"""
Standard Market Swaptions (Expiry × Tenor):

Short-dated (< 1Y):
  1M×5Y, 3M×5Y, 6M×5Y, 6M×10Y

Medium-dated (1-5Y):
  1Y×5Y, 1Y×10Y
  2Y×5Y, 2Y×10Y
  3Y×5Y, 3Y×10Y
  5Y×5Y, 5Y×10Y

Long-dated (> 5Y):
  10Y×5Y, 10Y×10Y, 10Y×30Y
  30Y×5Y, 30Y×10Y

EUR vs USD conventions may differ - check your market!
"""

# ==============================================================================
# OUTPUT INTERPRETATION
# ==============================================================================
"""
Typical output format:

  Payer swaption price (from norm vol): 0.001688
  (strike=0.03, expiry=1.0, tenor=5, notional=1.0)

What does 0.001688 mean?

Case 1: notional=1.0 (default)
  → Price per 1 unit of notional
  → For scaling: multiply by your notional

Case 2: notional=10,000,000
  → Price = 0.001688 (notional = 10M)
  → Total price = 0.001688 × 10,000,000 = 16,880 (currency units)

Case 3: notional=100,000,000 (100M)
  → Total price = 0.001688 × 100,000,000 = 168,800

PRICING UNITS:
  - Price is in the same currency as the interest rates
  - If using EUR swap curve → Price in EUR
  - If using USD swap curve → Price in USD
"""

# ==============================================================================
# ERROR TROUBLESHOOTING
# ==============================================================================
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║                          ERROR TROUBLESHOOTING                               ║
╠═════════════════════════════════════════════════════════════════════════════╣
║ ERROR: "norm_vol required when using norm_vol_quote pricing mode"            ║
║ CAUSE: Forgot --norm_vol parameter in norm_vol_quote mode                    ║
║ FIX:   Add --norm_vol 0.005                                                  ║
╠═════════════════════════════════════════════════════════════════════════════╣
║ ERROR: "Checkpoint not found: .../fullmodel_bbg_dim2_ep100.pt"               ║
║ CAUSE: Model checkpoint doesn't exist                                        ║
║ FIX:   Verify checkpoint exists in ../checkpoints/ or train model            ║
╠═════════════════════════════════════════════════════════════════════════════╣
║ ERROR: "Non-finite z encountered at step X"                                  ║
║ CAUSE: Simulation diverged (numerical instability)                           ║
║ FIX:   Reduce --dt (smaller time step) or increase --n_paths                 ║
╠═════════════════════════════════════════════════════════════════════════════╣
║ ERROR: "Could not compute implied normal volatility"                         ║
║ CAUSE: Optimization failed (price unlikely under any vol)                    ║
║ FIX:   Check forward rate ≠ strike, increase --n_paths                       ║
╠═════════════════════════════════════════════════════════════════════════════╣
║ ERROR: Swaption prices are 0 or very large                                   ║
║ CAUSE: Likely norm_vol in wrong units (basis points vs decimal)              ║
║ FIX:   Use --norm_vol 0.005 (decimal), not 50 (basis points)                 ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

# ==============================================================================
# WORKFLOW DIAGRAMS
# ==============================================================================
"""
WORKFLOW 1: Market Pricing (Fast)
───────────────────────────────────
Input: Market norm vol quote
  ↓
Get market parameters (forward, annuity) from model
  ↓
Apply Bachelier formula
  ↓
Output: Swaption price
Time: ~10-30 seconds

WORKFLOW 2: Model Calibration
──────────────────────────────
Input: Model + initial curve
  ↓
Simulate Monte Carlo paths
  ↓
Compute prices from dynamics
  ↓
Calculate implied normal vol
  ↓
Compare to market quotes
  ↓
Measure calibration error
Time: ~1-5 minutes


WORKFLOW 3: Sensitivity Analysis
─────────────────────────────────
Input: Base case parameters
  ↓
Loop over strikes
  ├─ Loop over expiries
  │   └─ Loop over volatilities
  │       ↓
  │       Price swaption
  │       Store result
  ├─ Plot strike sensitivity
  ├─ Plot tenor sensitivity
  └─ Plot vol sensitivity
Output: Sensitivity surfaces
Time: ~5-15 minutes


WORKFLOW 4: Volatility Term Structure
──────────────────────────────────────
Input: Single strike, multiple expiries
  ↓
For each expiry (0.5, 1, 2, 3, 5, 10 years):
  ├─ Extract market params
  ├─ Price swaption
  └─ Store result
  ↓
Plot term structure curve
  ↓
Analyze shape (upward, downward, humped)
Time: ~5-10 minutes
"""

# ==============================================================================
# SUCCESS CHECKLIST
# ==============================================================================
"""
✓ Syntax check passed?
  python -m py_compile price_options.py

✓ Model checkpoint exists?
  ls ../checkpoints/fullmodel_bbg_dim2_ep100.pt

✓ Correct norm_vol units (decimal)?
  50 bp → 0.005 (not 50)

✓ Pricing mode selected?
  --pricing_mode norm_vol_quote or monte_carlo

✓ Swaption params reasonable?
  strike close to forward rate, expiry > 0, tenor > 0

✓ Output makes sense?
  Price between 0.0001 and 0.1 for notional=1.0

✓ Implied vol calculated (MC mode)?
  Check: "Implied normal volatility: X (Y basis points)"

✓ Comparison to market quotes?
  Model vol vs Market vol within ±10 bp?

If all checks pass → Ready for production! 🚀
"""

# ==============================================================================
# NEXT STEPS
# ==============================================================================
"""
1. READ:    QUICKSTART.md (5 minutes)
2. TRY:     First example from SWAPTION_EXAMPLES.py
3. EXPLORE: Different strikes, expiries, volatilities
4. COMPARE: Model prices vs market quotes
5. ANALYZE: Term structures and sensitivities
6. THESIS:  Incorporate results into your master's thesis

For questions/issues:
  → Check SWAPTION_PRICING_GUIDE.md
  → Review error message in ERROR TROUBLESHOOTING section
  → Look at code examples in swaption_programming_examples.py
"""

