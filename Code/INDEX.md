# 📑 INDEX: Swaption Pricing with Normal Volatility - Complete Documentation

## 🚀 Start Here (Pick Your Path)

### ⏱️ **I have 5 minutes** → Read `QUICKSTART.md`
Quick commands to get pricing immediately with minimal explanation.

### ⏱️ **I have 15 minutes** → Read `SWAPTION_PRICING_GUIDE.md`
Comprehensive guide covering all features, modes, and usage patterns.

### ⏱️ **I want code examples** → See `swaption_programming_examples.py`
Runnable Python examples showing programmatic API usage.

### ⏱️ **I need CLI examples** → See `SWAPTION_EXAMPLES.py`
Command-line invocation examples with detailed explanations.

### ⏱️ **I need a visual reference** → See `VISUAL_REFERENCE.py`
Diagrams, tables, decision trees, and quick lookups.

---

## 📚 Complete Documentation Map

```
MAIN IMPLEMENTATION
│
└─ price_options.py (MODIFIED)
   ├─ extract_market_params_at_expiry()       [NEW]
   ├─ price_swaption_from_norm_vol()          [NEW]
   ├─ implied_normal_vol()                    [ENHANCED]
   ├─ bachelier_price()                       [ENHANCED]
   └─ --pricing_mode norm_vol_quote           [NEW CLI MODE]
   └─ --norm_vol                              [NEW CLI ARG]
   └─ --is_receiver                           [NEW CLI ARG]


DOCUMENTATION
│
├─ QUICKSTART.md                    [5-minute quick start]
│  ├─ Most common use case
│  ├─ Common tasks (loops, sensitivity)
│  ├─ Parameter cheat sheet
│  ├─ Basis points conversion
│  └─ Quick tips
│
├─ SWAPTION_PRICING_GUIDE.md        [Comprehensive reference]
│  ├─ Overview of new features
│  ├─ Detailed pricing modes
│  ├─ CLI arguments reference
│  ├─ Bachelier model explanation
│  ├─ Function documentation
│  └─ Troubleshooting
│
├─ VISUAL_REFERENCE.py              [Visual guides & tables]
│  ├─ Decision tree: which mode to use
│  ├─ Command builder step-by-step
│  ├─ Parameter reference table
│  ├─ Basis points conversion
│  ├─ Common swaption specs
│  ├─ Output interpretation
│  ├─ Error troubleshooting
│  └─ Workflow diagrams
│
├─ SWAPTION_EXAMPLES.py             [CLI examples (commented)]
│  ├─ 7 complete example invocations
│  ├─ Parameter reference table
│  ├─ Typical workflow explanation
│  └─ Use cases for your thesis
│
└─ swaption_programming_examples.py [Python code examples]
   ├─ Example 1: Direct norm vol pricing
   ├─ Example 2: Implied norm vol from MC
   ├─ Example 3: Term structure analysis
   ├─ Example 4: Volatility smile analysis
   └─ Runnable Python code (requires trained model)


THIS FILE
└─ INDEX.md                          [You are here!]
   └─ Navigation guide for all documentation
```

---

## 🎯 Common Use Cases → Which Doc?

| Use Case | Start With | Then Read | Example Code |
|----------|-----------|-----------|---|
| **Quick pricing from market norm vol** | QUICKSTART.md | None needed | SWAPTION_EXAMPLES.py |
| **Understand all features** | SWAPTION_PRICING_GUIDE.md | VISUAL_REFERENCE.py | Both examples files |
| **Integrate into Python script** | swaption_programming_examples.py | SWAPTION_PRICING_GUIDE.md | swaption_programming_examples.py |
| **Calibrate model to market** | QUICKSTART.md | SWAPTION_PRICING_GUIDE.md | SWAPTION_EXAMPLES.py (Examples 3-4) |
| **Analyze volatility surfaces** | VISUAL_REFERENCE.py (Workflow 3) | swaption_programming_examples.py | swaption_programming_examples.py (Example 4) |
| **Troubleshoot errors** | VISUAL_REFERENCE.py (Errors) | SWAPTION_PRICING_GUIDE.md | - |

---

## 📖 Reading Order Recommendations

### 🟢 Beginner: "I just want to price a swaption"
1. QUICKSTART.md (5 min)
2. Try the first example
3. SWAPTION_EXAMPLES.py (Example 1, 2 min)
4. Done! You're pricing swaptions

### 🟡 Intermediate: "I want to understand what's happening"
1. QUICKSTART.md (5 min)
2. SWAPTION_PRICING_GUIDE.md (15 min)
3. VISUAL_REFERENCE.py (10 min)
4. SWAPTION_EXAMPLES.py (10 min)
5. Try 2-3 examples yourself

### 🔴 Advanced: "I'm integrating this into my thesis"
1. SWAPTION_PRICING_GUIDE.md (full read)
2. swaption_programming_examples.py (study code)
3. VISUAL_REFERENCE.py (reference as needed)
4. Modify examples for your specific use case
5. Run batch pricing experiments
6. Incorporate results into thesis

---

## 🔧 Function Reference

### Core Functions (All in price_options.py)

```python
# NEW FUNCTIONS:
extract_market_params_at_expiry(z_paths, model, device, dt, expiry, tenor)
  → Returns forward swap rate and annuity at expiry

price_swaption_from_norm_vol(forward, strike, norm_vol, expiry, annuity, notional, is_call)
  → Returns swaption price from normal volatility quote

# ENHANCED EXISTING FUNCTIONS:
implied_normal_vol(market_price, forward, strike, expiry, annuity, notional, is_call)
  → Now documented and easily accessible

bachelier_price(forward, strike, sigma, expiry, annuity, notional, is_call)
  → Now exposed as the underlying pricing model
```

See **SWAPTION_PRICING_GUIDE.md** for full function signatures and examples.

---

## 🎮 Try It Now!

### 30-second test (verify installation):
```bash
cd Code
python price_options.py --help | grep "pricing_mode"
```

Should show:
```
--pricing_mode {monte_carlo,norm_vol_quote}
```

### 2-minute first example:
```bash
python price_options.py \
  --option_type swaption \
  --pricing_mode norm_vol_quote \
  --norm_vol 0.005 \
  --strike 0.03 \
  --expiry 1.0 \
  --tenor 5
```

Expected output includes:
```
Payer swaption price (from norm vol): ...
```

---

## 📋 What's New vs. What Existed

### ✨ NEW
- `extract_market_params_at_expiry()` function
- `price_swaption_from_norm_vol()` function
- `--pricing_mode norm_vol_quote` CLI mode
- `--norm_vol` parameter
- `--is_receiver` parameter
- Enhanced main() logic for norm vol pricing

### 🔧 ENHANCED
- `implied_normal_vol()` - now accessible and documented
- `bachelier_price()` - now exposed through wrapper function
- Main swaption pricing logic - now supports two modes

### ✓ UNCHANGED
- All existing functionality
- Backward compatibility
- Default behavior

---

## 🚨 Quick Troubleshooting

**"norm_vol required..."** → Add `--norm_vol 0.005`  
**"Checkpoint not found"** → Check model exists in ../checkpoints/  
**"Very large/small prices"** → Use decimal (0.005) not bp (50)  
**"Non-finite z"** → Reduce dt or increase n_paths  
**"Implied vol failed"** → Check forward ≠ strike, increase paths  

See **VISUAL_REFERENCE.py** error section for detailed troubleshooting.

---

## 💡 Pro Tips

1. **Keep QUICKSTART.md** bookmarked - you'll use it frequently
2. **Use VISUAL_REFERENCE.py** as a lookup during scripting
3. **Copy examples from SWAPTION_EXAMPLES.py** and modify
4. **For your thesis:** Start with Example 3 (term structure) or Example 4 (smile)
5. **Batch jobs:** Use shell loops + redirects for multiple runs

---

## 📞 Help & References

| Question | Resource |
|----------|----------|
| "How do I...?" | SWAPTION_EXAMPLES.py |
| "What does parameter X do?" | VISUAL_REFERENCE.py (Parameter table) |
| "Why is my output wrong?" | VISUAL_REFERENCE.py (Errors) |
| "What are the formulas?" | SWAPTION_PRICING_GUIDE.md (Technical Details) |
| "How do I use it in Python?" | swaption_programming_examples.py |
| "Show me all features" | SWAPTION_PRICING_GUIDE.md |

---

## 📊 Document Stats

| File | Type | Length | Purpose |
|------|------|--------|---------|
| QUICKSTART.md | Markdown | ~300 lines | 5-min quick start |
| SWAPTION_PRICING_GUIDE.md | Markdown | ~400 lines | Comprehensive reference |
| VISUAL_REFERENCE.py | Python | ~400 lines | Tables, diagrams, lookup |
| SWAPTION_EXAMPLES.py | Python | ~150 lines | CLI examples |
| swaption_programming_examples.py | Python | ~350 lines | Runnable code examples |
| price_options.py | Python | ~490 lines | Main implementation |

**Total documentation:** ~2000 lines of detailed guidance

---

## 🎓 Academic Use Cases

### For Your Master's Thesis:
1. **Calibration section:** Use Examples 3-4 to show term structure
2. **Model validation:** Compare implied norm vol to market quotes
3. **Risk analysis:** Price swaptions under different yield curves
4. **Sensitivity analysis:** Show Greeks via finite differences
5. **Results:** Include pricing tables (Example 1)

### Citation:
Use the Bachelier model reference:
> Bachelier, L. (1900). "Theory of Speculation"

Or modern reference:
> Heath, D., Jarrow, R., and Morton, A. (1992). "Bond pricing and the term structure of interest rates"

---

## 🏁 Getting Started Checklist

- [ ] Read QUICKSTART.md (5 min)
- [ ] Run first example (1 min)
- [ ] Verify output makes sense (1 min)
- [ ] Read relevant guide based on your use case
- [ ] Try 2-3 examples yourself
- [ ] Integrate into your code/thesis
- [ ] ✅ Done! You're ready to use norm vol pricing

---

## 📝 Version Info

- **Modified:** price_options.py
- **Added:** 5 documentation files
- **Status:** ✅ Complete and tested
- **Compatibility:** ✅ Fully backward compatible
- **Syntax:** ✅ Verified

---

**Welcome to swaption pricing with normal volatility!** 🚀

Start with QUICKSTART.md and enjoy the pricing!

