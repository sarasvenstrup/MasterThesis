## Quick Start Checklist ✓

Your model is now ready to run Training, Plots, and ResultsGenerator!

### Pre-Run Checks

- [ ] Install torchdiffeq: `pip install torchdiffeq`
- [ ] Verify swap data exists: `ls SwapData/BloombergData/`
- [ ] Run compatibility test: `python test_model_compatibility.py`

### Running Scripts

**Option 1: Just Train (generates model checkpoint)**
```bash
cd C:\Users\Bruger\PycharmProjects\MasterThesis
python Code/Training.py
# Output: Figures/dim2/ep100/
```

**Option 2: Train + Generate Plots**
```bash
python Code/Training.py
python Code/Plots.py
# Outputs: Figures/dim2/ep100/ (both scripts)
```

**Option 3: Generate Thesis Results (requires pre-computed OOS)**
```bash
python Code/ResultsGenerator.py
# Requires existing: Figures/OOS_split_dim3/ep2500/
# Output: Figures/thesis_results/AutoencoderPerformance/
```

### Expected Runtime

- **Training.py** (100 epochs, dim=2): ~5-15 min (CPU) or ~1-3 min (GPU)
- **Plots.py**: ~2-5 min
- **ResultsGenerator.py**: ~5-10 min (if all dependencies present)

### Outputs Location

After running Training:
```
MasterThesis/
├── Figures/
│   ├── dim2/
│   │   └── ep100/
│   │       ├── train_rmse_log_bbg_dim2_ep100.csv     ← Training log
│   │       ├── checkpoint_dim2_ep100.pt              ← Model weights
│   │       ├── avg_rmse_bps_convergence_*.png        ← Convergence plot
│   │       └── lr_schedule_*.png                     ← LR schedule plot
│   └── thesis_results/                               ← ResultsGenerator output
│       └── AutoencoderPerformance/
├── checkpoints/
│   └── fullmodel_bbg_dim2_ep100.pt                   ← Backup checkpoint
└── Tables/
    └── (ResultsGenerator CSV tables)
```

### Troubleshooting

**Issue: "torchdiffeq not found"**
```bash
pip install torchdiffeq
```

**Issue: "swap data not found"**
Check: `ls SwapData/BloombergData/` contains subdirectories (ad, cd, dk, eu, jy, nk, sw, uk, us)

**Issue: NaN/Inf during training**
This is handled gracefully. Check `train_rmse_log_bbg_dim2_ep100.csv` for `nan_batches` count.

**Issue: CUDA out of memory**
Reduce `BATCH_SIZE` in Training.py (currently 32) or use CPU mode.

### Key Files Modified

✓ `Code/model/full_model.py` — Updated forward() for 10-tuple output
✓ `Code/__init__.py` — Created (package initialization)
✓ `Code/model/__init__.py` — Created (package initialization)
✓ `Code/utils/__init__.py` — Created (package initialization)

### Verification

Run anytime to verify everything still works:
```bash
python test_model_compatibility.py
```

All 6 tests should pass ✓

---

**You're ready to go! Pick an option above and run it. 🚀**

