## You Can Now Run

Your model is fully compatible with all three scripts. Here's what to do:

### 🎯 Recommended Order

#### Step 1: Verify Everything Works (2 minutes)
```bash
python test_model_compatibility.py
```
Expected output: "ALL TESTS PASSED ✓"

#### Step 2: Train the Model (5-15 minutes)
```bash
python Code/Training.py
```
Outputs:
- `Figures/dim2/ep100/train_rmse_log_bbg_dim2_ep100.csv` — Training metrics
- `Figures/dim2/ep100/checkpoint_dim2_ep100.pt` — Model weights
- `Figures/dim2/ep100/avg_rmse_bps_convergence_*.png` — Convergence plots
- `Figures/dim2/ep100/lr_schedule_*.png` — Learning rate schedule
- `checkpoints/fullmodel_bbg_dim2_ep100.pt` — Backup checkpoint

#### Step 3: Generate Plots (2-5 minutes)
```bash
python Code/Plots.py
```
Generates detailed visualization plots from trained model

#### Step 4: Generate Thesis Results (5-10 minutes, optional)
```bash
python Code/ResultsGenerator.py
```
Requires pre-computed OOS split results. Output:
- `Figures/thesis_results/AutoencoderPerformance/` — Result figures
- `Tables/` — Result tables (CSV)

---

### 🚀 Quick Start (Just Run Training)

```bash
cd C:\Users\Bruger\PycharmProjects\MasterThesis
python Code/Training.py
```

That's it! This will:
1. Load Bloomberg swap data
2. Train FullModel for 100 epochs (dim=2)
3. Save checkpoint and plots
4. Log metrics to CSV

---

### 📊 Monitor Training

While Training.py runs, it prints updates like:
```
epoch=  0 train_rmse=1.234567e-01 avg_rmse_bps=45.23 lr=3.68e-04 ...
epoch= 10 train_rmse=8.234567e-02 avg_rmse_bps=35.10 lr=5.23e-04 ...
...
```

Check progress by looking at:
- `train_rmse`: Main loss metric
- `avg_rmse_bps`: Average basis points error
- `time_total`: Elapsed time

---

### 📁 What You'll Get

After running these scripts:

```
MasterThesis/
├── Figures/
│   ├── dim2/
│   │   └── ep100/
│   │       ├── train_rmse_log_bbg_dim2_ep100.csv
│   │       ├── checkpoint_dim2_ep100.pt
│   │       ├── avg_rmse_bps_convergence_bbg_dim2_ep100.png
│   │       ├── lr_schedule_bbg_dim2_ep100.png
│   │       └── [other plot outputs from Plots.py]
│   └── thesis_results/
│       ├── AutoencoderPerformance/
│       │   ├── Q1a_IS_rmse_all_dims.png
│       │   ├── Q1e_training_loss_curves.png
│       │   └── [many more thesis figures]
│       ├── parameters/
│       └── ExtraFigures/
├── checkpoints/
│   └── fullmodel_bbg_dim2_ep100.pt
├── Tables/
│   ├── Q1a_IS_rmse_all_dims.csv
│   └── [more result tables]
└── [your data files]
```

---

### ⚙️ Configuration (if needed)

Edit these files to customize:

**Training.py** (line 45-70):
- `LATENT_DIM = 2` — Latent dimension (change to 1, 3, or 4)
- `EPOCHS = 100` — Number of epochs
- `BATCH_SIZE = 32` — Batch size
- `max_lr = 1e-3` — Learning rate

**Plots.py** (line 55-58):
- `LATENT_DIM = 2` — Must match trained model
- `EPOCHS = 1500` — Epochs for loading checkpoints

**ResultsGenerator.py** (line 90):
- `LATENT_DIM = 3` — Change to match your model (1, 2, 3, or 4)

---

### 🔧 Troubleshooting

**Error: "torchdiffeq not found"**
```bash
pip install torchdiffeq
```

**Error: "No module named 'Code'"**
- Make sure you're in the repo root: `cd C:\Users\Bruger\PycharmProjects\MasterThesis`

**Error: "Data not found"**
- Check Bloomberg data exists: `dir SwapData\BloombergData\`
- Should contain: ad/, cd/, dk/, eu/, jy/, nk/, sw/, uk/, us/

**Error: CUDA out of memory**
- Edit Training.py: change `BATCH_SIZE = 32` to `BATCH_SIZE = 16`

**Error: Training diverges (NaNs)**
- This is handled gracefully
- Check CSV for nan_batches count
- Try reducing max_lr in Training.py

---

### ✅ Verification

Anytime you want to verify nothing broke:

```bash
python test_model_compatibility.py
```

Should show:
```
✓ TEST 1: Module imports
✓ TEST 2: Model instantiation  
✓ TEST 3: 10-tuple output (ResultsGenerator pattern)
✓ TEST 4: Auxiliary dict output (return_aux=True)
✓ TEST 5: Batch inference (32 samples)
✓ TEST 6: Single sample with squeeze_back

ALL TESTS PASSED ✓
```

---

## 🎉 You're Ready!

Choose your path:

**Just curious?** → `python test_model_compatibility.py`  
**Want to train?** → `python Code/Training.py`  
**Full pipeline?** → Run all three in order above  
**Generate thesis results?** → `python Code/ResultsGenerator.py` (after training)

---

**Start with Step 1 above. Everything else flows from there! 🚀**

