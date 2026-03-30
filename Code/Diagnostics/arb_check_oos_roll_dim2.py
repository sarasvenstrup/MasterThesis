"""
arb_check_oos_roll_dim2.py
──────────────────────────
Checks the approximate arbitrage condition (Sharpe ratio proxy SR_tau)
on the OOS test observations from the dim=2 rolling windows,
for both the baseline and stable model variants.

Logic:
  1. Read the train5Y_test6M rolling CSV to collect all test date ranges.
  2. Filter the full dataset to observations that fall in those test periods.
  3. Load the globally-trained IS checkpoint for each variant:
       baseline → Figures/TrainingResults/dim2_baseline/ep5000/checkpoint_dim2_ep5000.pt
       stable   → Figures/TrainingResults/dim2_stable/ep2500/checkpoint_dim2_ep2500.pt
  4. Run forward pass with do_arb_checks=True on the OOS test observations.
  5. Plot SR_tau mean ± 1 std band over tau=1..30 for both variants,
     side by side and overlaid.

Run from repo root:
    python Code/Diagnostics/arb_check_oos_roll_dim2.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ── path setup ──────────────────────────────────────────────────────────────
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
except NameError:
    # running interactively (PyCharm console, Jupyter, etc.)
    REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    # if cwd is already the repo root, use it directly
    if not os.path.isdir(os.path.join(REPO_ROOT, "Code")):
        REPO_ROOT = os.getcwd()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print(f"Repo root: {REPO_ROOT}")

from Code.load_swapdata import my_data, custom_palette, set_paper_theme
from Code.model.full_model import FullModel

set_paper_theme()
device = torch.device("cpu")

# ── config ───────────────────────────────────────────────────────────────────
LATENT_DIM   = 2
ROLL_SUBDIR  = "train5Y_test6M_step6M"
ROLL_EP      = 2500

VARIANTS = {
    "baseline": {
        "ckpt": os.path.join(REPO_ROOT, "Figures", "TrainingResults",
                             "dim2_baseline", "ep5000",
                             "checkpoint_dim2_ep5000.pt"),
        "color": custom_palette[0],
        "label": "Baseline",
    },
    "stable": {
        "ckpt": os.path.join(REPO_ROOT, "Figures", "TrainingResults",
                             "dim2_stable", "ep2500",
                             "checkpoint_dim2_ep2500.pt"),
        "color": custom_palette[2],
        "label": "Stable",
    },
}

ROLL_CSV = os.path.join(
    REPO_ROOT, "Figures", "OOSResults", "Roll",
    "OOS_roll_dim2_baseline",
    ROLL_SUBDIR, f"ep{ROLL_EP}",
    f"oos_rolling_bbg_dim2_{ROLL_SUBDIR}.csv",
)

OUT_DIR = os.path.join(
    REPO_ROOT, "Figures", "thesis_results", "AutoencoderPerformance"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. load rolling CSV and collect OOS test dates ───────────────────────────
if not os.path.exists(ROLL_CSV):
    raise FileNotFoundError(f"Rolling CSV not found: {ROLL_CSV}")

df_roll = pd.read_csv(ROLL_CSV)
df_roll["test_start"] = pd.to_datetime(df_roll["test_start"])
df_roll["test_end"]   = pd.to_datetime(df_roll["test_end"])
print(f"Rolling windows: {len(df_roll)}  "
      f"({df_roll['test_start'].min().date()} → {df_roll['test_end'].max().date()})")

# ── 2. load full dataset and filter to OOS test observations ─────────────────
print("Loading swap data...")
_, _, meta_full, X_full_raw, *_ = my_data(use="bbg")

meta_full = meta_full.copy()
meta_full["as_of_date"] = pd.to_datetime(meta_full["as_of_date"])

# mark each observation as OOS if it falls in any rolling test window
oos_mask = pd.Series(False, index=meta_full.index)
for _, row in df_roll.iterrows():
    in_window = (
        (meta_full["as_of_date"] >= row["test_start"]) &
        (meta_full["as_of_date"] <= row["test_end"])
    )
    oos_mask |= in_window

print(f"OOS test observations: {oos_mask.sum()} / {len(meta_full)}")

X_oos   = X_full_raw[oos_mask.values].float()
meta_oos = meta_full.loc[oos_mask].reset_index(drop=True)

# ── 3. arbitrage check helper ─────────────────────────────────────────────────
@torch.no_grad()
def compute_sr(model, X, batch=256):
    """Return SR_tau tensor (N, 30) for all observations in X."""
    model.eval()
    sr_list = []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].to(device)
        _, aux = model(xb, return_aux=True, do_arb_checks=True)
        sr_list.append(aux["arb"]["SR_tau"].cpu())
    return torch.cat(sr_list)   # (N, 30)

# ── 4. run both variants ──────────────────────────────────────────────────────
results = {}
for variant, cfg in VARIANTS.items():
    ckpt = cfg["ckpt"]
    if not os.path.exists(ckpt):
        warnings.warn(f"Checkpoint not found for {variant}: {ckpt} — skipping.")
        continue

    print(f"\nLoading {variant} checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model = FullModel(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"  Running arbitrage check on {len(X_oos)} OOS observations...")
    SR = compute_sr(model, X_oos)   # (N, 30)

    finite_mask = torch.isfinite(SR).all(dim=1)
    SR_valid = SR[finite_mask].numpy()
    print(f"  Valid observations: {finite_mask.sum().item()} / {len(SR)}")

    results[variant] = {
        "SR":    SR_valid,
        "mean":  SR_valid.mean(axis=0),
        "std":   SR_valid.std(axis=0),
        "color": cfg["color"],
        "label": cfg["label"],
    }

tau_grid = np.arange(1, 31)

# ── 5a. side-by-side plot ─────────────────────────────────────────────────────
n_panels = len(results)
fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.5), sharey=True)
if n_panels == 1:
    axes = [axes]

for ax, (variant, res) in zip(axes, results.items()):
    mean, std = res["mean"], res["std"]
    color = res["color"]

    ax.fill_between(tau_grid, mean - std, mean + std,
                    alpha=0.20, color=color)
    ax.plot(tau_grid, mean, linewidth=2.0, color=color, label="Mean SR")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(res["label"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Tenor (years)")
    if ax is axes[0]:
        ax.set_ylabel("Approx. Sharpe ratio $SR_\\tau$")
    ax.set_xticks(tau_grid[::2])
    ax.text(0.97, 0.97,
            f"N={res['SR'].shape[0]:,}\n"
            f"max |SR| = {np.abs(mean).max():.3f}",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

fig.suptitle(
    f"OOS arbitrage check — $\\ell=2$, rolling {ROLL_SUBDIR}",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "arb_check_oos_roll_dim2_sidebyside.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")

# ── 5b. overlaid plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

for variant, res in results.items():
    mean, std = res["mean"], res["std"]
    color = res["color"]
    ax.fill_between(tau_grid, mean - std, mean + std, alpha=0.15, color=color)
    ax.plot(tau_grid, mean, linewidth=2.0, color=color, label=res["label"])

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Tenor (years)")
ax.set_ylabel("Approx. Sharpe ratio $SR_\\tau$")
ax.set_title(
    f"OOS arbitrage check — $\\ell=2$, rolling {ROLL_SUBDIR}",
    fontsize=11, fontweight="bold"
)
ax.set_xticks(tau_grid[::2])
ax.legend(fontsize=10, frameon=False)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "arb_check_oos_roll_dim2_overlaid.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ── 5c. summary table ─────────────────────────────────────────────────────────
print("\n── Summary ─────────────────────────────────────────────────")
print(f"{'Variant':<12}  {'Mean |SR|':<12}  {'Max |SR|':<12}  "
      f"{'% SR > 0.1':<14}  {'N obs'}")
for variant, res in results.items():
    sr = res["mean"]
    pct = (np.abs(res["SR"]) > 0.1).mean() * 100
    print(f"{res['label']:<12}  {np.abs(sr).mean():<12.4f}  "
          f"{np.abs(sr).max():<12.4f}  {pct:<14.1f}  {res['SR'].shape[0]}")
