# =============================================================================
# Training_augmented_input.py
#
# Experiment: baseline autoencoder with augmented encoder input.
# The encoder receives the 8 swap rates PLUS 3 derived shape features:
#   - 10Y − 1Y   (short slope)
#   - 30Y − 10Y  (long slope)
#   - 2×10Y − 1Y − 30Y  (curvature / butterfly)
# The decoder still outputs 8 swap rates only.
# The loss is MSE over all 11 values (swap rates + derived features),
# giving each element equal weight.
#
# Nothing in the existing baseline pipeline is touched.
# Outputs go to:
#   Figures/TrainingResults/dim{N}_augmented_input/ep{E}/
#   checkpoints/fullmodel_augmented_input_dim{N}_ep{E}.pt
# =============================================================================

import time
import numpy as np
import os
import sys
import torch
import torch.nn as nn

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import json
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader

# ── environment setup ─────────────────────────────────────────────────────────
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    REPO_ROOT = os.getcwd()  # PyCharm console: CWD is the project root

for _p in [REPO_ROOT, os.path.dirname(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Code.load_swapdata import my_data, custom_palette
from Code.model.full_model import FullModel
from Code.utils import helpers as H

print("Torch:", torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
torch.backends.mkldnn.enabled = True

# ── settings ──────────────────────────────────────────────────────────────────
LATENT_DIM   = 4
EPOCHS       = 5000
BATCH_SIZE   = 32
EVAL_BATCH_SIZE = 256
EVAL_EVERY   = 1
LOG_EVERY    = 100
TARGET_MSE   = 1e-8
USE          = "bbg"

INPUT_DIM_ORIG = 8   # original swap-rate dimension
INPUT_DIM_AUG  = 11  # augmented encoder input dimension (8 rates + 3 derived features)

FIGURES_DIR = os.path.join(
    REPO_ROOT, "Figures", "TrainingResults",
    f"dim{LATENT_DIM}_augmented_input", f"ep{EPOCHS}"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── augmentation ──────────────────────────────────────────────────────────────
# Tenor order: [1, 2, 3, 5, 10, 15, 20, 30]
#              idx: 0  1  2  3   4   5   6   7

def augment(x: torch.Tensor) -> torch.Tensor:
    """Append 3 derived shape features to the 8-dim swap-rate vector."""
    f1 = x[:, 4] - x[:, 0]                          # 10Y − 1Y
    f2 = x[:, 7] - x[:, 4]                          # 30Y − 10Y
    f3 = 2.0 * x[:, 4] - x[:, 0] - x[:, 7]         # 2×10Y − 1Y − 30Y
    return torch.cat([x, f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)

def compute_feats(x: torch.Tensor) -> torch.Tensor:
    """Return only the 3 derived features (B, 3)."""
    f1 = x[:, 4] - x[:, 0]
    f2 = x[:, 7] - x[:, 4]
    f3 = 2.0 * x[:, 4] - x[:, 0] - x[:, 7]
    return torch.stack([f1, f2, f3], dim=1)

# ── data ──────────────────────────────────────────────────────────────────────
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor      = X_tensor.float()
X_tensor_full = X_tensor_full.float()

X_aug      = augment(X_tensor)       # (N, 11) — encoder input
X_aug_full = augment(X_tensor_full)  # for eval

dataset = TensorDataset(X_aug, X_tensor)   # (enc_input, orig_target)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# ── model ─────────────────────────────────────────────────────────────────────
SEED = 0
torch.manual_seed(SEED)
model = FullModel(input_dim=INPUT_DIM_AUG, latent_dim=LATENT_DIM).to(device)
model.train()

PCT_START        = 0.3
DIV_FACTOR       = 1.0
FINAL_DIV_FACTOR = 3000.0
max_lr = 1e-3

optim = torch.optim.Adam(model.parameters(), lr=max_lr)
scheduler = OneCycleLR(
    optim,
    max_lr=max_lr,
    steps_per_epoch=len(loader),
    epochs=EPOCHS,
    pct_start=PCT_START,
    div_factor=DIV_FACTOR,
    final_div_factor=FINAL_DIV_FACTOR,
)
loss_fn = nn.MSELoss()

# ── helpers ───────────────────────────────────────────────────────────────────
def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)

USE_SET_TO_NONE = True

@torch.no_grad()
def predict_S_hat(X_orig: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Run encoder on augmented input; return 8-dim S_hat."""
    was_training = model.training
    model.eval()
    outs = []
    for i in range(0, X_orig.shape[0], batch_size):
        xb      = X_orig[i:i + batch_size].to(device)
        xb_aug  = augment(xb)
        S_hat   = model(xb_aug)
        outs.append(S_hat.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(outs, dim=0)

def eval_rmse_bps(X_orig: torch.Tensor, meta_df: pd.DataFrame, batch_size: int = 256):
    S_hat_all = predict_S_hat(X_orig, batch_size)
    mask  = row_finite_mask(X_orig) & row_finite_mask(S_hat_all)
    n_bad  = int((~mask).sum().item())
    n_good = int(mask.sum().item())
    X_eval    = X_orig[mask]
    S_eval    = S_hat_all[mask]
    meta_eval = meta_df.loc[mask.numpy()].reset_index(drop=True)
    rmse_per_ccy = H.rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)
    avg_rmse_bps = float(rmse_per_ccy.mean())
    return rmse_per_ccy, avg_rmse_bps, n_bad, n_good

# ── CSV logger ────────────────────────────────────────────────────────────────
ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]
csv_path  = os.path.join(FIGURES_DIR, f"train_rmse_log_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.csv")

torch_version  = torch.__version__
python_version = sys.version.split()[0]
numpy_version  = np.__version__

csv_cols = (
    ["epoch", "time_total_sec", "time_interval_sec", "train_mse", "train_rmse",
     "avg_rmse_bps", "n_good", "n_bad"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
    + ["torch_version", "python_version", "numpy_version"]
)
pd.DataFrame(columns=csv_cols).to_csv(csv_path, index=False)
print("Logging to:", csv_path)

run_config = {
    "seed": SEED, "latent_dim": LATENT_DIM, "input_dim": INPUT_DIM_AUG,
    "variant": "augmented_input", "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "max_lr": max_lr, "pct_start": PCT_START, "div_factor": DIV_FACTOR,
    "final_div_factor": FINAL_DIV_FACTOR,
    "torch_version": torch_version, "python_version": python_version,
    "numpy_version": numpy_version,
}
with open(os.path.join(FIGURES_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

# ── training loop ─────────────────────────────────────────────────────────────
train_losses       = []
lrs_per_step       = []
lrs_per_epoch      = []
avg_rmse_bps_hist  = []
nan_batches_total  = 0

t0         = time.perf_counter()
t_last_log = t0

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    n_obs   = 0
    nan_batches = 0

    for batch_idx, (xb_aug, xb) in enumerate(loader):
        xb_aug = xb_aug.to(device)
        xb     = xb.to(device)

        optim.zero_grad(set_to_none=USE_SET_TO_NONE)

        try:
            S_hat = model(xb_aug)
        except Exception as e:
            nan_batches += 1
            print(f"    [Forward error epoch={epoch} batch={batch_idx}]: {str(e)[:200]}")
            continue

        if not torch.isfinite(S_hat).all():
            nan_batches += 1
            print(f"    [S_hat NaN/Inf epoch={epoch} batch={batch_idx}]")
            continue

        # augmented loss: MSE over swap rates + derived features (11 values)
        feat_true = compute_feats(xb)
        feat_hat  = compute_feats(S_hat)
        loss = loss_fn(
            torch.cat([S_hat, feat_hat],  dim=1),  # (B, 11)
            torch.cat([xb,   feat_true],  dim=1),  # (B, 11)
        )

        if not torch.isfinite(loss):
            nan_batches += 1
            print(f"    [Loss NaN/Inf epoch={epoch} batch={batch_idx}]")
            continue

        loss.backward()

        has_nan_grad = any(
            param.grad is not None and not torch.isfinite(param.grad).all()
            for _, param in model.named_parameters()
        )
        if has_nan_grad:
            nan_batches += 1
            print(f"    [NaN grad epoch={epoch} batch={batch_idx}]")
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

        lrs_per_step.append(optim.param_groups[0]["lr"])
        running += float(loss.detach().cpu()) * xb.shape[0]
        n_obs   += xb.shape[0]

    nan_batches_total += nan_batches
    epoch_mse  = running / max(n_obs, 1)
    epoch_rmse = epoch_mse ** 0.5
    train_losses.append(epoch_mse)
    lrs_per_epoch.append(optim.param_groups[0]["lr"])

    do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch == 0) or (epoch == EPOCHS - 1)
    do_log  = ((epoch + 1) % LOG_EVERY  == 0) or (epoch == 0) or (epoch == EPOCHS - 1)

    if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE and n_obs > 0:
        do_eval = do_log = True

    if do_eval:
        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = eval_rmse_bps(
            X_tensor, meta, batch_size=EVAL_BATCH_SIZE
        )
        avg_rmse_bps_hist.append((epoch, avg_rmse_bps))
    else:
        rmse_per_ccy, avg_rmse_bps, n_bad, n_good = None, np.nan, np.nan, np.nan

    if do_log:
        t_now          = time.perf_counter()
        time_total     = t_now - t0
        time_interval  = t_now - t_last_log
        t_last_log     = t_now

        row = {
            "epoch": epoch, "time_total_sec": time_total,
            "time_interval_sec": time_interval, "train_mse": epoch_mse,
            "train_rmse": epoch_rmse, "avg_rmse_bps": float(avg_rmse_bps),
            "n_good": int(n_good) if np.isfinite(n_good) else np.nan,
            "n_bad":  int(n_bad)  if np.isfinite(n_bad)  else np.nan,
        }
        for ccy in ccy_order:
            row[f"rmse_bps_{ccy}"] = float(rmse_per_ccy.get(ccy, np.nan)) if rmse_per_ccy is not None else np.nan
        row["torch_version"]  = torch_version
        row["python_version"] = python_version
        row["numpy_version"]  = numpy_version

        pd.DataFrame([row], columns=csv_cols).to_csv(csv_path, mode="a", header=False, index=False)
        print(
            f"epoch={epoch:4d} train_rmse={epoch_rmse:.6e} "
            f"avg_rmse_bps={avg_rmse_bps:.3f} lr={optim.param_groups[0]['lr']:.2e} "
            f"used_obs={n_obs} nan_batches={nan_batches} "
            f"time_total={time_total/60:.1f}min"
        )

    if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE and n_obs > 0:
        print(f"[STOP] epoch={epoch} train_rmse={epoch_rmse:.6e}")
        break

print("Training done.")

# ── plots ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(np.arange(len(lrs_per_step)), lrs_per_step, linewidth=1.0)
ax.set_xlabel("Training step"); ax.set_ylabel("Learning rate")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, f"lr_schedule_dim{LATENT_DIM}_ep{EPOCHS}.png"), dpi=300)
plt.close(fig)

if avg_rmse_bps_hist:
    epochs_logged = [e for e, _ in avg_rmse_bps_hist]
    avg_logged    = [v for _, v in avg_rmse_bps_hist]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(epochs_logged, avg_logged, linewidth=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Average RMSE (bps)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"avg_rmse_bps_convergence_dim{LATENT_DIM}_ep{EPOCHS}.png"), dpi=300)
    plt.close(fig)

# ── save checkpoint ───────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ckpt_name = f"fullmodel_augmented_input_{USE}_dim{LATENT_DIM}_ep{EPOCHS}.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {
        "latent_dim": LATENT_DIM,
        "input_dim":  INPUT_DIM_AUG,
    },
    "latent_dim": LATENT_DIM,
    "input_dim":  INPUT_DIM_AUG,
    "epochs":     EPOCHS,
    "use_data":   USE,
    "variant":    "augmented_input",
}, os.path.join(CHECKPOINT_DIR, ckpt_name))
print("Saved checkpoint:", ckpt_name)

# Also save plain state_dict alongside training logs
figures_ckpt = os.path.join(FIGURES_DIR, f"checkpoint_dim{LATENT_DIM}_ep{EPOCHS}.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {"latent_dim": LATENT_DIM, "input_dim": INPUT_DIM_AUG},
}, figures_ckpt)
print("Saved figures checkpoint:", figures_ckpt)
