# ============================= Import Packages ===============================
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# ============================= Environment Setup & Imports ===============================
try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Code.utils import helpers as H
from Code.load_swapdata import my_data, custom_palette, TARGET_TENORS
from Code import config
config.confirm_variant()
from Code.model.full_model_stable import FullModel

VARIANT = config.VARIANT

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.mkldnn.enabled = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)
print("MKLDNN enabled:", torch.backends.mkldnn.enabled)

# ============================= Config ===============================
USE = "bbg"
LATENT_DIM = 2

#

# Recommended rolling window setup (baseline OOS)
TRAIN_YEARS = 5
TEST_MONTHS = 6
STEP_MONTHS = 6
MIN_TRAIN_OBS = 200

# Training setup per window
EPOCHS = 3500
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
TARGET_MSE = -1          # set >0 if you want early stop
LOG_EVERY = 100          # training printouts inside each window

max_lr = 1e-3
final_div_factor = 3000.0

ccy_order = ["AUD", "CAD", "DKK", "EUR", "JPY", "NOK", "SEK", "GBP", "USD"]

# Output folders
FIGURES_DIR = os.path.join(
    REPO_ROOT,
    "Figures",
    "OOSResults",
    "Roll",
    f"OOS_roll_dim{LATENT_DIM}_{VARIANT}",
    f"train{TRAIN_YEARS}Y_test{TEST_MONTHS}M_step{STEP_MONTHS}M",
    f"ep{EPOCHS}"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

oos_csv_path = os.path.join(
    FIGURES_DIR,
    f"oos_rolling_{USE}_dim{LATENT_DIM}_train{TRAIN_YEARS}Y_test{TEST_MONTHS}M_step{STEP_MONTHS}M.csv"
)

# Combined per-roll CSVs (one file per type, appended window by window)
PRED_TEST_CSV  = os.path.join(FIGURES_DIR, "predictions_test_all.csv")
PRED_TRAIN_CSV = os.path.join(FIGURES_DIR, "predictions_train_all.csv")
LATENT_Z_CSV   = os.path.join(FIGURES_DIR, "latent_z_all.csv")
PARAMETERS_CSV = os.path.join(FIGURES_DIR, "parameters_all.csv")

# ============================= Load Data ===============================
meta, X_tensor, meta_full, X_tensor_full, tenors, df_wide, df_wide_all, SCALE_IS_PERCENT = my_data(use=USE)
X_tensor = X_tensor_full.float()

meta = meta_full.copy()
meta["as_of_date"] = pd.to_datetime(meta["as_of_date"])
meta = meta.reset_index(drop=True)

assert len(meta) == X_tensor.shape[0], "meta and X_tensor length mismatch"

# ============================= Helpers ===============================
def row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).all(dim=1)

@torch.no_grad()
def predict_S_hat(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    outs = []
    N = X.shape[0]
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size].to(device)
        out = model(xb)
        outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)

def rmse_bps_on_subset(model: nn.Module, X_sub: torch.Tensor, meta_sub: pd.DataFrame):
    S_hat = predict_S_hat(model, X_sub, batch_size=EVAL_BATCH_SIZE)

    mask = row_finite_mask(X_sub) & row_finite_mask(S_hat)
    n_bad = int((~mask).sum().item())
    n_good = int(mask.sum().item())

    X_eval = X_sub[mask]
    S_eval = S_hat[mask]
    meta_eval = meta_sub.loc[mask.numpy()].reset_index(drop=True)

    rmse_per_ccy = H.rmse_bps_per_currency_paper(X_eval, S_eval, meta_eval)
    avg_rmse_bps = float(rmse_per_ccy.mean())  # unweighted mean across currencies (recommended)

    return rmse_per_ccy, avg_rmse_bps, n_good, n_bad

def make_loader(X_sub: torch.Tensor, batch_size: int):
    ds = TensorDataset(X_sub)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

WINDOW_SEED = 0  # fixed seed for every rolling window — change to reproduce

def train_one_window(X_train: torch.Tensor):
    torch.manual_seed(WINDOW_SEED)
    np.random.seed(WINDOW_SEED)

    model = FullModel(latent_dim=LATENT_DIM).to(device)
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=max_lr)
    loader = make_loader(X_train, BATCH_SIZE)

    scheduler = OneCycleLR(
        optim,
        max_lr=max_lr,
        steps_per_epoch=len(loader),
        epochs=EPOCHS,
        pct_start=0.3,
        div_factor=1.0,
        final_div_factor=final_div_factor,
    )

    loss_fn = nn.MSELoss()
    train_mse_hist = []
    lr_hist = []
    nan_batches_total = 0

    for epoch in range(EPOCHS):
        running = 0.0
        n_obs = 0
        nan_batches = 0

        for batch_idx, (xb_cpu,) in enumerate(loader):
            xb = xb_cpu.to(device)

            optim.zero_grad(set_to_none=True)
            
            try:
                S_hat = model(xb)
            except Exception as e:
                # Catch any exceptions during forward pass (e.g., ODE solver issues)
                nan_batches += 1
                print(f"      [Forward error at epoch {epoch}, batch {batch_idx}]: {str(e)[:200]}")
                print(f"        Input range: [{xb.min():.3e}, {xb.max():.3e}], shape: {xb.shape}")
                continue

            # Check if S_hat contains NaN/Inf
            if not torch.isfinite(S_hat).all():
                nan_batches += 1
                nan_count = (~torch.isfinite(S_hat)).sum().item()
                print(f"      [S_hat has NaN/Inf at epoch {epoch}, batch {batch_idx}]")
                print(f"        S_hat contains {nan_count} NaN/Inf values out of {S_hat.numel()}")
                finite_vals = S_hat[torch.isfinite(S_hat)]
                if finite_vals.numel() > 0:
                    print(f"        S_hat range: [{finite_vals.min():.3e}, {finite_vals.max():.3e}]")
                else:
                    print(f"        S_hat range: all values are NaN/Inf")
                print(f"        Input range: [{xb.min():.3e}, {xb.max():.3e}]")
                continue

            loss = loss_fn(S_hat, xb)
            
            if not torch.isfinite(loss):
                nan_batches += 1
                print(f"      [Loss is NaN/Inf at epoch {epoch}, batch {batch_idx}]")
                print(f"        Loss value: {loss}")
                print(f"        S_hat stats: min={S_hat.min():.3e}, max={S_hat.max():.3e}, mean={S_hat.mean():.3e}")
                print(f"        Input stats: min={xb.min():.3e}, max={xb.max():.3e}, mean={xb.mean():.3e}")
                continue

            loss.backward()
            
            # Check for NaN/Inf in gradients before clipping
            has_nan_grad = False
            nan_grad_params = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        has_nan_grad = True
                        nan_count = (~torch.isfinite(param.grad)).sum().item()
                        grad_range = param.grad[torch.isfinite(param.grad)]
                        if len(grad_range) > 0:
                            grad_min, grad_max = grad_range.min().item(), grad_range.max().item()
                        else:
                            grad_min, grad_max = float('nan'), float('nan')
                        nan_grad_params.append({
                            'name': name,
                            'nan_count': nan_count,
                            'total': param.grad.numel(),
                            'grad_min': grad_min,
                            'grad_max': grad_max,
                        })
            
            if has_nan_grad:
                nan_batches += 1
                print(f"      [Gradients contain NaN/Inf at epoch {epoch}, batch {batch_idx}]")
                for param_info in nan_grad_params:
                    print(f"        {param_info['name']}: {param_info['nan_count']}/{param_info['total']} NaN")
                    print(f"          Valid grad range: [{param_info['grad_min']:.3e}, {param_info['grad_max']:.3e}]")
                print(f"        Loss was: {loss:.3e}")
                print(f"        S_hat stats: min={S_hat.min():.3e}, max={S_hat.max():.3e}")
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            running += float(loss.detach().cpu()) * xb.shape[0]
            n_obs += xb.shape[0]

        nan_batches_total += nan_batches
        epoch_mse = running / max(n_obs, 1)
        train_mse_hist.append(epoch_mse)
        lr_hist.append(optim.param_groups[0]["lr"])

        if TARGET_MSE > 0 and epoch_mse <= TARGET_MSE:
            break

        if (epoch == 0) or ((epoch + 1) % LOG_EVERY == 0) or (epoch == EPOCHS - 1):
            print(
                f"  epoch={epoch:4d} train_rmse={(epoch_mse**0.5):.6e} "
                f"lr={optim.param_groups[0]['lr']:.2e} used_obs={n_obs} "
                f"nan_batches={nan_batches} total_nan_batches={nan_batches_total}"
            )

    return model, np.array(train_mse_hist), np.array(lr_hist)

# ============================= Per-roll saving helpers ===============================
@torch.no_grad()
def get_latent(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    zs = []
    for i in range(0, len(X), batch_size):
        zs.append(model.encoder(X[i:i+batch_size].to(device)).cpu())
    return torch.cat(zs, dim=0)

def _param_label(name):
    if name.startswith("mu_"):
        k = name.split("_")[1]; return r"$\mu_{" + k + r"}$"
    if name.startswith("sigma_"):
        k = name.split("_")[1]; return r"$\sigma_{" + k + r"}$"
    if name.startswith("rho_"):
        ij = name.split("_")[1]; return r"$\rho_{" + ",".join(ij) + r"}$"
    if name == "r_tilde":
        return r"$\tilde{r}$"
    return name

@torch.no_grad()
def extract_parameters(model: nn.Module, X: torch.Tensor, meta_sub: pd.DataFrame) -> pd.DataFrame:
    model.eval()
    mask = row_finite_mask(X)
    X_m  = X[mask].to(device)
    z       = model.encoder(X_m)
    mu      = model.K(z)
    sigmas, rhos = model.H(z)
    r_til   = model.R(z).squeeze(-1)
    d = model.latent_dim
    rec = meta_sub.loc[mask.numpy()].copy().reset_index(drop=True)
    rec["as_of_date"] = pd.to_datetime(rec["as_of_date"])
    for k in range(d):
        rec[f"mu_{k+1}"]    = mu[:, k].cpu().numpy()
        rec[f"sigma_{k+1}"] = sigmas[:, k].cpu().numpy()
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            rec[f"rho_{i+1}{j+1}"] = rhos[:, idx].cpu().numpy()
            idx += 1
    rec["r_tilde"] = r_til.cpu().numpy()
    return rec

def _append_csv(df: pd.DataFrame, path: str):
    """Append df to path, writing header only if the file doesn't exist yet."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def save_roll_outputs(model: nn.Module, X_train: torch.Tensor, meta_train_sub: pd.DataFrame,
                      test_start,
                      X_test: torch.Tensor = None, meta_test: pd.DataFrame = None):
    """Append per-roll outputs to combined CSVs (latent, parameters, predictions train/test).
    Each row is tagged with test_start so windows remain identifiable."""

    ts = test_start.date().isoformat()

    # Latent vectors
    Z = get_latent(model, X_train)
    df_z = meta_train_sub.copy().reset_index(drop=True)
    df_z.insert(0, "test_start", ts)
    for ki in range(LATENT_DIM):
        df_z[f"z_{ki+1}"] = Z[:, ki].numpy()
    _append_csv(df_z, LATENT_Z_CSV)

    # Parameters CSV
    df_p = extract_parameters(model, X_train, meta_train_sub)
    df_p.insert(0, "test_start", ts)
    _append_csv(df_p, PARAMETERS_CSV)

    # Predictions (actual vs fitted) for train and test windows
    tenor_cols = [f"tenor_{t}" for t in range(X_train.shape[1])]

    def _save_predictions(X: torch.Tensor, meta_sub: pd.DataFrame, path: str):
        S_hat = predict_S_hat(model, X)
        mask  = row_finite_mask(X) & row_finite_mask(S_hat)
        X_np  = X[mask].numpy()
        S_np  = S_hat[mask].numpy()
        m     = meta_sub.loc[mask.numpy()].reset_index(drop=True)
        df_pred = m[["as_of_date", "ccy"]].copy()
        df_pred.insert(0, "test_start", ts)
        for i, t in enumerate(tenor_cols):
            df_pred[f"actual_{t}"]  = X_np[:, i]
            df_pred[f"fitted_{t}"]  = S_np[:, i]
        _append_csv(df_pred, path)

    _save_predictions(X_train, meta_train_sub, PRED_TRAIN_CSV)
    if X_test is not None and meta_test is not None:
        _save_predictions(X_test, meta_test, PRED_TEST_CSV)

# ============================= Build rolling schedule ===============================
date_min = max(meta["as_of_date"].min(), pd.Timestamp("2010-01-01"))
date_max = meta["as_of_date"].max()

start = date_min + pd.DateOffset(years=TRAIN_YEARS)
end = date_max - pd.DateOffset(months=TEST_MONTHS)

roll_starts = []
d = start
while d <= end:
    roll_starts.append(d)
    d = d + pd.DateOffset(months=STEP_MONTHS)

if len(roll_starts) == 0:
    raise RuntimeError("No rolling windows possible. Dataset may be too short for chosen TRAIN_YEARS/TEST_MONTHS.")

print(f"Rolling windows: {len(roll_starts)} from {roll_starts[0].date()} to {roll_starts[-1].date()}")

# ============================= CSV header ===============================
cols = (
    ["roll_start", "train_start", "train_end", "test_start", "test_end",
     "n_train", "n_test", "n_test_good", "n_test_bad",
     "n_train_good", "n_train_bad",
     "time_train_sec", "time_test_sec",
     "avg_rmse_bps", "avg_in_rmse_bps"]
    + [f"rmse_bps_{ccy}" for ccy in ccy_order]
    + [f"in_rmse_bps_{ccy}" for ccy in ccy_order]
    + [f"eig_real_{i+1}" for i in range(LATENT_DIM)]
)
pd.DataFrame(columns=cols).to_csv(oos_csv_path, index=False)
print("OOS CSV:", oos_csv_path)

# ============================= Run manifest ===============================
manifest = {
    "window_seed": WINDOW_SEED,
    "latent_dim": LATENT_DIM,
    "variant": VARIANT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_lr": max_lr,
    "pct_start": 0.3,
    "final_div_factor": final_div_factor,
    "mkldnn_enabled": torch.backends.mkldnn.enabled,
    "train_years": TRAIN_YEARS,
    "test_months": TEST_MONTHS,
    "step_months": STEP_MONTHS,
    "n_windows": len(roll_starts),
    "torch_version": torch.__version__,
    "python_version": sys.version.split()[0],
    "numpy_version": np.__version__,
    "run_started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "window_results": {},
}
manifest_path = os.path.join(FIGURES_DIR, "run_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest initialised: {manifest_path}")

# ============================= Rolling loop ===============================
avg_rmse_curve = []     # list of (test_start_date, avg_oos_rmse_bps)
avg_in_rmse_curve = []  # list of (test_start_date, avg_in_rmse_bps)

for k, test_start in enumerate(roll_starts):
    train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
    train_end = test_start - pd.DateOffset(days=1)

    test_end = test_start + pd.DateOffset(months=TEST_MONTHS) - pd.DateOffset(days=1)

    m_train = (meta["as_of_date"] >= train_start) & (meta["as_of_date"] <= train_end)
    m_test = (meta["as_of_date"] >= test_start) & (meta["as_of_date"] <= test_end)

    n_train = int(m_train.sum())
    n_test = int(m_test.sum())

    print(f"\n[{k+1:02d}/{len(roll_starts)}] "
          f"train {train_start.date()}..{train_end.date()} (n={n_train}) | "
          f"test {test_start.date()}..{test_end.date()} (n={n_test})")

    if n_train < MIN_TRAIN_OBS or n_test == 0:
        print("  Skipping (too few observations).")
        continue

    X_train = X_tensor[m_train.values]
    X_test = X_tensor[m_test.values]
    meta_train_sub = meta.loc[m_train.values].reset_index(drop=True)
    meta_test = meta.loc[m_test.values].reset_index(drop=True)

    # Train
    t0 = time.perf_counter()
    model, train_mse_hist, lr_hist = train_one_window(X_train)
    t1 = time.perf_counter()

    # Eigenvalues of drift matrix M (real parts, sorted descending)
    with torch.no_grad():
        if hasattr(model.K, "lin"):
            M = model.K.lin.weight.cpu()
        else:
            M = model.K.stable_matrix().cpu()
        eig_reals = torch.linalg.eigvals(M).real.numpy()
        eig_reals = np.sort(eig_reals)[::-1]  # descending

    # In-sample RMSE (same model, evaluated on training window)
    in_rmse_per_ccy, avg_in_rmse_bps, n_train_good, n_train_bad = rmse_bps_on_subset(model, X_train, meta_train_sub)

    # Test
    t2 = time.perf_counter()
    rmse_per_ccy, avg_rmse_bps, n_good, n_bad = rmse_bps_on_subset(model, X_test, meta_test)
    t3 = time.perf_counter()

    time_train = t1 - t0
    time_test = t3 - t2

    avg_rmse_curve.append((test_start, avg_rmse_bps))
    avg_in_rmse_curve.append((test_start, avg_in_rmse_bps))

    # Row for CSV
    row = {
        "roll_start": test_start.date().isoformat(),
        "train_start": train_start.date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": test_end.date().isoformat(),
        "n_train": n_train,
        "n_test": n_test,
        "n_test_good": n_good,
        "n_test_bad": n_bad,
        "n_train_good": n_train_good,
        "n_train_bad": n_train_bad,
        "time_train_sec": time_train,
        "time_test_sec": time_test,
        "avg_rmse_bps": avg_rmse_bps,
        "avg_in_rmse_bps": avg_in_rmse_bps,
    }
    for ccy in ccy_order:
        row[f"rmse_bps_{ccy}"] = float(rmse_per_ccy.get(ccy, np.nan))
    for ccy in ccy_order:
        row[f"in_rmse_bps_{ccy}"] = float(in_rmse_per_ccy.get(ccy, np.nan))
    for i, ev in enumerate(eig_reals):
        row[f"eig_real_{i+1}"] = round(float(ev), 6)

    pd.DataFrame([row], columns=cols).to_csv(oos_csv_path, mode="a", header=False, index=False)

    # update manifest after every window (crash-safe)
    manifest["window_results"][test_start.date().isoformat()] = {
        "avg_rmse_bps":     round(avg_rmse_bps, 4),
        "avg_in_rmse_bps":  round(avg_in_rmse_bps, 4),
        "per_ccy_bps":      {ccy: round(float(rmse_per_ccy.get(ccy, np.nan)), 4) for ccy in ccy_order},
        "in_per_ccy_bps":   {ccy: round(float(in_rmse_per_ccy.get(ccy, np.nan)), 4) for ccy in ccy_order},
        "train_minutes":    round(time_train / 60, 2),
        "n_train":          n_train,
        "n_test":           n_test,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  IS  avg_in_rmse_bps={avg_in_rmse_bps:.3f} | OOS avg_rmse_bps={avg_rmse_bps:.3f} | time_train={time_train/60:.1f}min time_test={time_test:.1f}s")

    save_roll_outputs(model, X_train, meta_train_sub, test_start,
                      X_test=X_test, meta_test=meta_test)
    print(f"  Appended window {k+1:02d} to combined CSVs")

manifest["run_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest finalised: {manifest_path}")
print("\nRolling OOS done.")

# ============================= Plot overall OOS avg RMSE curve ===============================
if len(avg_rmse_curve) > 0:
    dates    = [d for d, v in avg_rmse_curve]
    oos_vals = [v for d, v in avg_rmse_curve]
    in_vals  = [v for d, v in avg_in_rmse_curve]

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=160)
    ax.plot(dates, in_vals,  marker="o", linewidth=1.0, label="IS RMSE (bps)")
    ax.plot(dates, oos_vals, marker="o", linewidth=1.0, label="OOS RMSE (bps)")
    ax.set_xlabel("Test window start date")
    ax.set_ylabel("Average RMSE (bps)")
    ax.set_title(f"Rolling IS vs OOS avg RMSE (bps) — train={TRAIN_YEARS}Y test={TEST_MONTHS}M step={STEP_MONTHS}M")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "oos_avg_rmse_bps_curve.png"), dpi=300)
    plt.close(fig)

    print("Saved IS vs OOS avg RMSE curve plot:", os.path.join(FIGURES_DIR, "oos_avg_rmse_bps_curve.png"))
else:
    print("No OOS curve points to plot (all windows skipped).")